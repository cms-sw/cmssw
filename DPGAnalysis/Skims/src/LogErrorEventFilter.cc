//
// $Id: LogErrorEventFilter.cc,v 1.4 2013/04/09 10:06:10 davidlt Exp $
//

/**
  \class    gLogErrorEventFilter PATMuonKinematics.h "PhysicsTools/PatAlgos/interface/PATMuonKinematics.h"
  \brief    Use StandAlone track to define the 4-momentum of a PAT Muon (normally the global one is used)
            
  \author   Giovanni Petrucciani
*/


#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

#include <map>
#include <set>
#include <string>
#include <iomanip>
#include <iostream>
#include <iterator>

namespace leef {
  using Error = edm::ErrorSummaryEntry;
  struct ErrorSort {
    bool operator()(const Error &e1, const Error &e2) const {
      if (e1.severity.getLevel() != e2.severity.getLevel()) return e1.severity.getLevel() > e2.severity.getLevel();
      if (e1.module != e2.module) return e1.module < e2.module;
      if (e1.category != e2.category) return e1.category < e2.category;
      return false;
    }
  };
  using ErrorSet = std::set<edm::ErrorSummaryEntry,ErrorSort>;

  struct RunErrors {
    RunErrors():
      npass_{0}, nfail_{0}, collectionGuard_{false} {}
    //Errors should be sufficiently infrequent so that the use of a
    // spin lock on a thread-unsafe container should not pose a
    // performance problem
    CMS_THREAD_GUARD(collectionGuard_) mutable ErrorSet errorCollection_;
    mutable std::atomic<size_t> npass_;
    mutable std::atomic<size_t> nfail_;
    
    mutable std::atomic<bool> collectionGuard_;
  };

  struct LumiErrors {
    LumiErrors():
      npass_{0}, nfail_{0}, collectionGuard_{false} {}

    CMS_THREAD_GUARD(collectionGuard_) mutable ErrorSet errorCollection_;
    mutable std::atomic<size_t> npass_;
    mutable std::atomic<size_t> nfail_;

    mutable std::atomic<bool> collectionGuard_;
  };
}

namespace {
  //Use std::unique_ptr to guarantee that the use of an atomic
  // as a spin lock will get reset to false
  struct release {
    void operator()(std::atomic<bool>* b) const noexcept { b->store(false); }
  };
  std::unique_ptr<std::atomic<bool>, release> make_guard(std::atomic<bool>& b) noexcept { 
    bool expected = false;
    while( not b.compare_exchange_strong(expected,true) );
    
    return std::unique_ptr<std::atomic<bool>, release>(&b,release()); 
  }

}

using namespace leef;

class LogErrorEventFilter : public edm::global::EDFilter<edm::RunCache<leef::RunErrors>,
                                                         edm::LuminosityBlockCache<LumiErrors>,
                                                         edm::EndLuminosityBlockProducer> {
    public:
        explicit LogErrorEventFilter(const edm::ParameterSet & iConfig);
        ~LogErrorEventFilter() override { }

        bool filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const override;
        std::shared_ptr<LumiErrors> globalBeginLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) const override;
        void globalEndLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) const override;
        void globalEndLuminosityBlockProduce(edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) const override;
        std::shared_ptr<RunErrors> globalBeginRun(const edm::Run &run, const edm::EventSetup &iSetup) const override;
        void globalEndRun(const edm::Run &run, const edm::EventSetup &iSetup) const override;
        void endJob() override;

    private:
        typedef edm::ErrorSummaryEntry              Error;
        typedef std::vector<edm::ErrorSummaryEntry> ErrorList;

        edm::InputTag src_;
        edm::EDGetTokenT<ErrorList> srcT_;
        bool readSummaryMode_;
        std::set<std::string> modulesToWatch_;
        std::set<std::string> modulesToIgnore_;
        std::set<std::string> categoriesToWatch_;
        std::set<std::string> categoriesToIgnore_;
        CMS_THREAD_GUARD(statsGuard_) mutable std::map<std::pair<uint32_t,uint32_t>, std::pair<size_t,size_t> > statsPerLumi_;
        CMS_THREAD_GUARD(statsGuard_) mutable std::map<uint32_t, std::pair<size_t,size_t> > statsPerRun_;
        CMS_THREAD_GUARD(statsGuard_) mutable ErrorSet errorCollectionAll_;
        double thresholdPerLumi_, thresholdPerRun_;
        size_t maxSavedEventsPerLumi_;
        bool verbose_, veryVerbose_;
        bool taggedMode_, forcedValue_;
        mutable std::atomic<bool> statsGuard_;

        template<typename Collection> static void increment(ErrorSet &scoreboard, Collection &list);
        template<typename Collection> static void print(const Collection &errors) ;

        static std::unique_ptr<ErrorList > serialize(const ErrorSet &set) {
            return std::make_unique<ErrorList>(set.begin(), set.end());
        }
};

LogErrorEventFilter::LogErrorEventFilter(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    srcT_(consumes<ErrorList>(iConfig.getParameter<edm::InputTag>("src"))),
    readSummaryMode_(iConfig.existsAs<bool>("readSummaryMode") ? iConfig.getParameter<bool>("readSummaryMode") : false),
    thresholdPerLumi_(iConfig.getParameter<double>("maxErrorFractionInLumi")),
    thresholdPerRun_(iConfig.getParameter<double>("maxErrorFractionInRun")),
    maxSavedEventsPerLumi_(iConfig.getParameter<uint32_t>("maxSavedEventsPerLumiAndError")),
    verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
    veryVerbose_(iConfig.getUntrackedParameter<bool>("veryVerbose", false)),
    taggedMode_(iConfig.getUntrackedParameter<bool>("taggedMode", false)),
    forcedValue_(iConfig.getUntrackedParameter<bool>("forcedValue", true))
{
    produces<ErrorList, edm::Transition::EndLuminosityBlock>();
    produces<int, edm::Transition::EndLuminosityBlock>("pass");
    produces<int, edm::Transition::EndLuminosityBlock>("fail");
    //produces<ErrorList, edm::InRun>();
    produces<bool>();

    if (iConfig.existsAs<std::vector<std::string> >("modulesToWatch")) {
        std::vector<std::string> modules = iConfig.getParameter<std::vector<std::string> >("modulesToWatch");
        if (!(modules.size() == 1 && modules[0] == "*")) {
            modulesToWatch_.insert(modules.begin(), modules.end());
        }
    }
    if (iConfig.existsAs<std::vector<std::string> >("modulesToIgnore")) {
        std::vector<std::string> modules = iConfig.getParameter<std::vector<std::string> >("modulesToIgnore");
        if (!(modules.size() == 1 && modules[0] == "*")) {
            modulesToIgnore_.insert(modules.begin(), modules.end());
        }
    }
    if (iConfig.existsAs<std::vector<std::string> >("categoriesToWatch")) {
        std::vector<std::string> categories = iConfig.getParameter<std::vector<std::string> >("categoriesToWatch");
        if (!(categories.size() == 1 && categories[0] == "*")) {
            categoriesToWatch_.insert(categories.begin(), categories.end());
        }
    }
    if (iConfig.existsAs<std::vector<std::string> >("categoriesToIgnore")) {
        std::vector<std::string> categories = iConfig.getParameter<std::vector<std::string> >("categoriesToIgnore");
        if (!(categories.size() == 1 && categories[0] == "*")) {
            categoriesToIgnore_.insert(categories.begin(), categories.end());
        }
    }
    //std::ostream_iterator<std::string> dump(std::cout, ", ");
    //std::cout << "\nWatch modules:     " ; std::copy(modulesToWatch_.begin(),     modulesToWatch_.end(),     dump);
    //std::cout << "\nIgnore modules:    " ; std::copy(modulesToIgnore_.begin(),    modulesToIgnore_.end(),    dump);
    //std::cout << "\nIgnore categories: " ; std::copy(categoriesToIgnore_.begin(), categoriesToIgnore_.end(), dump);
    //std::cout << "\nWatch categories:  " ; std::copy(categoriesToWatch_.begin(),  categoriesToWatch_.end(),  dump);
    //std::cout << std::endl;

}

std::shared_ptr<LumiErrors>
LogErrorEventFilter::globalBeginLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) const{
  auto ret = std::make_shared<LumiErrors>();
    if (readSummaryMode_) {
        edm::Handle<ErrorList> handle;
        edm::Handle<int> hpass, hfail;
        lumi.getByLabel(src_, handle);
        lumi.getByLabel(edm::InputTag(src_.label(), "pass", src_.process()), hpass);
        lumi.getByLabel(edm::InputTag(src_.label(), "fail", src_.process()), hfail);
        increment(ret->errorCollection_, *handle);
        ret->npass_ = *hpass;
        ret->nfail_ = *hfail;

        auto runC = runCache(lumi.getRun().index());
        runC->npass_ +=*hpass;
        runC->nfail_ += *hfail;
    }
    return ret;
}
void
LogErrorEventFilter::globalEndLuminosityBlock(edm::LuminosityBlock const &lumi, const edm::EventSetup &iSetup) const {
   auto lumiC = luminosityBlockCache(lumi.index());
   auto nfail = lumiC->nfail_.load();
   auto npass = lumiC->npass_.load();
   {
     auto guard = make_guard(statsGuard_);
     statsPerLumi_[std::pair<uint32_t,uint32_t>(lumi.run(), lumi.luminosityBlock())] = std::pair<size_t,size_t>(npass, nfail);
   }
   {
     //synchronize lumiC->errorCollection_
     auto guard = make_guard(lumiC->collectionGuard_);
     
     {
       if (nfail < thresholdPerLumi_*(npass+nfail)) {
         //synchronize runC->errorCollection_
         auto runC = runCache(lumi.getRun().index()); 
         auto guard = make_guard(runC->collectionGuard_);
         increment(runC->errorCollection_, lumiC->errorCollection_);
       }
     }
     if (verbose_) {
       if (!lumiC->errorCollection_.empty()) {
         std::cout << "\n === REPORT FOR RUN " << lumi.run() << " LUMI " << lumi.luminosityBlock() << " === " << std::endl;
         print(lumiC->errorCollection_);
       }
     }
   }
}

void
LogErrorEventFilter::globalEndLuminosityBlockProduce(edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) const {
    auto lumiC = luminosityBlockCache(lumi.index());
    {
        //synchronize errorCollection_
        auto guard = make_guard(lumiC->collectionGuard_);
        lumi.put(serialize(lumiC->errorCollection_));
    }
    lumi.put(std::make_unique<int>(lumiC->npass_.load()), "pass");
    lumi.put(std::make_unique<int>(lumiC->nfail_.load()), "fail");
}


std::shared_ptr<RunErrors>
LogErrorEventFilter::globalBeginRun(const edm::Run &run, const edm::EventSetup &iSetup) const {
    return std::make_shared<RunErrors>();
}

void
LogErrorEventFilter::globalEndRun(const edm::Run &run, const edm::EventSetup &iSetup) const {
    auto runC = runCache(run.index());
    auto npass = runC->npass_.load();
    auto nfail = runC->nfail_.load();
    {
      auto guard = make_guard(statsGuard_);
      statsPerRun_[run.run()] = std::pair<size_t,size_t>(npass, nfail);
    }
    {
        //synchronize errorCollection_
        auto guard = make_guard(runC->collectionGuard_);
        if (nfail < thresholdPerRun_*(npass+nfail)) {
          auto guard = make_guard(statsGuard_);
          increment(errorCollectionAll_, runC->errorCollection_);
        }
        if (verbose_) {
            if (!runC->errorCollection_.empty()) {
                std::cout << "\n === REPORT FOR RUN " << run.run() << " === " << std::endl;
                print(runC->errorCollection_);
            }
        }
    }
}

void
LogErrorEventFilter::endJob() {
    if (verbose_) {
        std::cout << "\n === REPORT FOR JOB === " << std::endl;
        //synchronizes statsPerRun_ and errorCollectionAll_
        auto guard = make_guard(statsGuard_);
        print(errorCollectionAll_);
        
        typedef std::pair<size_t,size_t> counter;

        std::cout << "\n === SCOREBOARD PER RUN === " << std::endl;
        typedef std::pair<uint32_t, counter> hitRun;
        for(auto const& hit : statsPerRun_) {
            double fract = hit.second.second/double(hit.second.first + hit.second.second);
            printf("run %6d: fail %7zu, pass %7zu, fraction %7.3f%%%s\n", hit.first, hit.second.second, hit.second.first, fract*100., (fract >= thresholdPerRun_ ? " (run excluded from summary list)" : ""));
        }
 
        std::cout << "\n === SCOREBOARD PER LUMI === " << std::endl;
        typedef std::pair<std::pair<uint32_t,uint32_t>, counter> hitLumi;
        for(auto const& hit : statsPerLumi_) {
            double fract = hit.second.second/double(hit.second.first + hit.second.second);
            printf("run %6d, lumi %4d: fail %zu, pass %zu, fraction %7.3f%%%s\n", hit.first.first, hit.first.second, hit.second.second, hit.second.first, fract*100., (fract >= thresholdPerLumi_ ? " (lumi excluded from run list)" : ""));
        }
    }
}

bool 
LogErrorEventFilter::filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {
    if (readSummaryMode_) return true;

    bool fail = false, save = false;

    edm::Handle<ErrorList> errors;
    iEvent.getByToken(srcT_, errors);

    auto runC = runCache(iEvent.getRun().index()); 
    auto lumiC = luminosityBlockCache(iEvent.getLuminosityBlock().index());

    if (errors->empty()) {
        ++(runC->npass_);
        ++(lumiC->npass_);
	iEvent.put(std::make_unique<bool>(false));

	if(taggedMode_) return forcedValue_;
        return false;
    } 

    for(auto const& err : *errors) {
        if (!modulesToWatch_.empty()     && (modulesToWatch_.count(err.module)       == 0)) continue;
        if (!categoriesToWatch_.empty()  && (categoriesToWatch_.count(err.category)  == 0)) continue;
        if (!modulesToIgnore_.empty()    && (modulesToIgnore_.count(err.module)      != 0)) continue;
        if (!categoriesToIgnore_.empty() && (categoriesToIgnore_.count(err.category) != 0)) continue;
        
        fail = true;
        
        //synchronize errorCollection_
        auto guard = make_guard(lumiC->collectionGuard_);
        
        std::pair<ErrorSet::iterator, bool> result = lumiC->errorCollection_.insert(err);
        if (!result.second) { // already there
            // need the const_cast as set elements are const
            const_cast<unsigned int &>(result.first->count) += err.count;
            if (result.first->count < maxSavedEventsPerLumi_) save = true;
        } else {
            save = true;
        }
        
    }
    if (save && veryVerbose_) {
        std::cout << "\n === REPORT FOR EVENT " << iEvent.id().event() << " RUN " << iEvent.run() << " LUMI " << iEvent.luminosityBlock() << " === " << std::endl;
        print(*errors);
    }


    if (fail) { ++(lumiC->nfail_); ++(runC->nfail_); } else { ++(runC->npass_); ++(lumiC->npass_); }
    iEvent.put(std::make_unique<bool>(fail));  // fail is the unbiased boolean 

    if(taggedMode_) return forcedValue_;
    return save;
}

template<typename Collection> 
void
LogErrorEventFilter::increment(ErrorSet &scoreboard, Collection &list) {
    for(auto const& err : list) {
        std::pair<ErrorSet::iterator, bool> result = scoreboard.insert(err);
        // need the const_cast as set elements are const
         if (!result.second) const_cast<unsigned int &>(result.first->count) += err.count;
    }
}

template<typename Collection> 
void
LogErrorEventFilter::print(const Collection &errors) {
    using namespace std;
    cout << setw(40) << left  << "Category" << " " <<
            setw(60) << left  << "Module"   << " " <<
            setw(10) << left  << "Level"    << " " <<
            setw(9)  << right << "Count"    << "\n";
    cout << setw(40) << left  << "----------------------------------------"                       << " " <<
            setw(60) << left  << "------------------------------------------------------------"   << " " <<
            setw(10) << left  << "----------"                                                     << " " <<
            setw(9)  << right << "---------"                                                      << "\n";
    for(auto const& err : errors) {
        cout << setw(40) << left  << err.category           << " " <<
                setw(60) << left  << err.module             << " " <<
                setw(10) << left  << err.severity.getName() << " " <<
                setw(9)  << right << err.count << "\n";
    }
    cout << flush;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LogErrorEventFilter);
