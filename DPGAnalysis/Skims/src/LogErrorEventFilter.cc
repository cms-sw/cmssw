//
// $Id: LogErrorEventFilter.cc,v 1.5 2013/05/17 21:41:49 chrjones Exp $
//

/**
  \class    gLogErrorEventFilter PATMuonKinematics.h "PhysicsTools/PatAlgos/interface/PATMuonKinematics.h"
  \brief    Use StandAlone track to define the 4-momentum of a PAT Muon (normally the global one is used)
            
  \author   Giovanni Petrucciani
  \version  $Id: LogErrorEventFilter.cc,v 1.5 2013/05/17 21:41:49 chrjones Exp $
*/


#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

#include <map>
#include <set>
#include <string>
#include <iomanip>
#include <iterator>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH


class LogErrorEventFilter : public edm::one::EDFilter<edm::one::WatchRuns,
                                                      edm::one::WatchLuminosityBlocks,
                                                      edm::EndLuminosityBlockProducer> {
    public:
        explicit LogErrorEventFilter(const edm::ParameterSet & iConfig);
        virtual ~LogErrorEventFilter() { }

        virtual bool filter(edm::Event & iEvent, const edm::EventSetup& iSetup) override;
        virtual void beginLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) override;
        virtual void endLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) override;
        virtual void endLuminosityBlockProduce(edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) override;
        virtual void beginRun(const edm::Run &run, const edm::EventSetup &iSetup) override;
        virtual void endRun(const edm::Run &run, const edm::EventSetup &iSetup) override;
        virtual void endJob();

    private:
        typedef edm::ErrorSummaryEntry              Error;
        struct ErrorSort {
            bool operator()(const Error &e1, const Error &e2) {
                if (e1.severity.getLevel() != e2.severity.getLevel()) return e1.severity.getLevel() > e2.severity.getLevel();
                if (e1.module != e2.module) return e1.module < e2.module;
                if (e1.category != e2.category) return e1.category < e2.category;
                return false;
            }
        };
        typedef std::vector<edm::ErrorSummaryEntry> ErrorList;
        typedef std::set<edm::ErrorSummaryEntry,ErrorSort>    ErrorSet;

        edm::InputTag src_;
        bool readSummaryMode_;
        size_t npassLumi_, nfailLumi_;
        size_t npassRun_, nfailRun_;
        std::set<std::string> modulesToWatch_;
        std::set<std::string> modulesToIgnore_;
        std::set<std::string> categoriesToWatch_;
        std::set<std::string> categoriesToIgnore_;
        std::map<std::pair<uint32_t,uint32_t>, std::pair<size_t,size_t> > statsPerLumi_;
        std::map<uint32_t, std::pair<size_t,size_t> > statsPerRun_;
        ErrorSet errorCollectionAll_;
        ErrorSet errorCollectionThisLumi_, errorCollectionThisRun_;
        double thresholdPerLumi_, thresholdPerRun_;
        size_t maxSavedEventsPerLumi_;
        bool verbose_, veryVerbose_;
        bool taggedMode_, forcedValue_;

        template<typename Collection> static void increment(ErrorSet &scoreboard, Collection &list);
        template<typename Collection> static void print(const Collection &errors) ;

        static std::auto_ptr<ErrorList > serialize(const ErrorSet &set) {
            std::auto_ptr<ErrorList> ret(new ErrorList(set.begin(), set.end()));
            return ret;
        }
};

LogErrorEventFilter::LogErrorEventFilter(const edm::ParameterSet & iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    readSummaryMode_(iConfig.existsAs<bool>("readSummaryMode") ? iConfig.getParameter<bool>("readSummaryMode") : false),
    thresholdPerLumi_(iConfig.getParameter<double>("maxErrorFractionInLumi")),
    thresholdPerRun_(iConfig.getParameter<double>("maxErrorFractionInRun")),
    maxSavedEventsPerLumi_(iConfig.getParameter<uint32_t>("maxSavedEventsPerLumiAndError")),
    verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
    veryVerbose_(iConfig.getUntrackedParameter<bool>("veryVerbose", false)),
    taggedMode_(iConfig.getUntrackedParameter<bool>("taggedMode", false)),
    forcedValue_(iConfig.getUntrackedParameter<bool>("forcedValue", true))
{
    produces<ErrorList, edm::InLumi>();
    produces<int, edm::InLumi>("pass");
    produces<int, edm::InLumi>("fail");
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
    std::ostream_iterator<std::string> dump(std::cout, ", ");
    std::cout << "\nWatch modules:     " ; std::copy(modulesToWatch_.begin(),     modulesToWatch_.end(),     dump);
    std::cout << "\nIgnore modules:    " ; std::copy(modulesToIgnore_.begin(),    modulesToIgnore_.end(),    dump);
    std::cout << "\nIgnore categories: " ; std::copy(categoriesToIgnore_.begin(), categoriesToIgnore_.end(), dump);
    std::cout << "\nWatch categories:  " ; std::copy(categoriesToWatch_.begin(),  categoriesToWatch_.end(),  dump);
    std::cout << std::endl;

}

void
LogErrorEventFilter::beginLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) {
    npassLumi_ = 0; nfailLumi_ = 0;
    errorCollectionThisLumi_.clear();
    if (readSummaryMode_) {
        edm::Handle<ErrorList> handle;
        edm::Handle<int> hpass, hfail;
        lumi.getByLabel(src_, handle);
        lumi.getByLabel(edm::InputTag(src_.label(), "pass", src_.process()), hpass);
        lumi.getByLabel(edm::InputTag(src_.label(), "fail", src_.process()), hfail);
        increment(errorCollectionThisLumi_, *handle);
        npassLumi_ = *hpass;
        nfailLumi_ = *hfail;
        npassRun_ += npassLumi_;
        nfailRun_ += nfailLumi_;
    }
}
void
LogErrorEventFilter::endLuminosityBlock(edm::LuminosityBlock const &lumi, const edm::EventSetup &iSetup) {
   statsPerLumi_[std::pair<uint32_t,uint32_t>(lumi.run(), lumi.luminosityBlock())] = std::pair<size_t,size_t>(npassLumi_, nfailLumi_);
   if (nfailLumi_ < thresholdPerLumi_*(npassLumi_+nfailLumi_)) {
       increment(errorCollectionThisRun_, errorCollectionThisLumi_);
   }
   if (verbose_) {
       if (!errorCollectionThisLumi_.empty()) {
           std::cout << "\n === REPORT FOR RUN " << lumi.run() << " LUMI " << lumi.luminosityBlock() << " === " << std::endl;
           print(errorCollectionThisLumi_);
       }
   }
}

void
LogErrorEventFilter::endLuminosityBlockProduce(edm::LuminosityBlock &lumi, const edm::EventSetup &iSetup) {
    lumi.put(serialize(errorCollectionThisLumi_));
    std::auto_ptr<int> outpass(new int(npassLumi_)); lumi.put(outpass, "pass");
    std::auto_ptr<int> outfail(new int(nfailLumi_)); lumi.put(outfail, "fail");
}


void
LogErrorEventFilter::beginRun(const edm::Run &run, const edm::EventSetup &iSetup) {
    npassRun_ = 0; nfailRun_ = 0;
    errorCollectionThisRun_.clear();
}

void
LogErrorEventFilter::endRun(const edm::Run &run, const edm::EventSetup &iSetup) {
    statsPerRun_[run.run()] = std::pair<size_t,size_t>(npassRun_, nfailRun_);
    if (nfailRun_ < thresholdPerRun_*(npassRun_+nfailRun_)) {
        increment(errorCollectionAll_, errorCollectionThisRun_);
    }
    if (verbose_) {
        if (!errorCollectionThisRun_.empty()) {
            std::cout << "\n === REPORT FOR RUN " << run.run() << " === " << std::endl;
            print(errorCollectionThisRun_);
        }
    }
    //run.put(serialize(errorCollectionThisRun_));
}

void
LogErrorEventFilter::endJob() {
    if (verbose_) {
        std::cout << "\n === REPORT FOR JOB === " << std::endl;
        print(errorCollectionAll_);
        
        typedef std::pair<size_t,size_t> counter;

        std::cout << "\n === SCOREBOARD PER RUN === " << std::endl;
        typedef std::pair<uint32_t, counter> hitRun;
        foreach(const hitRun &hit, statsPerRun_) {
            double fract = hit.second.second/double(hit.second.first + hit.second.second);
            printf("run %6d: fail %7zu, pass %7zu, fraction %7.3f%%%s\n", hit.first, hit.second.second, hit.second.first, fract*100., (fract >= thresholdPerRun_ ? " (run excluded from summary list)" : ""));
        }
 
        std::cout << "\n === SCOREBOARD PER LUMI === " << std::endl;
        typedef std::pair<std::pair<uint32_t,uint32_t>, counter> hitLumi;
        foreach(const hitLumi &hit, statsPerLumi_) {
            double fract = hit.second.second/double(hit.second.first + hit.second.second);
            printf("run %6d, lumi %4d: fail %zu, pass %zu, fraction %7.3f%%%s\n", hit.first.first, hit.first.second, hit.second.second, hit.second.first, fract*100., (fract >= thresholdPerLumi_ ? " (lumi excluded from run list)" : ""));
        }
    }
}

bool 
LogErrorEventFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    if (readSummaryMode_) return true;

    bool fail = false, save = false;

    edm::Handle<ErrorList> errors;
    iEvent.getByLabel(src_, errors);

   
    if (errors->empty()) { 
        npassRun_++; npassLumi_++;
	iEvent.put( std::auto_ptr<bool>(new bool(false)) );

	if(taggedMode_) return forcedValue_;
        return false;
    } 

    foreach (const Error &err, *errors) {
        if (!modulesToWatch_.empty()     && (modulesToWatch_.count(err.module)       == 0)) continue;
        if (!categoriesToWatch_.empty()  && (categoriesToWatch_.count(err.category)  == 0)) continue;
        if (!modulesToIgnore_.empty()    && (modulesToIgnore_.count(err.module)      != 0)) continue;
        if (!categoriesToIgnore_.empty() && (categoriesToIgnore_.count(err.category) != 0)) continue;
        
        fail = true;
        std::pair<ErrorSet::iterator, bool> result = errorCollectionThisLumi_.insert(err);
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


    if (fail) { nfailLumi_++; nfailRun_++; } else { npassRun_++; npassLumi_++; }
    iEvent.put( std::auto_ptr<bool>(new bool(fail)) );  // fail is the unbiased boolean 

    if(taggedMode_) return forcedValue_;
    return save;
}

template<typename Collection> 
void
LogErrorEventFilter::increment(ErrorSet &scoreboard, Collection &list) {
    foreach (const Error &err, list) {
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
    foreach (const Error &err, errors) {
        cout << setw(40) << left  << err.category           << " " <<
                setw(60) << left  << err.module             << " " <<
                setw(10) << left  << err.severity.getName() << " " <<
                setw(9)  << right << err.count << "\n";
    }
    cout << flush;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LogErrorEventFilter);
