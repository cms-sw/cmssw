//
//

/**
  \class    pat::CandidateSummaryTable CandidateSummaryTable.h "PhysicsTools/PatAlgos/interface/CandidateSummaryTable.h"
  \brief    Produce a summary table of some candidate collections

  FIXME FIXME Move to CandAlgos

  \author   Giovanni Petrucciani
  \version  $Id: CandidateSummaryTable.cc,v 1.4 2010/02/20 21:00:15 wmtan Exp $
*/


#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <atomic>

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace pathelpers {
  struct Record {
    const edm::InputTag src;
    mutable std::atomic<size_t> present, empty, min, max, total;
    Record(const edm::InputTag& tag) : src(tag), present(0), empty(0), min(0), max(0), total(0) {}
    Record(const Record& other) : src(other.src), 
                                  present(other.present.load()), 
                                  empty(other.empty.load()), 
                                  min(other.min.load()), 
                                  max(other.max.load()), 
                                  total(other.total.load()) {}
    void update(const edm::View<reco::Candidate> &items) const {
      present++;
      const size_t size = items.size();
      if (size == 0) {
        empty++;
      } else  {
        auto previousMin = min.load();
        while( previousMin > size and not min.compare_exchange_weak(previousMin, size) ) {}
        auto previousMax = max.load();
        while( previousMax < size and not max.compare_exchange_weak(previousMax, size) ) {}        
      }
      total += size;
    }
  };

  struct RecordCache { 
    RecordCache(const edm::ParameterSet& iConfig) : 
      perEvent_(iConfig.getUntrackedParameter<bool>("perEvent", false)),
      perJob_(iConfig.getUntrackedParameter<bool>("perJob", true)),
      self_(iConfig.getParameter<std::string>("@module_label")),
      logName_(iConfig.getUntrackedParameter<std::string>("logName")),
      dumpItems_(iConfig.getUntrackedParameter<bool>("dumpItems", false)),
      totalEvents_(0) {
      const std::vector<edm::InputTag>& tags  = iConfig.getParameter<std::vector<edm::InputTag> >("candidates");
      for( const auto& tag : tags ) {
        collections_.emplace_back( tag );
      }
    }
    const bool perEvent_, perJob_;
    const std::string self_, logName_;
    const bool dumpItems_;
    mutable std::atomic<size_t> totalEvents_;
    std::vector<Record> collections_; // size of vector is never altered later! (atomics are in the class below)
  };  

}

namespace pat {
  class CandidateSummaryTable : public edm::stream::EDAnalyzer<edm::GlobalCache<pathelpers::RecordCache> > {
  public:
    explicit CandidateSummaryTable(const edm::ParameterSet & iConfig, const pathelpers::RecordCache*);
    ~CandidateSummaryTable();
    
    static std::unique_ptr<pathelpers::RecordCache> initializeGlobalCache(edm::ParameterSet const& conf) {
      return std::make_unique<pathelpers::RecordCache>(conf);
    }

    virtual void analyze(const edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    static void globalEndJob(const pathelpers::RecordCache*);
    
  private:    
    std::vector<std::pair<edm::InputTag,edm::EDGetTokenT<edm::View<reco::Candidate> > > > srcTokens;
  };
} // namespace

pat::CandidateSummaryTable::CandidateSummaryTable(const edm::ParameterSet & iConfig, const pathelpers::RecordCache*) {
    const std::vector<edm::InputTag>& inputs = iConfig.getParameter<std::vector<edm::InputTag> >("candidates");
    for (std::vector<edm::InputTag>::const_iterator it = inputs.begin(); it != inputs.end(); ++it) {      
      srcTokens.emplace_back(*it, consumes<edm::View<reco::Candidate> >(*it));
    }
}

pat::CandidateSummaryTable::~CandidateSummaryTable() {
}

void
pat::CandidateSummaryTable::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup) {
  using namespace edm;
  using std::setw; using std::left; using std::right; using std::setprecision;

  Handle<View<reco::Candidate> > candidates;
  if (globalCache()->perEvent_) {
    LogInfo(globalCache()->logName_) << "Per Event Table " << globalCache()->logName_ 
                                     << " (" << globalCache()->self_ << ", run:event " 
                                     << iEvent.id().run() << ":" << iEvent.id().event() << ")";
  }
  ++(globalCache()->totalEvents_);
  auto& collections = globalCache()->collections_;
  auto tags = srcTokens.cbegin();
  for (auto it = collections.begin(), ed = collections.end(); it != ed; ++it, ++tags) {
    iEvent.getByToken(tags->second, candidates);
    if (!candidates.failedToGet()) it->update(*candidates);
    if (globalCache()->perEvent_) {
        LogVerbatim(globalCache()->logName_) << "    " << setw(30) << left  << it->src.encode() << right;
        if (globalCache()->dumpItems_) {
            size_t i = 0;
            std::ostringstream oss;
            for (View<reco::Candidate>::const_iterator cand = candidates->begin(), endc = candidates->end(); cand != endc; ++cand, ++i) {
                oss << "      [" << setw(3) << i << "]" <<
                        "  pt "  << setw(7) << setprecision(5) << cand->pt()  <<
                        "  eta " << setw(7) << setprecision(5) << cand->eta() <<
                        "  phi " << setw(7) << setprecision(5) << cand->phi() <<
                        "  et "  << setw(7) << setprecision(5) << cand->et()  <<
                        "  phi " << setw(7) << setprecision(5) << cand->phi() <<
                        "  charge " << setw(2) << cand->charge() <<
                        "  id "     << setw(7) << cand->pdgId() <<
                        "  st "     << setw(7) << cand->status() << "\n";
            }
            LogVerbatim(globalCache()->logName_) << oss.str();
        }
    }
  }
  if (globalCache()->perEvent_) LogInfo(globalCache()->logName_) << "" ;  // add an empty line
}


void
pat::CandidateSummaryTable::globalEndJob(const pathelpers::RecordCache* rcd) {
    using std::setw; using std::left; using std::right; using std::setprecision;
    if (rcd->perJob_) {
        std::ostringstream oss;
        oss << "Summary Table " << rcd->logName_ << " (" << rcd->self_ << ", events total " << rcd->totalEvents_ << ")\n";
        for (auto it = rcd->collections_.cbegin(), ed = rcd->collections_.cend(); it != ed; ++it) {
            oss << "    " << setw(30) << left  << it->src.encode() << right <<
                "  present " << setw(7) << it->present << " (" << setw(4) << setprecision(3) << (it->present*100.0/rcd->totalEvents_) << "%)" <<
                "  empty "   << setw(7) << it->empty   << " (" << setw(4) << setprecision(3) << (it->empty*100.0/rcd->totalEvents_)   << "%)" <<
                "  min "     << setw(7) << it->min     <<
                "  max "     << setw(7) << it->max     <<
                "  total "   << setw(7) << it->total   <<
                "  avg "     << setw(5) << setprecision(3) << (it->total/double(rcd->totalEvents_)) << "\n";
        }
        oss << "\n";
        edm::LogVerbatim(rcd->logName_) << oss.str();
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
using pat::CandidateSummaryTable;
DEFINE_FWK_MODULE(CandidateSummaryTable);
