//
// $Id: CandidateSummaryTable.cc,v 1.5 2013/02/27 23:26:56 wmtan Exp $
//

/**
  \class    pat::CandidateSummaryTable CandidateSummaryTable.h "PhysicsTools/PatAlgos/interface/CandidateSummaryTable.h"
  \brief    Produce a summary table of some candidate collections

  FIXME FIXME Move to CandAlgos

  \author   Giovanni Petrucciani
  \version  $Id: CandidateSummaryTable.cc,v 1.5 2013/02/27 23:26:56 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace pat {
  class CandidateSummaryTable : public edm::EDAnalyzer {
    public:
      explicit CandidateSummaryTable(const edm::ParameterSet & iConfig);
      ~CandidateSummaryTable();  

      virtual void analyze(const edm::Event & iEvent, const edm::EventSetup& iSetup) override;
      virtual void endJob();

    private:
      struct Record {
        edm::InputTag src;
        size_t present, empty, min, max, total;
        Record(edm::InputTag tag) : src(tag), present(0), empty(0), min(0), max(0), total(0) {}

        void update(const edm::View<reco::Candidate> &items) {
            present++; 
            size_t size = items.size();
            if (size == 0) {
                empty++;
            } else  { 
                if (min > size) min = size;
                if (max < size) max = size;
            }
            total += size;
        }
      };
      std::vector<Record> collections_;
      size_t totalEvents_;
      bool perEvent_, perJob_;
      std::string self_, logName_;
      bool dumpItems_;
  };

} // namespace

pat::CandidateSummaryTable::CandidateSummaryTable(const edm::ParameterSet & iConfig) :
    totalEvents_(0),
    perEvent_(iConfig.getUntrackedParameter<bool>("perEvent", false)),
    perJob_(iConfig.getUntrackedParameter<bool>("perJob", true)),
    self_(iConfig.getParameter<std::string>("@module_label")),
    logName_(iConfig.getUntrackedParameter<std::string>("logName")),
    dumpItems_(iConfig.getUntrackedParameter<bool>("dumpItems", false))
{
    std::vector<edm::InputTag> inputs = iConfig.getParameter<std::vector<edm::InputTag> >("candidates");
    for (std::vector<edm::InputTag>::const_iterator it = inputs.begin(); it != inputs.end(); ++it) {
        collections_.push_back(Record(*it));
    }
}

pat::CandidateSummaryTable::~CandidateSummaryTable() {
}

void 
pat::CandidateSummaryTable::analyze(const edm::Event & iEvent, const edm::EventSetup & iSetup) {
  using namespace edm;
  using std::setw; using std::left; using std::right; using std::setprecision;

  Handle<View<reco::Candidate> > candidates;
  if (perEvent_) {
        LogInfo(logName_) << "Per Event Table " << logName_ <<
                             " (" << self_ << ", run:event " << iEvent.id().run() << ":" << iEvent.id().event() << ")";
  }
  totalEvents_++;
  for (std::vector<Record>::iterator it = collections_.begin(), ed = collections_.end(); it != ed; ++it) {
    iEvent.getByLabel(it->src, candidates);
    if (!candidates.failedToGet()) it->update(*candidates);
    if (perEvent_) {
        LogVerbatim(logName_) << "    " << setw(30) << left  << it->src.encode() << right;
        if (dumpItems_) {
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
            LogVerbatim(logName_) << oss.str();
        }
    }
  }
  if (perEvent_) LogInfo(logName_) << "" ;  // add an empty line
}


void 
pat::CandidateSummaryTable::endJob() { 
    using std::setw; using std::left; using std::right; using std::setprecision;
    if (perJob_) {
        std::ostringstream oss;
        oss << "Summary Table " << logName_ << " (" << self_ << ", events total " << totalEvents_ << ")\n";
        for (std::vector<Record>::iterator it = collections_.begin(), ed = collections_.end(); it != ed; ++it) {
            oss << "    " << setw(30) << left  << it->src.encode() << right << 
                "  present " << setw(7) << it->present << " (" << setw(4) << setprecision(3) << (it->present*100.0/totalEvents_) << "%)" << 
                "  empty "   << setw(7) << it->empty   << " (" << setw(4) << setprecision(3) << (it->empty*100.0/totalEvents_)   << "%)" << 
                "  min "     << setw(7) << it->min     <<  
                "  max "     << setw(7) << it->max     <<  
                "  total "   << setw(7) << it->total   <<
                "  avg "     << setw(5) << setprecision(3) << (it->total/double(totalEvents_)) << "\n";
        } 
        oss << "\n";
        edm::LogVerbatim(logName_) << oss.str();
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
using pat::CandidateSummaryTable;
DEFINE_FWK_MODULE(CandidateSummaryTable);
