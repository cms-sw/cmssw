#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//... L1Trigger

#include <DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerRecord.h>
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h>

//... Technical trigger bits
#include <DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTrigger.h>
#include <DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h>

//... For Track Study
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//
// class declaration
//

class RPCTTUMonitor : public DQMEDAnalyzer {
public:
  explicit  RPCTTUMonitor(const edm::ParameterSet&);
  ~ RPCTTUMonitor();
  
  int  discriminateGMT( const edm::Event& iEvent, const edm::EventSetup& iSetup );
  
  void discriminateDecision( bool ,  bool , int );
  
protected:

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  

private:
  MonitorElement * m_ttBitsDecisionData;
  MonitorElement * m_ttBitsDecisionEmulator;
  MonitorElement * m_bxDistDiffPac[8];
  MonitorElement * m_bxDistDiffDt[8];
  MonitorElement * m_dataVsemulator[8];  

  DQMStore * dbe;
  std::string  ttuFolder   ;
  std::string outputFile  ;
    
  int m_maxttBits;
  std::vector<unsigned> m_ttBits;

  bool m_dtTrigger;
  bool m_rpcTrigger;
  
  std::vector<int> m_GMTcandidatesBx;
  std::vector<int> m_DTcandidatesBx;
  std::vector<int> m_RPCcandidatesBx;
  
  //    edm::InputTag m_rpcDigiLabel;
  
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_gtReadoutLabel;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> m_gmtReadoutLabel;
  edm::EDGetTokenT<L1GtTechnicalTriggerRecord> m_rpcTechTrigEmu;

  
};

