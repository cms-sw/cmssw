// -*- C++ -*-
//
// Package:    DQMTtuAnalyzer
// Class:      DQMTtuAnalyzer
// 
/**\class DQMTtuAnalyzer DQMTtuAnalyzer.cc StudyL1Trigger/DQMTtuAnalyzer/src/DQMTtuAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andres Felipe Osorio Oliveros
//         Created:  Wed Sep 30 09:32:55 CEST 2009
// $Id: RPCTTUMonitor.h,v 1.4 2010/04/26 14:07:38 dellaric Exp $
//
//


// system include files
#include <memory>

//... User include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//
// class declaration
//

class RPCTTUMonitor : public edm::EDAnalyzer {
public:
  explicit  RPCTTUMonitor(const edm::ParameterSet&);
  ~ RPCTTUMonitor();
  
  int  discriminateGMT( const edm::Event& iEvent, const edm::EventSetup& iSetup );

  void discriminateDecision( bool ,  bool , int );
  
private:
  virtual void beginJob();
  virtual void beginRun(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  

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

  edm::InputTag m_rpcDigiLabel;
  edm::InputTag m_gtReadoutLabel;
  edm::InputTag m_gmtReadoutLabel;
  edm::InputTag m_rpcTechTrigEmu;
  
};

