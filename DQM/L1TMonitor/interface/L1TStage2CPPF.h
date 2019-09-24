#ifndef DQM_L1TMonitor_L1TStage2CPPF_H
#define DQM_L1TMonitor_L1TStage2CPPF_H

// system include files
#include <memory>
#include <unistd.h>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

///Data Format
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"

///Geometry
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TStage2CPPF : public DQMEDAnalyzer {
public:
  // Constructor
  L1TStage2CPPF(const edm::ParameterSet& ps);

  // Destructor
  ~L1TStage2CPPF() override;

protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // BeginRun
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

private:
  // ----------member data ---------------------------

  MonitorElement* CPPFInputNormOccupDisk[10];
  std::map<std::string, MonitorElement*> meInputDiskRing_1st;
  std::map<std::string, MonitorElement*> meInputDiskRing_2nd;

  MonitorElement* CPPFOutputNormOccupDisk[10];
  std::map<std::string, MonitorElement*> meOutputDiskRing_board;
  std::map<std::string, MonitorElement*> meOutputDiskRing_channel;
  std::map<std::string, MonitorElement*> meOutputDiskRing_emtf_sector;
  std::map<std::string, MonitorElement*> meOutputDiskRing_emtf_link;
  std::map<std::string, MonitorElement*> meOutputDiskRing_theta;
  std::map<std::string, MonitorElement*> meOutputDiskRing_phi;

  MonitorElement* CPPFInput_DiskRing_Vs_BX;
  MonitorElement* CPPFInput_Occupancy_DiskRing_Vs_Segment;
  MonitorElement* CPPFInput_Occupancy_Ring_Vs_Disk;

  MonitorElement* CPPFOutput_DiskRing_Vs_BX;
  MonitorElement* CPPFOutput_Occupancy_DiskRing_Vs_Segment;
  MonitorElement* CPPFOutput_1DTheta[12];
  MonitorElement* CPPFOutput_1DPhi[12];
  MonitorElement* CPPFOutput_Occupancy_Ring_Vs_Disk;

  std::map<uint32_t, std::map<std::string, MonitorElement*> > rpctpgmeCollection;

  int nev_;                 // Number of events processed
  std::string outputFile_;  //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::ofstream logFile_;
  edm::InputTag rpcdigiSource_;
  edm::EDGetTokenT<RPCDigiCollection> rpcdigiSource_token_;
  edm::EDGetTokenT<l1t::CPPFDigiCollection> cppfdigiSource_token_;
  edm::InputTag rpctfSource_;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> rpctfSource_token_;
};

#endif
