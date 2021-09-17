#ifndef L1TRPCTPG_H
#define L1TRPCTPG_H

/*
 * \file L1TRPCTPG.h
 *
 * \author J. Berryhill
 *
*/

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

class L1TRPCTPG : public DQMEDAnalyzer {
public:
  // Constructor
  L1TRPCTPG(const edm::ParameterSet& ps);

  // Destructor
  ~L1TRPCTPG() override;

protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // BeginRun
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

private:
  // ----------member data ---------------------------

  MonitorElement* rpctpgndigi[3];
  MonitorElement* rpctpgbx;
  MonitorElement* m_digiBxRPCBar;
  MonitorElement* m_digiBxRPCEnd;
  MonitorElement* m_digiBxDT;
  MonitorElement* m_digiBxCSC;

  std::map<uint32_t, std::map<std::string, MonitorElement*> > rpctpgmeCollection;

  int nev_;                 // Number of events processed
  std::string outputFile_;  //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::ofstream logFile_;
  edm::InputTag rpctpgSource_;
  edm::EDGetTokenT<RPCDigiCollection> rpctpgSource_token_;
  edm::InputTag rpctfSource_;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> rpctfSource_token_;
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcgeomToken_;
};

#endif
