#ifndef L1TRPCTPG_H
#define L1TRPCTPG_H

/*
 * \file L1TRPCTPG.h
 *
 * $Date: 2009/11/19 14:34:40 $
 * $Revision: 1.7 $
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

///Geometry
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"


#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TRPCTPG : public edm::EDAnalyzer {

public:

// Constructor
L1TRPCTPG(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TRPCTPG();

// Booking of MonitoringElemnt for one RPCDetId (= roll)
std::map<std::string, MonitorElement*> L1TRPCBookME(RPCDetId & detId);

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(void);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  MonitorElement* rpctpgndigi[3];
  MonitorElement* rpctpgbx;

  MonitorElement *  m_digiBxRPCBar;

  MonitorElement *  m_digiBxRPCEnd;

  MonitorElement *  m_digiBxDT;

  MonitorElement *  m_digiBxCSC;
  
  std::map<uint32_t, std::map<std::string, MonitorElement*> >  rpctpgmeCollection;

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag rpctpgSource_;
  edm::InputTag rpctfSource_ ;

};

#endif
