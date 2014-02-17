// -*- C++ -*-
//
// Package:    RPCNoise
// Class:      RPCNoise
// 
/**\class RPCNoise RPCNoise.cc RecoLocalMuon/RPCNoise/src/RPCNoise.cc

 Description: <simple analyis of RPC noise, and event filter>

 Implementation:
     <simple application of EDFilter>
*/
//
// Original Author:  Michael Henry Schmitt
//         Created:  Thu Oct 30 21:31:44 CET 2008
// $Id: RPCNoise.cc,v 1.4 2013/02/27 20:17:14 wmtan Exp $
//
//
// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <fstream>
#include <ctime>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <Geometry/CommonDetUnit/interface/GeomDet.h>//
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/RPCGeometry/interface/RPCRoll.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"
#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
//#include "EventFilter/RPCRawToDigi/interface/RPCRawSynchro.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "CondFormats/DTObjects/interface/DTT0.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "TVector3.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include "TProfile.h"

using namespace std;
using namespace edm;


//
// class declaration
//

class RPCNoise : public edm::EDFilter {
   public:
      explicit RPCNoise(const edm::ParameterSet&);
      ~RPCNoise();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

  // counters
  int nEventsAnalyzed;
  int nEventsSelected;
  int iRun;
  int iEvent;
  int firstOrbit;
  int lastOrbit;
  int thisOrbit;
  // control parameters
  bool fillHistograms;
  int nRPCHitsCut;
  int nCSCStripsCut;
  int nCSCWiresCut;
  int nDTDigisCut;
  // histogram
  std::string histogramFileName;
  // The root file for the histograms.
  TFile *theHistogramFile;
  // histograms
  TH1F *nWires;
  TH1F *nStrips;
  TH1F *nWiresH;
  TH1F *nStripsH;
  TH1F *nDTDigis;
  TH1F *nDTDigisH;
  TH1F *t0All;
  TH1F *t0AllH;
  TH1F *nDTDigisIn;
  TH1F *nDTDigisInH;
  TH1F *nDTDigisOut;
  TH1F *nDTDigisOutH;
  TH1F *fDTDigisOut;
  TH1F *fDTDigisOutH;
  TH1F *nRPCRecHits;
  TH1F *nRPCRecHitsLong;
  TProfile *hitsVsSerial;
  TProfile *orbitVsSerial;
  TProfile *hitsVsOrbit;
  TH1F *dOrbit;
  TH1F *RPCBX;
  TH1F *RPCClSize;
  TH1F *RPCBXH;
  TH1F *RPCClSizeH;
  TH1F *rpcStation;
  TH1F *rpcStationH;
  TH1F *rpcRing;
  TH1F *rpcRingH;
  TH1F *rpcSector;
  TH1F *rpcSectorH;
  TH1F *rpcLayer;
  TH1F *rpcLayerH;
  TProfile *rpcStationVsOrbit;
  TProfile *rpcSectorVsOrbit;
  TProfile *rpcRingVsOrbit;
  TH1F *rpcCorner;
  TH1F *rpcCornerH;
  TProfile *rpcCornerVsOrbit;

};

RPCNoise::RPCNoise(const edm::ParameterSet& pset)
{
  histogramFileName  = pset.getUntrackedParameter<std::string>("histogramFileName","histos.root");
  fillHistograms     = pset.getUntrackedParameter<bool>("fillHistograms",true);
  nRPCHitsCut        = pset.getUntrackedParameter<int>("nRPCHitsCut",40);
  nCSCStripsCut      = pset.getUntrackedParameter<int>("nCSCStripsCut",50);
  nCSCWiresCut       = pset.getUntrackedParameter<int>("nCSCWiresCut",10);
  nDTDigisCut        = pset.getUntrackedParameter<int>("nDTDigisCut",10);
}
RPCNoise::~RPCNoise()
{
}


void 
RPCNoise::beginJob()
{
  // initialize variables
  nEventsAnalyzed = 0;
  nEventsSelected = 0;
  iRun = 0;
  iEvent = 0;
  firstOrbit = lastOrbit = thisOrbit = 0;

  if (fillHistograms) {
    // Create the root file for the histograms
    theHistogramFile = new TFile(histogramFileName.c_str(), "RECREATE");
    theHistogramFile->cd();
    // book histograms
    nWires  = new TH1F("nWires","number of wire digis", 121, -0.5, 120.5);
    nStrips = new TH1F("nStrips","number of strip digis", 201, -0.5, 200.5);
    nWiresH  = new TH1F("nWiresH","number of wire digis HIGH", 121, -0.5, 120.5);
    nStripsH = new TH1F("nStripsH","number of strip digis HIGH", 201, -0.5, 200.5);
    nDTDigis = new TH1F("nDTDigis","number of DT digis",201,-0.5,200.5);
    nDTDigisH = new TH1F("nDTDigisH","number of DT digis HIGH",201,-0.5,200.5);
    nDTDigisIn    = new TH1F("nDTDigisIn","N DT digis in window",75,0.,150.);
    nDTDigisInH   = new TH1F("nDTDigisInH","N DT digis in window HIGH",75,0.,150.);
    nDTDigisOut   = new TH1F("nDTDigisOut","N DT digis out window",75,0.,150.);
    nDTDigisOutH  = new TH1F("nDTDigisOutH","N DT digis out window HIGH",75,0.,150.);
    fDTDigisOut   = new TH1F("fDTDigisOut", "fraction DT digis outside window",55,0.,1.1);
    fDTDigisOutH  = new TH1F("fDTDigisOutH","fraction DT digis outside window HIGH",55,0.,1.1);

    t0All  = new TH1F("t0All","t0",700,0.,7000.);
    t0AllH = new TH1F("t0AllH","t0 HIGH",700,0.,7000.);
    RPCBX  = new TH1F("RPCBX","RPC BX",21,-10.5,10.5);
    RPCBXH = new TH1F("RPCBXH","RPC BX HIGH",21,-10.5,10.5);
    RPCClSize  = new TH1F("RPCClSize","RPC cluster size",61,-0.5,60.5);
    RPCClSizeH = new TH1F("RPCClSizeH","RPC cluster size HIGH",61,-0.5,60.5);
    
    nRPCRecHits = new TH1F("nRPCRecHits","number of RPC RecHits",101,-0.5,100.5);
    nRPCRecHitsLong = new TH1F("nRPCRecHitsLong","number of RPC RecHits",601,-0.5,600.5);
    hitsVsSerial  = new TProfile("hitsVsSerial","mean RPC hits vs serial event number",4000,0.,40000.,0.,1000.);
    orbitVsSerial = new TProfile("orbitVsSerial","relative orbit number vs serial event number",4000,0.,40000.,0.,1.e10);
    hitsVsOrbit   = new TProfile("hitsVsOrbit","mean RPC hits vs orbit number",3000,0.,1200000.,0.,1000.);
    dOrbit = new TH1F("dOrbit","difference in orbit number",121,-0.5,120.5);

    rpcStation  = new TH1F("rpcStation",  "RPC station", 6,-0.5,5.5);
    rpcStationH = new TH1F("rpcStationH", "RPC station HIGH", 6,-0.5,5.5);
    rpcRing     = new TH1F("rpcRing",  "RPC ring", 9,-4.5,4.5);
    rpcRingH    = new TH1F("rpcRingH", "RPC ring HIGH", 9,-4.5,4.5);
    rpcSector   = new TH1F("rpcSector",  "RPC sector", 15,-0.5,14.5);
    rpcSectorH  = new TH1F("rpcSectorH", "RPC sector HIGH", 15,-0.5,14.5);
    rpcLayer    = new TH1F("rpcLayer",  "RPC layer", 4,-0.5,3.5);
    rpcLayerH   = new TH1F("rpcLayerH", "RPC layer HIGH", 4,-0.5,3.5);
    rpcStationVsOrbit = new TProfile("rpcStationVsOrbit","mean RPC station vs. Orbit",3000,0.,1200000.,0.,20.);
    rpcSectorVsOrbit  = new TProfile("rpcSectorVsOrbit","mean RPC sector vs. Orbit",  3000,0.,1200000.,0.,20.);
    rpcRingVsOrbit    = new TProfile("rpcRingVsOrbit","mean RPC ring vs. Orbit",      3000,0.,1200000.,-20.,20.);
    rpcCorner   = new TH1F("rpcCorner", "special corner designation",      4,-0.5,3.5);
    rpcCornerH  = new TH1F("rpcCornerH","special corner designation HIGH", 4,-0.5,3.5);
    rpcCornerVsOrbit = new TProfile("rpcCornerVsOrbit","special corner vs. Orbit",3000,0.,1200000.,-20.,20.);
  }

}


void 
RPCNoise::endJob() {
  std::cout << "\n\t===============================================================\n"
       << "\tnumber of events analyzed      = " << nEventsAnalyzed << std::endl
       << "\tnumber of events selected      = " << nEventsSelected << std::endl
       << "\tfirst and last orbit number    : " << firstOrbit << ", " << lastOrbit << ", " << lastOrbit-firstOrbit << std::endl
       << "\t===============================================================\n\n";


  if (fillHistograms) {
    // Write the histos to file
    printf("\n\n======= write out my histograms ====\n\n");
    theHistogramFile->cd();
    nWires->Write();
    nStrips->Write();
    nWiresH->Write();
    nStripsH->Write();
    nDTDigis->Write();
    nDTDigisH->Write();
    nDTDigisIn->Write();
    nDTDigisInH->Write();
    nDTDigisOut->Write();
    nDTDigisOutH->Write();
    fDTDigisOut->Write();
    fDTDigisOutH->Write();
    nRPCRecHits->Write();
    nRPCRecHitsLong->Write();
    hitsVsSerial->Write();
    hitsVsOrbit->Write();
    orbitVsSerial->Write();
    t0All->Write();
    t0AllH->Write();
    RPCBX->Write();
    RPCClSize->Write();
    RPCBXH->Write();
    RPCClSizeH->Write();
    rpcStation->Write();
    rpcStationH->Write();
    rpcRing->Write();
    rpcRingH->Write();
    rpcSector->Write();
    rpcSectorH->Write();
    rpcLayer->Write();
    rpcLayerH->Write();
    dOrbit->Write();
    rpcStationVsOrbit->Write();
    rpcSectorVsOrbit->Write();
    rpcRingVsOrbit->Write();
    rpcCorner->Write();
    rpcCornerH->Write();
    rpcCornerVsOrbit->Write();
    theHistogramFile->Close();
  }
}




bool
RPCNoise::filter(edm::Event& event, const edm::EventSetup& eventSetup){

  bool selectThisEvent = false;

  // increment counter
  nEventsAnalyzed++;

  iRun   = event.id().run();
  iEvent = event.id().event();

  bool printThisLine = (nEventsAnalyzed%100 == 0);
  if (printThisLine) {
  std::cout << "======================================"
       << " analyzed= " << nEventsAnalyzed 
       << ", selected= " << nEventsSelected
       << "\trun,event: " << iRun << ", " << iEvent << std::endl;
  }

  /*
  const edm::Timestamp jTime = event.time();
  unsigned int sec  = jTime.value() >> 32;
  unsigned int usec = 0xFFFFFFFF & jTime.value() ;
  double floatTime = sec + usec/(float)1000000.;
  */

  // first event gives
  // sec = 1225315493
  //    orbit = 202375185
  //    bx = 764
  //    mtime = 205094517
  int bx = event.bunchCrossing();
  int thisOrbit = event.orbitNumber();
  long mTime = 3564*thisOrbit + bx;
  if (firstOrbit == 0) {
    firstOrbit = thisOrbit;
    lastOrbit = thisOrbit;
  }
  int deltaOrbit = thisOrbit - lastOrbit;
  lastOrbit = thisOrbit;
  int relativeOrbit = thisOrbit - firstOrbit;

  if (fillHistograms) {dOrbit->Fill(deltaOrbit);}

  if (nEventsAnalyzed < 200) {
    std::cout << iEvent 
      //	 << "\tsec,usec: " << sec << ", " << usec
      //	 << "\tfloatTime= " << std::setprecision(16) << floatTime
      //	 << "\tctime: " << ctime(sec)
	 << "\torbit,bx,mTime: " << thisOrbit << "," << bx << "," << mTime
	 << "\tdelta= " << deltaOrbit
	 << std::endl;
  }


  // ================
  // RPC recHits
  // ================
  edm::Handle<RPCRecHitCollection> rpcRecHits; 
  event.getByLabel("rpcRecHits","",rpcRecHits);

  // count the number of RPC rechits
  int nRPC = 0;
  RPCRecHitCollection::const_iterator rpcIt;
  for (rpcIt = rpcRecHits->begin(); rpcIt != rpcRecHits->end(); rpcIt++) {
    //    RPCDetId id = (RPCDetId)(*rpcIt).rpcId();
    //    LocalPoint rhitlocal = (*rpcIt).localPosition();
    nRPC++;
  }

  // loop again, this time fill histograms
  for (rpcIt = rpcRecHits->begin(); rpcIt != rpcRecHits->end(); rpcIt++) {
    RPCDetId id = (RPCDetId)(*rpcIt).rpcId();
    int kRegion  = id.region();
    int kStation = id.station();
    int kRing = id.ring();
    int kSector = id.sector();
    int kLayer = id.layer();
    int bx = (*rpcIt).BunchX();
    int clSize = (*rpcIt).clusterSize();
    int cornerFlag = 0;
    if ( (kStation>3) && (kSector<3) ) {
      cornerFlag = 1;
      if (kRing < 0) cornerFlag = 2;
    }
    if (nEventsAnalyzed < 100) {
      std::cout << "Region/Station/Ring/Sector/Layer: "
	   << kRegion << " / "
	   << kStation << " / "
	   << kRing << " / "
	   << kSector << " / "
	   << kLayer 
	   << "\tbx,clSize: " << bx << ", " << clSize
	   << std::endl;
    }
    if (fillHistograms) {
      RPCBX->Fill(bx);
      RPCClSize->Fill(min((float)clSize,(float)60.));
      rpcStation->Fill(kStation);
      rpcRing->Fill(kRing);
      rpcSector->Fill(kSector);
      rpcLayer->Fill(kLayer);
      rpcStationVsOrbit->Fill(relativeOrbit,kStation);
      rpcSectorVsOrbit->Fill(relativeOrbit,kSector);
      rpcRingVsOrbit->Fill(relativeOrbit,kRing);
      rpcCorner->Fill(cornerFlag);
      rpcCornerVsOrbit->Fill(relativeOrbit,cornerFlag);
      if (nRPC > nRPCHitsCut) {
	RPCBXH->Fill(bx);
	RPCClSizeH->Fill(min((float)clSize,(float)60.));
	rpcStationH->Fill(kStation);
	rpcRingH->Fill(kRing);
	rpcSectorH->Fill(kSector);
	rpcLayerH->Fill(kLayer);
	rpcCornerH->Fill(cornerFlag);
      }
    }

  }

 
  // ===============
  // CSC DIGIs
  // ===============
  edm::Handle<CSCWireDigiCollection>  wires;
  edm::Handle<CSCStripDigiCollection> strips;
  event.getByLabel("muonCSCDigis","MuonCSCWireDigi",wires);
  event.getByLabel("muonCSCDigis","MuonCSCStripDigi",strips);

  // count the number of wire digis.
  int nW = 0;
  for (CSCWireDigiCollection::DigiRangeIterator jW=wires->begin(); jW!=wires->end(); jW++) {
    std::vector<CSCWireDigi>::const_iterator wireIterA = (*jW).second.first;
    std::vector<CSCWireDigi>::const_iterator lWireA = (*jW).second.second;
    for( ; wireIterA != lWireA; ++wireIterA) {
      nW++;
    }
  }

  // count the number of fired strips.
  // I am using a crude indicator of signal - this is fast and adequate for
  // this purpose, but it would be poor for actual CSC studies.
  int nS = 0;
  for (CSCStripDigiCollection::DigiRangeIterator jS=strips->begin(); jS!=strips->end(); jS++) {
    std::vector<CSCStripDigi>::const_iterator stripItA = (*jS).second.first;
    std::vector<CSCStripDigi>::const_iterator lastStripA = (*jS).second.second;
    for( ; stripItA != lastStripA; ++stripItA) {
      std::vector<int> myADCVals = stripItA->getADCCounts();
      int iDiff = myADCVals[4]+myADCVals[5]-myADCVals[0]-myADCVals[1];
      if (iDiff > 30) {
	nS++;
      }
    }
  }

 
  // ===============
  // DT DIGIs
  // ===============
  // see: CalibMuon/DTCalibration/plugins/DTT0Calibration.cc
  edm::Handle<DTDigiCollection>  dtDIGIs;
  event.getByLabel("muonDTDigis",dtDIGIs);

  // count the number of digis.
  int nDT = 0;
  int nDTin = 0;
  int nDTout = 0;
  for (DTDigiCollection::DigiRangeIterator jDT=dtDIGIs->begin(); jDT!=dtDIGIs->end(); ++jDT) {
    const DTDigiCollection::Range& digiRange = (*jDT).second;
    for (DTDigiCollection::const_iterator digi = digiRange.first;
	 digi != digiRange.second;
	 digi++) {
      double t0 = (*digi).countsTDC();
      nDT++;
      if ((t0>3050) && (t0<3700)) {
	nDTin++;
      } else {
	nDTout++;
      }
      if (fillHistograms) {
	t0All->Fill(t0);
	if (nRPC > nRPCHitsCut) {t0AllH->Fill(t0);}
      }
    }
  }

  //==============
  // Analysis
  //==============

  if (nEventsAnalyzed < 1000) {std::cout << "\tnumber of CSC DIGIS = " << nW << ", " << nS 
				    << "\tDT DIGIS = " << nDT
				    << "\tRPC Rechits = " << nRPC << std::endl;}

  if (fillHistograms) {

    nWires->Fill(min((float)nW,(float)120.));
    nStrips->Fill(min((float)nS,(float)200.));
    
    nDTDigis->Fill(min((float)nDT,(float)200.));
    nDTDigisIn->Fill(min((float)nDTin,(float)200.));
    nDTDigisOut->Fill(min((float)nDTout,(float)200.));
    if (nDT > 0) {
      float fracOut = float(nDTout)/float(nDT);
      fDTDigisOut->Fill(fracOut);
    }
    nRPCRecHits->Fill(min((float)nRPC,(float)100.));
    nRPCRecHitsLong->Fill(min((float)nRPC,(float)1000.));
    hitsVsSerial->Fill(nEventsAnalyzed,nRPC);
    hitsVsOrbit->Fill(relativeOrbit,nRPC);
    orbitVsSerial->Fill(nEventsAnalyzed,relativeOrbit);

    if (nRPC > nRPCHitsCut) {
      nWiresH->Fill(min((float)nW,(float)120.));
      nStripsH->Fill(min((float)nS,(float)200.));
      nDTDigisH->Fill(min((float)nDT,(float)200.));
      nDTDigisInH->Fill(min((float)nDTin,(float)200.));
      nDTDigisOutH->Fill(min((float)nDTout,(float)200.));
      if (nDT > 0) {
	float fracOut = float(nDTout)/float(nDT);
	fDTDigisOutH->Fill(fracOut);
      }
    }
  }

  // select this event for output?

  selectThisEvent = (nRPC > nRPCHitsCut) && (nW > nCSCWiresCut || nS > nCSCStripsCut) && (nDT > nDTDigisCut);
  if (selectThisEvent) {nEventsSelected++;}

  return selectThisEvent;
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCNoise);
