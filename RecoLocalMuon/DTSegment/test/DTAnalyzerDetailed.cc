/******* \class DTAnalyzerDetailed *******
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 20/11/2006 16:50:57 CET $
 *
 * Modification:
 *
 *********************************/

/* This Class Header */
#include "RecoLocalMuon/DTSegment/test/DTAnalyzerDetailed.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRangeMapAccessor.h"
#include "RecoLocalMuon/DTSegment/test/DTMeanTimer.h"
#include "RecoLocalMuon/DTSegment/test/DTSegmentResidual.h"


#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"


#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

/* C++ Headers */
#include <iostream>
#include <cmath>
using namespace std;

/* ====================================================================== */

/* Constructor */ 
DTAnalyzerDetailed::DTAnalyzerDetailed(const ParameterSet& pset) : _ev(0){

  theSync =
    DTTTrigSyncFactory::get()->create(pset.getUntrackedParameter<string>("tTrigMode"),
                                      pset.getUntrackedParameter<ParameterSet>("tTrigModeConfig"));
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  // the name of the 1D rec hits collection
  theRecHits1DLabel = pset.getParameter<string>("recHits1DLabel");

  // the name of the 2D rec hits collection
  theRecHits2DLabel = pset.getParameter<string>("recHits2DLabel");

  // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getParameter<string>("recHits4DLabel");

  doHits = pset.getParameter<bool>("doHits");
  doSegs = pset.getParameter<bool>("doSegs");

  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  bool dirStat=TH1::AddDirectoryStatus();
  TH1::AddDirectory(kTRUE);

  /// DT histos
  // 1d hits
  new TH1F("hnHitDT","Num 1d hits DT",200,0.,200.);
  new TH1F("hDigiTime","Digi time (ns)",700,-100.,600.);

  for (int w=-2; w<=2; ++w) {
    stringstream nameWheel;
    nameWheel << "_Wh"<< w ;
    //cout << "Wheel " << nameWheel.str() << endl;
    for (int sec=1; sec<=14; ++sec) { // section 1 to 14
      stringstream nameSector;
      nameSector << nameWheel.str() << "_Sec" << sec;
      //cout << "Sec " << nameSector.str() << endl;
      for (int st=1; st<=4; ++st) { // station 1 to 4
        stringstream nameChamber;
        nameChamber << nameSector.str() << "_St" << st;
        //cout << "Ch " << nameChamber.str() << endl;
        for (int sl=1; sl<=3; ++sl) { // SL 1 to 3 (2 missing for st==4)
          if (st==4 && sl==2) continue;
          stringstream nameSL;
          nameSL << nameChamber.str() << "_Sl" << sl;
          //cout << "SL " << nameSL.str() << endl;
          for (int l=1; l<=4; ++l) { // layer 1 to 4
            stringstream nameLayer;
            nameLayer << nameSL.str() << "_Lay" << l;
            //cout << "Lay " << nameLayer.str() << endl;

            // Create hist for each Layer
            createTH1F("hDigiTime", "Digi Time (ns) ",nameLayer.str(),100,-100.,600.);

            createTH1F("hPosLeft", "Pos of Left hit (cm) in local frame",
                       nameLayer.str(), 100,-220.,220.);
            createTH1F("hPosRight", "Pos of Right hit (cm) in local frame",
                       nameLayer.str(), 100,-220.,220.);
          }
          // Create hist for each SuperLayer
          createTH1F("hDigiTime", "Digi Time (ns) ", nameSL.str(), 100,-100.,600.);
          createTH1F("hPosLeft", "Pos of Left hit (cm) in local frame",
                     nameSL.str(), 100,-220.,220.);
          createTH1F("hPosRight", "Pos of Right hit (cm) in local frame",
                     nameSL.str(), 100,-220.,220.);
          createTH1F("hMeanTimer", "Tmax ", nameSL.str(), 100,200.,600.);
          createTH1F("hMeanTimerSeg", "Tmax from segments hits ", nameSL.str(),
                     100,200.,600.);
          createTH2F("hMeanTimerSegAlongWire",
                     "Tmax from segments hits vs pos along wire ", nameSL.str(),
                     40, -150., 150, 100,200.,600.);
          createTH2F("hMeanTimerSegVsAngle",
                     "Tmax from segments hits vs angle ", nameSL.str(),
                     40, -0., 1.0, 100, 200.,600.);
          createTH2F("hMeanTimerSegVsNHits",
                     "Tmax from segments hits vs n hits segment ", nameSL.str(),
                     10, 0.5, 10.5, 100, 200.,600.);
          createTH1F("hHitResidualSeg", "Hits residual wrt segments ", nameSL.str(), 100,-.2,+.2);
          createTH1F("hHitResidualSegCellDX", "Hits residual wrt segments semicell DX ", nameSL.str(), 100,-.2,+.2);
          createTH1F("hHitResidualSegCellSX", "Hits residual wrt segments semicell SX", nameSL.str(), 100,-.2,+.2);
          createTH2F("hHitResidualSegAlongWire",
                     "Hits residual wrt segments vs pos along wire ", nameSL.str(),
                     40, -150., 150, 100,-.2,+.2);
          createTH2F("hHitResidualSegVsWireDis",
                     "Hits residual wrt segments vs wire distance ", nameSL.str(),
                     40, 0., 2.1 , 100,-.2,+.2);
          createTH2F("hHitResidualSegVsAngle",
                     "Hits residual wrt segments vs impact angle ", nameSL.str(),
                     40, .0, 1.0 , 100,-.2,+.2);
          createTH2F("hHitResidualSegVsNHits",
                     "Hits residual wrt segments vs num hits ", nameSL.str(),
                     10, -0.5, 9.5 , 100,-.2,+.2);
          createTH2F("hHitResidualSegVsChi2",
                     "Hits residual wrt segments vs chi2 ", nameSL.str(),
                     25, .0, 25.0 , 100,-.2,+.2);

          createTH2F("hNsegs2dVsNhits",
                     "N segs 2d vs n Hits ", nameSL.str(),
                     20, -0.5, 19.5 , 5, -0.5,4.5);
        }
        // Create hist for each Chamber
        createTH1F("hDigiTime", "Digi Time (ns) ", nameChamber.str(), 100,-100.,600.);
        createTH1F("hPosLeft", "Pos of Left hit (cm) in local frame",
                   nameChamber.str(), 100,-220.,220.);
        createTH1F("hPosRight", "Pos of Right hit (cm) in local frame",
                   nameChamber.str(), 100,-220.,220.);
        //segments
        createTH1F("hNSegs", "N segments ", nameChamber.str(), 20,0.,20.);
        createTH1F("hChi2Seg", "Chi2 segments ", nameChamber.str(), 25,0.,25.);
        createTH1F("hChi2SegPhi", "Chi2 segments phi ", nameChamber.str(),
                   25,0.,25.);
        createTH1F("hChi2SegZed", "Chi2 segments zed ", nameChamber.str(),
                   25,0.,25.);
        createTH1F("hNHitsSeg", "N hits segment ", nameChamber.str(),
                   15,-0.5,14.5);
        createTH1F("hNHitsSegPhi", "N hits segment phi ", nameChamber.str(),
                   12,-0.5,11.5);
        createTH2F("hNHitsSegPhiVsAngle", "N hits segment phi vs angle ", nameChamber.str(),
                   40, .0, 1.0 ,12,-0.5,11.5);
        createTH2F("hNHitsSegPhiVsOtherHits", "N hits segment phi vs hits in SLs ", nameChamber.str(),
                   20, -0.5, 19.5 ,12,-0.5,11.5);
        createTH2F("hNHitsSegPhiVsNumSegs", "N hits segment zed vs num segs in ch ", nameChamber.str(),
                   6, -0.5, 5.5 ,12,-0.5,11.5);
        createTH2F("hNHitsSegPhiVsChi2", "N hits segment zed vs chi2/NDoF ", nameChamber.str(),
                   25,0.,25.,12,-0.5,11.5);
        createTH1F("hNHitsSegZed", "N hits segment zed ", nameChamber.str(),
                   10,-0.5,9.5);
        createTH2F("hNHitsSegZedVsAngle", "N hits segment zed vs angle ", nameChamber.str(),
                   40, .0, 1.0 ,12,-0.5,11.5);
        createTH2F("hNHitsSegZedVsOtherHits", "N hits segment zed vs hits in SLs ", nameChamber.str(),
                   20, -0.5, 19.5 ,12,-0.5,11.5);
        createTH2F("hNHitsSegZedVsNumSegs", "N hits segment zed vs num segs in ch ", nameChamber.str(),
                   6, -0.5, 5.5 ,12,-0.5,11.5);
        createTH2F("hNHitsSegZedVsChi2", "N hits segment zed vs chi2/NDoF ", nameChamber.str(),
                   25,0.,25.,12,-0.5,11.5);

        createTH1F("hHitResidualSeg", "Hits residual wrt segments ",
                   nameChamber.str(), 100,-.2,+.2);
        createTH1F("hHitResidualSegCellDX", "Hits residual wrt segments semicell DX ", nameChamber.str(), 100,-.2,+.2);
        createTH1F("hHitResidualSegCellSX", "Hits residual wrt segments semicell SX", nameChamber.str(), 100,-.2,+.2);
        createTH2F("hHitResidualSegAlongWire",
                   "Hits residual wrt segments vs pos along wire ", nameChamber.str(),
                   40, -150., 150, 100,-.2,+.2);
        createTH2F("hHitResidualSegVsWireDis",
                   "Hits residual wrt segments vs wire distance ", nameChamber.str(),
                   40, 0., 2.1 , 100,-.2,+.2);
        createTH2F("hHitResidualSegVsAngle",
                   "Hits residual wrt segments vs impact angle ", nameChamber.str(),
                   40, .0, 1.0 , 100,-.2,+.2);
        createTH2F("hHitResidualSegVsNHits",
                   "Hits residual wrt segments vs num hits ", nameChamber.str(),
                   10, -0.5, 9.5 , 100,-.2,+.2);
        createTH2F("hHitResidualSegVsChi2",
                   "Hits residual wrt segments vs chi2 ", nameChamber.str(),
                   25, .0, 25.0 , 100,-.2,+.2);

        // eff
        createTH2F("hNsegs4dVsNhits",
                   "N segs 4d vs n Hits ", nameChamber.str(),
                   20, 0.5, 20.5 , 5, -0.5,4.5);
        createTH2F("hNsegs4dVsNhitsPhi",
                   "N segs 4d vs n HitsPhi ", nameChamber.str(),
                   20, 0.5, 20.5 , 5, -0.5,4.5);
        createTH2F("hNsegs4dVsNhitsZed",
                   "N segs 4d vs n HitsZed ", nameChamber.str(),
                   20, 0.5, 20.5 , 5, -0.5,4.5);

        createTH2F("hNsegs4dVsNsegs2d",
                   "N segs 4d vs n segs2d ", nameChamber.str(),
                   4, 0.5, 4.5 , 5, -0.5,4.5);
        createTH2F("hNsegs4dVsNsegs2dPhi",
                   "N segs 4d vs n segs2d Phi ", nameChamber.str(),
                   4, 0.5, 4.5 , 5, -0.5,4.5);
        createTH2F("hNsegs4dVsNsegs2dZed",
                   "N segs 4d vs n segs2d Zed ", nameChamber.str(),
                   4, 0.5, 4.5 , 5, -0.5,4.5);

        createTH2F("hNsegs2dSL1VsNsegs2dSL3",
                   "N segs 2d SL1 vs SL3 ", nameChamber.str(),
                   5, -0.5, 4.5 , 5, -0.5,4.5);
        createTH2F("hNsegs2dSL1VsNsegs2dSL2",
                   "N segs 2d SL1 vs SL2 ", nameChamber.str(),
                   5, -0.5, 4.5 , 5, -0.5,4.5);
        createTH2F("hNsegs2dSL2VsNsegs2dSL3",
                   "N segs 2d SL2 vs SL3 ", nameChamber.str(),
                   5, -0.5, 4.5 , 5, -0.5,4.5);

        // trigger eff
        createTH1F("hSegEff","Eff LocaTrig if Seg", nameChamber.str(),
                   5, 0.,5.);
      }
    }
  }

  // segs
  new TH1F("hnSegDT","Num seg DT",50,0.,50.);
  TH1::AddDirectory(dirStat);
}

/* Destructor */ 
DTAnalyzerDetailed::~DTAnalyzerDetailed() {
  theFile->cd();
  theFile->Write();
  theFile->Close();
}

/* Operations */ 
void DTAnalyzerDetailed::analyze(const Event & event,
                         const EventSetup& eventSetup) {
  theSync->setES(eventSetup);
  _ev++;

  if (debug) 
    cout << "Run:Event analyzed " << event.id().run() << ":" << event.id().event() <<
      " Num " << _ev << endl;

  static int j=1;
  if ((_ev%j)==0) {
    if ((_ev/j)==9) j*=10;
    cout << "Run:Event analyzed " << event.id().run() << ":" << event.id().event() <<
      " Num " << _ev << endl;
  }

  if (doHits) analyzeDTHits(event, eventSetup);
  if (doSegs) analyzeDTSegments(event, eventSetup);
}

void DTAnalyzerDetailed::analyzeDTHits(const Event & event,
                               const EventSetup& eventSetup){
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the 1D rechits from the event --------------
  Handle<DTRecHitCollection> dtRecHits; 
  event.getByLabel(theRecHits1DLabel, dtRecHits);

  int nHitDT = dtRecHits->size();
  histo("hnHitDT")->Fill(nHitDT);

  //float ttrigg = 1895.; // should get this from CondDB...
  for (DTRecHitCollection::const_iterator hit = dtRecHits->begin();
       hit!=dtRecHits->end();  ++hit) {
    // Get the wireId of the rechit
    DTWireId wireId = (*hit).wireId();

    float ttrig = theSync->offset(wireId);
    //cout << "TTrig " << ttrig << endl;

    float time = (*hit).digiTime()  - ttrig ;
    double xLeft = (*hit).localPosition(DTEnums::Left).x();
    double xRight = (*hit).localPosition(DTEnums::Right).x();

    histo("hDigiTime")->Fill(time);
    { // per layer
      const DTLayerId& id((*hit).wireId().layerId());
      histo(hName("hDigiTime",id))->Fill(time);
      histo(hName("hPosLeft",id))->Fill(xLeft);
      histo(hName("hPosRight",id))->Fill(xRight);
    }

    { // per SL
      const DTSuperLayerId& id((*hit).wireId().superlayerId());
      histo(hName("hDigiTime", id))->Fill(time);
      histo(hName("hPosLeft",id))->Fill(xLeft);
      histo(hName("hPosRight",id))->Fill(xRight);
    }

    { // per Chamber
      const DTChamberId& id((*hit).wireId().chamberId());
      histo(hName("hDigiTime",id))->Fill(time);
      histo(hName("hPosLeft",id))->Fill(xLeft);
      histo(hName("hPosRight",id))->Fill(xRight);
    }
  }

  // MeanTimer analysis
  // loop on SL
  //cout << "MeanTimer analysis" << endl;
  const std::vector<DTSuperLayer*> & sls = dtGeom->superLayers();
  for (std::vector<DTSuperLayer*>::const_iterator sl = sls.begin();
       sl!=sls.end() ; ++sl) {
    DTSuperLayerId slid = (*sl)->id();

    DTMeanTimer meanTimer(dtGeom->superLayer(slid), dtRecHits, eventSetup,
                          theSync);
    vector<double> tMaxs=meanTimer.run();
    for (vector<double>::const_iterator tMax=tMaxs.begin() ; tMax!=tMaxs.end();
         ++tMax) {
      //cout << "Filling " << hName("hMeanTimer", slid) << " with " << *tMax << endl;
      histo(hName("hMeanTimer", slid))->Fill(*tMax);
    }
  }

}

void DTAnalyzerDetailed::analyzeDTSegments(const Event & event,
                                   const EventSetup& eventSetup){
  if (debug) cout << "analyzeDTSegments" << endl;
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the 4D rechit collection from the event -------------------
  edm::Handle<DTRecSegment4DCollection> segs;
  event.getByLabel(theRecHits4DLabel, segs);
  if (debug) cout << "4d " << segs->size() << endl;

  // Get the 2D rechit collection from the event -------------------
  edm::Handle<DTRecSegment2DCollection> segs2d;
  event.getByLabel(theRecHits2DLabel, segs2d);
  if (debug) cout << "2d " << segs2d->size() << endl;

  // Get the 1D rechits from the event --------------
  Handle<DTRecHitCollection> dtRecHits; 
  event.getByLabel(theRecHits1DLabel, dtRecHits);
  if (debug) cout << "1d " << dtRecHits->size() << endl;

  int nsegs = segs->size();
  histo("hnSegDT")->Fill(nsegs);
  const std::vector<DTChamber*> & chs = dtGeom->chambers();

  for (std::vector<DTChamber*>::const_iterator ch = chs.begin();
       ch!=chs.end() ; ++ch) {
    DTChamberId chid((*ch)->id());
    //cout << "chid " << chid << endl;
    //int w= chid.wheel();
    //int se= chid.sector();
    //int st= chid.station();
    DTRecSegment4DCollection::range segsch= segs->get(chid);
    int nSegsCh=segsch.second-segsch.first;
    histo(hName("hNSegs", chid))->Fill(nSegsCh);

    // some quality on segments
    //bool hasGoodSegment=false;

    for (DTRecSegment4DCollection::const_iterator seg=segsch.first ;
         seg!=segsch.second ; ++seg ) {

      // first the the two projection separately
      // phi
      if(debug) cout << *seg << endl;
      const DTChamberRecSegment2D* phiSeg= (*seg).phiSegment();

      vector<DTRecHit1D> phiHits;
      if (phiSeg) {
        if(debug) cout << "Phi " << *phiSeg << endl;
        phiHits = phiSeg->specificRecHits();
        //if (phiHits.size()>=6) hasGoodSegment = true;

        DTSuperLayerId slid1(phiSeg->chamberId(),1);
        /// Mean timer analysis
        DTMeanTimer meanTimer1(dtGeom->superLayer(slid1), phiHits, eventSetup, theSync);
        vector<double> tMaxs1=meanTimer1.run();
        for (vector<double>::const_iterator tMax=tMaxs1.begin() ; tMax!=tMaxs1.end();
             ++tMax) {
          histo(hName("hMeanTimerSeg", slid1))->Fill(*tMax);
          histo2d(hName("hMeanTimerSegVsNHits",
                        slid1))->Fill(phiHits.size(), *tMax);
          if ( (*seg).hasZed() ) {
            histo2d(hName("hMeanTimerSegAlongWire",
                          slid1))->Fill((*seg).localPosition().y(),*tMax);
            histo2d(hName("hMeanTimerSegVsAngle",
                          slid1))->Fill(M_PI-(*seg).localDirection().theta(), *tMax);
          }
        }

        DTSuperLayerId slid3(phiSeg->chamberId(),3);
        /// Mean timer analysis
        DTMeanTimer meanTimer3(dtGeom->superLayer(slid3), phiHits, eventSetup, theSync);
        vector<double> tMaxs3=meanTimer3.run();
        for (vector<double>::const_iterator tMax=tMaxs3.begin() ; tMax!=tMaxs3.end();
             ++tMax) {
          histo(hName("hMeanTimerSeg", slid3))->Fill(*tMax);
          histo2d(hName("hMeanTimerSegVsNHits",
                        slid1))->Fill(phiHits.size(), *tMax);
          if ( (*seg).hasZed() ) {
            histo2d(hName("hMeanTimerSegAlongWire",
                          slid1))->Fill((*seg).localPosition().y(), *tMax);
            histo2d(hName("hMeanTimerSegVsAngle",
                          slid1))->Fill(M_PI-(*seg).localDirection().theta(), *tMax);
          }
        }
      }
      // zed
      const DTSLRecSegment2D* zedSeg =(*seg).zSegment();
      //cout << "zedSeg " << zedSeg << endl;
      vector<DTRecHit1D> zedHits;
      if (zedSeg) {
        if(debug) cout << "Zed " << *zedSeg << endl;
        zedHits= zedSeg->specificRecHits();
        DTSuperLayerId slid(zedSeg->superLayerId());
        /// Mean timer analysis
        DTMeanTimer meanTimer(dtGeom->superLayer(slid), zedHits, eventSetup, theSync);
        vector<double> tMaxs=meanTimer.run();
        for (vector<double>::const_iterator tMax=tMaxs.begin() ; tMax!=tMaxs.end();
             ++tMax) {
          histo(hName("hMeanTimerSeg", slid))->Fill(*tMax);
          histo2d(hName("hMeanTimerSegVsNHits",
                        slid))->Fill(zedHits.size(), *tMax);
          if ( (*seg).hasPhi() ) {
            histo2d(hName("hMeanTimerSegAlongWire",
                          slid))->Fill((*seg).localPosition().x(), *tMax);
            histo2d(hName("hMeanTimerSegVsAngle",
                          slid))->Fill(M_PI-(*seg).localDirection().theta(), *tMax);
          }
        }
      }

      histo(hName("hChi2Seg",
                  chid))->Fill((*seg).chi2()/(*seg).degreesOfFreedom());
      if(phiSeg) {
        histo(hName("hNHitsSegPhi", chid))->Fill(phiSeg->recHits().size());
        histo2d(hName("hNHitsSegPhiVsAngle",
                      chid))->Fill(M_PI-phiSeg->localDirection().theta(),
                                   phiSeg->recHits().size());
        // get the total numer of hits in this SL(s)
        DTSuperLayerId slid1(chid,1);
        DTRecHitCollection::range rangeH1 =
          dtRecHits->get(DTRangeMapAccessor::layersBySuperLayer(slid1));
        DTSuperLayerId slid3(chid,3);
        DTRecHitCollection::range rangeH3 =
          dtRecHits->get(DTRangeMapAccessor::layersBySuperLayer(slid3));
        histo2d(hName("hNHitsSegPhiVsOtherHits",
                      chid))->Fill((rangeH1.second-rangeH1.first)+
                                   (rangeH3.second-rangeH3.first),
                                   phiSeg->recHits().size());
        histo2d(hName("hNHitsSegPhiVsNumSegs",
                      chid))->Fill(nSegsCh,
                                   phiSeg->recHits().size());
        histo2d(hName("hNHitsSegPhiVsChi2",
                      chid))->Fill(phiSeg->chi2()/phiSeg->degreesOfFreedom(),
                                   phiSeg->recHits().size());
        histo(hName("hChi2SegPhi",
                    chid))->Fill(phiSeg->chi2()/phiSeg->degreesOfFreedom());
      }

      if (zedSeg) {
        histo(hName("hNHitsSegZed", chid))->Fill(zedSeg->recHits().size());
        histo2d(hName("hNHitsSegZedVsAngle",
                      chid))->Fill(M_PI-zedSeg->localDirection().theta(),
                                   zedSeg->recHits().size());
        DTSuperLayerId slid(chid,2);
        DTRecHitCollection::range rangeH =
          dtRecHits->get(DTRangeMapAccessor::layersBySuperLayer(slid));
        histo2d(hName("hNHitsSegZedVsOtherHits",
                      chid))->Fill(rangeH.second-rangeH.first,
                                   zedSeg->recHits().size());
        histo2d(hName("hNHitsSegZedVsNumSegs",
                      chid))->Fill(nSegsCh,
                                   zedSeg->recHits().size());
        histo2d(hName("hNHitsSegZedVsChi2",
                      chid))->Fill(zedSeg->chi2()/zedSeg->degreesOfFreedom(),
                                   zedSeg->recHits().size());
        histo(hName("hChi2SegZed",
                    chid))->Fill(zedSeg->chi2()/zedSeg->degreesOfFreedom());
      }
      if (phiSeg && zedSeg) 
        histo(hName("hNHitsSeg", chid))->Fill(phiSeg->recHits().size()+zedSeg->recHits().size());


      // residual analysis
      if (phiSeg) {
        DTSegmentResidual res(phiSeg, *ch);
        res.run();
        vector<DTSegmentResidual::DTResidual> deltas=res.residuals();
        for (vector<DTSegmentResidual::DTResidual>::const_iterator delta=deltas.begin();
             delta!=deltas.end(); ++delta) {
          histo(hName("hHitResidualSeg", (*ch)->id()))->Fill((*delta).value);
          if ((*delta).side == DTEnums::Right ) 
            histo(hName("hHitResidualSegCellDX", (*ch)->id()))->Fill((*delta).value);
          else if ((*delta).side == DTEnums::Left )
            histo(hName("hHitResidualSegCellSX", (*ch)->id()))->Fill((*delta).value);

          histo2d(hName("hHitResidualSegVsWireDis",
                        (*ch)->id()))->Fill((*delta).wireDistance,(*delta).value);

          histo2d(hName("hHitResidualSegVsAngle",
                        (*ch)->id()))->Fill((*delta).angle,(*delta).value);
          histo2d(hName("hHitResidualSegVsNHits",
                        (*ch)->id()))->Fill(phiSeg->recHits().size(),(*delta).value);
          histo2d(hName("hHitResidualSegVsChi2",
                        (*ch)->id()))->Fill(phiSeg->chi2(),(*delta).value);

          if ( (*seg).hasPhi() ) 
            histo2d(hName("hHitResidualSegAlongWire",
                          (*ch)->id()))->Fill((*seg).localPosition().x(),
                                              (*delta).value);
        }
      }

      if (zedSeg) {
        const DTSuperLayer* sl= (*ch)->superLayer(2);
        DTSegmentResidual res(zedSeg, sl);
        res.run();
        vector<DTSegmentResidual::DTResidual> deltas=res.residuals();
        for (vector<DTSegmentResidual::DTResidual>::const_iterator delta=deltas.begin();
             delta!=deltas.end(); ++delta) {
          histo(hName("hHitResidualSeg", sl->id()))->Fill((*delta).value);
          if ((*delta).side == DTEnums::Right ) 
            histo(hName("hHitResidualSegCellDX", sl->id()))->Fill((*delta).value);
          else if ((*delta).side == DTEnums::Left )
            histo(hName("hHitResidualSegCellSX", sl->id()))->Fill((*delta).value);

          histo2d(hName("hHitResidualSegVsWireDis",
                        sl->id()))->Fill((*delta).wireDistance,(*delta).value);

          histo2d(hName("hHitResidualSegVsAngle",
                        sl->id()))->Fill((*delta).angle,(*delta).value);
          histo2d(hName("hHitResidualSegVsNHits",
                        sl->id()))->Fill(zedSeg->recHits().size(),(*delta).value);
          histo2d(hName("hHitResidualSegVsChi2",
                        sl->id()))->Fill(zedSeg->chi2(),(*delta).value);

          if ( (*seg).hasPhi() ) 
            histo2d(hName("hHitResidualSegAlongWire",
                          sl->id()))->Fill((*seg).localPosition().x(),
                                           (*delta).value);
        }
      } // if ZedSeg

    } // end loop segment in chamber

    // Efficiency analysis
    // first get number of segments vs number of hits

    // get No hits in this chamber
    // and no of 2d segments in chamber
    int nHitSl[3]={0,0,0};
    int nSeg2dSl[3]={0,0,0};
    for (int sl = 1; sl<=3; ++sl ) {
      if (sl==2 && chid.station()==4) continue;
      DTSuperLayerId slid(chid,sl);
      DTRecHitCollection::range rangeH =
        dtRecHits->get(DTRangeMapAccessor::layersBySuperLayer(slid));
      DTRecSegment2DCollection::range rangeS =
        segs2d->get(slid);
      nHitSl[sl-1]=rangeH.second-rangeH.first;
      nSeg2dSl[sl-1]=rangeS.second-rangeS.first;
      histo2d(hName("hNsegs2dVsNhits",slid))->Fill(nHitSl[sl-1],nSeg2dSl[sl-1]);
    }
    histo2d(hName("hNsegs4dVsNhits",chid))->Fill(nHitSl[0]+nHitSl[1]+nHitSl[2],nSegsCh);
    histo2d(hName("hNsegs4dVsNhitsPhi",chid))->Fill(nHitSl[0]+nHitSl[2],nSegsCh);
    histo2d(hName("hNsegs4dVsNhitsZed",chid))->Fill(nHitSl[1],nSegsCh);

    histo2d(hName("hNsegs4dVsNsegs2d",chid))->Fill(nSeg2dSl[0]+nSeg2dSl[1]+nSeg2dSl[2],nSegsCh);
    histo2d(hName("hNsegs4dVsNsegs2dPhi",chid))->Fill(nSeg2dSl[0]+nSeg2dSl[2],nSegsCh);
    histo2d(hName("hNsegs4dVsNsegs2dZed",chid))->Fill(nSeg2dSl[1],nSegsCh);

    histo2d(hName("hNsegs2dSL1VsNsegs2dSL3",chid))->Fill(nSeg2dSl[2],nSeg2dSl[0]);
    histo2d(hName("hNsegs2dSL1VsNsegs2dSL2",chid))->Fill(nSeg2dSl[1],nSeg2dSl[0]);
    histo2d(hName("hNsegs2dSL2VsNsegs2dSL3",chid))->Fill(nSeg2dSl[2],nSeg2dSl[1]);


  } // loop on all chamber

}

TH1F* DTAnalyzerDetailed::histo(const string& name) const{
  if (TH1F* h =  dynamic_cast<TH1F*>(theFile->Get(name.c_str())) ) return h;
  else throw cms::Exception("DTAnalyzerDetailed") << " Not a TH1F " << name;
}

TH2F* DTAnalyzerDetailed::histo2d(const string& name) const{
  if (TH2F* h =  dynamic_cast<TH2F*>(theFile->Get(name.c_str())) ) return h;
  else throw  cms::Exception("DTAnalyzerDetailed") << " Not a TH2F " << name;
}

string DTAnalyzerDetailed::toString(const DTLayerId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station()
    << "_Sl" << id.superLayer()
    << "_Lay" << id.layer();
  return result.str();
}

string DTAnalyzerDetailed::toString(const DTSuperLayerId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station()
    << "_Sl" << id.superLayer();
  return result.str();
}

string DTAnalyzerDetailed::toString(const DTChamberId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station();
  return result.str();
}

template<class T> string DTAnalyzerDetailed::hName(const string& s, const T& id) const {
  string name(toString(id));
  stringstream hName;
  hName << s << name;
  return hName.str();
}

void DTAnalyzerDetailed::createTH1F(const std::string& name,
                            const std::string& title,
                            const std::string& suffix,
                            int nbin,
                            const double& binMin,
                            const double& binMax) const {
  stringstream hName;
  stringstream hTitle;
  hName << name << suffix;
  hTitle << title << suffix;
  new TH1F(hName.str().c_str(), hTitle.str().c_str(), nbin,binMin,binMax);
}

void DTAnalyzerDetailed::createTH2F(const std::string& name,
                            const std::string& title,
                            const std::string& suffix,
                            int nBinX,
                            const double& binXMin,
                            const double& binXMax,
                            int nBinY,
                            const double& binYMin,
                            const double& binYMax) const {
  stringstream hName;
  stringstream hTitle;
  hName << name << suffix;
  hTitle << title << suffix;
  new TH2F(hName.str().c_str(), hTitle.str().c_str(), nBinX,binXMin,binXMax, nBinY,binYMin,binYMax);
}

DEFINE_FWK_MODULE(DTAnalyzerDetailed);
