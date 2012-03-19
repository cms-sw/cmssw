/******* \class DTSegAnalyzer *******
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
#include "RecoLocalMuon/DTSegment/test/DTSegAnalyzer.h"

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
DTSegAnalyzer::DTSegAnalyzer(const ParameterSet& pset) : _ev(0){

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
  // new TH1F("hMeanTimerSL","Mean timer per SL (ns)",100,200.,600.);
  // new TH1F("hDigiTimeSL", "Digi Time (ns) ",100,-100.,600.);

  new TH1F("hPosLeft", "Pos of Left hit (cm) in local frame", 100,-220.,220.);
  new TH1F("hPosRight", "Pos of Right hit (cm) in local frame", 100,-220.,220.);
  new TH1F("hMeanTimer", "Tmax ", 100,200.,600.);
  new TH1F("hMeanTimerSeg", "Tmax from segments hits ", 100,200.,600.);
  new TH2F("hMeanTimerSegAlongWire", "Tmax from segments hits vs pos along wire ", 40, -150., 150, 100,200.,600.);
  new TH2F("hMeanTimerSegVsAngle", "Tmax from segments hits vs angle ", 40, -0., 1.0, 100, 200.,600.);
  new TH2F("hMeanTimerSegVsNHits", "Tmax from segments hits vs n hits segment ", 10, 0.5, 10.5, 100, 200.,600.);

  // segs
  new TH1F("hnSegDT","Num seg DT", 50,0.,50.);
  new TH1F("hNSegs", "N segments ", 20,0.,20.);
  new TH1F("hChi2Seg", "Chi2 segments ", 25,0.,25.);
  new TH1F("hChi2SegPhi", "Chi2 segments phi ", 25,0.,25.);
  new TH1F("hChi2SegZed", "Chi2 segments zed ", 25,0.,25.);
  new TH1F("hNHitsSeg", "N hits segment ", 15,-0.5,14.5);
  new TH1F("hNHitsSegPhi", "N hits segment phi ", 12,-0.5,11.5);
  new TH2F("hNHitsSegPhiVsAngle", "N hits segment phi vs angle ", 40, .0, 1.0 ,12,-0.5,11.5);
  new TH2F("hNHitsSegPhiVsOtherHits", "N hits segment phi vs hits in SLs ", 20, -0.5, 19.5 ,12,-0.5,11.5);
  new TH2F("hNHitsSegPhiVsNumSegs", "N hits segment zed vs num segs in ch ", 6, -0.5, 5.5 ,12,-0.5,11.5);
  new TH2F("hNHitsSegPhiVsChi2", "N hits segment zed vs chi2/NDoF ", 25,0.,25.,12,-0.5,11.5);
  new TH1F("hNHitsSegZed", "N hits segment zed ", 10,-0.5,9.5);
  new TH2F("hNHitsSegZedVsAngle", "N hits segment zed vs angle ", 40, .0, 1.0 ,12,-0.5,11.5);
  new TH2F("hNHitsSegZedVsOtherHits", "N hits segment zed vs hits in SLs ", 20, -0.5, 19.5 ,12,-0.5,11.5);
  new TH2F("hNHitsSegZedVsNumSegs", "N hits segment zed vs num segs in ch ", 6, -0.5, 5.5 ,12,-0.5,11.5);
  new TH2F("hNHitsSegZedVsChi2", "N hits segment zed vs chi2/NDoF ", 25,0.,25.,12,-0.5,11.5);

  new TH1F("hHitResidualSeg", "Hits residual wrt segments ", 100,-.2,+.2);
  new TH1F("hHitResidualSegCellDX", "Hits residual wrt segments semicell DX ",  100,-.2,+.2);
  new TH1F("hHitResidualSegCellSX", "Hits residual wrt segments semicell SX",  100,-.2,+.2);
  new TH2F("hHitResidualSegAlongWire", "Hits residual wrt segments vs pos along wire ", 40, -150., 150, 100,-.2,+.2);
  new TH2F("hHitResidualSegVsWireDis", "Hits residual wrt segments vs wire distance ", 40, 0., 2.1 , 100,-.2,+.2);
  new TH2F("hHitResidualSegVsAngle", "Hits residual wrt segments vs impact angle ", 40, .0, 1.0 , 100,-.2,+.2);
  new TH2F("hHitResidualSegVsNHits", "Hits residual wrt segments vs num hits ", 10, -0.5, 9.5 , 100,-.2,+.2);
  new TH2F("hHitResidualSegVsChi2", "Hits residual wrt segments vs chi2 ", 25, .0, 25.0 , 100,-.2,+.2);
  new TH2F("hNsegs2dVsNhits", "N segs 2d vs n Hits ", 20, -0.5, 19.5 , 5, -0.5,4.5);

  // eff
  new TH2F("hNsegs4dVsNhits", "N segs 4d vs n Hits ", 20, 0.5, 20.5 , 5, -0.5,4.5);
  new TH2F("hNsegs4dVsNhitsPhi", "N segs 4d vs n HitsPhi ", 20, 0.5, 20.5 , 5, -0.5,4.5);
  new TH2F("hNsegs4dVsNhitsZed", "N segs 4d vs n HitsZed ", 20, 0.5, 20.5 , 5, -0.5,4.5);

  new TH2F("hNsegs4dVsNsegs2d", "N segs 4d vs n segs2d ", 4, 0.5, 4.5 , 5, -0.5,4.5);
  new TH2F("hNsegs4dVsNsegs2dPhi", "N segs 4d vs n segs2d Phi ", 4, 0.5, 4.5 , 5, -0.5,4.5);
  new TH2F("hNsegs4dVsNsegs2dZed", "N segs 4d vs n segs2d Zed ", 4, 0.5, 4.5 , 5, -0.5,4.5);

  new TH2F("hNsegs2dSL1VsNsegs2dSL3", "N segs 2d SL1 vs SL3 ", 5, -0.5, 4.5 , 5, -0.5,4.5);
  new TH2F("hNsegs2dSL1VsNsegs2dSL2", "N segs 2d SL1 vs SL2 ", 5, -0.5, 4.5 , 5, -0.5,4.5);
  new TH2F("hNsegs2dSL2VsNsegs2dSL3", "N segs 2d SL2 vs SL3 ", 5, -0.5, 4.5 , 5, -0.5,4.5);

  TH1::AddDirectory(dirStat);
}

/* Destructor */ 
DTSegAnalyzer::~DTSegAnalyzer() {
  theFile->cd();
  theFile->Write();
  theFile->Close();
}

/* Operations */ 
void DTSegAnalyzer::analyze(const Event & event,
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

void DTSegAnalyzer::analyzeDTHits(const Event & event,
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
    histo("hPosLeft")->Fill(xLeft);
    histo("hPosRight")->Fill(xRight);
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
      histo("hMeanTimer")->Fill(*tMax);
    }
  }

}

void DTSegAnalyzer::analyzeDTSegments(const Event & event,
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
    // Segment 4d in this chamber
    DTRecSegment4DCollection::range segsch= segs->get(chid);
    int nSegsCh=segsch.second-segsch.first;
    histo("hNSegs")->Fill(nSegsCh);

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
          histo("hMeanTimerSeg")->Fill(*tMax);
          histo2d("hMeanTimerSegVsNHits")->Fill(phiHits.size(), *tMax);
          if ( (*seg).hasZed() ) {
            histo2d("hMeanTimerSegAlongWire")->Fill((*seg).localPosition().y(),*tMax);
            histo2d("hMeanTimerSegVsAngle")->Fill(M_PI-(*seg).localDirection().theta(), *tMax);
          }
        }

        DTSuperLayerId slid3(phiSeg->chamberId(),3);
        /// Mean timer analysis
        DTMeanTimer meanTimer3(dtGeom->superLayer(slid3), phiHits, eventSetup, theSync);
        vector<double> tMaxs3=meanTimer3.run();
        for (vector<double>::const_iterator tMax=tMaxs3.begin() ; tMax!=tMaxs3.end();
             ++tMax) {
          histo("hMeanTimerSeg")->Fill(*tMax);
          histo2d("hMeanTimerSegVsNHits")->Fill(phiHits.size(), *tMax);
          if ( (*seg).hasZed() ) {
            histo2d("hMeanTimerSegAlongWire")->Fill((*seg).localPosition().y(), *tMax);
            histo2d("hMeanTimerSegVsAngle")->Fill(M_PI-(*seg).localDirection().theta(), *tMax);
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
          histo("hMeanTimerSeg")->Fill(*tMax);
          histo2d("hMeanTimerSegVsNHits")->Fill(zedHits.size(), *tMax);
          if ( (*seg).hasPhi() ) {
            histo2d("hMeanTimerSegAlongWire")->Fill((*seg).localPosition().x(), *tMax);
            histo2d("hMeanTimerSegVsAngle")->Fill(M_PI-(*seg).localDirection().theta(), *tMax);
          }
        }
      }

      histo("hChi2Seg")->Fill((*seg).chi2()/(*seg).degreesOfFreedom());
      if(phiSeg) {
        histo("hNHitsSegPhi")->Fill(phiSeg->recHits().size());
        histo2d("hNHitsSegPhiVsAngle")->Fill(M_PI-phiSeg->localDirection().theta(), phiSeg->recHits().size());
        // get the total numer of hits in this SL(s)
        DTSuperLayerId slid1(chid,1);
        DTRecHitCollection::range rangeH1 =
          dtRecHits->get(DTRangeMapAccessor::layersBySuperLayer(slid1));
        DTSuperLayerId slid3(chid,3);
        DTRecHitCollection::range rangeH3 =
          dtRecHits->get(DTRangeMapAccessor::layersBySuperLayer(slid3));
        histo2d("hNHitsSegPhiVsOtherHits")->Fill((rangeH1.second-rangeH1.first)+
                                                 (rangeH3.second-rangeH3.first),
                                                 phiSeg->recHits().size());
        histo2d("hNHitsSegPhiVsNumSegs")->Fill(nSegsCh,
                                               phiSeg->recHits().size());
        histo2d("hNHitsSegPhiVsChi2")->Fill(phiSeg->chi2()/phiSeg->degreesOfFreedom(),
                                            phiSeg->recHits().size());
        histo("hChi2SegPhi")->Fill(phiSeg->chi2()/phiSeg->degreesOfFreedom());
      }

      if (zedSeg) {
        histo("hNHitsSegZed")->Fill(zedSeg->recHits().size());
        histo2d("hNHitsSegZedVsAngle")->Fill(M_PI-zedSeg->localDirection().theta(),
                                             zedSeg->recHits().size());
        DTSuperLayerId slid(chid,2);
        DTRecHitCollection::range rangeH =
          dtRecHits->get(DTRangeMapAccessor::layersBySuperLayer(slid));
        histo2d("hNHitsSegZedVsOtherHits")->Fill(rangeH.second-rangeH.first,
                                                 zedSeg->recHits().size());
        histo2d("hNHitsSegZedVsNumSegs")->Fill(nSegsCh,
                                               zedSeg->recHits().size());
        histo2d("hNHitsSegZedVsChi2")->Fill(zedSeg->chi2()/zedSeg->degreesOfFreedom(),
                                            zedSeg->recHits().size());
        histo("hChi2SegZed")->Fill(zedSeg->chi2()/zedSeg->degreesOfFreedom());
      }
      if (phiSeg && zedSeg) 
        histo("hNHitsSeg")->Fill(phiSeg->recHits().size()+zedSeg->recHits().size());


      // residual analysis
      if (phiSeg) {
        DTSegmentResidual res(phiSeg, *ch);
        res.run();
        vector<DTSegmentResidual::DTResidual> deltas=res.residuals();
        for (vector<DTSegmentResidual::DTResidual>::const_iterator delta=deltas.begin();
             delta!=deltas.end(); ++delta) {
          histo("hHitResidualSeg")->Fill((*delta).value);
          if ((*delta).side == DTEnums::Right ) 
            histo("hHitResidualSegCellDX")->Fill((*delta).value);
          else if ((*delta).side == DTEnums::Left )
            histo("hHitResidualSegCellSX")->Fill((*delta).value);

          histo2d("hHitResidualSegVsWireDis")->Fill((*delta).wireDistance,(*delta).value);

          histo2d("hHitResidualSegVsAngle")->Fill((*delta).angle,(*delta).value);
          histo2d("hHitResidualSegVsNHits")->Fill(phiSeg->recHits().size(),(*delta).value);
          histo2d("hHitResidualSegVsChi2")->Fill(phiSeg->chi2(),(*delta).value);

          if ( (*seg).hasPhi() ) 
            histo2d("hHitResidualSegAlongWire")->Fill((*seg).localPosition().x(),
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
          histo("hHitResidualSeg")->Fill((*delta).value);
          if ((*delta).side == DTEnums::Right ) 
            histo("hHitResidualSegCellDX")->Fill((*delta).value);
          else if ((*delta).side == DTEnums::Left )
            histo("hHitResidualSegCellSX")->Fill((*delta).value);

          histo2d("hHitResidualSegVsWireDis")->Fill((*delta).wireDistance,(*delta).value);

          histo2d("hHitResidualSegVsAngle")->Fill((*delta).angle,(*delta).value);
          histo2d("hHitResidualSegVsNHits")->Fill(zedSeg->recHits().size(),(*delta).value);
          histo2d("hHitResidualSegVsChi2")->Fill(zedSeg->chi2(),(*delta).value);

          if ( (*seg).hasPhi() ) 
            histo2d("hHitResidualSegAlongWire")->Fill((*seg).localPosition().x(),
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
      histo2d("hNsegs2dVsNhits")->Fill(nHitSl[sl-1],nSeg2dSl[sl-1]);
    }
    histo2d("hNsegs4dVsNhits")->Fill(nHitSl[0]+nHitSl[1]+nHitSl[2],nSegsCh);
    histo2d("hNsegs4dVsNhitsPhi")->Fill(nHitSl[0]+nHitSl[2],nSegsCh);
    histo2d("hNsegs4dVsNhitsZed")->Fill(nHitSl[1],nSegsCh);

    histo2d("hNsegs4dVsNsegs2d")->Fill(nSeg2dSl[0]+nSeg2dSl[1]+nSeg2dSl[2],nSegsCh);
    histo2d("hNsegs4dVsNsegs2dPhi")->Fill(nSeg2dSl[0]+nSeg2dSl[2],nSegsCh);
    histo2d("hNsegs4dVsNsegs2dZed")->Fill(nSeg2dSl[1],nSegsCh);

    histo2d("hNsegs2dSL1VsNsegs2dSL3")->Fill(nSeg2dSl[2],nSeg2dSl[0]);
    histo2d("hNsegs2dSL1VsNsegs2dSL2")->Fill(nSeg2dSl[1],nSeg2dSl[0]);
    histo2d("hNsegs2dSL2VsNsegs2dSL3")->Fill(nSeg2dSl[2],nSeg2dSl[1]);


  } // loop on all chamber

}

TH1F* DTSegAnalyzer::histo(const string& name) const{
  if (TH1F* h =  dynamic_cast<TH1F*>(theFile->Get(name.c_str())) ) return h;
  else throw cms::Exception("DTSegAnalyzer") << " Not a TH1F " << name;
}

TH2F* DTSegAnalyzer::histo2d(const string& name) const{
  if (TH2F* h =  dynamic_cast<TH2F*>(theFile->Get(name.c_str())) ) return h;
  else throw  cms::Exception("DTSegAnalyzer") << " Not a TH2F " << name;
}

string DTSegAnalyzer::toString(const DTLayerId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station()
    << "_Sl" << id.superLayer()
    << "_Lay" << id.layer();
  return result.str();
}

string DTSegAnalyzer::toString(const DTSuperLayerId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station()
    << "_Sl" << id.superLayer();
  return result.str();
}

string DTSegAnalyzer::toString(const DTChamberId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station();
  return result.str();
}

template<class T> string DTSegAnalyzer::hName(const string& s, const T& id) const {
  string name(toString(id));
  stringstream hName;
  hName << s << name;
  return hName.str();
}

void DTSegAnalyzer::createTH1F(const std::string& name,
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

void DTSegAnalyzer::createTH2F(const std::string& name,
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

DEFINE_FWK_MODULE(DTSegAnalyzer);
