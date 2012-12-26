/******* \class STAnalyzer *******
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
#include "RecoLocalMuon/DTSegment/test/STAnalyzer.h"

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

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

/* C++ Headers */
#include <iostream>
#include <cmath>
using namespace std;

/* ====================================================================== */

/* Constructor */ 
STAnalyzer::STAnalyzer(const ParameterSet& pset) : _ev(0){
  if (debug) cout << "STAnalyzer::STAnalyzer" << endl;

  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");

  // the name of the 1D rec hits collection
  theRecHits1DLabel = pset.getParameter<string>("recHits1DLabel");

  // the name of the 2D rec hits collection
  theRecHits2DLabel = pset.getParameter<string>("recHits2DLabel");

  // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getParameter<string>("recHits4DLabel");

  // the name of the STA rec hits collection
  theSTAMuonLabel = pset.getParameter<string>("SALabel");

  thePropagatorName = pset.getParameter<std::string>("PropagatorName");
  thePropagator = 0;

  doSA = pset.getParameter<bool>("doSA");

  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  bool dirStat=TH1::AddDirectoryStatus();
  TH1::AddDirectory(kTRUE);

  // CosmicMuon
  new TH1F("hNSA","Num SA tracks in events", 6, -0.5, 5.5);
  new TH1F("hNSADifferentSectors","Num SA tracks with hits in different sectors", 6, -0.5, 5.5);
  new TH1F("hNhitsSA","Num hits in SA tracks", 50, .5, 50.5);
  new TH1F("hChi2SA","#chi^{2}/NDoF for SA tracks", 100, 0, 100.);
  new TH1F("hPIPSA","p for SA tracks @ IP", 100, 0., 100);
  new TH1F("hPtIPSA","pt for SA tracks @ IP", 100, 0., 100);
  new TH1F("hPhiIPSA","#phi for SA tracks @ IP", 100, -M_PI_2, M_PI_2);
  new TH1F("hEtaIPSA","#eta for SA tracks @ IP", 100, -2.5, 2.5);
  new TH1F("hPInnerTSOSSA","p for SA tracks @ innermost TSOS", 100, 0., 100);
  new TH1F("hPtInnerTSOSSA","pt for SA tracks @ innermost TSOS", 100, 0., 100);
  new TH1F("hPhiInnerTSOSSA","#phi for SA tracks @ innermost TSOS", 100, -M_PI_2, M_PI_2);
  new TH1F("hEtaInnerTSOSSA","#eta for SA tracks @ innermost TSOS", 100, -2.5, 2.5);
  new TH1F("hInnerRSA","Radius of innermost TSOS for SA tracks", 100, 400, 1000.);
  new TH1F("hOuterRSA","Radius of outermost TSOS for SA tracks", 100, 400, 1000.);
  new TH2F("hInnerOuterRSA","Radius of outermost TSOS vs innermost one for SA tracks",
           40, 300, 1000.,40, 300, 1000.);

  new TH2F("hNSAVsNHits","Num 1d Hits vs N SA", 100,-0.5,39.5, 5,-0.5, 4.5);
  new TH2F("hNSAVsNSegs2D","Num 2d Segs vs N SA", 20,-0.5,19.5, 5,-0.5, 4.5);
  new TH2F("hNSAVsNSegs4D","Num 4d Segs vs N SA", 10,-0.5,9.5, 5,-0.5, 4.5);

  new TH2F("hNHitsSAVsNHits","Num 1d Hits vs N Hits SA", 40,-0.5,39.5, 100,0,100);
  new TH2F("hNHitsSAVsNSegs2D","Num 2d Segs vs N Hits SA", 20,-0.5,19.5,100,0,100);
  new TH2F("hNHitsSAVsNSegs4D","Num 4d Segs vs N Hits SA", 10,-0.5,9.5, 100,0,100);

  new TH1F("hSANLayers","Num Layers used SA", 50,-0.5,49.5);
  new TH1F("hSANSLs","Num SuperLayers used SA", 15,-0.5,14.5);
  new TH1F("hSANLayersPerSL","Num Layers per SL used SA", 15,-0.5,14.5);
  new TH1F("hSANChambers","Num Chambers used SA", 6,-0.5,5.5);
  new TH1F("hSANLayersPerChamber","Num Layers per SL used SA", 15,-0.5,14.5);
  new TH1F("hSANSLsPerChamber","Num SLs per Chamber used SA", 5,-0.5,4.5);

  new TH2F("hSANLayersVsNChambers","Num Layers Vs Num Chambers used SA", 6,-0.5,5.5, 50,-0.5,49.5);
  new TH2F("hSANSLsVsNChambers","Num SuperLayers Vs Num Chambers used SA", 6,-0.5,5.5, 16,-0.5,15.5);

  new TH2F("hSANHitsVsNLayers","Num Hits vs Num Layers used SA", 50,-0.5,49.5, 50,0.,50.);
  new TH2F("hSANHitsVsNSLs","Num Hits vs Num SuperLayers used SA", 16,-0.5,15.5,50,0.,50.);
  new TH2F("hSANHitsVsNChambers","Num Hits vs Num Chambers used SA", 6,-0.5,5.5,50,0.,50.);

  new TH2F("hHitsPosXYSA","Hits position (x,y) SA",100,-800,800,100,-800,800);
  new TH2F("hHitsPosXZSA","Hits position (x,z) SA",100,-800,800,100,-670,670);
  new TH2F("hHitsPosYZSA","Hits position (y,z) SA",100,-670,670,100,-800,800);

  new TH1F("nHitsSAInChamber","Num STA recHits in each chamber", 4, 0.5, 4.5);
  new TH1F("nHitsSAInSL","Num STA recHits in each sl", 50, 0.5, 50.5);
  new TH1F("nHitsSAInLayer","Num STA recHits in each layer", 75, 25.5, 100.5);

  new TH1F("hHitsLostChamber","Num hits in det w/o associated hits chamber", 30, 0.,30.);
  new TH1F("hHitsLostSL","Num hits in det w/o associated hits SL", 30, 0.,30.);
  new TH1F("hHitsLostLayer","Num hits in det w/o associated hits Layer", 30, 0.,30.);

  new TH1F("hMinDistChamberFound","Min distance between extrap post and closest hit in det w associated hits chamber", 100, 0., 300.);
  new TH1F("hMinDistChamberNotFound","Min distance between extrap post and closest hit in det w/o associated hits chamber", 100, 0., 300.);
  new TH1F("hMinPhiChamberFound","Min phi between extrap post and closest hit in det w associated hits chamber", 100, -1., 1.);
  new TH1F("hMinPhiChamberNotFound","Min phi between extrap post and closest hit in det w/o associated hits chamber", 100, -1., 1.);
  new TH1F("hMinThetaChamberFound","Min theta between extrap post and closest hit in det w associated hits chamber", 100, -1., 1.);
  new TH1F("hMinThetaChamberNotFound","Min theta between extrap post and closest hit in det w/o associated hits chamber", 100, -1., 1.);

  new TH1F("hMinDistSLFound","Min distance between extrap post and closest hit in det w associated hits SL", 100, 0., 300.);
  new TH1F("hMinDistSLNotFound","Min distance between extrap post and closest hit in det w/o associated hits SL", 100, 0., 300.);
  new TH1F("hMinThetaSLFound","Min theta between extrap post and closest hit in det w associated hits SL", 100, -1., 1.);
  new TH1F("hMinThetaSLNotFound","Min theta between extrap post and closest hit in det w/o associated hits SL", 100, -1., 1.);

  new TH1F("hMinDistLayerFound","Min distance between extrap post and closest hit in det w associated hits Layer", 100, 0., 300.);
  new TH1F("hMinDistLayerNotFound","Min distance between extrap post and closest hit in det w/o associated hits Layer", 100, 0., 300.);

  TH1::AddDirectory(dirStat);
}

/* Destructor */ 
STAnalyzer::~STAnalyzer() {
  theFile->cd();
  theFile->Write();
  theFile->Close();
}

void STAnalyzer::beginRun(const edm::Run& run, const EventSetup& setup) {
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  setup.get<MuonGeometryRecord>().get(dtGeom);

  static bool FirstPass = true;

  if(FirstPass){
  const std::vector<DTChamber*> & chs = dtGeom->chambers();
  for (std::vector<DTChamber*>::const_iterator ch = chs.begin();
       ch!=chs.end() ; ++ch) 
    hitsPerChamber[(*ch)->id()]=0;
  }

  FirstPass = false;

}

/* Operations */ 
void STAnalyzer::beginJob() {
  if (debug) cout << "STAnalyzer::beginJob" << endl;
}

void STAnalyzer::analyze(const Event & event,
                         const EventSetup& eventSetup) {
  if (debug) cout << "STAnalyzer::analyze" << endl;

  if (debug) 
    cout << "Run:Event analyzed " << event.id().run() << ":" << event.id().event() <<
      " Num " << _ev++ << endl;

  if (doSA) analyzeSATrack(event, eventSetup);
}

void STAnalyzer::analyzeSATrack(const Event & event,
                                const EventSetup& eventSetup){
  if (debug) cout << "STAnalyzer::analyzeSATrack" << endl;
  if (!thePropagator){
    ESHandle<Propagator> prop;
    eventSetup.get<TrackingComponentsRecord>().get(thePropagatorName, prop);
    thePropagator = prop->clone();
    thePropagator->setPropagationDirection(anyDirection);
  }

  MuonPatternRecoDumper muonDumper;

  // Get the 4D rechit collection from the event -------------------
  edm::Handle<DTRecSegment4DCollection> segs;
  event.getByLabel(theRecHits4DLabel, segs);

  // Get the 2D rechit collection from the event -------------------
  edm::Handle<DTRecSegment2DCollection> segs2d;
  event.getByLabel(theRecHits2DLabel, segs2d);

  // Get the 1D rechits from the event --------------
  Handle<DTRecHitCollection> dtRecHits; 
  event.getByLabel(theRecHits1DLabel, dtRecHits);

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theSTAMuonLabel, staTracks);
  if (debug) cout << "SA " << staTracks->size() << endl;
  // Get the RecTrack collection from the event
  std::vector<Handle<reco::TrackCollection> > tracks;
  event.getManyByType (tracks);
  std::vector<Handle<reco::TrackCollection> >::iterator i (tracks.begin ()), end (tracks.end ());
  for (; i != end; ++i) {
    if (debug) cout << "ALL Tracks id " << (*i).id() << endl;
    if (debug) cout << "ALL Tracks size " << (*i).product()->size() << endl;
  }

  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get(theMGField);

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  reco::TrackCollection::const_iterator staTrack;

  double recPt=0.;
  histo("hNSA")->Fill(staTracks->size());
  histo2d("hNSAVsNHits")->Fill(dtRecHits->size(),staTracks->size());
  histo2d("hNSAVsNSegs2D")->Fill(segs2d->size(),staTracks->size());
  histo2d("hNSAVsNSegs4D")->Fill(segs->size(),staTracks->size());

  if (debug && staTracks->size() ) 
    cout << endl<<"R:E " << event.id().run() << ":" << event.id().event() << 
      " SA " << staTracks->size() << endl;

  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 

    if (debug) {
      cout << muonDumper.dumpFTS(track.impactPointTSCP().theState());

      recPt = track.impactPointTSCP().momentum().perp();    
      cout<<" p: "<<track.impactPointTSCP().momentum().mag()<< " pT: "<<recPt<<endl;
      cout<<" normalized chi2: "<<track.normalizedChi2()<<endl;
    }

    histo("hChi2SA")->Fill(track.normalizedChi2());

    histo("hPIPSA")->Fill(track.impactPointTSCP().momentum().mag());
    histo("hPtIPSA")->Fill(recPt);
    histo("hPhiIPSA")->Fill(track.impactPointTSCP().momentum().phi());
    histo("hEtaIPSA")->Fill(track.impactPointTSCP().momentum().eta());


    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();
    if (debug) {
      cout << "Inner TSOS:"<<endl;
      cout << muonDumper.dumpTSOS(innerTSOS);
      cout<<" p: "<<innerTSOS.globalMomentum().mag()<< " pT: "<<innerTSOS.globalMomentum().perp()<<endl;
    }

    // try to extrapolate using thePropagator

    // Get a surface (here a cylinder of radius 1290mm) ECAL
    float radiusECAL = 129.0; // radius in centimeter
    Cylinder::PositionType pos0;
    Cylinder::RotationType rot0;
    const Cylinder::CylinderPointer ecal = Cylinder::build(radiusECAL,pos0, rot0);
    //cout << "Cyl " << ecal->radius() << endl;

    TrajectoryStateOnSurface tsosAtEcal =
      thePropagator->propagate(*innerTSOS.freeState(), *ecal);
    // if (tsosAtEcal.isValid())
    //   cout << "extrap to ECAL " << tsosAtEcal.globalPosition() << " r " <<
    //     tsosAtEcal.globalPosition().perp() << endl;
    // else 
    //   cout << "Extrapolation to ECAL failed" << endl;

    // Get a surface (here a cylinder of radius 1811 mm) HCAL
    float radiusHCAL = 181.1; // radius in centimeter
    const Cylinder::CylinderPointer hcal = Cylinder::build(radiusHCAL,pos0, rot0);
    //cout << "Cyl " << hcal->radius() << endl;

    TrajectoryStateOnSurface tsosAtHcal =
      thePropagator->propagate(*innerTSOS.freeState(), *hcal);
    // if (tsosAtHcal.isValid())
    //   cout << "extrap to HCAL " << tsosAtHcal.globalPosition() << " r " <<
    //     tsosAtHcal.globalPosition().perp() << endl;
    // else 
    //   cout << "Extrapolation to HCAL failed" << endl;

    histo("hPInnerTSOSSA")->Fill(innerTSOS.globalMomentum().mag());
    histo("hPtInnerTSOSSA")->Fill(innerTSOS.globalMomentum().perp());
    histo("hPhiInnerTSOSSA")->Fill(innerTSOS.globalMomentum().phi());
    histo("hEtaInnerTSOSSA")->Fill(innerTSOS.globalMomentum().eta());

    histo("hNhitsSA")->Fill(staTrack->recHitsSize());

    histo2d("hNHitsSAVsNHits")->Fill(dtRecHits->size(),staTrack->recHitsSize());
    histo2d("hNHitsSAVsNSegs2D")->Fill(segs2d->size(),staTrack->recHitsSize());
    histo2d("hNHitsSAVsNSegs4D")->Fill(segs->size(),staTrack->recHitsSize());

    histo("hInnerRSA")->Fill(sqrt(staTrack->innerPosition().perp2()));
    histo("hOuterRSA")->Fill(sqrt(staTrack->outerPosition().perp2()));
    histo2d("hInnerOuterRSA")->Fill(sqrt(staTrack->innerPosition().perp2()),
                                    sqrt(staTrack->outerPosition().perp2()));

    if (debug) cout<<"RecHits:"<<endl;

    trackingRecHit_iterator rhbegin = staTrack->recHitsBegin();
    trackingRecHit_iterator rhend = staTrack->recHitsEnd();
    
    // zero's the maps
    for ( std::map<DTChamberId, int>::iterator h=hitsPerChamber.begin(); h!=hitsPerChamber.end(); ++h) (*h).second=0;
    for ( std::map<DTSuperLayerId, int>::iterator h=hitsPerSL.begin(); h!=hitsPerSL.end(); ++h) (*h).second=0;
    for ( std::map<DTLayerId, int>::iterator h=hitsPerLayer.begin(); h!=hitsPerLayer.end(); ++h) (*h).second=0;
    
    int firstHitWheel = 0; // the Wheel of first hit
    int lastHitWheel = 0; // the Wheel of last hit
    int firstHitSector = 0; // the sector of first hit
    int lastHitSector = 0; // the sector of last hit
    for(trackingRecHit_iterator recHit = rhbegin; recHit != rhend; ++recHit){
      const GeomDet* geomDet = theTrackingGeometry->idToDet((*recHit)->geographicalId());
      GlobalPoint gpos=geomDet->toGlobal((*recHit)->localPosition());
      if (debug) cout<<"r: "<< gpos.perp() <<" z: "<<gpos.z() <<endl;
      histo2d("hHitsPosXYSA")->Fill(gpos.x(),gpos.y());
      histo2d("hHitsPosXZSA")->Fill(gpos.z(),gpos.x());
      histo2d("hHitsPosYZSA")->Fill(gpos.z(),gpos.y());

      if (const DTLayer* lay=dynamic_cast<const DTLayer*>(geomDet)) {
        if (firstHitSector == 0 ) firstHitSector=lay->id().sector();
        lastHitSector=lay->id().sector();
        if (firstHitWheel == 0 ) firstHitWheel=lay->id().wheel();
        lastHitWheel=lay->id().wheel();

        if(debug)cout << "Layer " << lay->id() << endl;
        histo("nHitsSAInChamber")->Fill(lay->id().station());
        hitsPerChamber[lay->id()]++;
        histo("nHitsSAInSL")->Fill(lay->id().station()*10+lay->id().superlayer());
        hitsPerSL[lay->id()]++;
        histo("nHitsSAInLayer")->Fill(lay->id().station()*20+lay->id().superlayer()*5+lay->id().layer());
        hitsPerLayer[lay->id()]++;
      }
      if (const DTSuperLayer* lay=dynamic_cast<const DTSuperLayer*>(geomDet)) 
        cout << "SuperLayer " << lay->id() << endl;
      if (const DTChamber* lay=dynamic_cast<const DTChamber*>(geomDet)) 
        cout << "Chamber " << lay->id() << endl;

      TrajectoryStateOnSurface extraptsos = thePropagator->propagate(innerTSOS,
                                                                     geomDet->specificSurface());
    }
    if(debug) {
      cout << "PerChamber " << muonDumper.dumpTSOS(innerTSOS) << endl;
      for ( std::map<DTChamberId, int>::iterator h=hitsPerChamber.begin(); h!=hitsPerChamber.end(); ++h)
        if ((*h).second ) cout << (*h).first << ":" << (*h).second << endl;
      cout << "=====" << endl;

      cout << "PerSL " << endl;
      for ( std::map<DTSuperLayerId, int>::iterator h=hitsPerSL.begin(); h!=hitsPerSL.end(); ++h) 
        if ((*h).second )  cout << (*h).first << ":" << (*h).second << endl;
      cout << "=====" << endl;
      cout << "PerLayer " << endl;
      for ( std::map<DTLayerId, int>::iterator h=hitsPerLayer.begin(); h!=hitsPerLayer.end(); ++h) 
        if ((*h).second ) cout << (*h).first << ":" << (*h).second << endl;
      cout << "=====" << endl;
    }
    
    

    // Get the DT Geometry
    ESHandle<DTGeometry> dtGeom;
    eventSetup.get<MuonGeometryRecord>().get(dtGeom);

    // Get the 1D rechits from the event --------------
    Handle<DTRecHitCollection> hits1d; 
    event.getByLabel(theRecHits1DLabel, hits1d);

    // Get the 2D rechit collection from the event -------------------
    edm::Handle<DTRecSegment2DCollection> segs2d;
    event.getByLabel(theRecHits2DLabel, segs2d);

    // Get the 4D rechit collection from the event -------------------
    edm::Handle<DTRecSegment4DCollection> segs;
    event.getByLabel(theRecHits4DLabel, segs);

    // Now loop on chamber of relevant sector and look if there are hits in all
    // of them
    if (firstHitWheel == lastHitWheel && (
         (firstHitSector == lastHitSector) || 
         (firstHitSector==10 && lastHitSector ==14) || 
         (firstHitSector==14 && lastHitSector ==10) || 
         (firstHitSector==13 && lastHitSector ==4) ||
         (firstHitSector==4 && lastHitSector ==13) )) {
      int nLay=0, nSL=0, nCh=0;
      for (int c=1; c<=5;++c) {
        // cout << "c, lh, fh " << c <<" " << firstHitSector << " " <<
        //   lastHitSector << endl;
        if (c==5 && firstHitSector==14) firstHitSector=10;
        else if (c==5 && lastHitSector==14) lastHitSector=10;
        else if (c==5 && firstHitSector==13) firstHitSector=4;
        else if (c==5 && lastHitSector==13) lastHitSector=4;
        else if (c==5) continue;
        int cc=c;
        if (c==5) cc=4;

        int sect = min(firstHitSector,lastHitSector); 
        DTChamberId chid(firstHitWheel, cc, sect);
        const DTChamber * ch = dtGeom->chamber(chid);
        if (hitsPerChamber[chid]==0) {
          missingHit(dtGeom, segs, ch, innerTSOS, false);
        } else {
          missingHit(dtGeom, segs, ch, innerTSOS, true);
          nCh++;
          //cout << "Hits in " << chid << " = " << hitsPerChamber[chid]<< endl;
        } 
        int nLayPerCh=0, nSLPerCh=0;
        for (int sl=1; sl<=3; ++sl) {
          if (sl==2 && cc==4) continue;
          DTSuperLayerId slid(chid, sl);
          const DTSuperLayer* dtsl = dtGeom->superLayer(slid);
          if (hitsPerSL[slid]==0) {
            //cout << "Hits missing in " << slid << endl;
            missingHit(dtGeom, segs2d, dtsl, innerTSOS, false);
          } else {
            missingHit(dtGeom, segs2d, dtsl, innerTSOS, true);
            nSL++;
            nSLPerCh++;
            //cout << "Hits in " << slid << " = " << hitsPerSL[slid]<< endl;
          }
          int nLayPerSL=0;
          for (int l=1; l<=4; ++l) {
            DTLayerId lid(slid, l);
            const DTLayer* lay = dtGeom->layer(lid);
            if (hitsPerLayer[lid]==0) {
              //cout << "Hits missing in " << lid << endl;
              missingHit(dtGeom, hits1d, lay, innerTSOS, false);
            } else {
              missingHit(dtGeom, hits1d, lay, innerTSOS, true);
              nLay++;
              nLayPerCh++;
              nLayPerSL++;
              //cout << "Hits in " << lid << " = " << hitsPerLayer[lid]<< endl;
            }
          }
          histo("hSANLayersPerSL")->Fill(nLayPerSL);
        }
        histo("hSANLayersPerChamber")->Fill(nLayPerCh);
        histo("hSANSLsPerChamber")->Fill(nSLPerCh);
      }

      // how many different Chambers, SL, Layers are used in a SA?
      histo("hSANLayers")->Fill(nLay);
      histo("hSANSLs")->Fill(nSL);
      histo("hSANChambers")->Fill(nCh);
      histo2d("hSANLayersVsNChambers")->Fill(nCh, nLay);
      histo2d("hSANSLsVsNChambers")->Fill(nCh,nSL);
      // n det vs num hits used
      histo2d("hSANHitsVsNLayers")->Fill(nLay,staTrack->recHitsSize());
      histo2d("hSANHitsVsNSLs")->Fill(nSL, staTrack->recHitsSize());
      histo2d("hSANHitsVsNChambers")->Fill(nCh, staTrack->recHitsSize());
      if (nCh>4) cout << endl<<"R:E " << event.id().run() << ":" << event.id().event() << 
        " nCh " << nCh << endl;
    } else { // different wheel/sector
      // cout << "First/Last hit in different wheel/sector " << firstHitSector
      //    << " " << lastHitSector<< endl;
      histo("hNSADifferentSectors")->Fill(1);
    }

  }
  if (debug) cout<<"---"<<endl;  
}

template <typename T, typename C> 
void STAnalyzer::missingHit(const ESHandle<DTGeometry>& dtGeom,
                            const Handle<C>& segs,
                            const T* ch,
                            const TrajectoryStateOnSurface& startTsos,
                            bool found) {

  if (!ch) return;
  TrajectoryStateOnSurface extraptsos =
    thePropagator->propagate(startTsos, ch->specificSurface());
  //cout << "extrap " << extraptsos.isValid() << endl;
  if (extraptsos.isValid() ) {
    //cout << "extraptsos " << extraptsos.localPosition() << endl;
    bool inside =
      ch->specificSurface().bounds().inside(extraptsos.localPosition());
    //cout << "Is extrap inside? " << inside << endl;
    if (inside) {
      // Here I should loop on all hits of this chamber (if any) and get
      // the closest
      typename C::range segsch= segs->get(ch->id());
      //int nSegsCh=segsch.second-segsch.first;
      //cout << "Hits in chamber " << segsch.second-segsch.first << endl;
      fillMinDist(segsch, ch, extraptsos, found);
    }
  }
}

void STAnalyzer::fillMinDist(const DTRecSegment4DCollection::range segsch,
                       const DTChamber* ch,
                       const TrajectoryStateOnSurface& extraptsos,
                       bool found) {
  int nSegsCh=segsch.second-segsch.first;
  histo("hHitsLostChamber")->Fill(nSegsCh);
  if (nSegsCh) {
    LocalPoint extrapPos=extraptsos.localPosition();
    LocalVector extrapDir=extraptsos.localDirection();
    //cout << "Extrap pos " << extrapPos << endl;
    double minDist=99999.;
    double minPhi=0., minTheta=0.;
    for (DTRecSegment4DCollection::const_iterator hit=segsch.first; hit!=segsch.second;
         ++hit) {
      //cout << "Hit pos " << hit->localPosition() << endl;
      LocalVector dist = hit->localPosition() - extrapPos;
      //cout << "dist " << dist << " " << dist.perp() << endl;
      if (dist.perp()<minDist) {
        minDist=dist.perp();
        minPhi=hit->localDirection().phi()-extrapDir.phi();
        minTheta=hit->localDirection().theta()-extrapDir.theta();
      }
    }
    string inOut=(found) ? "Found": "NotFound";
    histo(string("hMinDistChamber")+inOut)->Fill(minDist);
    histo(string("hMinPhiChamber")+inOut)->Fill(minPhi);
    histo(string("hMinThetaChamber")+inOut)->Fill(minTheta);
  }
}

void STAnalyzer::fillMinDist(const DTRecSegment2DCollection::range segsch,
                       const DTSuperLayer* ch,
                       const TrajectoryStateOnSurface& extraptsos,
                       bool found) {
  int nSegsCh=segsch.second-segsch.first;
  histo("hHitsLostSL")->Fill(nSegsCh);
  if (nSegsCh) {
    LocalPoint extrapPos=extraptsos.localPosition();
    LocalVector extrapDir=extraptsos.localDirection();
    //cout << "Extrap pos " << extrapPos << endl;
    double minDist=99999.;
    double minTheta=0.;
    for (DTRecSegment2DCollection::const_iterator hit=segsch.first; hit!=segsch.second;
         ++hit) {
      //cout << "Hit pos " << hit->localPosition() << endl;
      LocalVector dist = hit->localPosition() - extrapPos;
      //cout << "dist " << dist << " " << dist.perp() << endl;
      if (dist.perp()<minDist) {
        minDist=dist.x();
        minTheta=hit->localDirection().theta()-extrapDir.theta();
      }
    }
    string inOut=(found) ? "Found": "NotFound";
    histo(string("hMinDistSL")+inOut)->Fill(minDist);
    histo(string("hMinThetaSL")+inOut)->Fill(minTheta);
  }
}

void STAnalyzer::fillMinDist(const DTRecHitCollection::range segsch,
                       const DTLayer* ch,
                       const TrajectoryStateOnSurface& extraptsos,
                       bool found) {
  int nSegsCh=segsch.second-segsch.first;
  histo("hHitsLostLayer")->Fill(nSegsCh);
  if (nSegsCh) {
    LocalPoint extrapPos=extraptsos.localPosition();
    //cout << "Extrap pos " << extrapPos << endl;
    double minDist=99999.;
    for (DTRecHitCollection::const_iterator hit=segsch.first; hit!=segsch.second;
         ++hit) {
      //cout << "Hit pos " << hit->localPosition() << endl;
      LocalVector dist = hit->localPosition() - extrapPos;
      //cout << "dist " << dist << " " << dist.perp() << endl;
      if (dist.perp()<minDist) minDist=dist.x();
    }
    string inOut=(found) ? "Found": "NotFound";
    histo(string("hMinDistLayer")+inOut)->Fill(minDist);
  }
}

TH1F* STAnalyzer::histo(const string& name) const{
  if (TH1F* h =  dynamic_cast<TH1F*>(theFile->Get(name.c_str())) ) return h;
  else throw cms::Exception("STAnalyzer") << " Not a TH1F " << name;
}

TH2F* STAnalyzer::histo2d(const string& name) const{
  if (TH2F* h =  dynamic_cast<TH2F*>(theFile->Get(name.c_str())) ) return h;
  else throw  cms::Exception("STAnalyzer") << " Not a TH2F " << name;
}

string STAnalyzer::toString(const DTLayerId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station()
    << "_Sl" << id.superLayer()
    << "_Lay" << id.layer();
  return result.str();
}

string STAnalyzer::toString(const DTSuperLayerId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station()
    << "_Sl" << id.superLayer();
  return result.str();
}

string STAnalyzer::toString(const DTChamberId& id) const {
  stringstream result;
  result << "_Wh" << id.wheel() 
    << "_Sec" << id.sector() 
    << "_St" << id.station();
  return result.str();
}

template<class T> string STAnalyzer::hName(const string& s, const T& id) const {
  string name(toString(id));
  stringstream hName;
  hName << s << name;
  return hName.str();
}

void STAnalyzer::createTH1F(const std::string& name,
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

void STAnalyzer::createTH2F(const std::string& name,
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

DEFINE_FWK_MODULE(STAnalyzer);
