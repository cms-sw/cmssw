// -*- C++ -*-
//
// Package:    CSCSkim
// Class:      CSCSkim
// 
/**\class CSCSkim CSCSkim.cc RecoLocalMuon/CSCSkim/src/CSCSkim.cc

 Description: Offline skim module for CSC cosmic ray data

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Schmitt
//         Created:  Sat Jul 12 17:43:33 CEST 2008
// $Id: CSCSkim.cc,v 1.12 2012/01/10 14:04:51 eulisse Exp $
//
//
//======================================================================
//
// CSCSkim:
//
// A simple skim module for extracting generally useful events from
// the cosmic-ray runs (CRUZET-n and CRAFT).  The selected events
// should be useful for most CSC-DPG and muon-POG studies.  However,
// the selection requirements may bias the sample with respect to
// trigger requirements and certain noise and efficiency-related
// studies.
//
// Types of skims:   (typeOfSkim control word)
//    1  = loose skim demanding hit chambers and/or segments
//    2  = ask for hit chambers in both endcaps
//    3  = segments in neighboring chambers - good for alignment
//    4  = messy events
//    5  = select events with DIGIs from one particular chamber
//    6  = overlap with DT
//    7  = nearly horizontal track going through ME1/1,2/1,3/1,4/1
//    8  = ask for one long cosmic stand-alone muon track
//    9  = selection for magnetic field studies
//
//
//======================================================================

#include "DPGAnalysis/Skims/interface/CSCSkim.h"

using namespace std;
using namespace edm;


//===================
//  CONSTRUCTOR
//===================
CSCSkim::CSCSkim(const edm::ParameterSet& pset)
{

  // input tags
  cscRecHitTag  = pset.getParameter<edm::InputTag>("cscRecHitTag");
  cscSegmentTag = pset.getParameter<edm::InputTag>("cscSegmentTag");
  SAMuonTag     = pset.getParameter<edm::InputTag>("SAMuonTag");
  GLBMuonTag    = pset.getParameter<edm::InputTag>("GLBMuonTag");
  trackTag      = pset.getParameter<edm::InputTag>("trackTag");

  // Get the various input parameters
  outputFileName     = pset.getUntrackedParameter<std::string>("outputFileName","outputSkim.root");
  histogramFileName  = pset.getUntrackedParameter<std::string>("histogramFileName","histos.root");
  typeOfSkim              = pset.getUntrackedParameter<int>("typeOfSkim",1);
  nLayersWithHitsMinimum  = pset.getUntrackedParameter<int>("nLayersWithHitsMinimum",3);
  minimumHitChambers      = pset.getUntrackedParameter<int>("minimumHitChambers",1);
  minimumSegments         = pset.getUntrackedParameter<int>("minimumSegments",3);
  demandChambersBothSides = pset.getUntrackedParameter<bool>("demandChambersBothSides",false);
  makeHistograms          = pset.getUntrackedParameter<bool>("makeHistograms",false);
  makeHistogramsForMessyEvents = pset.getUntrackedParameter<bool>("makeHistogramsForMessyEvebts",false);
  whichEndcap             = pset.getUntrackedParameter<int>("whichEndcap",2);
  whichStation            = pset.getUntrackedParameter<int>("whichStation",3);
  whichRing               = pset.getUntrackedParameter<int>("whichRing",2);
  whichChamber            = pset.getUntrackedParameter<int>("whichChamber",24);

  // for BStudy selection (skim type 9)
  pMin               = pset.getUntrackedParameter<double>("pMin",3.);
  zLengthMin         = pset.getUntrackedParameter<double>("zLengthMin",200.);
  nCSCHitsMin        = pset.getUntrackedParameter<int>("nCSCHitsMin",9);
  zInnerMax          = pset.getUntrackedParameter<double>("zInnerMax",9000.);
  nTrHitsMin         = pset.getUntrackedParameter<int>("nTrHitsMin",8);
  zLengthTrMin       = pset.getUntrackedParameter<double>("zLengthTrMin",180.);
  rExtMax            = pset.getUntrackedParameter<double>("rExtMax",3000.);
  redChiSqMax        = pset.getUntrackedParameter<double>("redChiSqMax",20.);
  nValidHitsMin      = pset.getUntrackedParameter<int>("nValidHitsMin",8);

  LogInfo("[CSCSkim] Setup")
    << "\n\t===== CSCSkim =====\n"
    << "\t\ttype of skim ...............................\t" << typeOfSkim
    << "\t\tminimum number of layers with hits .........\t" << nLayersWithHitsMinimum
    << "\n\t\tminimum number of chambers w/ hit layers..\t" << minimumHitChambers
    << "\n\t\tminimum number of segments ...............\t" << minimumSegments
    << "\n\t\tdemand chambers on both sides.............\t" << demandChambersBothSides
    << "\n\t\tmake histograms...........................\t" << makeHistograms
    << "\n\t\t..for messy events........................\t" << makeHistogramsForMessyEvents
    << "\n\t===================\n\n";

}

//===================
//  DESTRUCTOR
//===================
CSCSkim::~CSCSkim()
{
}


//================
//  BEGIN JOB
//================
void 
CSCSkim::beginJob()
{
  // set counters to zero
  nEventsAnalyzed = 0;
  nEventsSelected = 0;
  nEventsChambersBothSides = 0;
  nEventsOverlappingChambers = 0;
  nEventsMessy = 0;
  nEventsCertainChamber = 0;
  nEventsDTOverlap = 0;
  nEventsHaloLike = 0;
  nEventsLongSATrack = 0;
  nEventsForBFieldStudies = 0;
  iRun = 0;
  iEvent = 0;

  if (makeHistograms || makeHistogramsForMessyEvents) {

    // Create the root file for the histograms
    theHistogramFile = new TFile(histogramFileName.c_str(), "RECREATE");
    theHistogramFile->cd();

    if (makeHistograms) {
      // book histograms for the skimming module
      hxnRecHits     = new TH1F("hxnRecHits","n RecHits",61,-0.5,60.5);
      hxnSegments    = new TH1F("hxnSegments","n Segments",11,-0.5,10.5);
      hxnHitChambers = new TH1F("hxnHitsChambers","n chambers with hits",11,-0.5,10.5);
      hxnRecHitsSel  = new TH1F("hxnRecHitsSel","n RecHits selected",61,-0.5,60.5);

      xxP = new TH1F("xxP","P global",100,0.,200.);
      xxnValidHits = new TH1F("xxnValidHits","n valid hits global",61,-0.5,60.5);
      xxnTrackerHits = new TH1F("xxnTrackerHits","n tracker hits global",61,-0.5,60.5);
      xxnCSCHits = new TH1F("xxnCSCHits","n CSC hits global",41,-0.5,40.5);
      xxredChiSq = new TH1F("xxredChiSq","red chisq global",100,0.,100.);



    }
    if (makeHistogramsForMessyEvents) {
      // book histograms for the messy event skimming module
      mevnRecHits0     = new TH1F("mevnRecHits0","n RecHits",121,-0.5,120.5);
      mevnChambers0    = new TH1F("mevnChambers0","n chambers with hits",21,-0.5,20.5);
      mevnSegments0    = new TH1F("mevnSegments0","n Segments",21,-0.5,20.5);
      mevnRecHits1     = new TH1F("mevnRecHits1","n RecHits",100,0.,300.);
      mevnChambers1    = new TH1F("mevnChambers1","n chambers with hits",50,0.,50.);
      mevnSegments1    = new TH1F("mevnSegments1","n Segments",30,0.,30.);
    }

  }
}

//================
//  END JOB
//================
void 
CSCSkim::endJob() {

  // Write out results

  float fraction = 0.;
  if (nEventsAnalyzed > 0) {fraction = (float)nEventsSelected / (float)nEventsAnalyzed;}

  LogInfo("[CSCSkim] Summary")
    << "\n\n\t====== CSCSkim ==========================================================\n"
    << "\t\ttype of skim ...............................\t" << typeOfSkim << "\n"
    << "\t\tevents analyzed ..............\t" << nEventsAnalyzed << "\n"
    << "\t\tevents selected ..............\t" << nEventsSelected << "\tfraction= " << fraction << std::endl
    << "\t\tevents chambers both sides ...\t" << nEventsChambersBothSides << "\n"
    << "\t\tevents w/ overlaps .......... \t" << nEventsOverlappingChambers << "\n"
    << "\t\tevents lots of hit chambers . \t" << nEventsMessy << "\n"
    << "\t\tevents from certain chamber . \t" << nEventsCertainChamber << "\n"
    << "\t\tevents in DT-CSC overlap .... \t" << nEventsDTOverlap << "\n"
    << "\t\tevents halo-like ............ \t" << nEventsHaloLike << "\n"
    << "\t\tevents w/ long SA track ..... \t" << nEventsLongSATrack << "\n"
    << "\t\tevents good for BField  ..... \t" << nEventsForBFieldStudies << "\n"
    <<     "\t=========================================================================\n\n";

  if (makeHistograms || makeHistogramsForMessyEvents) {
    // Write the histos to file
    LogDebug("[CSCSkim]") << "======= write out my histograms ====\n" ;
    theHistogramFile->cd();
    if (makeHistograms) {
      hxnRecHits->Write();
      hxnSegments->Write();
      hxnHitChambers->Write();
      hxnRecHitsSel->Write();
    }
    if (makeHistogramsForMessyEvents) {
      mevnRecHits0->Write();
      mevnChambers0->Write();
      mevnSegments0->Write();
      mevnRecHits1->Write();
      mevnChambers1->Write();
      mevnSegments1->Write();
    }
    theHistogramFile->Close();
  }
}

//================
//  FILTER MAIN
//================
bool
CSCSkim::filter(edm::Event& event, const edm::EventSetup& eventSetup)
{
  // increment counter
  nEventsAnalyzed++;

  iRun   = event.id().run();
  iEvent = event.id().event();

  LogDebug("[CSCSkim] EventInfo") << "Run: " << iRun << "\tEvent: " << iEvent << "\tn Analyzed: " << nEventsAnalyzed;

  // Get the CSC Geometry :
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);

  // Get the DIGI collections
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;

  if (event.eventAuxiliary().isRealData()){
    event.getByLabel("muonCSCDigis","MuonCSCWireDigi",wires);
    event.getByLabel("muonCSCDigis","MuonCSCStripDigi",strips);
  }
  else {
    event.getByLabel("simMuonCSCDigis","MuonCSCWireDigi",wires);
    event.getByLabel("simMuonCSCDigis","MuonCSCStripDigi",strips);
  }

  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> cscRecHits;
  event.getByLabel(cscRecHitTag,cscRecHits);

  // get CSC segment collection
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel(cscSegmentTag, cscSegments);

  // get the cosmic muons collection
  Handle<reco::TrackCollection> saMuons;
  if (typeOfSkim == 8) {
    event.getByLabel(SAMuonTag,saMuons);
  }

  // get the stand-alone muons collection
  Handle<reco::TrackCollection> tracks;
  Handle<reco::MuonCollection> gMuons;
  if (typeOfSkim == 9) {
    event.getByLabel(SAMuonTag,saMuons);
    event.getByLabel(trackTag,tracks);
    event.getByLabel(GLBMuonTag,gMuons);
  }


  //======================================
  // evaluate the skimming routines
  //======================================


  // basic skimming
  bool basicEvent = false;
  if (typeOfSkim == 1 || typeOfSkim == 2) {
    basicEvent = doCSCSkimming(cscRecHits,cscSegments);
  }

  // overlapping chamber skim
  bool goodOverlapEvent = false;
  if (typeOfSkim == 3) {
    goodOverlapEvent = doOverlapSkimming(cscSegments);
    if (goodOverlapEvent) {nEventsOverlappingChambers++;}
  }

  // messy events skim
  bool messyEvent = false;
  if (typeOfSkim == 4) {
    messyEvent = doMessyEventSkimming(cscRecHits,cscSegments);
    if (messyEvent) {nEventsMessy++;}
  }

  // select events with DIGIs in a certain chamber
  bool hasChamber = false;
  if (typeOfSkim == 5) {
    hasChamber = doCertainChamberSelection(wires,strips);
    if (hasChamber) {nEventsCertainChamber++;}
  }

  // select events in the DT-CSC overlap region
  bool DTOverlapCandidate = false;
  if (typeOfSkim == 6) {
    DTOverlapCandidate = doDTOverlap(cscSegments);
    if (DTOverlapCandidate) {nEventsDTOverlap++;}
  }

  // select halo-like events
  bool HaloLike = false;
  if (typeOfSkim == 7) {
    HaloLike = doHaloLike(cscSegments);
    if (HaloLike) {nEventsHaloLike++;}
  }

  // select long cosmic tracks
  bool LongSATrack = false;
  if (typeOfSkim == 8) {
    LongSATrack = doLongSATrack(saMuons);
    if (LongSATrack) {nEventsLongSATrack++;}
  }

  // select events suitable for a B-field study.  They have tracks in the tracker.
  bool GoodForBFieldStudy = false;
  if (typeOfSkim == 9) {
    GoodForBFieldStudy = doBFieldStudySelection(saMuons,tracks,gMuons);
    if (GoodForBFieldStudy) {nEventsForBFieldStudies++;}
  }


  // set filter flag
  bool selectThisEvent = false;
  if (typeOfSkim == 1 || typeOfSkim == 2) {selectThisEvent = basicEvent;}
  if (typeOfSkim == 3) {selectThisEvent = goodOverlapEvent;}
  if (typeOfSkim == 4) {selectThisEvent = messyEvent;}
  if (typeOfSkim == 5) {selectThisEvent = hasChamber;}
  if (typeOfSkim == 6) {selectThisEvent = DTOverlapCandidate;}
  if (typeOfSkim == 7) {selectThisEvent = HaloLike;}
  if (typeOfSkim == 8) {selectThisEvent = LongSATrack;}
  if (typeOfSkim == 9) {selectThisEvent = GoodForBFieldStudy;}

  if (selectThisEvent) {nEventsSelected++;}

  return selectThisEvent;
}


// ==============================================
//
// CSC Skimming
//
// ==============================================

bool CSCSkim::doCSCSkimming(edm::Handle<CSCRecHit2DCollection> cscRecHits, edm::Handle<CSCSegmentCollection> cscSegments){

  // how many RecHits in the collection?
  int nRecHits = cscRecHits->size();

  // zero the recHit counter
  int cntRecHit[600];
  for (int i = 0; i < 600; i++) {
    cntRecHit[i] = 0;
  }

  // ---------------------
  // Loop over rechits 
  // ---------------------

  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = cscRecHits->begin(); recIt != cscRecHits->end(); recIt++) {

    // which chamber is it?
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();

    // compute chamber serial number
    int kSerial = chamberSerial( kEndcap, kStation, kRing, kChamber ) ;

    // increment recHit counter
    //     (each layer is represented by a different power of 10)
    int kDigit = (int) pow((float)10.,(float)(kLayer-1));
    cntRecHit[kSerial] += kDigit;

  } //end rechit loop


  // ------------------------------------------------------
  // Are there chambers with the minimum number of hits?
  // ------------------------------------------------------

  int nChambersWithMinimalHits = 0;
  int nChambersWithMinimalHitsPOS = 0;
  int nChambersWithMinimalHitsNEG = 0;
  if (nRecHits > 0) {
    for (int i = 0; i < 600; i++) {
      if (cntRecHit[i] > 0) {
	int nLayersWithHits = 0;
	float dummy = (float) cntRecHit[i];
	for (int j = 5; j > -1; j--) {
	  float digit = dummy / pow( (float)10., (float)j );
	  int kCount = (int) digit;
	  if (kCount > 0) nLayersWithHits++;
	  dummy = dummy - ( (float) kCount) * pow( (float)10., (float)j );
	}
	if (nLayersWithHits > nLayersWithHitsMinimum) {
	  if (i < 300) {nChambersWithMinimalHitsPOS++;}
	  else         {nChambersWithMinimalHitsNEG++;}
	}
      }
    }
    nChambersWithMinimalHits = nChambersWithMinimalHitsPOS + nChambersWithMinimalHitsNEG;
  }

  // how many Segments?
  int nSegments = cscSegments->size();

  // ----------------------
  // fill histograms
  // ----------------------

  if (makeHistograms) {
    hxnRecHits->Fill(nRecHits);
    if (nRecHits > 0) {
      hxnSegments->Fill(nSegments);
      hxnHitChambers->Fill(nChambersWithMinimalHits);
    }
    if (nChambersWithMinimalHits > 0) {
      hxnRecHitsSel->Fill(nRecHits);
    }
  }

  // ----------------------
  // set the filter flag
  // ----------------------
  bool basicEvent = ( nChambersWithMinimalHits >= minimumHitChambers ) && ( nSegments >= minimumSegments );

  bool chambersOnBothSides = ((nChambersWithMinimalHitsPOS >= minimumHitChambers) && (nChambersWithMinimalHitsNEG >= minimumHitChambers));

  if (chambersOnBothSides) {nEventsChambersBothSides++;}

  bool selectEvent = false;
  if (typeOfSkim == 1) {selectEvent = basicEvent;}
  if (typeOfSkim == 2) {selectEvent = chambersOnBothSides;}


  // debug
  LogDebug("[CSCSkim]") << "----- nRecHits = " << nRecHits
		       << "\tnChambersWithMinimalHits = " << nChambersWithMinimalHits
		       << "\tnSegments = " << nSegments 
		       << "\tselect? " << selectEvent << std::endl;

  /*
  if ((nChambersWithMinimalHitsPOS >= minimumHitChambers) && (nChambersWithMinimalHitsNEG >= minimumHitChambers)) {
    std::cout << "\n==========================================================================\n"
	 << "\tinteresting event - chambers hit on both sides\n"
	 << "\t  " <<  nEventsAnalyzed
	 << "\trun " << iRun << "\tevent " << iEvent << std::endl;
    std::cout << "----- nRecHits = " << nRecHits
	 << "\tnChambersWithMinimalHits = " << nChambersWithMinimalHits
	 << "\tnSegments = " << nSegments 
	 << "\tselect? " << selectEvent << std::endl;
    for (int i = 0; i < 600; i++) {
      if (cntRecHit[i] > 0) {
	cout << "\t\t" << i << "\tcntRecHit= " << cntRecHit[i] << std::endl;
      }
    }
    std::cout << "==========================================================================\n\n" ;
  }
  */

  return selectEvent;
}

//-------------------------------------------------------------------------------
// A module to select events useful in aligning chambers relative to each other
// using the overlap regions at the edges (in Xlocal) of the chamber.
//-------------------------------------------------------------------------------
bool CSCSkim::doOverlapSkimming(edm::Handle<CSCSegmentCollection> cscSegments){

  const int nhitsMinimum = 4;
  const float chisqMaximum = 100.;
  const int nAllMaximum  = 3;

  // how many Segments?
  //  int nSegments = cscSegments->size();

  // zero arrays
  int nAll[600];
  int nGood[600];
  for (int i=0; i<600; i++) {
    nAll[i] = 0;
    nGood[i] = 0;
  }

  // -----------------------
  // loop over segments
  // -----------------------
  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {

    // which chamber?
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    int kEndcap  = id.endcap();
    int kStation = id.station();
    int kRing    = id.ring();
    int kChamber = id.chamber();
    int kSerial = chamberSerial( kEndcap, kStation, kRing, kChamber);

    // segment information
    float chisq    = (*it).chi2();
    int nhits      = (*it).nRecHits();

    // is this a good segment?
    bool goodSegment = (nhits >= nhitsMinimum) && (chisq < chisqMaximum) ;

    /*
    LocalPoint localPos = (*it).localPosition();
    float segX     = localPos.x();
    float segY     = localPos.y();
    std::cout << "E/S/R/Ch: " << kEndcap << "/" << kStation << "/" << kRing << "/" << kChamber
	 << "\tnhits/chisq: " << nhits << "/" << chisq
	 << "\tX/Y: " << segX << "/" << segY
	 << "\tgood? " << goodSegment << std::endl;
    */

    // count
    nAll[kSerial-1]++;
    if (goodSegment) nGood[kSerial]++;

  } // end loop over segments

  //----------------------
  // select the event
  //----------------------

  // does any chamber have too many segments?
  bool messyChamber = false;
  for (int i = 0; i < 600; i++) {
    if (nAll[i] > nAllMaximum) messyChamber = true;
  }

  // are there consecutive chambers with good segments
  // (This is a little sloppy but is probably fine for skimming...)
  bool consecutiveChambers = false;
  for (int i = 0; i < 599; i++) {
    if ( (nGood[i]>0) && (nGood[i+1]>0) ) consecutiveChambers = true;
  }

  bool selectThisEvent = !messyChamber && consecutiveChambers;

  return selectThisEvent;

}

//============================================================
//
// This module selects events with a large numbere
// of recHits and larger number of chambers with hits.
//
//============================================================
bool CSCSkim::doMessyEventSkimming(edm::Handle<CSCRecHit2DCollection> cscRecHits, edm::Handle<CSCSegmentCollection> cscSegments){

  // how many RecHits in the collection?
  int nRecHits = cscRecHits->size();

  // zero the recHit counter
  int cntRecHit[600];
  for (int i = 0; i < 600; i++) {
    cntRecHit[i] = 0;
  }

  // ---------------------
  // Loop over rechits 
  // ---------------------

  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = cscRecHits->begin(); recIt != cscRecHits->end(); recIt++) {

    // which chamber is it?
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();

    // compute chamber serial number
    int kSerial = chamberSerial( kEndcap, kStation, kRing, kChamber ) ;

    // increment recHit counter
    //     (each layer is represented by a different power of 10)
    int kDigit = (int) pow((float)10.,(float)(kLayer-1));
    cntRecHit[kSerial] += kDigit;

  } //end rechit loop


  // ------------------------------------------------------
  // Are there chambers with the minimum number of hits?
  // ------------------------------------------------------

  int nChambersWithMinimalHits = 0;
  int nChambersWithMinimalHitsPOS = 0;
  int nChambersWithMinimalHitsNEG = 0;
  if (nRecHits > 0) {
    for (int i = 0; i < 600; i++) {
      if (cntRecHit[i] > 0) {
	int nLayersWithHits = 0;
	float dummy = (float) cntRecHit[i];
	for (int j = 5; j > -1; j--) {
	  float digit = dummy / pow( (float)10., (float)j );
	  int kCount = (int) digit;
	  if (kCount > 0) nLayersWithHits++;
	  dummy = dummy - ( (float) kCount) * pow( (float)10., (float)j );
	}
	if (nLayersWithHits > nLayersWithHitsMinimum) {
	  if (i < 300) {nChambersWithMinimalHitsPOS++;}
	  else         {nChambersWithMinimalHitsNEG++;}
	}
      }
    }
    nChambersWithMinimalHits = nChambersWithMinimalHitsPOS + nChambersWithMinimalHitsNEG;
  }

  // how many Segments?
  int nSegments = cscSegments->size();

  // ----------------------
  // fill histograms
  // ----------------------

  if (makeHistogramsForMessyEvents) {
    if (nRecHits > 8) {
      mevnRecHits0->Fill(nRecHits);
      mevnChambers0->Fill(nChambersWithMinimalHits);
      mevnSegments0->Fill(nSegments);
    }
    if (nRecHits > 54) {
      double dummy = (double) nRecHits;
      if (dummy > 299.9) dummy = 299.9;
      mevnRecHits1->Fill(dummy);
      dummy = (double) nChambersWithMinimalHits;
      if (dummy > 49.9) dummy = 49.9;
      mevnChambers1->Fill(dummy);
      dummy = (double) nSegments;
      if (dummy > 29.9) dummy = 29.9;
      mevnSegments1->Fill(dummy);
    }
  }

  // ----------------------
  // set the filter flag
  // ----------------------

  bool selectEvent = false;
  if ( (nRecHits > 54) && (nChambersWithMinimalHits > 5) ) {selectEvent = true;}

  // debug
  LogDebug("[CSCSkim]") << "----- nRecHits = " << nRecHits
		       << "\tnChambersWithMinimalHits = " << nChambersWithMinimalHits
		       << "\tnSegments = " << nSegments 
		       << "\tselect? " << selectEvent << std::endl;

  /*
  if (selectEvent) {
    std::cout << "\n==========================================================================\n"
	 << "\tmessy event!\n"
	 << "\t  " <<  nEventsAnalyzed
	 << "\trun " << iRun << "\tevent " << iEvent << std::endl;
    std::cout << "----- nRecHits = " << nRecHits
	 << "\tnChambersWithMinimalHits = " << nChambersWithMinimalHits
	 << "\tnSegments = " << nSegments 
	 << "\tselect? " << selectEvent << std::endl;
    for (int i = 0; i < 600; i++) {
      if (cntRecHit[i] > 0) {
	cout << "\t\t" << i << "\tcntRecHit= " << cntRecHit[i] << std::endl;
      }
    }
    std::cout << "==========================================================================\n\n" ;
  }
  */

  return selectEvent;
}


//============================================================
//
// Select events with DIGIs are a particular chamber.
//
//============================================================
bool CSCSkim::doCertainChamberSelection(edm::Handle<CSCWireDigiCollection> wires,
					edm::Handle<CSCStripDigiCollection> strips) {

  // Loop through the wire DIGIs, looking for a match
  bool certainChamberIsPresentInWires = false;
  for (CSCWireDigiCollection::DigiRangeIterator jw=wires->begin(); jw!=wires->end(); jw++) {
    CSCDetId id = (CSCDetId)(*jw).first;
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    if ( (kEndcap     == whichEndcap) &&
         (kStation    == whichStation) &&
         (kRing       == whichRing) &&
         (kChamber    == whichChamber) )
      {certainChamberIsPresentInWires = true;}
  } // end wire loop


  // Loop through the strip DIGIs, looking for a match
  bool certainChamberIsPresentInStrips = false;
  for (CSCStripDigiCollection::DigiRangeIterator js=strips->begin(); js!=strips->end(); js++) {
    CSCDetId id = (CSCDetId)(*js).first;
    int kEndcap  = id.endcap();
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    if ( (kEndcap     == whichEndcap) &&
         (kStation    == whichStation) &&
         (kRing       == whichRing) &&
         (kChamber    == whichChamber) )
      {certainChamberIsPresentInStrips = true;}
  }

  bool certainChamberIsPresent = certainChamberIsPresentInWires || certainChamberIsPresentInStrips;

  return certainChamberIsPresent;
}



//============================================================
//
// Select events which *might* probe the DT-CSC overlap region.
//
//============================================================
bool CSCSkim::doDTOverlap(Handle<CSCSegmentCollection> cscSegments) {
  const float chisqMax = 100.;
  const int nhitsMin = 5;
  const int maxNSegments = 3;

  // initialize
  bool DTOverlapCandidate = false;
  int cntMEP13[36];
  int cntMEN13[36];
  int cntMEP22[36];
  int cntMEN22[36];
  int cntMEP32[36];
  int cntMEN32[36];
  for (int i=0; i<36; ++i) {
    cntMEP13[i] = 0;
    cntMEN13[i] = 0;
    cntMEP22[i] = 0;
    cntMEN22[i] = 0;
    cntMEP32[i] = 0;
    cntMEN32[i] = 0;
  }

  // -----------------------
  // loop over segments
  // -----------------------

  int nSegments = cscSegments->size();
  if (nSegments < 2) return DTOverlapCandidate;

  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    // which chamber?
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    int kEndcap  = id.endcap();
    int kStation = id.station();
    int kRing    = id.ring();
    int kChamber = id.chamber();
    // segment information
    float chisq    = (*it).chi2();
    int nhits      = (*it).nRecHits();
    bool goodSegment = (chisq < chisqMax) && (nhits >= nhitsMin) ;
    if (goodSegment) {
      if ( (kStation == 1) && (kRing == 3) ) {
	if (kEndcap == 1) {cntMEP13[kChamber-1]++;}
	if (kEndcap == 2) {cntMEN13[kChamber-1]++;}
      }
      if ( (kStation == 2) && (kRing == 2) ) {
	if (kEndcap == 1) {cntMEP22[kChamber-1]++;}
	if (kEndcap == 2) {cntMEN22[kChamber-1]++;}
      }
      if ( (kStation == 3) && (kRing == 2) ) {
	if (kEndcap == 1) {cntMEP32[kChamber-1]++;}
	if (kEndcap == 2) {cntMEN32[kChamber-1]++;}
      }
    } // this is a good segment
  } // end loop over segments

  // ---------------------------------------------
  // veto messy events
  // ---------------------------------------------
  bool tooManySegments = false;
  for (int i=0; i<36; ++i) {
    if ( (cntMEP13[i] > maxNSegments) ||
         (cntMEN13[i] > maxNSegments) ||
         (cntMEP22[i] > maxNSegments) ||
         (cntMEN22[i] > maxNSegments) ||
         (cntMEP32[i] > maxNSegments) ||
         (cntMEN32[i] > maxNSegments) ) tooManySegments = true;
  }
  if (tooManySegments) {
    return DTOverlapCandidate;
  }

  // ---------------------------------------------
  // check for relevant matchup of segments
  // ---------------------------------------------
  bool matchup = false;
  for (int i=0; i<36; ++i) {
    if ( (cntMEP13[i] > 0) && (cntMEP22[i]+cntMEP32[i] > 0) ) {matchup = true;}
    if ( (cntMEN13[i] > 0) && (cntMEN22[i]+cntMEN32[i] > 0) ) {matchup = true;}
  }
  /*
  if (matchup) {
    std::cout << "\tYYY looks like a good event.  Select!\n";
    std::cout << "-- pos endcap --\n"
	 << "ME1/3: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEP13[k];}
    std::cout << "\nME2/2: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEP22[k];}
    std::cout << "\nME3/2: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEP32[k];}
    std::cout << std::endl;
  }
  */

  // set the selection flag
  DTOverlapCandidate = matchup;
  return DTOverlapCandidate;
}




//============================================================
//
// Select events which register in the inner parts of
// stations 1, 2, 3 and 4.
//
//============================================================
bool CSCSkim::doHaloLike(Handle<CSCSegmentCollection> cscSegments) {
  const float chisqMax = 100.;
  const int nhitsMin = 5; // on a segment
  const int maxNSegments = 3; // in a chamber

  // initialize
  bool HaloLike = false;
  int cntMEP11[36];
  int cntMEN11[36];
  int cntMEP12[36];
  int cntMEN12[36];
  int cntMEP21[36];
  int cntMEN21[36];
  int cntMEP31[36];
  int cntMEN31[36];
  int cntMEP41[36];
  int cntMEN41[36];
  for (int i=0; i<36; ++i) {
    cntMEP11[i] = 0;
    cntMEN11[i] = 0;
    cntMEP12[i] = 0;
    cntMEN12[i] = 0;
    cntMEP21[i] = 0;
    cntMEN21[i] = 0;
    cntMEP31[i] = 0;
    cntMEN31[i] = 0;
    cntMEP41[i] = 0;
    cntMEN41[i] = 0;
  }

  // -----------------------
  // loop over segments
  // -----------------------
  int nSegments = cscSegments->size();
  if (nSegments < 4) return HaloLike;

  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    // which chamber?
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    int kEndcap  = id.endcap();
    int kStation = id.station();
    int kRing    = id.ring();
    int kChamber = id.chamber();
    // segment information
    float chisq    = (*it).chi2();
    int nhits      = (*it).nRecHits();
    bool goodSegment = (chisq < chisqMax) && (nhits >= nhitsMin) ;
    if (goodSegment) {
      if ( (kStation == 1) && (kRing == 1) ) {
	if (kEndcap == 1) {cntMEP11[kChamber-1]++;}
	if (kEndcap == 2) {cntMEN11[kChamber-1]++;}
      }
      if ( (kStation == 1) && (kRing == 2) ) {
	if (kEndcap == 1) {cntMEP12[kChamber-1]++;}
	if (kEndcap == 2) {cntMEN12[kChamber-1]++;}
      }
      if ( (kStation == 2) && (kRing == 1) ) {
	if (kEndcap == 1) {cntMEP21[kChamber-1]++;}
	if (kEndcap == 2) {cntMEN21[kChamber-1]++;}
      }
      if ( (kStation == 3) && (kRing == 1) ) {
	if (kEndcap == 1) {cntMEP31[kChamber-1]++;}
	if (kEndcap == 2) {cntMEN31[kChamber-1]++;}
      }
      if ( (kStation == 4) && (kRing == 1) ) {
	if (kEndcap == 1) {cntMEP41[kChamber-1]++;}
	if (kEndcap == 2) {cntMEN41[kChamber-1]++;}
      }
    } // this is a good segment
  } // end loop over segments

  // ---------------------------------------------
  // veto messy events
  // ---------------------------------------------
  bool tooManySegments = false;
  for (int i=0; i<36; ++i) {
    if ( (cntMEP11[i] > 3*maxNSegments) ||
         (cntMEN11[i] > 3*maxNSegments) ||
	 (cntMEP12[i] > maxNSegments) ||
         (cntMEN12[i] > maxNSegments) ||
         (cntMEP21[i] > maxNSegments) ||
         (cntMEN21[i] > maxNSegments) ||
         (cntMEP31[i] > maxNSegments) ||
         (cntMEN31[i] > maxNSegments) ||
         (cntMEP41[i] > maxNSegments) ||
         (cntMEN41[i] > maxNSegments) ) tooManySegments = true;
  }
  if (tooManySegments) {
    return HaloLike;
  }

  // ---------------------------------------------
  // check for relevant matchup of segments
  // ---------------------------------------------
  bool matchup = false;
  for (int i=0; i<36; ++i) {
    if ( (cntMEP11[i]+cntMEP12[i] > 0) && 
         (cntMEP21[i] > 0) &&
         (cntMEP31[i] > 0) &&
         (cntMEP41[i] > 0) ) {matchup = true;}
    if ( (cntMEN11[i]+cntMEN12[i] > 0) && 
         (cntMEN21[i] > 0) &&
         (cntMEN31[i] > 0) &&
         (cntMEN41[i] > 0) ) {matchup = true;}
  }
  /*
  if (matchup) {
    std::cout << "\tYYY looks like a good event.  Select!\n";
    std::cout << "-- pos endcap --\n"
	 << "ME1/1: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEP11[k];}
    std::cout << "\nME1/2: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEP12[k];}
    std::cout << "\nME2/1: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEP21[k];}
    std::cout << "\nME3/1: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEP31[k];}
    std::cout << "\nME4/1: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEP41[k];}
    std::cout << std::endl;
    std::cout << "-- neg endcap --\n"
	 << "ME1/1: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEN11[k];}
    std::cout << "\nME1/2: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEN12[k];}
    std::cout << "\nME2/1: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEN21[k];}
    std::cout << "\nME3/1: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEN31[k];}
    std::cout << "\nME4/1: ";
    for (int k=0; k<36; ++k) {std::cout << " " << setw(3) << cntMEN41[k];}
    std::cout << std::endl;
    std::cout << "\tn Analyzed = " << nEventsAnalyzed << "\tn Halo-like = " << nEventsHaloLike << std::endl;
  }
  */

  // set the selection flag
  HaloLike = matchup;
  return HaloLike;
}


//--------------------------------------------------------------
// select events with at least one "long" stand-alone muon
//--------------------------------------------------------------
bool CSCSkim::doLongSATrack(edm::Handle<reco::TrackCollection> saMuons) {

  const float zDistanceMax = 2500.;
  const float zDistanceMin =  700.;
  const int nCSCHitsMin = 25;
  const int nCSCHitsMax = 50;
  const float zInnerMax = 80000.;

  const int nNiceMuonsMin = 1;

  //
  // Loop through the track collection and test each one
  //

  int nNiceMuons = 0;

  for(reco::TrackCollection::const_iterator muon = saMuons->begin(); muon != saMuons->end(); ++ muon ) {

    // basic information
    math::XYZVector innerMo = muon->innerMomentum();
    GlobalVector im(innerMo.x(),innerMo.y(),innerMo.z());
    math::XYZPoint innerPo = muon->innerPosition();
    GlobalPoint ip(innerPo.x(), innerPo.y(),innerPo.z());
    math::XYZPoint outerPo = muon->outerPosition();
    GlobalPoint op(outerPo.x(), outerPo.y(),outerPo.z());
    float zInner = ip.z();
    float zOuter = op.z();
    float zDistance = fabs(zOuter-zInner);



    // loop over hits
    int nDTHits = 0;
    int nCSCHits = 0;
    for (trackingRecHit_iterator hit = muon->recHitsBegin(); hit != muon->recHitsEnd(); ++hit ) {
      const DetId detId( (*hit)->geographicalId() );
      if (detId.det() == DetId::Muon) {
	if (detId.subdetId() == MuonSubdetId::DT) {
	  //DTChamberId dtId(detId.rawId());
	  //int chamberId = dtId.sector();
	  nDTHits++;
	}
	else if (detId.subdetId() == MuonSubdetId::CSC) {
	  //CSCDetId cscId(detId.rawId());
	  //int chamberId = cscId.chamber();
	  nCSCHits++;
	}
      }
    }

    // is this a nice muon?
    if ( (zDistance < zDistanceMax) && (zDistance > zDistanceMin) 
	 && (nCSCHits > nCSCHitsMin) && (nCSCHits < nCSCHitsMax)
	 && (min ( fabs(zInner), fabs(zOuter) ) < zInnerMax) 
	 && (fabs(innerMo.z()) > 0.000000001) ) {
      nNiceMuons++;
    }
  }

  bool select = (nNiceMuons >= nNiceMuonsMin);

  return select;
}





//============================================================
//
// Select events which are good for B-field studies.
//
// These events have a good track in the tracker.
//
//  D.Dibur and M.Schmitt
//============================================================
bool CSCSkim::doBFieldStudySelection(edm::Handle<reco::TrackCollection> saMuons,  edm::Handle<reco::TrackCollection> tracks, edm::Handle<reco::MuonCollection> gMuons) {

  bool acceptThisEvent = false;

  //-----------------------------------
  // examine the stand-alone tracks
  //-----------------------------------
  int nGoodSAMuons = 0;
  for (reco::TrackCollection::const_iterator muon = saMuons->begin(); muon != saMuons->end(); ++ muon ) {
    float preco  = muon->p();

    math::XYZPoint innerPo = muon->innerPosition();
    GlobalPoint iPnt(innerPo.x(), innerPo.y(),innerPo.z());
    math::XYZPoint outerPo = muon->outerPosition();
    GlobalPoint oPnt(outerPo.x(), outerPo.y(),outerPo.z());
    float zLength = abs( iPnt.z() - oPnt.z() );

    math::XYZVector innerMom = muon->innerMomentum();
    GlobalVector iP(innerMom.x(), innerMom.y(), innerMom.z() );
    math::XYZVector outerMom = muon->outerMomentum();
    GlobalVector oP(outerMom.x(), outerMom.y(), outerMom.z() );

    const float zRef = 300.;
    float xExt = 10000.;
    float yExt = 10000.;
    if (abs(oPnt.z()) < abs(iPnt.z())) {
      float deltaZ = 0.;
      if (oPnt.z() > 0) {
	deltaZ = zRef - oPnt.z();
      } else {
	deltaZ = -zRef - oPnt.z();
      }
      xExt = oPnt.x() + deltaZ * oP.x() / oP.z();
      yExt = oPnt.y() + deltaZ * oP.y() / oP.z();
    } else {
      float deltaZ = 0.;
      if (iPnt.z() > 0) {
	deltaZ = zRef - iPnt.z();
      } else {
	deltaZ = -zRef - iPnt.z();
      }
      xExt = iPnt.x() + deltaZ * iP.x() / iP.z();
      yExt = iPnt.y() + deltaZ * iP.y() / iP.z();
    }
    float rExt = sqrt( xExt*xExt + yExt*yExt );
    
    int kHit = 0;
    int nDTHits = 0;
    int nCSCHits = 0;
    for (trackingRecHit_iterator hit = muon->recHitsBegin(); hit != muon->recHitsEnd(); ++hit ) {
      ++kHit;
      const DetId detId( (*hit)->geographicalId() );
      if (detId.det() == DetId::Muon) {
	if (detId.subdetId() == MuonSubdetId::DT) {
	  nDTHits++;
	}
	else if (detId.subdetId() == MuonSubdetId::CSC) {
	  nCSCHits++;
	}
      }
    } // end loop over hits

    float zInner = -1.;
    if (nCSCHits >= nCSCHitsMin) {
      if (abs(iPnt.z()) < abs(iPnt.z())) {
	zInner = iPnt.z();
      } else {
	zInner = oPnt.z();
      }
    }

    bool goodSAMuon = (preco > pMin)
      && ( zLength > zLengthMin )
      && ( nCSCHits >= nCSCHitsMin ) 
      && ( zInner < zInnerMax )
      && ( rExt < rExtMax ) ;

    if (goodSAMuon) {nGoodSAMuons++;}
    
  } // end loop over stand-alone muon collection




  //-----------------------------------
  // examine the tracker tracks
  //-----------------------------------
  int nGoodTracks = 0;
  for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); ++ track ) {
    float preco  = track->p();
    int   n = track->recHitsSize();

    math::XYZPoint innerPo = track->innerPosition();
    GlobalPoint iPnt(innerPo.x(), innerPo.y(),innerPo.z());
    math::XYZPoint outerPo = track->outerPosition();
    GlobalPoint oPnt(outerPo.x(), outerPo.y(),outerPo.z());
    float zLength = abs( iPnt.z() - oPnt.z() );

    math::XYZVector innerMom = track->innerMomentum();
    GlobalVector iP(innerMom.x(), innerMom.y(), innerMom.z() );
    math::XYZVector outerMom = track->outerMomentum();
    GlobalVector oP(outerMom.x(), outerMom.y(), outerMom.z() );

    const float zRef = 300.;
    float xExt = 10000.;
    float yExt = 10000.;
    if (abs(oPnt.z()) > abs(iPnt.z())) {
      float deltaZ = 0.;
      if (oPnt.z() > 0) {
	deltaZ = zRef - oPnt.z();
      } else {
	deltaZ = -zRef - oPnt.z();
      }
      xExt = oPnt.x() + deltaZ * oP.x() / oP.z();
      yExt = oPnt.y() + deltaZ * oP.y() / oP.z();
    } else {
      float deltaZ = 0.;
      if (iPnt.z() > 0) {
	deltaZ = zRef - iPnt.z();
      } else {
	deltaZ = -zRef - iPnt.z();
      }
      xExt = iPnt.x() + deltaZ * iP.x() / iP.z();
      yExt = iPnt.y() + deltaZ * iP.y() / iP.z();
    }
    float rExt = sqrt( xExt*xExt + yExt*yExt );

    bool goodTrack = (preco > pMin)
      && (n >= nTrHitsMin)
      && (zLength > zLengthTrMin)
      && ( rExt < rExtMax ) ;

    if (goodTrack) {nGoodTracks++;}

  } // end loop over tracker tracks


  //-----------------------------------
  // examine the global muons
  //-----------------------------------
  int nGoodGlobalMuons = 0;
  for (reco::MuonCollection::const_iterator global = gMuons->begin(); global != gMuons->end(); ++global ) {

    if (global->isGlobalMuon()) {

      float pDef  = global->p();
      float redChiSq = global->globalTrack()->normalizedChi2();
      const reco::HitPattern& hp = (global->globalTrack())->hitPattern();
      // int nTotalHits = hp.numberOfHits();
      //    int nValidHits = hp.numberOfValidHits();
      int nTrackerHits = hp.numberOfValidTrackerHits();
      // int nPixelHits   = hp.numberOfValidPixelHits();
      // int nStripHits   = hp.numberOfValidStripHits();
      
      int nDTHits = 0;
      int nCSCHits = 0;
      for (trackingRecHit_iterator hit = (global->globalTrack())->recHitsBegin(); hit != (global->globalTrack())->recHitsEnd(); ++hit ) {
	const DetId detId( (*hit)->geographicalId() );
	if (detId.det() == DetId::Muon) {
	  if (detId.subdetId() == MuonSubdetId::DT) {
	    nDTHits++;
	  }
	  else if (detId.subdetId() == MuonSubdetId::CSC) {
	    nCSCHits++;
	  }
	}
      } // end loop over hits
      
      bool goodGlobalMuon = (pDef > pMin)
	&& ( nTrackerHits >= nValidHitsMin )
	&& ( nCSCHits >= nCSCHitsMin ) 
	&& ( redChiSq < redChiSqMax );

      if (goodGlobalMuon) {nGoodGlobalMuons++;}
      
    } // this is a global muon
  } // end loop over stand-alone muon collection

  //-----------------------------------
  // do we accept this event?
  //-----------------------------------

  acceptThisEvent = ( (nGoodSAMuons > 0) && (nGoodTracks > 0) ) || (nGoodGlobalMuons > 0) ;

  return acceptThisEvent;
}

//--------------------------------------------------------------
// Compute a serial number for the chamber.
// This is useful when filling histograms and working with arrays.
//--------------------------------------------------------------
int CSCSkim::chamberSerial( int kEndcap, int kStation, int kRing, int kChamber ) {
    int kSerial = kChamber;
    if (kStation == 1 && kRing == 1) {kSerial = kChamber;}
    if (kStation == 1 && kRing == 2) {kSerial = kChamber + 36;}
    if (kStation == 1 && kRing == 3) {kSerial = kChamber + 72;}
    if (kStation == 1 && kRing == 4) {kSerial = kChamber;}
    if (kStation == 2 && kRing == 1) {kSerial = kChamber + 108;}
    if (kStation == 2 && kRing == 2) {kSerial = kChamber + 126;}
    if (kStation == 3 && kRing == 1) {kSerial = kChamber + 162;}
    if (kStation == 3 && kRing == 2) {kSerial = kChamber + 180;}
    if (kStation == 4 && kRing == 1) {kSerial = kChamber + 216;}
    if (kStation == 4 && kRing == 2) {kSerial = kChamber + 234;}  // one day...
    if (kEndcap == 2) {kSerial = kSerial + 300;}
    return kSerial;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCSkim);
