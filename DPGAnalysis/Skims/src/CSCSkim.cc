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
// $Id: CSCSkim.cc,v 1.2 2008/07/30 15:57:13 rahatlou Exp $
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

  // Get the various input parameters
  outputFileName     = pset.getUntrackedParameter<string>("outputFileName","outputSkim.root");
  histogramFileName  = pset.getUntrackedParameter<string>("histogramFileName","histos.root");
  typeOfSkim              = pset.getUntrackedParameter<int>("typeOfSkim",1);
  nLayersWithHitsMinimum  = pset.getUntrackedParameter<int>("nLayersWithHitsMinimum",3);
  minimumHitChambers      = pset.getUntrackedParameter<int>("minimumHitChambers",1);
  minimumSegments         = pset.getUntrackedParameter<int>("minimumSegments",3);
  demandChambersBothSides = pset.getUntrackedParameter<bool>("demandChambersBothSides",false);
  makeHistograms          = pset.getUntrackedParameter<bool>("makeHistograms",false);
  makeHistogramsForMessyEvents = pset.getUntrackedParameter<bool>("makeHistogramsForMessyEvebts",false);

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
CSCSkim::beginJob(const edm::EventSetup&)
{
  // set counters to zero
  nEventsAnalyzed = 0;
  nEventsSelected = 0;
  nEventsChambersBothSides = 0;
  nEventsOverlappingChambers = 0;
  nEventsMessy = 0;
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
    << "\t\tevents selected ..............\t" << nEventsSelected << "\tfraction= " << fraction << endl
    << "\t\tevents chambers both sides ...\t" << nEventsChambersBothSides << "\n"
    << "\t\tevents w/ overlaps .......... \t" << nEventsOverlappingChambers << "\n"
    << "\t\tevents lots of hit chambers . \t" << nEventsMessy << "\n"
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

  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> cscRecHits;
  event.getByLabel("csc2DRecHits",cscRecHits);

  // get CSC segment collection
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel("cscSegments", cscSegments);

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

  // set filter flag
  bool selectThisEvent = false;
  if (typeOfSkim == 1 || typeOfSkim == 2) {selectThisEvent = basicEvent;}
  if (typeOfSkim == 3) {selectThisEvent = goodOverlapEvent;}
  if (typeOfSkim == 4) {selectThisEvent = messyEvent;}

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
		       << "\tselect? " << selectEvent << endl;

  /*
  if ((nChambersWithMinimalHitsPOS >= minimumHitChambers) && (nChambersWithMinimalHitsNEG >= minimumHitChambers)) {
    cout << "\n==========================================================================\n"
	 << "\tinteresting event - chambers hit on both sides\n"
	 << "\t  " <<  nEventsAnalyzed
	 << "\trun " << iRun << "\tevent " << iEvent << endl;
    cout << "----- nRecHits = " << nRecHits
	 << "\tnChambersWithMinimalHits = " << nChambersWithMinimalHits
	 << "\tnSegments = " << nSegments 
	 << "\tselect? " << selectEvent << endl;
    for (int i = 0; i < 600; i++) {
      if (cntRecHit[i] > 0) {
	cout << "\t\t" << i << "\tcntRecHit= " << cntRecHit[i] << endl;
      }
    }
    cout << "==========================================================================\n\n" ;
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
    cout << "E/S/R/Ch: " << kEndcap << "/" << kStation << "/" << kRing << "/" << kChamber
	 << "\tnhits/chisq: " << nhits << "/" << chisq
	 << "\tX/Y: " << segX << "/" << segY
	 << "\tgood? " << goodSegment << endl;
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

//--------------------------------------------------
//
// This module selects events with a large numbere
// of recHits and larger number of chambers with hits.
//
//--------------------------------------------------
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
		       << "\tselect? " << selectEvent << endl;

  /*
  if (selectEvent) {
    cout << "\n==========================================================================\n"
	 << "\tmessy event!\n"
	 << "\t  " <<  nEventsAnalyzed
	 << "\trun " << iRun << "\tevent " << iEvent << endl;
    cout << "----- nRecHits = " << nRecHits
	 << "\tnChambersWithMinimalHits = " << nChambersWithMinimalHits
	 << "\tnSegments = " << nSegments 
	 << "\tselect? " << selectEvent << endl;
    for (int i = 0; i < 600; i++) {
      if (cntRecHit[i] > 0) {
	cout << "\t\t" << i << "\tcntRecHit= " << cntRecHit[i] << endl;
      }
    }
    cout << "==========================================================================\n\n" ;
  }
  */


  return selectEvent;
}



//--------------------------------------------------------------
// Compute a serial number for the chamber.
// This is useful when filling histograms and working with arrays.
//--------------------------------------------------------------
int CSCSkim::chamberSerial( int kEndcap, int kStation, int kRing, int kChamber ) {
    int kSerial = kChamber;
    if (kStation == 1 && kRing == 1) {kSerial = kChamber;}
    if (kStation == 1 && kRing == 2) {kSerial = kChamber + 36;}
    if (kStation == 1 && kRing == 3) {kSerial = kChamber + 73;}
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
