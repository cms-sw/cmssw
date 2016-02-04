
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/11/19 14:02:08 $
 *  $Revision: 1.3 $
 *  \author A. Vilela Pereira
 */

#include "DTVDriftSegmentCalibration.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include <string>
#include <sstream>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;
using namespace edm;

DTVDriftSegmentCalibration::DTVDriftSegmentCalibration(const ParameterSet& pset):
  select_(pset),
  theRecHits4DLabel_(pset.getParameter<InputTag>("recHits4DLabel")),
  //writeVDriftDB_(pset.getUntrackedParameter<bool>("writeVDriftDB", false)),
  theCalibChamber_(pset.getUntrackedParameter<string>("calibChamber", "All")) {

  LogVerbatim("Calibration") << "[DTVDriftSegmentCalibration] Constructor called!";

  // the root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName","DTVDriftHistos.root");
  rootFile_ = new TFile(rootFileName.c_str(), "RECREATE");
  rootFile_->cd();
}

void DTVDriftSegmentCalibration::beginJob(){
  TH1::SetDefaultSumw2(true);
}

void DTVDriftSegmentCalibration::beginRun(const edm::Run& run, const edm::EventSetup& setup) {}

DTVDriftSegmentCalibration::~DTVDriftSegmentCalibration(){
  rootFile_->Close();
  LogVerbatim("Calibration") << "[DTVDriftSegmentCalibration] Destructor called!";
}

void DTVDriftSegmentCalibration::analyze(const Event & event, const EventSetup& eventSetup) {
  rootFile_->cd();

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the rechit collection from the event
  Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel_, all4DSegments); 

  DTChamberId chosenChamberId;
  if(theCalibChamber_ != "All") {
    stringstream linestr;
    int selWheel, selStation, selSector;
    linestr << theCalibChamber_;
    linestr >> selWheel >> selStation >> selSector;
    chosenChamberId = DTChamberId(selWheel, selStation, selSector);
    LogVerbatim("Calibration") << " Chosen chamber: " << chosenChamberId << endl;
  }
  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for(chamberIdIt = all4DSegments->id_begin(); chamberIdIt != all4DSegments->id_end(); ++chamberIdIt){

    // Calibrate just the chosen chamber/s    
    if((theCalibChamber_ != "All") && ((*chamberIdIt) != chosenChamberId)) continue;

    // Book histos
    if(theVDriftHistoMapTH1F_.find(*chamberIdIt) == theVDriftHistoMapTH1F_.end()){
      LogTrace("Calibration") << "   Booking histos for Chamber: " << *chamberIdIt;
      bookHistos(*chamberIdIt);
    }
   
    // Get the chamber from the setup
    const DTChamber* chamber = dtGeom->chamber(*chamberIdIt);
    // Get the range for the corresponding ChamberId
    DTRecSegment4DCollection::range range = all4DSegments->get((*chamberIdIt));
    // Loop over the rechits of this DetUnit
    for(DTRecSegment4DCollection::const_iterator segment  = range.first;
                                                 segment != range.second; ++segment){

      
      LogTrace("Calibration") << "Segment local pos (in chamber RF): " << (*segment).localPosition()
                              << "\nSegment global pos: " << chamber->toGlobal((*segment).localPosition());
      
      if( !select_(*segment, event, eventSetup) ) continue;

      // Fill v-drift values
      if( (*segment).hasPhi() ) {
         //if( segment->phiSegment()->ist0Valid() ){
         double segmentVDrift = segment->phiSegment()->vDrift();
         if( segmentVDrift != 0.00 ){   
	    (theVDriftHistoMapTH1F_[*chamberIdIt])[0]->Fill(segmentVDrift);
            (theVDriftHistoMapTH2F_[*chamberIdIt])[0]->Fill(segment->localPosition().x(),segmentVDrift);
            (theVDriftHistoMapTH2F_[*chamberIdIt])[1]->Fill(segment->localPosition().y(),segmentVDrift);
	}
      }
      // Probably not meaningful 
      if( (*segment).hasZed() ){
         //if( segment->zSegment()->ist0Valid() ){
         double segmentVDrift = segment->zSegment()->vDrift();
         if( segmentVDrift != 0.00 ){
	    (theVDriftHistoMapTH1F_[*chamberIdIt])[1]->Fill(segmentVDrift);
	}
      }
    } // DTRecSegment4DCollection::const_iterator segment
  } // DTRecSegment4DCollection::id_iterator chamberIdIt
} // DTVDriftSegmentCalibration::analyze

void DTVDriftSegmentCalibration::endJob() {
  rootFile_->cd();
  
  LogVerbatim("Calibration") << "[DTVDriftSegmentCalibration] Writing histos to file!" << endl;

  for(ChamberHistosMapTH1F::const_iterator itChHistos = theVDriftHistoMapTH1F_.begin(); itChHistos != theVDriftHistoMapTH1F_.end(); ++itChHistos){
     vector<TH1F*>::const_iterator itHistTH1F = (*itChHistos).second.begin();
     vector<TH1F*>::const_iterator itHistTH1F_end = (*itChHistos).second.end();
     for(; itHistTH1F != itHistTH1F_end; ++itHistTH1F) (*itHistTH1F)->Write();

     vector<TH2F*>::const_iterator itHistTH2F = theVDriftHistoMapTH2F_[(*itChHistos).first].begin();
     vector<TH2F*>::const_iterator itHistTH2F_end = theVDriftHistoMapTH2F_[(*itChHistos).first].end();
     for(; itHistTH2F != itHistTH2F_end; ++itHistTH2F) (*itHistTH2F)->Write();
  }

  /*if(writeVDriftDB_){
     // ...
  }*/ 
}

// Book a set of histograms for a given Chamber
void DTVDriftSegmentCalibration::bookHistos(DTChamberId chId) {

  // Compose the chamber name
  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();
  stringstream sector; sector << chId.sector();

  string chHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();

  vector<TH1F*> histosTH1F;
  histosTH1F.push_back(new TH1F(("hRPhiVDriftCorr" + chHistoName).c_str(), "v-drift corr. from Phi segments", 200, -0.4, 0.4));
  if(chId.station() != 4) histosTH1F.push_back(new TH1F(("hRZVDriftCorr" + chHistoName).c_str(), "v-drift corr. from Z segments", 200, -0.4, 0.4));
  
  vector<TH2F*> histosTH2F;
  histosTH2F.push_back(new TH2F(("hRPhiVDriftCorrVsSegmPosX" + chHistoName).c_str(), "v-drift corr. vs. segment x position", 250, -125., 125., 200, -0.4, 0.4));
  histosTH2F.push_back(new TH2F(("hRPhiVDriftCorrVsSegmPosY" + chHistoName).c_str(), "v-drift corr. vs. segment y position", 250, -125., 125., 200, -0.4, 0.4));
  //if(chId.station() != 4) ...

  theVDriftHistoMapTH1F_[chId] = histosTH1F;
  theVDriftHistoMapTH2F_[chId] = histosTH2F;
}
