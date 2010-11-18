
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/11/18 11:40:19 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "DTVDriftT0FitCalibration.h"

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

DTVDriftT0FitCalibration::DTVDriftT0FitCalibration(const ParameterSet& pset):
  select_(pset),
  theRecHits4DLabel_(pset.getParameter<InputTag>("recHits4DLabel")),
  writeVDriftDB_(pset.getUntrackedParameter<bool>("writeVDriftDB", false)),
  theCalibChamber_(pset.getUntrackedParameter<string>("calibChamber", "All")) {

  LogVerbatim("Calibration") << "[DTVDriftT0FitCalibration] Constructor called!";

  // the root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName","DTVDriftHistos.root");
  rootFile_ = new TFile(rootFileName.c_str(), "RECREATE");
  rootFile_->cd();
}

void DTVDriftT0FitCalibration::beginRun(const edm::Run& run, const edm::EventSetup& setup) {}

DTVDriftT0FitCalibration::~DTVDriftT0FitCalibration(){
  rootFile_->Close();
  LogVerbatim("Calibration") << "[DTVDriftT0FitCalibration] Destructor called!";
}

void DTVDriftT0FitCalibration::analyze(const Event & event, const EventSetup& eventSetup) {
  rootFile_->cd();
  DTChamberId chosenChamberId;

  if(theCalibChamber_ != "All") {
    stringstream linestr;
    int selWheel, selStation, selSector;
    linestr << theCalibChamber_;
    linestr >> selWheel >> selStation >> selSector;
    chosenChamberId = DTChamberId(selWheel, selStation, selSector);
    LogVerbatim("Calibration") << " Chosen chamber: " << chosenChamberId << endl;
  }

  /*
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);
  */

  // Get the rechit collection from the event
  Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel_, all4DSegments); 

  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for(chamberIdIt = all4DSegments->id_begin(); chamberIdIt != all4DSegments->id_end(); ++chamberIdIt){

    // Get the chamber from the setup
    //const DTChamber* chamber = dtGeom->chamber(*chamberIdIt);
    LogTrace("Calibration") << "Chamber Id: " << *chamberIdIt;

    // Book histos
    if(theVDriftHistoMapTH1F_.find(*chamberIdIt) == theVDriftHistoMapTH1F_.end()){
      bookHistos(*chamberIdIt);
    }
   
    // Calibrate just the chosen chamber/s    
    if((theCalibChamber_ != "All") && ((*chamberIdIt) != chosenChamberId)) continue;

    // Get the range for the corresponding ChamberId
    DTRecSegment4DCollection::range range = all4DSegments->get((*chamberIdIt));

    // Loop over the rechits of this DetUnit
    for(DTRecSegment4DCollection::const_iterator segment  = range.first;
                                                 segment != range.second; ++segment){

      /*LogTrace("Calibration") << "Segment local pos (in chamber RF): " << (*segment).localPosition()
                              << "\nSegment global pos: " << chamber->toGlobal((*segment).localPosition());*/
      LogTrace("Calibration") << "Segment local pos (in chamber RF): " << (*segment).localPosition();
      
      if( !select_(event, eventSetup, *segment) ) continue;

      // Fill t0-seg values
      if( (*segment).hasPhi() ) {
        if( segment->phiSegment()->ist0Valid() ){
	  (theVDriftHistoMapTH1F_[*chamberIdIt])[0]->Fill(segment->phiSegment()->vDrift());
          (theVDriftHistoMapTH2F_[*chamberIdIt])[0]->Fill(segment->localPosition().z(),segment->phiSegment()->vDrift());
	}
      }
      if( (*segment).hasZed() ){
        if( segment->zSegment()->ist0Valid() ){
	  (theVDriftHistoMapTH1F_[*chamberIdIt])[1]->Fill(segment->zSegment()->vDrift());
          (theVDriftHistoMapTH2F_[*chamberIdIt])[1]->Fill(segment->localPosition().z(),segment->zSegment()->vDrift()); 
	}
      }
    } // DTRecSegment4DCollection::const_iterator segment
  } // DTRecSegment4DCollection::id_iterator chamberIdIt
} // DTVDriftT0FitCalibration::analyze

void DTVDriftT0FitCalibration::endJob() {
  rootFile_->cd();
  
  LogVerbatim("Calibration") << "[DTVDriftT0FitCalibration] Writing histos to file!" << endl;

  for(ChamberHistosMapTH1F::const_iterator itChHistos = theVDriftHistoMapTH1F_.begin(); itChHistos != theVDriftHistoMapTH1F_.end(); ++itChHistos){
     vector<TH1F*>::const_iterator itHistTH1F = (*itChHistos).second.begin();
     vector<TH1F*>::const_iterator itHistTH1F_end = (*itChHistos).second.end();
     for(; itHistTH1F != itHistTH1F_end; ++itHistTH1F) (*itHistTH1F)->Write();

     vector<TH2F*>::const_iterator itHistTH2F = theVDriftHistoMapTH2F_[(*itChHistos).first].begin();
     vector<TH2F*>::const_iterator itHistTH2F_end = theVDriftHistoMapTH2F_[(*itChHistos).first].end();
     for(; itHistTH2F != itHistTH2F_end; ++itHistTH2F) (*itHistTH2F)->Write();
  }

  if(writeVDriftDB_){
    // Create the object to be written to DB
    DTMtime* mTimeMap = new DTMtime();

    for(ChamberHistosMapTH1F::const_iterator itChHistos = theVDriftHistoMapTH1F_.begin(); itChHistos != theVDriftHistoMapTH1F_.end(); ++itChHistos){
       DTChamberId chId = itChHistos->first;
       // Get SuperLayerId's for each ChamberId
       vector<DTSuperLayerId> slIds;
       slIds.push_back(DTSuperLayerId(chId,1));
       slIds.push_back(DTSuperLayerId(chId,3));
       if(chId.station() != 4) slIds.push_back(DTSuperLayerId(chId,2));

       for(vector<DTSuperLayerId>::const_iterator itSl = slIds.begin(); itSl != slIds.end(); ++itSl){      
          // Set values
          // FIXME: Placeholder; fit vDrift here
          double vDriftMean = 0.;
          double vDriftSigma = 0.;
          // vdrift is cm/ns , resolution is cm
          mTimeMap->set(*itSl,
                        vDriftMean,
	                vDriftSigma,
		        DTVelocityUnits::cm_per_ns);
 
      }
    }
    LogVerbatim("Calibration")<< "[DTVDriftT0FitCalibration] Writing ttrig object to DB!" << endl;
    // Write the object to DB
    string record = "DTMtimeRcd";
    DTCalibDBUtils::writeToDB<DTMtime>(record, mTimeMap);
  } 
}

// Book a set of histograms for a given Chamber
void DTVDriftT0FitCalibration::bookHistos(DTChamberId chId) {

  LogTrace("Calibration") << "   Booking histos for Chamber: " << chId;

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
  histosTH2F.push_back(new TH2F(("hRPhiVDriftCorrVsSegmPosZ" + chHistoName).c_str(), "v-drift corr. vs. segment z position from Phi segments", 250, -125., 125., 200, -0.4, 0.4));
  if(chId.station() != 4) histosTH2F.push_back(new TH2F(("hRZVDriftCorrVsSegmPosZ" + chHistoName).c_str(), "v-drift corr. vs. segment z position from Z segments", 250, -125., 125., 200, -0.4, 0.4));

  theVDriftHistoMapTH1F_[chId] = histosTH1F;
  theVDriftHistoMapTH2F_[chId] = histosTH2F;
}
