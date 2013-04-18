
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/02/28 20:10:01 $
 *  $Revision: 1.5 $
 *  \author A. Vilela Pereira
 */

#include "CalibMuon/DTCalibration/plugins/DTTTrigOffsetCalibration.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

/* C++ Headers */
#include <map>
#include <string>
#include <sstream>
#include "TFile.h"
#include "TH1F.h"

using namespace std;
using namespace edm;

DTTTrigOffsetCalibration::DTTTrigOffsetCalibration(const ParameterSet& pset) {

  LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration] Constructor called!";

  // the root file which will contain the histos
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName","DTT0SegHistos.root");
  theFile_ = new TFile(rootFileName.c_str(), "RECREATE");
  theFile_->cd();

  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");

  // Do t0-seg correction to ttrig
  doTTrigCorrection_ = pset.getUntrackedParameter<bool>("doT0SegCorrection", false);

  // Chamber/s to calibrate
  theCalibChamber_ =  pset.getUntrackedParameter<string>("calibChamber", "All");

  // the name of the 4D rec hits collection
  theRecHits4DLabel_ = pset.getParameter<InputTag>("recHits4DLabel");

  //get the switch to check the noisy channels
  checkNoisyChannels_ = pset.getParameter<bool>("checkNoisyChannels");

  // get maximum chi2 value 
  theMaxChi2_ =  pset.getParameter<double>("maxChi2");

  // Maximum incidence angle for Phi SL 
  theMaxPhiAngle_ =  pset.getParameter<double>("maxAnglePhi");

  // Maximum incidence angle for Theta SL 
  theMaxZAngle_ =  pset.getParameter<double>("maxAngleZ");
}

void DTTTrigOffsetCalibration::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  if(doTTrigCorrection_){
    ESHandle<DTTtrig> tTrig;
    setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
    tTrigMap = &*tTrig;
    LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration]: TTrig version: " << tTrig->version() << endl; 
  }
}

DTTTrigOffsetCalibration::~DTTTrigOffsetCalibration(){
  theFile_->Close();
  LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration] Destructor called!";
}

void DTTTrigOffsetCalibration::analyze(const Event & event, const EventSetup& eventSetup) {
  theFile_->cd();
  DTChamberId chosenChamberId;

  if(theCalibChamber_ != "All") {
    stringstream linestr;
    int selWheel, selStation, selSector;
    linestr << theCalibChamber_;
    linestr >> selWheel >> selStation >> selSector;
    chosenChamberId = DTChamberId(selWheel, selStation, selSector);
    LogVerbatim("Calibration") << " chosen chamber " << chosenChamberId << endl;
  }

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the rechit collection from the event
  Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel_, all4DSegments); 

  // Get the map of noisy channels
  ESHandle<DTStatusFlag> statusMap;
  if(checkNoisyChannels_) eventSetup.get<DTStatusFlagRcd>().get(statusMap);

  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = all4DSegments->id_begin();
       chamberIdIt != all4DSegments->id_end();
       ++chamberIdIt){

    // Get the chamber from the setup
    const DTChamber* chamber = dtGeom->chamber(*chamberIdIt);
    LogTrace("Calibration") << "Chamber Id: " << *chamberIdIt;

    // Book histos
    if(theT0SegHistoMap_.find(*chamberIdIt) == theT0SegHistoMap_.end()){
      bookHistos(*chamberIdIt);
    }
   
    // Calibrate just the chosen chamber/s    
    if((theCalibChamber_ != "All") && ((*chamberIdIt) != chosenChamberId)) continue;

    // Get the range for the corresponding ChamberId
    DTRecSegment4DCollection::range range = all4DSegments->get((*chamberIdIt));

    // Loop over the rechits of this DetUnit
    for (DTRecSegment4DCollection::const_iterator segment = range.first;
         segment!=range.second; ++segment){

      LogTrace("Calibration") << "Segment local pos (in chamber RF): " << (*segment).localPosition()
                              << "\nSegment global pos: " << chamber->toGlobal((*segment).localPosition());
      
      //get the segment chi2
      double chiSquare = ((*segment).chi2()/(*segment).degreesOfFreedom());
      // cut on the segment chi2 
      if(chiSquare > theMaxChi2_) continue;

      // get the Phi 2D segment and plot the angle in the chamber RF
      if(!((*segment).phiSegment())){
        LogTrace("Calibration") << "No phi segment";
      }
      LocalPoint phiSeg2DPosInCham;  
      LocalVector phiSeg2DDirInCham;

      bool segmNoisy = false;
      map<DTSuperLayerId,vector<DTRecHit1D> > hitsBySLMap; 

      if((*segment).hasPhi()){
        const DTChamberRecSegment2D* phiSeg = (*segment).phiSegment();  // phiSeg lives in the chamber RF
        phiSeg2DPosInCham = phiSeg->localPosition();  
        phiSeg2DDirInCham = phiSeg->localDirection();

        vector<DTRecHit1D> phiHits = phiSeg->specificRecHits();
        for(vector<DTRecHit1D>::const_iterator hit = phiHits.begin();
            hit != phiHits.end(); ++hit) {
          DTWireId wireId = (*hit).wireId();
          DTSuperLayerId slId =  wireId.superlayerId();
          hitsBySLMap[slId].push_back(*hit); 

          // Check for noisy channels to skip them
          if(checkNoisyChannels_) {
            bool isNoisy = false;
            bool isFEMasked = false;
            bool isTDCMasked = false;
            bool isTrigMask = false;
            bool isDead = false;
            bool isNohv = false;
            statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
            if(isNoisy) {
              LogTrace("Calibration") << "Wire: " << wireId << " is noisy, skipping!";
              segmNoisy = true;
            }      
          }
        }
      }

      // get the Theta 2D segment and plot the angle in the chamber RF
      LocalVector zSeg2DDirInCham;
      LocalPoint zSeg2DPosInCham;
      if((*segment).hasZed()) {
        const DTSLRecSegment2D* zSeg = (*segment).zSegment();  // zSeg lives in the SL RF
        const DTSuperLayer* sl = chamber->superLayer(zSeg->superLayerId());
        zSeg2DPosInCham = chamber->toLocal(sl->toGlobal((*zSeg).localPosition())); 
        zSeg2DDirInCham = chamber->toLocal(sl->toGlobal((*zSeg).localDirection()));
        hitsBySLMap[zSeg->superLayerId()] = zSeg->specificRecHits();

        // Check for noisy channels to skip them
        vector<DTRecHit1D> zHits = zSeg->specificRecHits();
        for(vector<DTRecHit1D>::const_iterator hit = zHits.begin();
            hit != zHits.end(); ++hit) {
          DTWireId wireId = (*hit).wireId();
          if(checkNoisyChannels_) {
            bool isNoisy = false;
            bool isFEMasked = false;
            bool isTDCMasked = false;
            bool isTrigMask = false;
            bool isDead = false;
            bool isNohv = false;
            statusMap->cellStatus(wireId, isNoisy, isFEMasked, isTDCMasked, isTrigMask, isDead, isNohv);
            if(isNoisy) {
              LogTrace("Calibration") << "Wire: " << wireId << " is noisy, skipping!";
              segmNoisy = true;
            }      
          }
        }
      } 

      if (segmNoisy) continue;

      LocalPoint segment4DLocalPos = (*segment).localPosition();
      LocalVector segment4DLocalDir = (*segment).localDirection();
      if(fabs(atan(segment4DLocalDir.y()/segment4DLocalDir.z())* 180./Geom::pi()) > theMaxZAngle_) continue; // cut on the angle
      if(fabs(atan(segment4DLocalDir.x()/segment4DLocalDir.z())* 180./Geom::pi()) > theMaxPhiAngle_) continue; // cut on the angle
      // Fill t0-seg values
      if((*segment).hasPhi()) {
	//if((segment->phiSegment()->t0()) != 0.00){
        if(segment->phiSegment()->ist0Valid()){
	  (theT0SegHistoMap_[*chamberIdIt])[0]->Fill(segment->phiSegment()->t0());
	}
      }
      if((*segment).hasZed()){
    	//if((segment->zSegment()->t0()) != 0.00){
        if(segment->zSegment()->ist0Valid()){
	  (theT0SegHistoMap_[*chamberIdIt])[1]->Fill(segment->zSegment()->t0());
	}
      }
      
      // Fill t0-seg values
    //      if((*segment).hasPhi()) (theT0SegHistoMap_[*chamberIdIt])[0]->Fill(segment->phiSegment()->t0());
    //  if((*segment).hasZed()) (theT0SegHistoMap_[*chamberIdIt])[1]->Fill(segment->zSegment()->t0());
      //if((*segment).hasZed() && (*segment).hasPhi()) {}

      /*//loop over the segments 
      for(map<DTSuperLayerId,vector<DTRecHit1D> >::const_iterator slIdAndHits = hitsBySLMap.begin(); slIdAndHits != hitsBySLMap.end();  ++slIdAndHits) {
        if (slIdAndHits->second.size() < 3) continue;
        DTSuperLayerId slId =  slIdAndHits->first;

      }*/
    }
  }
}

void DTTTrigOffsetCalibration::endJob() {
  theFile_->cd();
  
  LogVerbatim("Calibration") << "[DTTTrigOffsetCalibration]Writing histos to file!" << endl;

  for(ChamberHistosMap::const_iterator itChHistos = theT0SegHistoMap_.begin(); itChHistos != theT0SegHistoMap_.end(); ++itChHistos){
    for(vector<TH1F*>::const_iterator itHist = (*itChHistos).second.begin();
                                      itHist != (*itChHistos).second.end(); ++itHist) (*itHist)->Write();
  }

  if(doTTrigCorrection_){
    // Create the object to be written to DB
    DTTtrig* tTrig = new DTTtrig();

    for(ChamberHistosMap::const_iterator itChHistos = theT0SegHistoMap_.begin(); itChHistos != theT0SegHistoMap_.end(); ++itChHistos){
      DTChamberId chId = itChHistos->first;
      // Get SuperLayerId's for each ChamberId
      vector<DTSuperLayerId> slIds;
      slIds.push_back(DTSuperLayerId(chId,1));
      slIds.push_back(DTSuperLayerId(chId,3));
      if(chId.station() != 4) slIds.push_back(DTSuperLayerId(chId,2));

      for(vector<DTSuperLayerId>::const_iterator itSl = slIds.begin(); itSl != slIds.end(); ++itSl){      
        // Get old values from DB
        float ttrigMean = 0;
        float ttrigSigma = 0;
	float kFactor = 0;
        tTrigMap->get(*itSl,ttrigMean,ttrigSigma,kFactor,DTTimeUnits::ns);
        //FIXME: verify if values make sense
        // Set new values
        float ttrigMeanNew = ttrigMean;
        float ttrigSigmaNew = ttrigSigma;
	float t0SegMean = (itSl->superLayer() != 2)?itChHistos->second[0]->GetMean():itChHistos->second[1]->GetMean();

	float kFactorNew = (kFactor*ttrigSigma+t0SegMean)/ttrigSigma;

        tTrig->set(*itSl,ttrigMeanNew,ttrigSigmaNew,kFactorNew,DTTimeUnits::ns);
      }
    }
    LogVerbatim("Calibration")<< "[DTTTrigOffsetCalibration]Writing ttrig object to DB!" << endl;
    // Write the object to DB
    string tTrigRecord = "DTTtrigRcd";
    DTCalibDBUtils::writeToDB(tTrigRecord, tTrig);
  } 
}

// Book a set of histograms for a given Chamber
void DTTTrigOffsetCalibration::bookHistos(DTChamberId chId) {

  LogTrace("Calibration") << "   Booking histos for Chamber: " << chId;

  // Compose the chamber name
  stringstream wheel; wheel << chId.wheel();
  stringstream station; station << chId.station();
  stringstream sector; sector << chId.sector();

  string chHistoName =
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();

  /*// Define the step
  stringstream Step; Step << step;

  string chHistoName =
    "_STEP" + Step.str() +
    "_W" + wheel.str() +
    "_St" + station.str() +
    "_Sec" + sector.str();

  theDbe->setCurrentFolder("DT/DTCalibValidation/Wheel" + wheel.str() +
                           "/Station" + station.str() +
                           "/Sector" + sector.str());
  // Create the monitor elements
  vector<MonitorElement *> histos;
  // Note hte order matters
  histos.push_back(theDbe->book1D("hRPhiSegT0"+chHistoName, "t0 from Phi segments", 200, -25., 25.));
  histos.push_back(theDbe->book1D("hRZSegT0"+chHistoName, "t0 from Z segments", 200, -25., 25.));*/

  vector<TH1F*> histos;
  // Note the order matters
  histos.push_back(new TH1F(("hRPhiSegT0"+chHistoName).c_str(), "t0 from Phi segments", 250, -60., 60.));
  if(chId.station() != 4) histos.push_back(new TH1F(("hRZSegT0"+chHistoName).c_str(), "t0 from Z segments", 250, -60., 60.));

  theT0SegHistoMap_[chId] = histos;
}
