#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

//#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TRandom.h"

#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/DeadModuleDB.h"

#include <iostream>

using namespace std;

namespace TMTT {

// Static variables

thread_local string Stub::trackerGeometryVersion_ = "UNKNOWN";

bool       Stub::stubKillerInit_ = false;
StubKiller Stub::stubKiller_;

//=== Store useful info about the stub (for use with HYBRID code), with hard-wired constants to allow use outside CMSSW.

Stub::Stub(double phi, double r, double z, double bend, int layerid, bool psModule, bool barrel, unsigned int iphi, double alpha, const Settings* settings, const TrackerTopology* trackerTopology, unsigned int ID, unsigned int iPhiSec) : 
  phi_(phi), r_(r), z_(z), bend_(bend), iphi_(iphi), alpha_(alpha), psModule_(psModule), layerId_(layerid), endcapRing_(0), barrel_(barrel), 
  digitalStub_(settings, r, phi, z, iPhiSec), stubWindowSuggest_(settings)
{ //work in progress on better constructor for new hybrid
  if (psModule && barrel) {
    double zMax[4];
    settings->get_zMaxNonTilted(zMax);
    tiltedBarrel_ = (fabs(z) > zMax[layerid]);
  } else {
    tiltedBarrel_ = false;
  }
  if (!psModule) {
    stripPitch_ = settings->ssStripPitch(); nStrips_=settings->ssNStrips(); sigmaPar_=settings->ssStripLength()/std::sqrt(12.0);
  } else {
    stripPitch_ = settings->psStripPitch(); nStrips_=settings->psNStrips(); sigmaPar_=settings->psPixelLength()/std::sqrt(12.0);
  }
  sigmaPerp_ = stripPitch_/std::sqrt(12.0);
  index_in_vStubs_ = ID; // A unique ID to label the stub.
}

//=== Store useful info about stub (for use with TMTT tracking).

Stub::Stub(const TTStubRef& ttStubRef, unsigned int index_in_vStubs, const Settings* settings, 
           const TrackerGeometry*  trackerGeometry, const TrackerTopology*  trackerTopology) : 
  TTStubRef(ttStubRef), 
  settings_(settings), 
  index_in_vStubs_(index_in_vStubs), 
  assocTP_(nullptr), // Initialize in case job is using no MC truth info.
  digitalStub_(settings),
  digitizedForGPinput_(false), // notes that stub has not yet been digitized for GP input.
  digitizedForHTinput_(false), // notes that stub has not yet been digitized for HT input.
  digitizedForSForTFinput_(""), // notes that stub has not yet been digitized for seed filter or track fitter input.
  digitizeWarningsOn_(true),
  stubWindowSuggest_(settings, trackerTopology), // TMTT recommendations for stub window sizes to CMS.
  degradeBend_(trackerTopology) // Used to degrade stub bend information.
{
  // Determine tracker geometry version (T3, T4, T5 ...)
  this->setTrackerGeometryVersion(trackerGeometry, trackerTopology);

  // Initialize tool to optionally emulate dead modules.
  
  if (not stubKillerInit_) {
    stubKillerInit_ = true;
    stubKiller_.initialise(settings->killScenario(), trackerTopology, trackerGeometry);
  }
  
  
  // Get coordinates of stub.
  const TTStub<Ref_Phase2TrackerDigi_> *ttStubP = ttStubRef.get();

  // The stub gives access to the DetId of the stacked module, but we want the DetId of the lower of
  // the two sensors in the module.

  /*
  // This the way CMS usually does this conversion, but it uses huge amounts of CPU.
  DetId geoDetId;
  for (const GeomDet* gd : trackerGeometry->dets()) {
      DetId detid = gd->geographicalId();
      if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID) continue; // Phase 2 Outer Tracker uses TOB for entire barrel & TID for entire endcap.
      if ( trackerTopology->isLower(detid) ) { // Select only lower of the two sensors in a module.
          DetId stackDetid = trackerTopology->stack(detid); // Det ID of stacked module containing stub.
          if ( ttStubRef->getDetId() == stackDetid ) {
              geoDetId = detid; // Note Det ID of lower sensor in stacked module containing stub.
              break;
          }
      }
  }
  if (geoDetId.null()) throw cms::Exception("Stub: Det ID corresponding to Stub not found");
  */

  // This is a faster way we found of doing the conversion. It seems to work ...
  DetId stackDetid = ttStubRef->getDetId();
  DetId geoDetId(stackDetid.rawId() + 1);
  if ( not (trackerTopology->isLower(geoDetId) && trackerTopology->stack(geoDetId) == stackDetid) ) throw cms::Exception("Stub: determination of detId went wrong");

  const GeomDetUnit* det0 = trackerGeometry->idToDetUnit( geoDetId );
  // To get other module, can do this
  // const GeomDetUnit* det1 = trackerGeometry->idToDetUnit( trackerTopology->partnerDetId( geoDetId ) );

  const PixelGeomDetUnit* theGeomDet = dynamic_cast< const PixelGeomDetUnit* >( det0 );
  const PixelTopology* topol = dynamic_cast< const PixelTopology* >( &(theGeomDet->specificTopology()) );
  MeasurementPoint measurementPoint = ttStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();
  LocalPoint clustlp   = topol->localPosition(measurementPoint);
  GlobalPoint pos  =  theGeomDet->surface().toGlobal(clustlp);

  phi_ = pos.phi();
  r_   = pos.perp();
  z_   = pos.z();

  if (r_ < settings_->trackerInnerRadius() || r_ > settings_->trackerOuterRadius() || fabs(z_) > settings_->trackerHalfLength()) {
    throw cms::Exception("Stub: Stub found outside assumed tracker volume. Please update tracker dimensions specified in Settings.h!")<<" r="<<r_<<" z="<<z_<<" "<<ttStubRef->getDetId().subdetId()<<endl;
  }

  // Set info about the module this stub is in
  this->setModuleInfo(trackerGeometry, trackerTopology, geoDetId);
  // Uncertainty in stub coordinates due to strip or pixel length in r-z.
  if (barrel_) {
    rErr_ = 0.;
    zErr_ = 0.5*stripLength_;
  } else {
    rErr_ = 0.5*stripLength_; 
    zErr_ = 0.;
  }

  // Get the coordinates of the two clusters that make up this stub, measured in units of strip pitch, and measured
  // in the local frame of the sensor. They have a granularity  of 0.5*pitch.
  for (unsigned int iClus = 0; iClus <= 1; iClus++) { // Loop over two clusters in stub.  
    localU_cluster_[iClus] = ttStubP->clusterRef(iClus)->findAverageLocalCoordinatesCentered().x();
    localV_cluster_[iClus] = ttStubP->clusterRef(iClus)->findAverageLocalCoordinatesCentered().y();
  }

  // Get location of stub in module in units of strip number (or pixel number along finest granularity axis).
  // Range from 0 to (nStrips - 1) inclusive.
  // N.B. Since iphi is integer, this degrades the granularity by a factor 2. This seems silly, but track fit wants it.
  iphi_ = localU_cluster_[0]; // granularity 1*strip (unclear why we want to degrade it ...)

  // Determine alpha correction for non-radial strips in endcap 2S modules.
  // (If true hit at larger r than stub r by deltaR, then stub phi needs correcting by +alpha*deltaR).  
  alpha_ = 0.;
  if ((not barrel_) && (not psModule_)) {
    float fracPosInModule = (float(iphi_) - 0.5*float(nStrips_)) / float(nStrips_);
    float phiRelToModule = sensorWidth_ * fracPosInModule / r_;
    if (z_ < 0)                  phiRelToModule *= -1;
    if (outerModuleAtSmallerR_)  phiRelToModule *= -1; // Module flipped.
    // If true hit at larger r than stub r by deltaR, then stub phi needs correcting by +alpha*deltaR.
    alpha_ = -phiRelToModule /r_ ;
  }

  // Calculate constants used to interpret bend information.

  // float sensorSpacing = barrel_ ? (moduleMaxR_ - moduleMinR_) : (moduleMaxZ_ - moduleMinZ_);
  // EJC Above not true for tilted modules
  float sensorSpacing = sqrt( (moduleMaxR_ - moduleMinR_) * (moduleMaxR_ - moduleMinR_) + (moduleMaxZ_ - moduleMinZ_) * (moduleMaxZ_ - moduleMinZ_) );
  
  pitchOverSep_ = stripPitch_/sensorSpacing;
  // IRT - use stub (r,z) instead of module (r,z). Logically correct but has negligable effect on results.
  // This old equation was valid for flat geom, where all modules are parallel or perpendicular to beam.
  //dphiOverBend_ = barrel_  ?  pitchOverSep_  :  pitchOverSep_*fabs(z_)/r_;
  // EJC - This new equation is valid in general case, so works for both flat and tilted geom.
  dphiOverBendCorrection_ = fabs( cos( this->theta() - moduleTilt_ ) / sin( this->theta() ) );
  dphiOverBendCorrection_approx_ = getApproxB();
  if ( settings->useApproxB() ) {
    dphiOverBend_ = pitchOverSep_ * dphiOverBendCorrection_approx_;
  }
  else{
    dphiOverBend_ = pitchOverSep_ * dphiOverBendCorrection_;
  }

  // Get stub bend that is available in front-end electronics, where bend is displacement between 
  // two hits in stubs in units of strip pitch.
  bendInFrontend_ = ttStubRef->bendFE();
  if ((not barrel_) && pos.z() > 0) bendInFrontend_ *= -1;
  // EJC Bend in barrel seems to be flipped in tilted geom.
  if (barrel_) bendInFrontend_ *= -1;

  // Get stub bend that is available in off-detector electronics, allowing for degredation of 
  // bend resolution due to bit encoding by FE chip if required.
  bool rejectStub = false;          // indicates if bend is outside window assumed in DegradeBend.h
  numMergedBend_ = 1;               // Number of bend values merged into single degraded one.
  if (settings->degradeBendRes() == 2) {
    float degradedBend;       // degraded bend
    this->degradeResolution(bendInFrontend_,
          degradedBend, rejectStub, numMergedBend_); // sets value of last 3 arguments.
    bend_ = degradedBend;
  } else if (settings->degradeBendRes() == 1) {
    bend_ = ttStubRef->bendBE(); // Degraded bend from official CMS recipe.
    if ((not barrel_) && pos.z() > 0) bend_ *= -1;
    if (barrel_) bend_ *= -1;
  } else {
    bend_ = bendInFrontend_;
  }

  // Fill frontendPass_ flag, indicating if frontend readout electronics will output this stub.
  this->setFrontend(rejectStub); 

  // Calculate bin range along q/Pt axis of r-phi Hough transform array consistent with bend of this stub.
  this->calcQoverPtrange();

  // Initialize class used to produce digital version of stub, with original stub parameters pre-digitization.
  digitalStub_.init(phi_, r_, z_, min_qOverPt_bin_, max_qOverPt_bin_, layerId_, this->layerIdReduced(), bend_, stripPitch_, sensorSpacing, rErr_, zErr_, barrel_, tiltedBarrel_, psModule_);

  // Update recommended stub window sizes that TMTT recommends that CMS should use in FE electronics.
  if (settings_->printStubWindows()) stubWindowSuggest_.process(this);

  // Initialize truth info to false in case job is using no MC truth info.
  for (unsigned int iClus = 0; iClus <= 1; iClus++) {
    assocTPofCluster_[iClus] = nullptr;
  }
}

//=== Calculate bin range along q/Pt axis of r-phi Hough transform array consistent with bend of this stub.

void Stub::calcQoverPtrange() {
  // First determine bin range along q/Pt axis of HT array 
  const int nbinsPt = (int) settings_->houghNbinsPt(); // Use "int" as nasty things happen if multiply "int" and "unsigned int".
  const int min_array_bin = 0;
  const int max_array_bin = nbinsPt - 1;  
  // Now calculate range of q/Pt bins allowed by bend filter.
  float qOverPtMin = this->qOverPtOverBend() * (this->bend() - this->bendRes());
  float qOverPtMax = this->qOverPtOverBend() * (this->bend() + this->bendRes());
  int houghNbinsPt = settings_->houghNbinsPt();
  const float houghMaxInvPt = 1./settings_->houghMinPt();
  float qOverPtBinSize = (2. * houghMaxInvPt)/houghNbinsPt;
  if ( settings_->shape() == 2 || settings_->shape() == 1 || settings_->shape() == 3 ) // Non-square HT cells.
    qOverPtBinSize = 2. * houghMaxInvPt / ( houghNbinsPt - 1 );
  // Convert to bin number along q/Pt axis of HT array.
  // N.B. For square HT cells, setting "tmp = -0.5" causeas cell to be accepted if q/Pt at its centre is consistent 
  // with the stub bend. Instead using "tmp = 0.0" accepts cells if q/Pt at any point in cell is consistent with bend.
  // So if you use change from -0.5 to 0.0, you have to tighten the bend cut (by ~0.05) to get similar performance.
  // Decision to set tmp = 0.0 taken in softare & GP firmware on 9th August 2016.
  //float tmp = ( settings_->shape() == 2 || settings_->shape() == 1 || settings_->shape() == 3 ) ? 1. : -0.5;

  float tmp = ( settings_->shape() == 2 || settings_->shape() == 1 || settings_->shape() == 3 ) ? 1. : 0.;
  int min_bin = std::floor(-tmp + (qOverPtMin + houghMaxInvPt)/qOverPtBinSize);
  int max_bin = std::floor( tmp + (qOverPtMax + houghMaxInvPt)/qOverPtBinSize);

  // Limit it to range of HT array.
  min_bin = max(min_bin, min_array_bin);
  max_bin = min(max_bin, max_array_bin);
  // If min_bin > max_bin at this stage, it means that the Pt estimated from the bend is below the cutoff for track-finding.
  // Keep min_bin > max_bin, so such stubs can be rejected, but set both variables to values inside the HT bin range.
  if (min_bin > max_bin) {
    min_bin = max_array_bin;
    max_bin = min_array_bin;
    //if (frontendPass_) throw cms::Exception("Stub: m bin calculation found low Pt stub not killed by FE electronics cuts")<<qOverPtMin<<" "<<qOverPtMax<<endl;
  }
  min_qOverPt_bin_ = (unsigned int) min_bin;
  max_qOverPt_bin_ = (unsigned int) max_bin;
}

//=== Digitize stub for input to Geographic Processor, with digitized phi coord. measured relative to closest phi sector.
//=== (This approximation is valid if their are an integer number of digitisation bins inside each phi nonant).
//=== However, you should also call digitizeForHTinput() before accessing digitized stub data, even if you only care about that going into GP! Otherwise, you will not identify stubs assigned to more than one nonant.

void Stub::digitizeForGPinput(unsigned int iPhiSec) {
  if (settings_->enableDigitize()) {

    // Save CPU by not redoing digitization if stub was already digitized for this phi sector.
    if ( ! (digitizedForGPinput_ && digitalStub_.iGetNonant(iPhiSec) == digitalStub_.iDigi_Nonant()) ) {

      // Digitize
      digitalStub_.makeGPinput(iPhiSec);

      // Replace stub coordinates with those degraded by digitization process.
      phi_  = digitalStub_.phi();
      r_    = digitalStub_.r();
      z_    = digitalStub_.z();
      bend_ = digitalStub_.bend();

      // If the Stub class contains any data members that are not input to the GP, but are derived from variables that
      // are, then be sure to update these here too, unless Stub.h uses the check*() functions to declare them invalid.
      dphiOverBendCorrection_ = fabs( cos( this->theta() - moduleTilt_ ) / sin( this->theta() ) );
      dphiOverBend_ = pitchOverSep_ * dphiOverBendCorrection_; 

      // Note that stub has been digitized for GP input
      digitizedForGPinput_ = true;
    }
    digitizedForHTinput_ = false;
  }
}

//=== Digitize stub for input to Hough transform, with digitized phi coord. measured relative to specified phi sector.

void Stub::digitizeForHTinput(unsigned int iPhiSec) {

  if (settings_->enableDigitize()) {

    // Save CPU by not redoing digitization if stub was already digitized for this phi sector.
    if ( ! (digitizedForHTinput_ && iPhiSec == digitalStub_.iDigi_PhiSec()) ) {

      // Call digitization for GP in case not already done. (Needed for variables that are common to GP & HT).
      this->digitizeForGPinput(iPhiSec);

      // Digitize
      digitalStub_.makeHTinput(iPhiSec);

      // Since GP and HT use same digitisation in r and z, don't bother updating their values.
      // (Actually, the phi digitisation boundaries also match, except for systolic array, so could skip updating phi too).

      // Replace stub coordinates and bend with those degraded by digitization process. (Don't bother with r & z, as already done by GP digitisation).
      phi_  = digitalStub_.phi();

      // Recalculate bin range along q/Pt axis of r-phi Hough transform array 
      // consistent with bend of this stub, since it depends on r & z which have now been digitized.
      // (This recalculation should really be done in DigitalStub::makeHTinput(), but too lazy to move it there ...).
      this->calcQoverPtrange();

      // If the Stub class contains any data members that are not input to the HT, but are derived from variables that
      // are, then be sure to update these here too, unless Stub.h uses the check*() functions to declare them invalid. 
      // - currently none.

      // Note that stub has been digitized.
      digitizedForHTinput_ = true;
    }
  }
}

//=== Digitize stub for input to r-z Seed Filter or Track Fitter.
//=== Argument is "SeedFilter" or name of Track Fitter.

void Stub::digitizeForSForTFinput(string SForTF) {
  if (settings_->enableDigitize()) {

    if ( digitizedForSForTFinput_ != SForTF) {
      // Digitize variables specific to seed filter or track fittr if not already done.
      digitalStub_.makeSForTFinput(SForTF);

      // Replace stub (r,z) uncertainties, estimated from half-pixel/strip-length, by those degraded by the digitization process. 
      rErr_ = digitalStub_.rErr();
      zErr_ = digitalStub_.zErr();
      // Must also replace stub r coordinate, as seed filter & fitters work with digitized r instead of digitized rT.
      r_    = digitalStub_.r();
      // And KF may also redigitize z.
      z_    = digitalStub_.z();

      digitizedForSForTFinput_ = SForTF;
    }
  }
}

//=== Digitize stub for input to r-z Seed Filter.

void Stub::digitizeForDRinput(unsigned int stubId) {
  if (settings_->enableDigitize()) {
    
    // Digitize variables specific to seed filter if not already done.
    digitalStub_.makeDRinput(stubId);
    // digitizedForDRinput_ = true;
    
  }
}

//===  Restore stub to pre-digitized state. i.e. Undo what function digitize() did.

void Stub::reset_digitize() {
  if (settings_->enableDigitize()) {
    // Save CPU by not undoing digitization if stub was not already digitized.
    if (digitizedForGPinput_ || digitizedForHTinput_) {

      // Replace stub coordinates and bend with original coordinates stored prior to any digitization.
      phi_  = digitalStub_.orig_phi();
      r_    = digitalStub_.orig_r();
      z_    = digitalStub_.orig_z();
      bend_ = digitalStub_.orig_bend();

      // Also restore original uncertainties in stub coordinates (estimated from strip or pixel half-length).
      rErr_ = digitalStub_.orig_rErr();
      zErr_ = digitalStub_.orig_zErr();

      // Note that stub is (no longer) digitized.
      digitizedForGPinput_ = false;
      digitizedForHTinput_ = false;
      digitizedForSForTFinput_ = "";

      // If the Stub class contains any data members that are not input to the GP or HT, but are derived from 
      // variables that are, then be sure to update these here too.
      dphiOverBendCorrection_ = fabs( cos( this->theta() - moduleTilt_ ) / sin( this->theta() ) );
      dphiOverBend_ = pitchOverSep_ * dphiOverBendCorrection_; 
    }
  }
}

//=== Degrade assumed stub bend resolution.
//=== Also return boolean indicating if stub bend was outside assumed window, so stub should be rejected
//=== and return an integer indicating how many values of bend are merged into this single one.

void Stub::degradeResolution(float bend, 
			     float& degradedBend, bool& reject, unsigned int& num) const {

  // If TMTT code is tightening official CMS FE stub window cuts, then calculate TMTT stub windows.
  float windowFE;
  if (settings_->killLowPtStubs()) {
    // Window size corresponding to Pt cut used for tracking.
    float invPtMax = 1./(settings_->houghMinPt());
    windowFE = invPtMax/fabs(this->qOverPtOverBend());
    // Increase half-indow size to allow for resolution in bend.				       
    windowFE += this->bendResInFrontend();
  } else {
    windowFE = 99999.; // TMTT is not tightening windows.
  }

  static bool firstErr = true;
  if (trackerGeometryVersion_ != "T5") { // Tilted geometry
    if (firstErr) {
      cout<<"Stub: WARNING - Stub windows in DegradeBend class have not been tuned for this tracker geometry, so may need retuning "<<trackerGeometryVersion_ << endl;
      firstErr = false;
    }
  }

  degradeBend_.degrade(bend, psModule_, idDet_, windowFE,
                       degradedBend, reject, num);
}


//=== Set flag indicating if stub will be output by front-end readout electronics 
//=== (where we can reconfigure the stub window size and rapidity cut).
//=== Argument indicates if stub bend was outside window size encoded in DegradeBend.h
//=== Note that this should run on quantities as available inside front-end chip, which are not
//=== degraded by loss of bits or digitisation.

void Stub::setFrontend(bool rejectStub) {
  frontendPass_ = true; // Did stub pass cuts applied in front-end chip
  stubFailedDegradeWindow_ = false; // Did it only fail cuts corresponding to windows encoded in DegradeBend.h?
  // Don't use stubs at large eta, since it is impossible to form L1 tracks from them, so they only contribute to combinatorics.
  if ( fabs(this->eta()) > settings_->maxStubEta() ) frontendPass_ = false;
  // Don't use stubs whose Pt is significantly below the Pt cut used in the L1 tracking, allowing for uncertainty in q/Pt due to stub bend resolution.
  if (settings_->killLowPtStubs()) {
    const float qOverPtCut = 1./settings_->houghMinPt();
    // Apply this cut in the front-end electronics.
    if (fabs(this->bendInFrontend()) - this->bendResInFrontend() > qOverPtCut/this->qOverPtOverBend()) frontendPass_ = false;
    // Reapply the same cut using the degraded bend information available in the off-detector electronics.
    // The reason is  that the bend degredation can move the Pt below the Pt cut, making the stub useless to the off-detector electronics.
    if (fabs(this->bend())           - this->bendRes()           > qOverPtCut/this->qOverPtOverBend()) frontendPass_ = false;
  } 
  // Don't use stubs whose bend is outside the window encoded into DegradeBend.h
  if (rejectStub) {
    if (frontendPass_) stubFailedDegradeWindow_ = true;
    frontendPass_ = false;
  }

  // Emulate stubs in dead tracker regions using private TMTT emulation.
  if (settings_->deadSimulateFrac() > 0.) { // Is option to emulate dead modules enabled?
    const DeadModuleDB dead;
    if (dead.killStub(this)) {
      static TRandom randomGenerator;
      if (randomGenerator.Rndm() < settings_->deadSimulateFrac()) frontendPass_ = false;
    }
  }

  // Or emulate stubs in dead tracker regions using communal emulation shared with Tracklet.
  if (settings_->killScenario() > 0) {
    TTStubRef ttStubRef(*this); // Cast to base class
    bool kill = stubKiller_.killStub(ttStubRef.get());
    if (kill) frontendPass_ = false;
  }
}

//=== Function to calculate approximation for dphiOverBendCorrection aka B
double Stub::getApproxB() {
  if ( tiltedBarrel_ ) {
      return settings_->bApprox_gradient() * fabs(z_)/r_ + settings_->bApprox_intercept();
  }
  else {
    return barrel_  ?  1  :  fabs(z_)/r_;
  }
}

//=== Note which tracking particle(s), if any, produced this stub.
//=== The 1st argument is a map relating TrackingParticles to TP.

void Stub::fillTruth(const map<edm::Ptr< TrackingParticle >, const TP* >& translateTP, const edm::Handle<TTStubAssMap>& mcTruthTTStubHandle, const edm::Handle<TTClusterAssMap>& mcTruthTTClusterHandle){

  TTStubRef ttStubRef(*this); // Cast to base class

  //--- Fill assocTP_ info. If both clusters in this stub were produced by the same single tracking particle, find out which one it was.

  bool genuine =  mcTruthTTStubHandle->isGenuine(ttStubRef); // Same TP contributed to both clusters?
  assocTP_ = nullptr;

  // Require same TP contributed to both clusters.
  if ( genuine ) {
    edm::Ptr< TrackingParticle > tpPtr = mcTruthTTStubHandle->findTrackingParticlePtr(ttStubRef);
    if (translateTP.find(tpPtr) != translateTP.end()) {
      assocTP_ = translateTP.at(tpPtr);
      // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
    }
  }

  // Fill assocTPs_ info.

  if (settings_->stubMatchStrict()) {

    // We consider only stubs in which this TP contributed to both clusters.
    if (assocTP_ != nullptr) assocTPs_.insert(assocTP_);

  } else {

    // We consider stubs in which this TP contributed to either cluster.

    for (unsigned int iClus = 0; iClus <= 1; iClus++) { // Loop over both clusters that make up stub.
       const TTClusterRef& ttClusterRef = ttStubRef->clusterRef(iClus);

      // Now identify all TP's contributing to either cluster in stub.
      vector< edm::Ptr< TrackingParticle > > vecTpPtr = mcTruthTTClusterHandle->findTrackingParticlePtrs(ttClusterRef);

      for (edm::Ptr< TrackingParticle> tpPtr : vecTpPtr) {
  if (translateTP.find(tpPtr) != translateTP.end()) {
    assocTPs_.insert( translateTP.at(tpPtr) );
    // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
  }
      }
    }
  }

  //--- Also note which tracking particles produced the two clusters that make up the stub

  for (unsigned int iClus = 0; iClus <= 1; iClus++) { // Loop over both clusters that make up stub.
    const TTClusterRef& ttClusterRef = ttStubRef->clusterRef(iClus);

    bool genuineCluster =  mcTruthTTClusterHandle->isGenuine(ttClusterRef); // Only 1 TP made cluster?
    assocTPofCluster_[iClus] = nullptr;

    // Only consider clusters produced by just one TP.
    if ( genuineCluster ) {
      edm::Ptr< TrackingParticle > tpPtr = mcTruthTTClusterHandle->findTrackingParticlePtr(ttClusterRef);

      if (translateTP.find(tpPtr) != translateTP.end()) {
  assocTPofCluster_[iClus] = translateTP.at(tpPtr);
  // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
      }
    }
  }

  // Sanity check - is truth info of stub consistent with that of its clusters?
  // Commented this out, as it throws errors for unknown reason with iErr=1. Apparently, "genuine" stubs can be composed of two clusters that are
  // not "genuine", providing that one of the TP that contributed to each cluster was the same.
  /*
  unsigned int iErr = 0;
  if (this->genuine()) { // Stub matches truth particle
    if ( ! ( this->genuineCluster()[0] && (this->assocTPofCluster()[0] == this->assocTPofCluster()[1]) ) ) iErr = 1;
  } else {
    if ( ! ( ! this->genuineCluster()[0] || (this->assocTPofCluster()[0] != this->assocTPofCluster()[1]) )  ) iErr = 2;
  }
  if (iErr > 0) {
    cout<<" DEBUGA "<<(this->assocTP() == nullptr)<<endl;
    cout<<" DEBUGB "<<(this->assocTPofCluster()[0] == nullptr)<<" "<<(this->assocTPofCluster()[1] == nullptr)<<endl;
    cout<<" DEBUGC "<<this->genuineCluster()[0]<<" "<<this->genuineCluster()[1]<<endl;
    if (this->assocTPofCluster()[0] != nullptr) cout<<" DEBUGD "<<this->assocTPofCluster()[0]->index()<<endl;
    if (this->assocTPofCluster()[1] != nullptr) cout<<" DEBUGE "<<this->assocTPofCluster()[1]->index()<<endl;
    //    throw cms::Exception("Stub: Truth info of stub & its clusters inconsistent!")<<iErr<<endl;
  }
  */
}

//=== Estimated phi angle at which track intercepts a given radius rad, based on stub bend info. Also estimate uncertainty on this angle due to endcap 2S module strip length.
//=== N.B. This is identical to Stub::beta() if rad=0.

pair <float, float> Stub::trkPhiAtR(float rad) const { 
  float rStubMax = r_ + rErr_; // Uncertainty in radial stub coordinate due to strip length.
  float rStubMin = r_ - rErr_;
  float trkPhi1 = (phi_ + dphi()*(1. - rad/rStubMin));
  float trkPhi2 = (phi_ + dphi()*(1. - rad/rStubMax));
  float trkPhi    = 0.5*    (trkPhi1 + trkPhi2);
  float errTrkPhi = 0.5*fabs(trkPhi1 - trkPhi2); 
  return pair<float, float>(trkPhi, errTrkPhi);
}


//=== Note if stub is a crazy distance from the tracking particle trajectory that produced it.
//=== If so, it was probably produced by a delta ray.

bool Stub::crazyStub() const {

  bool crazy;
  if (assocTP_ == nullptr) {
    crazy = false; // Stub is fake, but this is not crazy. It happens ...
  } else {
    // Stub was produced by TP. Check it lies not too far from TP trajectory.
    crazy = fabs( reco::deltaPhi(phi_, assocTP_->trkPhiAtStub( this )) )  >  settings_->crazyStubCut();
  } 
  return crazy;
}

//=== Get reduced layer ID (in range 1-7), which can be packed into 3 bits so simplifying the firmware).

unsigned int Stub::layerIdReduced() const {
  // Don't bother distinguishing two endcaps, as no track can have stubs in both.
  unsigned int lay = (layerId_ < 20) ? layerId_ : layerId_ - 10; 

  // No genuine track can have stubs in both barrel layer 6 and endcap disk 11 etc., so merge their layer IDs.
  // WARNING: This is tracker geometry dependent, so may need changing in future ...
  if (lay == 6) lay = 11; 
  if (lay == 5) lay = 12; 
  if (lay == 4) lay = 13; 
  if (lay == 3) lay = 15; 
  // At this point, the reduced layer ID can have values of 1, 2, 11, 12, 13, 14, 15. So correct to put in range 1-7.
  if (lay > 10) lay -= 8;

  if (lay < 1 || lay > 7) throw cms::Exception("Stub: Reduced layer ID out of expected range");

  return lay;
}


//=== Set info about the module that this stub is in.

void Stub::setModuleInfo(const TrackerGeometry* trackerGeometry, const TrackerTopology* trackerTopology, const DetId& detId) {

  idDet_ = detId;

  // Note if module is PS or 2S, and whether in barrel or endcap.
  psModule_ = trackerGeometry->getDetectorType( detId ) == TrackerGeometry::ModuleType::Ph2PSP; // From https://github.com/cms-sw/cmssw/blob/CMSSW_8_1_X/Geometry/TrackerGeometryBuilder/README.md
  barrel_ = detId.subdetId()==StripSubdetector::TOB || detId.subdetId()==StripSubdetector::TIB;

  // Get min & max (r,phi,z) coordinates of the centre of the two sensors containing this stub.
  const GeomDetUnit* det0 = trackerGeometry->idToDetUnit( detId );
  const GeomDetUnit* det1 = trackerGeometry->idToDetUnit( trackerTopology->partnerDetId( detId ) );

  float R0 = det0->position().perp();
  float R1 = det1->position().perp();
  float PHI0 = det0->position().phi();
  float PHI1 = det1->position().phi();
  float Z0 = det0->position().z();
  float Z1 = det1->position().z();
  moduleMinR_   = std::min(R0,R1);
  moduleMaxR_   = std::max(R0,R1);
  moduleMinPhi_ = std::min(PHI0,PHI1);
  moduleMaxPhi_ = std::max(PHI0,PHI1);
  moduleMinZ_   = std::min(Z0,Z1);
  moduleMaxZ_   = std::max(Z0,Z1);

  // Note if tilted barrel module & get title angle (in range 0 to PI).
  tiltedBarrel_ = barrel_ && (trackerTopology->tobSide(detId) != 3);
  float deltaR = fabs(R1 - R0);
  float deltaZ = (R1 - R0 > 0)  ?  (Z1 - Z0)  :  - (Z1 - Z0);
  moduleTilt_  = atan2( deltaR, deltaZ);
  if (moduleTilt_ >  M_PI/2.) moduleTilt_ -= M_PI; // Put in range -PI/2 to +PI/2.
  if (moduleTilt_ < -M_PI/2.) moduleTilt_ += M_PI; // 

  // cout<<"DEBUG STUB "<<barrel_<<" "<<psModule_<<"  sep(r,z)=( "<<moduleMaxR_ - moduleMinR_<<" , "<<moduleMaxZ_ - moduleMinZ_<<" )    stub(r,z)=( "<<0.5*(moduleMaxR_ + moduleMinR_) - r_<<" , "<<0.5*(moduleMaxZ_ + moduleMinZ_) - z_<<" )"<<endl;

  // Encode layer ID.
  if (barrel_) {
    layerId_ = trackerTopology->layer( detId ); // barrel layer 1-6 encoded as 1-6
  } else {
    // layerId_ = 10*detId.iSide() + detId.iDisk(); // endcap layer 1-5 encoded as 11-15 (endcap A) or 21-25 (endcapB)
    // EJC This seems to give the same encoding as what we had in CMSSW6
    layerId_ = 10*trackerTopology->side( detId ) + trackerTopology->tidWheel( detId );
  }

  // Note module ring in endcap
  endcapRing_ = barrel_  ?  0  :  trackerTopology->tidRing( detId );

  if (trackerGeometryVersion_ == "T5") {
    if ( ! barrel_) {
      // Apply bodge, since Topology class annoyingly starts ring count at 1, even in endcap wheels where
      // inner rings are absent.
      unsigned int iWheel = trackerTopology->tidWheel( detId );
      if (iWheel >= 3 && iWheel <=5) endcapRing_ += 3;
    }
  }

  // Get sensor strip or pixel pitch using innermost sensor of pair.

  const PixelGeomDetUnit* unit = reinterpret_cast<const PixelGeomDetUnit*>( det0 );
  const PixelTopology& topo = unit->specificTopology();
  const Bounds& bounds = det0->surface().bounds();

  std::pair<float, float> pitch = topo.pitch();
  stripPitch_ = pitch.first; // Strip pitch (or pixel pitch along shortest axis)
  stripLength_ = pitch.second;  //  Strip length (or pixel pitch along longest axis)
  nStrips_ = topo.nrows(); // No. of strips in sensor
  sensorWidth_ = bounds.width(); // Width of sensitive region of sensor (= stripPitch * nStrips).

  // Note if modules are flipped back-to-front.
  outerModuleAtSmallerR_ = ( det0->position().mag() > det1->position().mag() );
  /*
  if ( barrel_ && det0->position().perp() > det1->position().perp() ) {
    outerModuleAtSmallerR_ = true;
  }
  */

  sigmaPerp_ = stripPitch_/sqrt(12.); // resolution perpendicular to strip (or to longest pixel axis)
  sigmaPar_  = stripLength_/sqrt(12.); // resolution parallel to strip (or to longest pixel axis)
}

//=== Determine tracker geometry version by counting modules.

void Stub::setTrackerGeometryVersion(const TrackerGeometry* trackerGeometry, const TrackerTopology* trackerTopology) {

  // N.B. The relationship between Tracker geometry name (T3, T4, T5 ...) and CMS geometry name
  // (D13, D17 ...) is documented in 
  //  https://github.com/cms-sw/cmssw/blob/CMSSW_9_1_X/Configuration/Geometry/README.md .

  if (trackerGeometryVersion_ == "UNKNOWN") {
    unsigned int numDet = 0;
    for (const GeomDet* gd : trackerGeometry->dets()) {
      DetId detid = gd->geographicalId();
      if (detid.subdetId() != StripSubdetector::TOB || detid.subdetId() != StripSubdetector::TID) { // Phase 2 Outer Tracker uses TOB for entire barrel & TID for entire endcap.        
        if ( trackerTopology->isLower(detid) ) { // Select only lower of the two sensors in a module.
          numDet++;
        }
      }
    }

    if (numDet == 13296) {
      trackerGeometryVersion_ = "T5"; // Tilted geometry
    } else if (numDet == 13556) {
      trackerGeometryVersion_ = "T3"; // Older tilted geometry
    } else if (numDet == 14850) {
      trackerGeometryVersion_ = "T4"; // Flat geometry
    } else {
      trackerGeometryVersion_ = "UNRECOGNISED";
      cout<<"Stub: WARNING -- The tracker geometry you are using is yet not known to the stub class. Please update Stub::degradeResolution() & Stub::setTrackerGeometryVersion(). Number of tracker modules = "<<numDet<<endl;
    }
  }
}

}
