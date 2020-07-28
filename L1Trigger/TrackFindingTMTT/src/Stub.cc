#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/StubKiller.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"

#include <iostream>

using namespace std;

namespace tmtt {

  //=== Hybrid L1 tracking: stub constructor.

  Stub::Stub(const Settings* settings,
             unsigned int idStub,
             double phi,
             double r,
             double z,
             double bend,
             unsigned int iphi,
             double alpha,
             unsigned int layerId,
             unsigned int iPhiSec,
             bool psModule,
             bool barrel,
             bool tiltedBarrel,
             float stripPitch,
             float stripLength,
             unsigned int nStrips)
      : index_in_vStubs_(idStub),  // A unique ID to label the stub.
        phi_(phi),
        r_(r),
        z_(z),
        bend_(bend),
        iphi_(iphi),
        alpha_(alpha),
        digitalStub_(std::make_unique<DigitalStub>(settings, r, phi, z, iPhiSec)),
        layerId_(layerId),
        layerIdReduced_(TrackerModule::calcLayerIdReduced(layerId)),
        stripPitch_(stripPitch),
        stripLength_(stripLength),
        nStrips_(nStrips),
        psModule_(psModule),
        barrel_(barrel),
        tiltedBarrel_(tiltedBarrel) {}

  //=== TMTT L1 tracking: stub constructor.

  Stub::Stub(const TTStubRef& ttStubRef,
             unsigned int index_in_vStubs,
             const Settings* settings,
             const TrackerTopology* trackerTopology,
             const TrackerModule* trackerModule,
             const DegradeBend* degradeBend,
             const StubKiller* stubKiller)
      : ttStubRef_(ttStubRef),
        settings_(settings),
        index_in_vStubs_(index_in_vStubs),
        assocTP_(nullptr),  // Initialize in case job is using no MC truth info.
        lastDigiStep_(Stub::DigiStage::NONE),
        digitizeWarningsOn_(true),
        trackerModule_(trackerModule),  // Info about tracker module containing stub
        degradeBend_(degradeBend),      // Used to degrade stub bend information.
        // Module related variables (need to be stored for Hybrid)
        layerId_(trackerModule->layerId()),
        layerIdReduced_(trackerModule->layerIdReduced()),
        tiltAngle_(trackerModule->tiltAngle()),
        stripPitch_(trackerModule->stripPitch()),
        stripLength_(trackerModule->stripLength()),
        nStrips_(trackerModule->nStrips()),
        psModule_(trackerModule->psModule()),
        barrel_(trackerModule->barrel()),
        tiltedBarrel_(trackerModule->tiltedBarrel()) {
    // Get coordinates of stub.
    const TTStub<Ref_Phase2TrackerDigi_>* ttStubP = ttStubRef_.get();

    const PixelGeomDetUnit* specDet = trackerModule_->specDet();
    const PixelTopology* specTopol = trackerModule_->specTopol();
    MeasurementPoint measurementPoint = ttStubRef_->clusterRef(0)->findAverageLocalCoordinatesCentered();
    LocalPoint clustlp = specTopol->localPosition(measurementPoint);
    GlobalPoint pos = specDet->surface().toGlobal(clustlp);

    phi_ = pos.phi();
    r_ = pos.perp();
    z_ = pos.z();

    // Get the coordinates of the two clusters that make up this stub, measured in units of strip pitch, and measured
    // in the local frame of the sensor. They have a granularity  of 0.5*pitch.
    for (unsigned int iClus = 0; iClus <= 1; iClus++) {  // Loop over two clusters in stub.
      localU_cluster_[iClus] = ttStubP->clusterRef(iClus)->findAverageLocalCoordinatesCentered().x();
      localV_cluster_[iClus] = ttStubP->clusterRef(iClus)->findAverageLocalCoordinatesCentered().y();
    }

    // Get location of stub in module in units of strip number (or pixel number along finest granularity axis).
    // Range from 0 to (nStrips - 1) inclusive.
    // N.B. Since iphi is integer, this degrades the granularity by a factor 2. This seems silly, but track fit wants it.
    iphi_ = localU_cluster_[0];  // granularity 1*strip (unclear why we want to degrade it ...)

    // Determine alpha correction for non-radial strips in endcap 2S modules.
    // (If true hit at larger r than stub r by deltaR, then stub phi needs correcting by +alpha*deltaR).
    alpha_ = 0.;
    if ((not barrel()) && (not psModule())) {
      float fracPosInModule = (float(iphi_) - 0.5 * float(nStrips())) / float(nStrips());
      float phiRelToModule = trackerModule_->sensorWidth() * fracPosInModule / r_;
      if (z_ < 0)
        phiRelToModule *= -1;
      if (trackerModule_->outerModuleAtSmallerR())
        phiRelToModule *= -1;  // Module flipped.
      // If true hit at larger r than stub r by deltaR, then stub phi needs correcting by +alpha*deltaR.
      alpha_ = -phiRelToModule / r_;
    }

    // Calculate variables giving ratio of track intercept angle to stub bend.
    this->calcDphiOverBend();

    // Get stub bend that is available in front-end electronics, where bend is displacement between
    // two hits in stubs in units of strip pitch.
    bendInFrontend_ = ttStubRef_->bendFE();
    if ((not barrel()) && pos.z() > 0)
      bendInFrontend_ *= -1;
    if (barrel())
      bendInFrontend_ *= -1;

    // Get stub bend that is available in off-detector electronics, allowing for degredation of
    // bend resolution due to bit encoding by FE chip if required.
    numMergedBend_ = 1;  // Number of bend values merged into single degraded one.
    if (settings->degradeBendRes() == 2) {
      float degradedBend;  // degraded bend
      // This returns values of degradedBend & numMergedBend_
      this->degradeResolution(bendInFrontend_, degradedBend, numMergedBend_);
      bend_ = degradedBend;
    } else if (settings->degradeBendRes() == 1) {
      bend_ = ttStubRef_->bendBE();  // Degraded bend from official CMS recipe.
      if ((not barrel()) && pos.z() > 0)
        bend_ *= -1;
      if (barrel())
        bend_ *= -1;
    } else {
      bend_ = bendInFrontend_;
    }

    // Fill frontendPass_ flag, indicating if frontend readout electronics will output this stub.
    this->setFrontend(stubKiller);

    // Calculate bin range along q/Pt axis of r-phi Hough transform array consistent with bend of this stub.
    this->calcQoverPtrange();

    // Initialize truth info to false in case job is using no MC truth info.
    for (unsigned int iClus = 0; iClus <= 1; iClus++) {
      assocTPofCluster_[iClus] = nullptr;
    }
  }

  //=== Calculate bin range along q/Pt axis of r-phi Hough transform array consistent with bend of this stub.

  void Stub::calcQoverPtrange() {
    // First determine bin range along q/Pt axis of HT array
    // (Use "int" as nasty things happen if multiply "int" and "unsigned int").
    const int nbinsPt = (int)settings_->houghNbinsPt();
    const int min_array_bin = 0;
    const int max_array_bin = nbinsPt - 1;
    // Now calculate range of q/Pt bins allowed by bend filter.
    float qOverPtMin = this->qOverPtOverBend() * (this->bend() - this->bendCut());
    float qOverPtMax = this->qOverPtOverBend() * (this->bend() + this->bendCut());
    int houghNbinsPt = settings_->houghNbinsPt();
    const float houghMaxInvPt = 1. / settings_->houghMinPt();
    float qOverPtBinSize = (2. * houghMaxInvPt) / houghNbinsPt;
    if (settings_->shape() == 2 || settings_->shape() == 1 || settings_->shape() == 3)  // Non-square HT cells.
      qOverPtBinSize = 2. * houghMaxInvPt / (houghNbinsPt - 1);
    // Convert to bin number along q/Pt axis of HT array.
    // N.B. For square HT cells, setting "tmp = -0.5" causeas cell to be accepted if q/Pt at its centre is consistent
    // with the stub bend. Instead using "tmp = 0.0" accepts cells if q/Pt at any point in cell is consistent with bend.
    // So if you use change from -0.5 to 0.0, you have to tighten the bend cut (by ~0.05) to get similar performance.
    // Decision to set tmp = 0.0 taken in softare & GP firmware on 9th August 2016.

    float tmp = (settings_->shape() == 2 || settings_->shape() == 1 || settings_->shape() == 3) ? 1. : 0.;
    int min_bin = std::floor(-tmp + (qOverPtMin + houghMaxInvPt) / qOverPtBinSize);
    int max_bin = std::floor(tmp + (qOverPtMax + houghMaxInvPt) / qOverPtBinSize);

    // Limit it to range of HT array.
    min_bin = max(min_bin, min_array_bin);
    max_bin = min(max_bin, max_array_bin);
    // If min_bin > max_bin at this stage, it means that the Pt estimated from the bend is below the cutoff for track-finding.
    // Keep min_bin > max_bin, so such stubs can be rejected, but set both variables to values inside the HT bin range.
    if (min_bin > max_bin) {
      min_bin = max_array_bin;
      max_bin = min_array_bin;
    }
    min_qOverPt_bin_ = (unsigned int)min_bin;
    max_qOverPt_bin_ = (unsigned int)max_bin;
  }

  //=== Digitize stub for input to Geographic Processor, with digitized phi coord. measured relative to closest phi sector.
  //=== (This approximation is valid if their are an integer number of digitisation bins inside each phi nonant).

  void Stub::digitize(unsigned int iPhiSec, Stub::DigiStage digiStep) {
    if (settings_->enableDigitize()) {
      bool updated = true;
      if (not digitalStub_) {
        // Digitize stub if not yet done.
        digitalStub_ =
            std::make_unique<DigitalStub>(settings_, phi_, r_, z_, min_qOverPt_bin_, max_qOverPt_bin_, bend_, iPhiSec);
      } else {
        // If digitization already done, redo phi digi if phi sector has changed.
        updated = digitalStub_->changePhiSec(iPhiSec);
      }

      // Save CPU by only updating if something has changed.
      if (updated || digiStep != lastDigiStep_) {
        lastDigiStep_ = digiStep;

        // Replace stub coords with those degraded by digitization process.
        if (digiStep == DigiStage::GP) {
          phi_ = digitalStub_->phi_GP();
        } else {
          phi_ = digitalStub_->phi_HT_TF();
        }
        if (digiStep == DigiStage::GP || digiStep == DigiStage::HT) {
          r_ = digitalStub_->r_GP_HT();
        } else {
          r_ = digitalStub_->r_SF_TF();
        }
        z_ = digitalStub_->z();
        bend_ = digitalStub_->bend();

        // Update data members that depend on updated coords.
        // (Logically part of digitisation, so disable warnings)
        digitizeWarningsOn_ = false;
        if (digiStep == DigiStage::GP)
          this->calcDphiOverBend();
        if (digiStep == DigiStage::HT)
          this->calcQoverPtrange();
        digitizeWarningsOn_ = true;
      }
    }
  }

  //=== Degrade assumed stub bend resolution.
  //=== And return an integer indicating how many values of bend are merged into this single one.

  void Stub::degradeResolution(float bend, float& degradedBend, unsigned int& num) const {
    // If TMTT code is tightening official CMS FE stub window cuts, then calculate TMTT stub windows.
    float windowFE;
    if (settings_->killLowPtStubs()) {
      // Window size corresponding to Pt cut used for tracking.
      float invPtMax = 1. / (settings_->houghMinPt());
      windowFE = invPtMax / std::abs(this->qOverPtOverBend());
      // Increase half-indow size to allow for resolution in bend.
      windowFE += this->bendCutInFrontend();
    } else {
      windowFE = rejectedStubBend_;  // TMTT is not tightening windows.
    }

    degradeBend_->degrade(bend, psModule(), trackerModule_->detId(), windowFE, degradedBend, num);
  }

  //=== Set flag indicating if stub will be output by front-end readout electronics
  //=== (where we can reconfigure the stub window size and rapidity cut).

  void Stub::setFrontend(const StubKiller* stubKiller) {
    frontendPass_ = true;              // Did stub pass cuts applied in front-end chip
    stubFailedDegradeWindow_ = false;  // Did it only fail cuts corresponding to windows encoded in DegradeBend.h?
    // Don't use stubs at large eta, since it is impossible to form L1 tracks from them, so they only contribute to combinatorics.
    if (std::abs(this->eta()) > settings_->maxStubEta())
      frontendPass_ = false;
    // Don't use stubs whose Pt is significantly below the Pt cut used in the L1 tracking, allowing for uncertainty in q/Pt due to stub bend resolution.
    const float qOverPtCut = 1. / settings_->houghMinPt();
    if (settings_->killLowPtStubs()) {
      // Apply this cut in the front-end electronics.
      if (std::abs(this->bendInFrontend()) - this->bendCutInFrontend() > qOverPtCut / this->qOverPtOverBend())
        frontendPass_ = false;
    }

    if (frontendPass_ && this->bend() == rejectedStubBend_) {
      throw cms::Exception(
          "BadConfig: FE stub bend window sizes provided in cfg ES source are tighter than those to make the stubs. "
          "Please fix them");
    }

    if (settings_->killLowPtStubs()) {
      // Reapply the same cut using the degraded bend information available in the off-detector electronics.
      // The reason is  that the bend degredation can move the Pt below the Pt cut, making the stub useless to the off-detector electronics.
      if (std::abs(this->bend()) - this->bendCut() > qOverPtCut / this->qOverPtOverBend())
        frontendPass_ = false;
    }

    // Emulate stubs in dead tracker regions..
    StubKiller::KillOptions killScenario = static_cast<StubKiller::KillOptions>(settings_->killScenario());
    if (killScenario != StubKiller::KillOptions::none) {
      bool kill = stubKiller->killStub(ttStubRef_.get());
      if (kill)
        frontendPass_ = false;
    }
  }

  //=== Function to calculate approximation for dphiOverBendCorrection aka B
  double Stub::approxB() {
    if (tiltedBarrel()) {
      return settings_->bApprox_gradient() * std::abs(z_) / r_ + settings_->bApprox_intercept();
    } else {
      return barrel() ? 1 : std::abs(z_) / r_;
    }
  }

  //=== Calculate variables giving ratio of track intercept angle to stub bend.

  void Stub::calcDphiOverBend() {
    // Uses stub (r,z) instead of module (r,z). Logically correct but has negligable effect on results.
    if (settings_->useApproxB()) {
      float dphiOverBendCorrection_approx_ = approxB();
      dphiOverBend_ = trackerModule_->pitchOverSep() * dphiOverBendCorrection_approx_;
    } else {
      float dphiOverBendCorrection_ = std::abs(cos(this->theta() - trackerModule_->tiltAngle()) / sin(this->theta()));
      dphiOverBend_ = trackerModule_->pitchOverSep() * dphiOverBendCorrection_;
    }
  }

  //=== Note which tracking particle(s), if any, produced this stub.
  //=== The 1st argument is a map relating TrackingParticles to TP.

  void Stub::fillTruth(const map<edm::Ptr<TrackingParticle>, const TP*>& translateTP,
                       const edm::Handle<TTStubAssMap>& mcTruthTTStubHandle,
                       const edm::Handle<TTClusterAssMap>& mcTruthTTClusterHandle) {
    //--- Fill assocTP_ info. If both clusters in this stub were produced by the same single tracking particle, find out which one it was.

    bool genuine = mcTruthTTStubHandle->isGenuine(ttStubRef_);  // Same TP contributed to both clusters?
    assocTP_ = nullptr;

    // Require same TP contributed to both clusters.
    if (genuine) {
      edm::Ptr<TrackingParticle> tpPtr = mcTruthTTStubHandle->findTrackingParticlePtr(ttStubRef_);
      if (translateTP.find(tpPtr) != translateTP.end()) {
        assocTP_ = translateTP.at(tpPtr);
        // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
      }
    }

    // Fill assocTPs_ info.

    if (settings_->stubMatchStrict()) {
      // We consider only stubs in which this TP contributed to both clusters.
      if (assocTP_ != nullptr)
        assocTPs_.insert(assocTP_);

    } else {
      // We consider stubs in which this TP contributed to either cluster.

      for (unsigned int iClus = 0; iClus <= 1; iClus++) {  // Loop over both clusters that make up stub.
        const TTClusterRef& ttClusterRef = ttStubRef_->clusterRef(iClus);

        // Now identify all TP's contributing to either cluster in stub.
        vector<edm::Ptr<TrackingParticle> > vecTpPtr = mcTruthTTClusterHandle->findTrackingParticlePtrs(ttClusterRef);

        for (const edm::Ptr<TrackingParticle>& tpPtr : vecTpPtr) {
          if (translateTP.find(tpPtr) != translateTP.end()) {
            assocTPs_.insert(translateTP.at(tpPtr));
            // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
          }
        }
      }
    }

    //--- Also note which tracking particles produced the two clusters that make up the stub

    for (unsigned int iClus = 0; iClus <= 1; iClus++) {  // Loop over both clusters that make up stub.
      const TTClusterRef& ttClusterRef = ttStubRef_->clusterRef(iClus);

      bool genuineCluster = mcTruthTTClusterHandle->isGenuine(ttClusterRef);  // Only 1 TP made cluster?
      assocTPofCluster_[iClus] = nullptr;

      // Only consider clusters produced by just one TP.
      if (genuineCluster) {
        edm::Ptr<TrackingParticle> tpPtr = mcTruthTTClusterHandle->findTrackingParticlePtr(ttClusterRef);

        if (translateTP.find(tpPtr) != translateTP.end()) {
          assocTPofCluster_[iClus] = translateTP.at(tpPtr);
          // N.B. Since not all tracking particles are stored in InputData::vTPs_, sometimes no match will be found.
        }
      }
    }
  }
}  // namespace tmtt
