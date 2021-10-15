//
//  Created by J.Li on 1/23/21.
//

#include "L1Trigger/TrackTrigger/interface/HitPatternHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>

namespace hph {

  SensorModule::SensorModule(
      bool isBarrel, bool isPS, int numColumns, int layerId, double r, double z, double pitchCol, double tilt)
      : isBarrel_(isBarrel),
        isPS_(isPS),
        isMaybe_(false),
        numColumns_(numColumns),
        layerId_(layerId),
        r_(r),
        z_(z),
        pitchCol_(pitchCol),
        tilt_(tilt) {
    sin_ = std::sin(tilt_);
    cos_ = std::cos(tilt_);
  }

  Setup::Setup(const edm::ParameterSet& iConfig,
               const TrackerGeometry& trackerGeometry,
               const TrackerTopology& trackerTopology)
      : trackerGeometry_(&trackerGeometry), trackerTopology_(&trackerTopology) {
    iConfig_ = iConfig;

    for (const auto& gd : trackerGeometry_->dets()) {
      DetId detid = (*gd).geographicalId();
      if (detid.subdetId() != StripSubdetector::TOB && detid.subdetId() != StripSubdetector::TID)
        continue;  // only run on OT
      if (!trackerTopology_->isLower(detid))
        continue;  // loop on the stacks: choose the lower arbitrarily

      // Get the DetSets of the Clusters
      const GeomDetUnit* det0 = trackerGeometry_->idToDetUnit(detid);
      const auto* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(det0);
      const PixelTopology* topol = dynamic_cast<const PixelTopology*>(&(theGeomDet->specificTopology()));
      const GlobalPoint pos0 = det0->position();
      const GlobalPoint pos1 = trackerGeometry_->idToDetUnit(trackerTopology_->partnerDetId(detid))->position();
      double r = pos0.perp();
      double z = pos0.z();

      bool flipped = pos0.mag() > pos1.mag();
      bool isBarrel = detid.subdetId() == StripSubdetector::TOB;
      bool isPS = trackerGeometry_->getDetectorType(detid) == TrackerGeometry::ModuleType::Ph2PSP;
      double tilt = flipped ? atan2(pos1.z() - pos0.z(), pos0.perp() - pos1.perp())
                            : atan2(pos0.z() - pos1.z(), pos1.perp() - pos0.perp());

      int layerId = isBarrel ? trackerTopology_->layer(detid) : trackerTopology_->layer(detid) + 10;
      int numColumns = topol->ncolumns();
      double pitchCol = topol->pitch().second;

      sensorModules_.emplace_back(isBarrel, isPS, numColumns, layerId, r, z, pitchCol, tilt);
    }

    sort(sensorModules_.begin(), sensorModules_.end(), smallerR);
    sort(sensorModules_.begin(), sensorModules_.end(), smallerZ);
    sensorModules_.erase(unique(sensorModules_.begin(), sensorModules_.end(), equalRZ), sensorModules_.end());
  }

  HitPatternHelper::HitPatternHelper(const Setup* setup, int hitpattern, double cot, double z0)
      : hitpattern_(hitpattern),
        numExpLayer_(0),
        numMissingLayer_(0),
        numMissingPS_(0),
        numMissing2S_(0),
        numPS_(0),
        num2S_(0),
        numMissingInterior1_(0),
        numMissingInterior2_(0),
        cot_(cot),
        z0_(z0),
        setup_(setup),
        layers_(),
        binary_(11, 0),
        hphDebug_(setup_->hphDebug()),
        useNewKF_(setup_->useNewKF()),
        chosenRofZ_(setup_->chosenRofZ()),
        deltaTanL_(setup_->deltaTanL()) {
    //Calculating eta sector based on cot and z0
    float kfzRef = z0_ + chosenRofZ_ * cot_;
    int kf_eta_reg = 0;
    for (int iEtaSec = 1; iEtaSec < ((int)etaRegions_.size() - 1); iEtaSec++) {  // Doesn't apply eta < 2.4 cut.
      float etaMax = etaRegions_[iEtaSec];
      float zRefMax = chosenRofZ_ / tan(2. * atan(exp(-etaMax)));
      if (kfzRef > zRefMax) {
        kf_eta_reg = iEtaSec;
      }
    }
    etaSector_ = kf_eta_reg;
    if (kf_eta_reg < ((int)etaRegions_.size() - 1) / 2) {
      kf_eta_reg = ((int)etaRegions_.size() - 1) / 2 - 1 - kf_eta_reg;
    } else {
      kf_eta_reg = kf_eta_reg - (int)(etaRegions_.size() - 1) / 2;
    }
    //Looping over sensor modules to make predictions on which layers particles are expected to hit
    for (SensorModule sm : setup_->sensorModules()) {
      double d = (z0_ - sm.z() + sm.r() * cot_) / (sm.cos() - sm.sin() * cot_);
      double d_p = (z0_ - sm.z() + sm.r() * (cot_ + deltaTanL_ / 2)) / (sm.cos() - sm.sin() * (cot_ + deltaTanL_ / 2));
      double d_m = (z0_ - sm.z() + sm.r() * (cot_ - deltaTanL_ / 2)) / (sm.cos() - sm.sin() * (cot_ - deltaTanL_ / 2));
      if (!(abs(d_p) < sm.numColumns() * sm.pitchCol() / 2. && abs(d_m) < sm.numColumns() * sm.pitchCol() / 2.))
        sm.setMaybe();
      if (useNewKF_ &&
          (abs(d_p) < sm.numColumns() * sm.pitchCol() / 2. || abs(d_m) < sm.numColumns() * sm.pitchCol() / 2.)) {
        layers_.push_back(sm);
      }
      if (!useNewKF_ && abs(d) < sm.numColumns() * sm.pitchCol() / 2.) {
        layers_.push_back(sm);
      }
    }
    //layers_ constains all the sensor modules that particles are expected to hit
    sort(layers_.begin(), layers_.end(), smallerID);
    layers_.erase(unique(layers_.begin(), layers_.end(), equalID), layers_.end());  //Keep only one sensor per layer

    numExpLayer_ = layers_.size();

    int nbits = floor(log2(hitpattern_)) + 1;
    int lay_i = 0;
    bool seq = false;
    for (int i = 0; i < nbits; i++) {
      lay_i = ((1 << i) & hitpattern_) >> i;  //0 or 1 in ith bit (right to left)

      if (lay_i && !seq)
        seq = true;  //sequence starts when first 1 found
      if (!lay_i && seq) {
        numMissingInterior1_++;  //This is the same as the "tmp_trk_nlaymiss_interior" calculated in Trackquality.cc
      }
      if (!lay_i) {
        bool realhit = false;
        for (int j : hitmap_[kf_eta_reg][i]) {
          if (j < 1)
            continue;
          int k = findLayer(j);
          if (k > 0)
            realhit = true;
        }
        if (realhit)
          numMissingInterior2_++;
      }
    }

    if (hphDebug_) {
      if (useNewKF_) {
        edm::LogVerbatim("TrackTriggerHPH") << "Running with New KF";
      } else {
        edm::LogVerbatim("TrackTriggerHPH") << "Running with Old KF";
      }
      edm::LogVerbatim("TrackTriggerHPH") << "======================================================";
      edm::LogVerbatim("TrackTriggerHPH")
          << "Looking at hitpattern " << std::bitset<7>(hitpattern_) << "; Looping over KF layers:";
    }

    if (useNewKF_) {
      //New KF uses sensor modules to determine the hitmask already
      for (int i = 0; i < numExpLayer_; i++) {
        if (hphDebug_) {
          edm::LogVerbatim("TrackTriggerHPH") << "--------------------------";
          edm::LogVerbatim("TrackTriggerHPH") << "Looking at KF layer " << i;
          if (layers_[i].layerId() < 10) {
            edm::LogVerbatim("TrackTriggerHPH") << "KF expects L" << layers_[i].layerId();
          } else {
            edm::LogVerbatim("TrackTriggerHPH") << "KF expects D" << layers_[i].layerId() - 10;
          }
        }

        if (((1 << i) & hitpattern_) >> i) {
          if (hphDebug_) {
            edm::LogVerbatim("TrackTriggerHPH") << "Layer found in hitpattern";
          }

          binary_[ReducedId(layers_[i].layerId())] = 1;
          if (layers_[i].isPS()) {
            numPS_++;
          } else {
            num2S_++;
          }
        } else {
          if (hphDebug_) {
            edm::LogVerbatim("TrackTriggerHPH") << "Layer missing in hitpattern";
          }

          if (layers_[i].isPS()) {
            numMissingPS_++;
          } else {
            numMissing2S_++;
          }
        }
      }

    } else {
      //Old KF uses the hard coded layermap to determien hitmask
      for (int i = 0; i < 7; i++) {  //Loop over each digit of hitpattern

        if (hphDebug_) {
          edm::LogVerbatim("TrackTriggerHPH") << "--------------------------";
          edm::LogVerbatim("TrackTriggerHPH") << "Looking at KF layer " << i;
        }

        for (int j :
             hitmap_[kf_eta_reg][i]) {  //Find out which layer the Old KF is dealing with when hitpattern is encoded
          if (j < 1) {
            if (hphDebug_) {
              edm::LogVerbatim("TrackTriggerHPH") << "KF does not expect this layer";
            }

            continue;
          }

          if (hphDebug_) {
            if (j < 10) {
              edm::LogVerbatim("TrackTriggerHPH") << "KF expects L" << j;
            } else {
              edm::LogVerbatim("TrackTriggerHPH") << "KF expects D" << j - 10;
            }
          }

          int k = findLayer(j);
          if (k < 0) {
            //k<0 means even though layer j is predicted by Old KF, this prediction is rejected because it contradicts
            if (hphDebug_) {  //a more accurate prediction made with the help of information from sensor modules.
              edm::LogVerbatim("TrackTriggerHPH") << "Rejected by sensor modules";
            }

            continue;
          }

          if (hphDebug_) {
            edm::LogVerbatim("TrackTriggerHPH") << "Confirmed by sensor modules";
          }
          //prediction is accepted
          if (((1 << i) & hitpattern_) >> i) {
            if (hphDebug_) {
              edm::LogVerbatim("TrackTriggerHPH") << "Layer found in hitpattern";
            }

            binary_[ReducedId(j)] = 1;
            if (layers_[k].isPS()) {
              numPS_++;
            } else {
              num2S_++;
            }
          } else {
            if (hphDebug_) {
              edm::LogVerbatim("TrackTriggerHPH") << "Layer missing in hitpattern";
            }

            if (layers_[k].isPS()) {
              numMissingPS_++;
            } else {
              numMissing2S_++;
            }
          }
        }
      }
    }

    if (hphDebug_) {
      edm::LogVerbatim("TrackTriggerHPH") << "------------------------------";
      edm::LogVerbatim("TrackTriggerHPH") << "numPS = " << numPS_ << ", num2S = " << num2S_
                                          << ", missingPS = " << numMissingPS_ << ", missing2S = " << numMissing2S_;
      edm::LogVerbatim("TrackTriggerHPH") << "======================================================";
    }
  }

  int HitPatternHelper::ReducedId(int layerId) {
    if (hphDebug_ && (layerId > 15 || layerId < 1)) {
      edm::LogVerbatim("TrackTriggerHPH") << "Warning: invalid layer id !";
    }
    if (layerId <= 6) {
      layerId = layerId - 1;
      return layerId;
    } else {
      layerId = layerId - 5;
      return layerId;
    }
  };

  int HitPatternHelper::findLayer(int layerId) {
    for (int i = 0; i < (int)layers_.size(); i++) {
      if (layerId == (int)layers_[i].layerId()) {
        return i;
      }
    }
    return -1;
  }

}  // namespace hph
