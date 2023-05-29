//
//  Created by J.Li on 1/23/21.
//

#include "L1Trigger/TrackTrigger/interface/HitPatternHelper.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFbase.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <cmath>

namespace hph {

  Setup::Setup(const edm::ParameterSet& iConfig, const tt::Setup& setupTT)
      : iConfig_(iConfig),
        setupTT_(setupTT),
        layerIds_(),
        layermap_(),
        nEtaRegions_(tmtt::KFbase::nEta_ / 2),
        nKalmanLayers_(tmtt::KFbase::nKFlayer_) {
    for (const tt::SensorModule& sm : setupTT_.sensorModules()) {
      layerIds_.push_back(std::make_pair(sm.layerId(), sm.barrel()));
    }
    sort(layerIds_.begin(), layerIds_.end(), smallerID);
    layerIds_.erase(unique(layerIds_.begin(), layerIds_.end(), equalID), layerIds_.end());  //Keep only unique layerIds
    // Converting tmtt::KFbase::layerMap_ to a format that is acceptatble by HitPatternHelper
    for (int i = 0; i < nEtaRegions_; i++) {
      for (int j = 0; j < (int)layerIds_.size(); j++) {
        int layer;
        if (layerIds_[j].second) {
          layer = tmtt::KFbase::layerMap_[i][tmtt::TrackerModule::calcLayerIdReduced(layerIds_[j].first)].first;
        } else {
          layer = tmtt::KFbase::layerMap_[i][tmtt::TrackerModule::calcLayerIdReduced(layerIds_[j].first)].second;
        }
        if (layer < nKalmanLayers_) {
          layermap_[i][layer].push_back(layerIds_[j].first);
        }
      }
    }
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
        binary_(11, 0),
        hphDebug_(setup_->hphDebug()),
        useNewKF_(setup_->useNewKF()),
        chosenRofZ_(setup_->chosenRofZ()),
        deltaTanL_(setup_->deltaTanL()),
        etaRegions_(setup_->etaRegions()),
        nKalmanLayers_(setup_->nKalmanLayers()),
        layermap_(setup_->layermap()) {
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
    for (const tt::SensorModule& sm : setup_->sensorModules()) {
      double d = (z0_ - sm.z() + sm.r() * cot_) / (sm.cosTilt() - sm.sinTilt() * cot_);
      double d_p =
          (z0_ - sm.z() + sm.r() * (cot_ + deltaTanL_ / 2)) / (sm.cosTilt() - sm.sinTilt() * (cot_ + deltaTanL_ / 2));
      double d_m =
          (z0_ - sm.z() + sm.r() * (cot_ - deltaTanL_ / 2)) / (sm.cosTilt() - sm.sinTilt() * (cot_ - deltaTanL_ / 2));
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
        if (layermap_[kf_eta_reg][i].empty())
          continue;
        for (int j : layermap_[kf_eta_reg][i]) {
          int k = findLayer(j);
          if (k > 0)
            realhit = true;
        }
        if (realhit)
          numMissingInterior2_++;  //This variable doesn't make sense for new KF because it uses the layermap from Old KF
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
          if (layers_[i].psModule()) {
            numPS_++;
          } else {
            num2S_++;
          }
        } else {
          if (hphDebug_) {
            edm::LogVerbatim("TrackTriggerHPH") << "Layer missing in hitpattern";
          }

          if (layers_[i].psModule()) {
            numMissingPS_++;
          } else {
            numMissing2S_++;
          }
        }
      }

    } else {
      //Old KF uses the hard coded layermap to determien hitmask
      for (int i = 0; i < nKalmanLayers_; i++) {  //Loop over each digit of hitpattern

        if (hphDebug_) {
          edm::LogVerbatim("TrackTriggerHPH") << "--------------------------";
          edm::LogVerbatim("TrackTriggerHPH") << "Looking at KF layer " << i;
        }

        if (layermap_[kf_eta_reg][i].empty()) {
          if (hphDebug_) {
            edm::LogVerbatim("TrackTriggerHPH") << "KF does not expect this layer";
          }

          continue;
        }

        for (int j :
             layermap_[kf_eta_reg][i]) {  //Find out which layer the Old KF is dealing with when hitpattern is encoded

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
            if (hphDebug_) {  //a more accurate prediction made with the help of information from sensor modules
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
            if (layers_[k].psModule()) {
              numPS_++;
            } else {
              num2S_++;
            }
          } else {
            if (hphDebug_) {
              edm::LogVerbatim("TrackTriggerHPH") << "Layer missing in hitpattern";
            }

            if (layers_[k].psModule()) {
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
