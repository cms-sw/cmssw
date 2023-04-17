//
//  Created by J.Li on 1/23/21.
//

#include "L1Trigger/TrackFindingTracklet/interface/HitPatternHelper.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFbase.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <algorithm>
#include <cmath>

namespace hph {

  Setup::Setup(const edm::ParameterSet& iConfig,
               const tt::Setup& setupTT,
               const trackerTFP::DataFormats& dataFormats,
               const trackerTFP::LayerEncoding& layerEncoding)
      : iConfig_(iConfig),
        oldKFPSet_(iConfig.getParameter<edm::ParameterSet>("oldKFPSet")),
        setupTT_(setupTT),
        dataFormats_(dataFormats),
        dfcot_(dataFormats_.format(trackerTFP::Variable::cot, trackerTFP::Process::kfin)),
        dfzT_(dataFormats_.format(trackerTFP::Variable::zT, trackerTFP::Process::kfin)),
        layerEncoding_(layerEncoding),
        hphDebug_(iConfig.getParameter<bool>("hphDebug")),
        useNewKF_(iConfig.getParameter<bool>("useNewKF")),
        chosenRofZNewKF_(setupTT_.chosenRofZ()),
        etaRegionsNewKF_(setupTT_.boundarieEta()),
        layermap_(),
        nEtaRegions_(tmtt::KFbase::nEta_ / 2),
        nKalmanLayers_(tmtt::KFbase::nKFlayer_) {
    if (useNewKF_) {
      chosenRofZ_ = chosenRofZNewKF_;
      etaRegions_ = etaRegionsNewKF_;
    } else {
      chosenRofZ_ = oldKFPSet_.getParameter<double>("ChosenRofZ");
      etaRegions_ = oldKFPSet_.getParameter<std::vector<double>>("EtaRegions");
    }
    static constexpr auto layerIds = {1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15};  //layer ID 11~15 correspond to D1~D5
    // Converting tmtt::KFbase::layerMap_ to a format that is acceptatble by HitPatternHelper
    for (int i = 0; i < nEtaRegions_; i++) {
      for (int j : layerIds) {
        int layer;
        if (j < 7) {
          layer = tmtt::KFbase::layerMap_[i][tmtt::TrackerModule::calcLayerIdReduced(j)].first;
        } else {
          layer = tmtt::KFbase::layerMap_[i][tmtt::TrackerModule::calcLayerIdReduced(j)].second;
        }
        if (layer < nKalmanLayers_) {
          layermap_[i][layer].push_back(j);
        }
      }
    }
  }

  HitPatternHelper::HitPatternHelper(const Setup* setup, int hitpattern, double cot, double z0)
      : setup_(setup),
        hphDebug_(setup_->hphDebug()),
        useNewKF_(setup_->useNewKF()),
        etaRegions_(setup_->etaRegions()),
        layermap_(setup_->layermap()),
        nKalmanLayers_(setup_->nKalmanLayers()),
        etaBin_(setup_->etaRegion(z0, cot, true)),
        cotBin_(setup_->digiCot(cot, etaBin_)),
        zTBin_(setup_->digiZT(z0, cot, etaBin_)),
        layerEncoding_(setup->layerEncoding(etaBin_, zTBin_, cotBin_)),
        layerEncodingMap_(setup->layerEncodingMap(etaBin_, zTBin_, cotBin_)),
        numExpLayer_(layerEncoding_.size()),
        hitpattern_(hitpattern),
        etaSector_(setup_->etaRegion(z0, cot, useNewKF_)),
        numMissingLayer_(0),
        numMissingPS_(0),
        numMissing2S_(0),
        numPS_(0),
        num2S_(0),
        numMissingInterior1_(0),
        numMissingInterior2_(0),
        binary_(11, 0),  //there are 11 unique layer IDs, as defined in variable "layerIds"
        bonusFeatures_() {
    int kf_eta_reg = etaSector_;
    if (kf_eta_reg < ((int)etaRegions_.size() - 1) / 2) {
      kf_eta_reg = ((int)etaRegions_.size() - 1) / 2 - 1 - kf_eta_reg;
    } else {
      kf_eta_reg = kf_eta_reg - (int)(etaRegions_.size() - 1) / 2;
    }

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
          if (layerEncoding_[i] < 10) {
            edm::LogVerbatim("TrackTriggerHPH") << "KF expects L" << layerEncoding_[i];
          } else {
            edm::LogVerbatim("TrackTriggerHPH") << "KF expects D" << layerEncoding_[i] - 10;
          }
        }

        if (((1 << i) & hitpattern_) >> i) {
          if (hphDebug_) {
            edm::LogVerbatim("TrackTriggerHPH") << "Layer found in hitpattern";
          }

          binary_[reducedId(layerEncoding_[i])] = 1;
          if (layerEncodingMap_[layerEncoding_[i]]->psModule()) {
            numPS_++;
          } else {
            num2S_++;
          }
        } else {
          if (hphDebug_) {
            edm::LogVerbatim("TrackTriggerHPH") << "Layer missing in hitpattern";
          }

          if (layerEncodingMap_[layerEncoding_[i]]->psModule()) {
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

            binary_[reducedId(j)] = 1;
            if (layerEncodingMap_[layerEncoding_[k]]->psModule()) {
              numPS_++;
            } else {
              num2S_++;
            }
          } else {
            if (hphDebug_) {
              edm::LogVerbatim("TrackTriggerHPH") << "Layer missing in hitpattern";
            }

            if (layerEncodingMap_[layerEncoding_[k]]->psModule()) {
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

  int Setup::etaRegion(double z0, double cot, bool useNewKF) const {
    //Calculating eta sector based on cot and z0
    double chosenRofZ;
    std::vector<double> etaRegions;
    if (useNewKF) {
      chosenRofZ = chosenRofZNewKF_;
      etaRegions = etaRegionsNewKF_;
    } else {
      chosenRofZ = chosenRofZ_;
      etaRegions = etaRegions_;
    }
    double kfzRef = z0 + chosenRofZ * cot;
    int kf_eta_reg = 0;
    for (int iEtaSec = 1; iEtaSec < ((int)etaRegions.size() - 1); iEtaSec++) {  // Doesn't apply eta < 2.4 cut.
      double etaMax = etaRegions[iEtaSec];
      double zRefMax = chosenRofZ / tan(2. * atan(exp(-etaMax)));
      if (kfzRef < zRefMax) break;
      kf_eta_reg = iEtaSec;
    }
    return kf_eta_reg;
  }

  int Setup::digiCot(double cot, int binEta) const {
    double cotLocal = dfcot_.digi(cot - setupTT_.sectorCot(binEta));
    return dfcot_.toUnsigned(dfcot_.integer(cotLocal));
  }

  int Setup::digiZT(double z0, double cot, int binEta) const {
    double zT = z0 + setupTT_.chosenRofZ() * cot;
    double zTLocal = dfzT_.digi(zT - setupTT_.sectorCot(binEta) * setupTT_.chosenRofZ());
    return dfzT_.toUnsigned(dfzT_.integer(zTLocal));
  }

  int HitPatternHelper::reducedId(int layerId) {
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
  }

  int HitPatternHelper::findLayer(int layerId) {
    for (int i = 0; i < (int)layerEncoding_.size(); i++) {
      if (layerId == (int)layerEncoding_[i]) {
        return i;
      }
    }
    return -1;
  }

}  // namespace hph
