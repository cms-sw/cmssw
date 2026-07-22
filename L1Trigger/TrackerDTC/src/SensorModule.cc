#include "L1Trigger/TrackerDTC/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <set>
#include <cmath>
#include <algorithm>
#include <iterator>

namespace trackerDTC {

  SensorModule::SensorModule(const Setup* setup,
                             const DetId& detId,
                             int dtcId,
                             int modId,
                             bool barrel,
                             bool psModule,
                             int index,
                             int ws,
                             const std::vector<double>& encodingBend,
                             const std::vector<double>& degradedBend,
                             const std::set<int>& encodingLayer)
      : detId_(detId),
        dtcId_(dtcId),
        modId_(modId),
        barrel_(barrel),
        psModule_(psModule),
        layerIndex_(index - 1),
        windowSize_(ws),
        encodingBend_(encodingBend),
        degradedBend_(degradedBend) {
    const TrackerGeometry* trackerGeometry = setup->trackerGeometry();
    const TrackerTopology* trackerTopology = setup->trackerTopology();
    const GeomDetUnit* geomDetUnit = trackerGeometry->idToDetUnit(detId);
    const PixelTopology* pixelTopology =
        dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit)->specificTopology()));
    const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(geomDetUnit)->surface();
    const GlobalPoint pos0 = GlobalPoint(geomDetUnit->position());
    const GlobalPoint pos1 =
        GlobalPoint(trackerGeometry->idToDetUnit(trackerTopology->partnerDetId(detId))->position());
    // detector region [0-8]
    const int region = dtcId_ / setup->regNumDTC();
    // module radius in cm
    r_ = pos0.perp();
    // module phi w.r.t. detector region_ centre in rad
    phi_ = tt::deltaPhi(pos0.phi() - (region + .5) * setup->regRangePhiT());
    // module z in cm
    z_ = pos0.z();
    // sensor separation in cm
    sep_ = (pos1 - pos0).mag();
    // number of columns [2S=2*1,PS=2*16]
    numColumns_ = pixelTopology->ncolumns();
    // number of rows [2S=8*127,PS=8*120]
    numRows_ = pixelTopology->nrows();
    // +z or -z
    side_ = pos0.z() >= 0.;
    // main sensor inside or outside
    flipped_ = pos0.mag() > pos1.mag();
    // sensor pitch in cm [strip=.009,pixel=.01]
    pitchRow_ = psModule_ ? setup->mpaPitch() : setup->cbcPitch();
    // sensor length in cm [strip=5,pixel=.15625]
    pitchCol_ = psModule_ ? setup->mpaLength() : setup->cbcLength();
    // module tilt angle measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    tilt_ = flipped_ ? std::atan2(pos1.z() - pos0.z(), pos0.perp() - pos1.perp())
                     : std::atan2(pos0.z() - pos1.z(), pos1.perp() - pos0.perp());
    // sinus of module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    sinTilt_ = std::sin(tilt_);
    // cosinus of module tilt measured w.r.t. beam axis (+-1=endcap), tk layout measures w.r.t. radial axis
    cosTilt_ = std::cos(tilt_);
    // layer index combined [barrel: 0-5, endcap: 6-10]
    layerIndexCombined_ = layerIndex_ + (barrel_ ? 0 : 6);
    // layer id [1-6,11-15]
    layerId_ = layerIndex_ + 1 + (barrel_ ? 0 : 10);
    // reduced layer id [0-6] { 1, 2, 11 & 6, 12 & 5, 13 & 4, 14, 15 & 3 }
    layerIdReduced_ = layerId_;
    if (layerIdReduced_ == 6)
      layerIdReduced_ = 11;
    else if (layerIdReduced_ == 5)
      layerIdReduced_ = 12;
    else if (layerIdReduced_ == 4)
      layerIdReduced_ = 13;
    else if (layerIdReduced_ == 3)
      layerIdReduced_ = 15;
    if (layerIdReduced_ > 10)
      layerIdReduced_ -= 8;
    layerIdReduced_--;
    // TTStub row needs flip of sign
    signRow_ = std::signbit(tt::deltaPhi(plane.rotation().x().phi() - pos0.phi()));
    // TTStub col needs flip of sign
    signCol_ = !barrel_ && !side_;
    // TTStub bend needs flip of sign
    signBend_ = signCol_;
    // determing sensor type
    if (barrel_ && psModule_)
      type_ = BarrelPS;
    if (barrel_ && !psModule_)
      type_ = Barrel2S;
    if (!barrel_ && psModule_)
      type_ = DiskPS;
    if (!barrel_ && !psModule_)
      type_ = Disk2S;
    // r and z offsets
    if (barrel_) {
      offsetR_ = setup->stubLayerR(layerIndex_) - setup->regChosenRofPhi();
      offsetZ_ = 0.;
    } else {
      offsetR_ = -setup->regChosenRofPhi();
      offsetZ_ = side_ ? setup->stubDiskZ(layerIndex_) : -setup->stubDiskZ(layerIndex_);
      // Adding offset of 7.5 cm (256 * granularity) to allow adding negDisk bit required for dual FPGA project
      if (psModule_)
        offsetR_ = setup->stubOffsetRDiskPS() - setup->regChosenRofPhi();
      // encoding for 2S endcap radii
      else {
        const int offset = setup->stubNumRingsPS(layerIndex_);
        const int ring = trackerTopology->endcapRingP2(detId);
        offsetR_ = numColumns_ * (ring - offset);
      }
    }
    const Phase2Tracker::BarrelModuleTilt typeTilt = trackerTopology->barrelTiltTypeP2(detId);
    // encode layer id
    encodedLayer_ = std::distance(encodingLayer.begin(), encodingLayer.find(layerId_));
    // calculate tilt correction parameter used to project r to z uncertainty
    tilted_ = typeTilt == Phase2Tracker::BarrelModuleTilt::tiltedZminus ||
              typeTilt == Phase2Tracker::BarrelModuleTilt::tiltedZplus;
    tiltCorrectionSlope_ = setup->smTiltApproxSlope();
    tiltCorrectionIntercept_ = setup->smTiltApproxIntercept();
    scattering_ = setup->smScattering();
    clusterWidth_ = setup->smClusterWidth(this->module());
    addPhiUncertainty_ = setup->smAddPhiUncertainty(this->module());
    // stub uncertainty
    dR_ = 0.;
    if (!barrel_)
      dR_ = pitchCol_;
    else if (tilted_)
      dR_ = setup->smTiltUncertaintyR();
  }

  // stub z uncertainty in cm for given track cot
  double SensorModule::dZ(double cot) const {
    if (barrel_) {
      return pitchCol_ + std::abs(cot) * dR_;
      if (tilted_)
        return pitchCol_ * (tiltCorrectionSlope_ * std::abs(cot) + tiltCorrectionIntercept_);
    } else
      return std::abs(cot) * dR_;
  }

  // stub phi uncertainty in rad for given stub radius and track inv2R
  double SensorModule::dPhi(double r, double inv2R) const {
    return ((dR_ + scattering_) * std::abs(inv2R)) + clusterWidth_ * pitchRow_ / r + addPhiUncertainty_;
  }

  // encode bend
  int SensorModule::decodeBend(double bendBE) const {
    const auto pos = std::find(encodingBend_.begin(), encodingBend_.end(), std::abs(bendBE));
    const int uBend = std::distance(encodingBend_.begin(), pos);
    return bendBE < 0 && uBend != 0 ? -uBend : uBend;
  }

  // bend dergadadtion
  double SensorModule::degradeBend(int bendFE) const {
    const double uBend = degradedBend_[std::abs(bendFE)];
    return bendFE < 0 && uBend != 0 ? -uBend : uBend;
  }

  // bend encoding
  double SensorModule::encodeBend(int decodedBend) const {
    const double uBend = encodingBend_[std::abs(decodedBend)];
    return decodedBend < 0 ? -uBend : uBend;
  }

  // module type
  int SensorModule::module() const {
    if (barrel_)
      if (psModule_)
        if (tilted_)
          return 2;
        else
          return 1;
      else
        return 0;
    else if (psModule_)
      return 4;
    else
      return 3;
  }

}  // namespace trackerDTC
