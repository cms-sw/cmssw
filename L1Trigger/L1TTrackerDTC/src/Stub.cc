#include "L1Trigger/L1TTrackerDTC/interface/Stub.h"
#include "L1Trigger/L1TTrackerDTC/interface/Settings.h"
#include "L1Trigger/L1TTrackerDTC/interface/Module.h"

#include <cmath>
#include <iterator>
#include <algorithm>


using namespace std;

namespace L1TTrackerDTC {

  Stub::Stub(Settings* settings, const TTStubRef& ttStubRef, Module* module)
      : settings_(settings), ttStubRef_(ttStubRef), module_(module), valid_(true) {
    regions_.reserve(settings_->numOverlappingRegions());

    const int& numColumns = module->numColumns_;  // number of columns [2S=2,PS=8]
    const int& numRows = module->numRows_;        // number of rows [2S=8*127,PS=8*120]
    const bool& signRow = module->signRow_;       // TTStub bend needs flip of sign
    const bool& signCol = module->signCol_;       // TTStub col needs flip of sign
    const bool& signBend = module->signBend_;     // TTStub bend needs flip of sign
    const double& pitchCol = module->pitchCol_;   // sensor length in cm [strip=5,pixel=.15625]
    const double& pitchRow = module->pitchRow_;   // sensor pitch in cm [strip=.009,pixel=.01]
    const double& sep = module->sep_;             // sensor separation in cm
    const double& R = module->R_;                 // module radius in cm
    const double& Phi = module->Phi_;             // module phi w.r.t. detector region centre in rad
    const double& Z = module->Z_;                 // module z in cm
    // sinus of module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    const double& sin = module ->sin_;
    // cosinus of module tilt measured w.r.t. beam axis (+-1=endcap), tk layout measures w.r.t. radial axis
    const double& cos = module->cos_;

    // get stub local coordinates
    const MeasurementPoint& mp = ttStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();

    // convert to uniformed local coordinates
    col_ = (int)floor(pow(-1, signCol) * (mp.y() - numColumns / 2) / settings_->baseCol());  // column number
    row_ = (int)floor(pow(-1, signRow) * (mp.x() - numRows / 2) / settings_->baseRow());     // row number
    bend_ = (int)floor(pow(-1, signBend) * (ttStubRef->bendBE()) / settings_->baseBend());   // bend number
    //reduced row number for look up
    rowLUT_ = (int)floor((double)row_ / pow(2., settings_->widthRow() - settings_->widthRowLUT()));
    // sub row number inside reduced row number
    rowSub_ = row_ - (rowLUT_ + .5) * pow(2, settings_->widthRow() - settings_->widthRowLUT());

    // convert local to global coordinates
    const double y = (col_ + .5) * settings_->baseCol() * pitchCol;
    d_ = R + y * sin;  // CIC r in cm
    z_ = Z + y * cos;  // stub z in cm

    const double x0 = rowLUT_ * settings_->baseRow() * settings_->numMergedRows() * pitchRow;
    const double x1 = (rowLUT_ + 1) * settings_->baseRow() * settings_->numMergedRows() * pitchRow;
    const double x = (rowLUT_ + .5) * settings_->baseRow() * settings_->numMergedRows() * pitchRow;
    r_ = sqrt(d_ * d_ + x * x);  // stub r in cm

    const double phi0 = Phi + atan2(x0, d_);
    const double phi1 = Phi + atan2(x1, d_);
    const double c = (phi0 + phi1) / 2.;
    const double m = (phi1 - phi0) / settings_->numMergedRows();

    c_ = digi(c, settings_->baseC());  // intercept of linearized stub phi in rad
    m_ = digi(m, settings_->baseM());  // slope of linearized stub phi in rad / strip

    if (settings_->dataFormat() == "TMTT") {
      const double zT = settings_->tmtt()->chosenRofZ() * z_ / r_;  // extrapolated z at radius T assuming z0=0
      // extrapolated z0 window at radius T
      const double dZT = settings_->tmtt()->beamWindowZ() * fabs(1. - settings_->tmtt()->chosenRofZ() / r_);

      double zTMin = zT - dZT;
      double zTMax = zT + dZT;
      if (zTMin >= settings_->tmtt()->maxZT() || zTMax < -settings_->tmtt()->maxZT())
        valid_ = false;  // did not pass "eta" cut
      else {
        zTMin = max(zTMin, -settings_->tmtt()->maxZT());
        zTMax = min(zTMax, settings_->tmtt()->maxZT());
      }

      cot_ = {zTMin / settings_->tmtt()->chosenRofZ(),
              zTMax / settings_->tmtt()->chosenRofZ()};  // range of stub cot(theta)

    } else if (settings_->dataFormat() == "Hybrid") {
      if (fabs(z_ / r_) > settings_->maxCot())
        valid_ = false;  // did not pass eta cut
    }

    r_ = digi(r_ - settings_->chosenRofPhi(), settings_->baseR());  // stub r w.r.t. chosenRofPhi in cm

    const double dr = sep / (cos - sin * z_ / d_);      // radial (cylindrical) component of sensor separation
    const double qOverPtOverBend = pitchRow / dr / d_;  // converts bend into qOverPt in 1/cm

    const double qOverPt = bend_ * settings_->baseBend() * qOverPtOverBend;  // qOverPt in 1/cm
    const double dQoverPt = settings_->bendCut() * qOverPtOverBend;          // qOverPt uncertainty in 1/cm

    double qOverPtMin = digi(qOverPt - dQoverPt, settings_->baseQoverPt());
    double qOverPtMax = digi(qOverPt + dQoverPt, settings_->baseQoverPt());
    if (qOverPtMin >= settings_->maxQoverPt() || qOverPtMax < -settings_->maxQoverPt())
      valid_ = false;  // did not pass pt cut
    else {
      qOverPtMin = max(qOverPtMin, -settings_->maxQoverPt());
      qOverPtMax = min(qOverPtMax, settings_->maxQoverPt());
    }

    qOverPt_ = {qOverPtMin, qOverPtMax};  // range of stub qOverPt in 1/cm

    phi_ = c_ + rowSub_ * m_;  // stub phi w.r.t. detector region centre in rad

    phiT_.first = phi_ + r_ * qOverPt_.first;  // range of stub extrapolated phi to radius chosenRofPhi in rad
    phiT_.second = phi_ + r_ * qOverPt_.second;
    if (phiT_.first > phiT_.second)
      swap(phiT_.first, phiT_.second);

    if (phiT_.first < 0.)
      regions_.push_back(0);
    if (phiT_.second >= 0.)
      regions_.push_back(1);

    // apply data format specific manipulations
    if (settings_->dataFormat() != "Hybrid")
      return;

    r_ -= module->offsetR_;  // stub r w.r.t. an offset in cm
    z_ -= module->offsetZ_;  // stub z w.r.t. an offset in cm
    if (module->type_ == SettingsHybrid::disk2S)
      r_ = (module->decodedR_ + .5) * settings_->hybrid()->baseR(module->type_);  // decoded r

    // decode bend
    const vector<double>& bendEncoding = module->bendEncoding_;
    bend_ = pow(-1, signbit(bend_)) *
            (distance(bendEncoding.begin(), find(bendEncoding.begin(), bendEncoding.end(), fabs(bend_))) -
             (int)bendEncoding.size() / 2);
  }

  // returns bit accurate representation of Stub
  TTDTC::BV Stub::frame(const int& region) const {
    return settings_->dataFormat() == "Hybrid" ? formatHybrid(region) : formatTMTT(region);
  }

  // outer tracker dtc routing block id [0-1]
  int Stub::blockId() const { return module_->blockId_; }

  // outer tracker dtc routing block channel id [0-35]
  int Stub::channelId() const { return module_->channelId_; }

  // returns true if stub belongs to region
  bool Stub::inRegion(const int& region) const {
    return find(regions_.begin(), regions_.end(), region) != regions_.end();
  }

  // truncates double precision to f/w integer equivalent
  double Stub::digi(const double& value, const double& precision) const {
    return (floor(value / precision) + .5) * precision;
  }

  // returns 64 bit stub in hybrid data format
  TTDTC::BV Stub::formatHybrid(const int& region) const {
    SettingsHybrid* format = settings_->hybrid();
    SettingsHybrid::SensorType& type = module_->type_;
    const int& layer = module_->layerId_;

    // stub phi w.r.t. processing region centre in rad
    const double phi = phi_ - (region - .5) * settings_->baseRegion();

    // convert stub variables into bit vectors
    const TTBV hwR(r_, format->baseR(type), format->widthR(type), true);
    const TTBV hwPhi(phi, format->basePhi(type), format->widthPhi(type), true);
    const TTBV hwZ(z_, format->baseZ(type), format->widthZ(type), true);
    const TTBV hwAlpha(row_, format->baseAlpha(type), format->widthAlpha(type), true);
    const TTBV hwBend(bend_, format->widthBend(type), true);
    const TTBV hwLayer(layer, settings_->widthLayer());
    const TTBV hwGap(0, format->numUnusedBits(type));
    const TTBV hwValid(1, 1);

    // assemble final bitset
    return TTDTC::BV(hwGap.str() + hwR.str() + hwZ.str() + hwPhi.str() + hwAlpha.str() + hwBend.str() + hwLayer.str() +
                     hwValid.str());
  }

  TTDTC::BV Stub::formatTMTT(const int& region) const {
    SettingsTMTT* format = settings_->tmtt();
    const int& layerM = module_->layerId_;

    // convert unique layer id [1-6,11-15] into reduced layer id [0-6]
    // a fiducial track may not cross more then 7 detector layers, for stubs from a given track the reduced layer id is actually unique
    int layer(-1);
    if (layerM == 1)
      layer = 0;
    else if (layerM == 2)
      layer = 1;
    else if (layerM == 6 || layerM == 11)
      layer = 2;
    else if (layerM == 5 || layerM == 12)
      layer = 3;
    else if (layerM == 4 || layerM == 13)
      layer = 4;
    else if (layerM == 14)
      layer = 5;
    else if (layerM == 3 || layerM == 15)
      layer = 6;

    // assign stub to phi sectors within a processing region, to be generalized
    TTBV sectorsPhi(0, settings_->numOverlappingRegions() * format->numSectorsPhi());
    if (phiT_.first < 0.) {
      if (phiT_.first < -format->baseSector())
        sectorsPhi.set(0);
      else
        sectorsPhi.set(1);
      if (phiT_.second < 0. && phiT_.second >= -format->baseSector())
        sectorsPhi.set(1);
    }
    if (phiT_.second >= 0.) {
      if (phiT_.second < format->baseSector())
        sectorsPhi.set(2);
      else
        sectorsPhi.set(3);
      if (phiT_.first >= 0. && phiT_.first < format->baseSector())
        sectorsPhi.set(2);
    }

    // assign stub to eta sectors within a processing region
    pair<int, int> setcorEta({0, format->numSectorsEta() - 1});
    for (int bin = 0; bin < format->numSectorsEta(); bin++)
      if (asinh(cot_.first) < format->bounderiesEta(bin + 1)) {
        setcorEta.first = bin;
        break;
      }
    for (int bin = setcorEta.first; bin < format->numSectorsEta(); bin++)
      if (asinh(cot_.second) < format->bounderiesEta(bin + 1)) {
        setcorEta.second = bin;
        break;
      }

    // stub phi w.r.t. processing region centre in rad
    const double phi = phi_ - (region - .5) * settings_->baseRegion();

    // convert stub variables into bit vectors
    const TTBV hwValid(1, 1);
    const TTBV hwGap(0, format->numUnusedBits());
    const TTBV hwLayer(layer, settings_->widthLayer());
    const TTBV hwSectorEtaMin(setcorEta.first, settings_->widthEta());
    const TTBV hwSectorEtaMax(setcorEta.second, settings_->widthEta());
    const TTBV hwR(r_, settings_->baseR(), settings_->widthR(), true);
    const TTBV hwPhi(phi, settings_->basePhi(), settings_->widthPhi(), true);
    const TTBV hwZ(z_, settings_->baseZ(), settings_->widthZ(), true);
    const TTBV hwQoverPtMin(qOverPt_.first, format->baseQoverPt(), format->widthQoverPtBin(), true);
    const TTBV hwQoverPtMax(qOverPt_.second, format->baseQoverPt(), format->widthQoverPtBin(), true);

    TTBV hwSectorPhis(0, format->numSectorsPhi());
    for (int sectorPhi = 0; sectorPhi < format->numSectorsPhi(); sectorPhi++)
      hwSectorPhis[sectorPhi] = sectorsPhi[region * format->numSectorsPhi() + sectorPhi];

    // assemble final bitset
    return TTDTC::BV(hwGap.str() + hwValid.str() + hwR.str() + hwPhi.str() + hwZ.str() + hwQoverPtMin.str() +
                     hwQoverPtMax.str() + hwSectorEtaMin.str() + hwSectorEtaMax.str() + hwSectorPhis.str() +
                     hwLayer.str());
  }

}  // namespace L1TTrackerDTC