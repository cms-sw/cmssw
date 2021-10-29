#include "L1Trigger/TrackerDTC/interface/Stub.h"

#include <cmath>
#include <iterator>
#include <algorithm>
#include <utility>

using namespace edm;
using namespace std;
using namespace tt;

namespace trackerDTC {

  Stub::Stub(const ParameterSet& iConfig,
             const Setup* setup,
             const LayerEncoding* layerEncoding,
             SensorModule* sm,
             const TTStubRef& ttStubRef)
      : setup_(setup),
        layerEncoding_(layerEncoding),
        sm_(sm),
        ttStubRef_(ttStubRef),
        hybrid_(iConfig.getParameter<bool>("UseHybrid")),
        valid_(true) {
    regions_.reserve(setup->numOverlappingRegions());
    // get stub local coordinates
    const MeasurementPoint& mp = ttStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();

    // convert to uniformed local coordinates

    // column number in pitch units
    col_ = (int)floor(pow(-1, sm->signCol()) * (mp.y() - sm->numColumns() / 2) / setup->baseCol());
    // row number in half pitch units
    row_ = (int)floor(pow(-1, sm->signRow()) * (mp.x() - sm->numRows() / 2) / setup->baseRow());
    // bend number in quarter pitch units
    bend_ = (int)floor(pow(-1, sm->signBend()) * (ttStubRef->bendBE()) / setup->baseBend());
    // reduced row number for look up
    rowLUT_ = (int)floor((double)row_ / pow(2., setup->widthRow() - setup->dtcWidthRowLUT()));
    // sub row number inside reduced row number
    rowSub_ = row_ - (rowLUT_ + .5) * pow(2, setup->widthRow() - setup->dtcWidthRowLUT());

    // convert local to global coordinates

    const double y = (col_ + .5) * setup->baseCol() * sm->pitchCol();
    // radius of a column of strips/pixel in cm
    d_ = sm->r() + y * sm->sinTilt();
    // stub z in cm
    z_ = digi(sm->z() + y * sm->cosTilt(), setup->tmttBaseZ());

    const double x0 = rowLUT_ * setup->baseRow() * setup->dtcNumMergedRows() * sm->pitchRow();
    const double x1 = (rowLUT_ + 1) * setup->baseRow() * setup->dtcNumMergedRows() * sm->pitchRow();
    const double x = (rowLUT_ + .5) * setup->baseRow() * setup->dtcNumMergedRows() * sm->pitchRow();
    // stub r in cm
    r_ = sqrt(d_ * d_ + x * x);

    const double phi0 = sm->phi() + atan2(x0, d_);
    const double phi1 = sm->phi() + atan2(x1, d_);
    const double c = (phi0 + phi1) / 2.;
    const double m = (phi1 - phi0) / setup->dtcNumMergedRows();

    // intercept of linearized stub phi in rad
    c_ = digi(c, setup->tmttBasePhi());
    // slope of linearized stub phi in rad / strip
    m_ = digi(m, setup->dtcBaseM());

    if (hybrid_) {
      if (abs(z_ / r_) > setup->hybridMaxCot())
        // did not pass eta cut
        valid_ = false;
    } else {
      // extrapolated z at radius T assuming z0=0
      const double zT = setup->chosenRofZ() * z_ / r_;
      // extrapolated z0 window at radius T
      const double dZT = setup->beamWindowZ() * abs(1. - setup->chosenRofZ() / r_);
      double zTMin = zT - dZT;
      double zTMax = zT + dZT;
      if (zTMin >= setup->maxZT() || zTMax < -setup->maxZT())
        // did not pass "eta" cut
        valid_ = false;
      else {
        zTMin = max(zTMin, -setup->maxZT());
        zTMax = min(zTMax, setup->maxZT());
      }
      // range of stub cot(theta)
      cot_ = {zTMin / setup->chosenRofZ(), zTMax / setup->chosenRofZ()};
    }

    // stub r w.r.t. chosenRofPhi in cm
    static const double chosenRofPhi = hybrid_ ? setup->hybridChosenRofPhi() : setup->chosenRofPhi();
    r_ = digi(r_ - chosenRofPhi, setup->tmttBaseR());

    // radial (cylindrical) component of sensor separation
    const double dr = sm->sep() / (sm->cosTilt() - sm->sinTilt() * z_ / d_);
    // converts bend into inv2R in 1/cm
    const double inv2ROverBend = sm->pitchRow() / dr / d_;
    // inv2R in 1/cm
    const double inv2R = -bend_ * setup->baseBend() * inv2ROverBend;
    // inv2R uncertainty in 1/cm
    const double dInv2R = setup->bendCut() * inv2ROverBend;
    const double minPt = hybrid_ ? setup->hybridMinPtStub() : setup->minPt();
    const double maxInv2R = setup->invPtToDphi() / minPt - setup->dtcBaseInv2R() / 2.;
    double inv2RMin = digi(inv2R - dInv2R, setup->dtcBaseInv2R());
    double inv2RMax = digi(inv2R + dInv2R, setup->dtcBaseInv2R());
    if (inv2RMin > maxInv2R || inv2RMax < -maxInv2R) {
      // did not pass pt cut
      valid_ = false;
    } else {
      inv2RMin = max(inv2RMin, -maxInv2R);
      inv2RMax = min(inv2RMax, maxInv2R);
    }
    // range of stub inv2R in 1/cm
    inv2R_ = {inv2RMin, inv2RMax};

    // stub phi w.r.t. detector region centre in rad
    phi_ = c_ + rowSub_ * m_;

    // range of stub extrapolated phi to radius chosenRofPhi in rad
    phiT_.first = phi_ - r_ * inv2R_.first;
    phiT_.second = phi_ - r_ * inv2R_.second;
    if (phiT_.first > phiT_.second)
      swap(phiT_.first, phiT_.second);

    if (phiT_.first < 0.)
      regions_.push_back(0);
    if (phiT_.second >= 0.)
      regions_.push_back(1);

    // apply data format specific manipulations
    if (!hybrid_)
      return;

    // stub r w.r.t. an offset in cm
    r_ -= sm->offsetR() - chosenRofPhi;
    // stub z w.r.t. an offset in cm
    z_ -= sm->offsetZ();
    if (sm->type() == SensorModule::Disk2S) {
      // encoded r
      r_ = sm->encodedR() + (sm->side() ? -col_ : (col_ + sm->numColumns() / 2));
      r_ = (r_ + 0.5) * setup->hybridBaseR(sm->type());
    }

    // encode bend
    const vector<double>& encodingBend = setup->encodingBend(sm->windowSize(), sm->psModule());
    const auto pos = find(encodingBend.begin(), encodingBend.end(), abs(ttStubRef->bendBE()));
    const int uBend = distance(encodingBend.begin(), pos);
    bend_ = pow(-1, signbit(bend_)) * uBend;
  }

  // returns bit accurate representation of Stub
  Frame Stub::frame(int region) const { return hybrid_ ? formatHybrid(region) : formatTMTT(region); }

  // returns true if stub belongs to region
  bool Stub::inRegion(int region) const { return find(regions_.begin(), regions_.end(), region) != regions_.end(); }

  // truncates double precision to f/w integer equivalent
  double Stub::digi(double value, double precision) const { return (floor(value / precision) + .5) * precision; }

  // returns 64 bit stub in hybrid data format
  Frame Stub::formatHybrid(int region) const {
    const SensorModule::Type type = sm_->type();
    // layer encoding
    const int decodedLayerId = layerEncoding_->decode(sm_);
    // stub phi w.r.t. processing region border in rad
    double phi = phi_ - (region - .5) * setup_->baseRegion() + setup_->hybridRangePhi() / 2.;
    if (phi >= setup_->hybridRangePhi())
      phi = setup_->hybridRangePhi() - setup_->hybridBasePhi(type) / 2.;
    // convert stub variables into bit vectors
    const TTBV hwR(r_, setup_->hybridBaseR(type), setup_->hybridWidthR(type), true);
    const TTBV hwPhi(phi, setup_->hybridBasePhi(type), setup_->hybridWidthPhi(type), true);
    const TTBV hwZ(z_, setup_->hybridBaseZ(type), setup_->hybridWidthZ(type), true);
    const TTBV hwAlpha(row_, setup_->hybridBaseAlpha(type), setup_->hybridWidthAlpha(type), true);
    const TTBV hwBend(bend_, setup_->hybridWidthBend(type), true);
    const TTBV hwLayer(decodedLayerId, setup_->hybridWidthLayerId());
    const TTBV hwGap(0, setup_->hybridNumUnusedBits(type));
    const TTBV hwValid(1, 1);
    // assemble final bitset
    return Frame(hwGap.str() + hwR.str() + hwZ.str() + hwPhi.str() + hwAlpha.str() + hwBend.str() + hwLayer.str() +
                 hwValid.str());
  }

  Frame Stub::formatTMTT(int region) const {
    int layerM = sm_->layerId();
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
    TTBV sectorsPhi(0, setup_->numOverlappingRegions() * setup_->numSectorsPhi());
    if (phiT_.first < 0.) {
      if (phiT_.first < -setup_->baseSector())
        sectorsPhi.set(0);
      else
        sectorsPhi.set(1);
      if (phiT_.second < 0. && phiT_.second >= -setup_->baseSector())
        sectorsPhi.set(1);
    }
    if (phiT_.second >= 0.) {
      if (phiT_.second < setup_->baseSector())
        sectorsPhi.set(2);
      else
        sectorsPhi.set(3);
      if (phiT_.first >= 0. && phiT_.first < setup_->baseSector())
        sectorsPhi.set(2);
    }
    // assign stub to eta sectors within a processing region
    pair<int, int> sectorEta({0, setup_->numSectorsEta() - 1});
    for (int bin = 0; bin < setup_->numSectorsEta(); bin++)
      if (asinh(cot_.first) < setup_->boundarieEta(bin + 1)) {
        sectorEta.first = bin;
        break;
      }
    for (int bin = sectorEta.first; bin < setup_->numSectorsEta(); bin++)
      if (asinh(cot_.second) < setup_->boundarieEta(bin + 1)) {
        sectorEta.second = bin;
        break;
      }
    // stub phi w.r.t. processing region centre in rad
    const double phi = phi_ - (region - .5) * setup_->baseRegion();
    // convert stub variables into bit vectors
    const TTBV hwValid(1, 1);
    const TTBV hwGap(0, setup_->tmttNumUnusedBits());
    const TTBV hwLayer(layer, setup_->tmttWidthLayer());
    const TTBV hwSectorEtaMin(sectorEta.first, setup_->tmttWidthSectorEta());
    const TTBV hwSectorEtaMax(sectorEta.second, setup_->tmttWidthSectorEta());
    const TTBV hwR(r_, setup_->tmttBaseR(), setup_->tmttWidthR(), true);
    const TTBV hwPhi(phi, setup_->tmttBasePhi(), setup_->tmttWidthPhi(), true);
    const TTBV hwZ(z_, setup_->tmttBaseZ(), setup_->tmttWidthZ(), true);
    const TTBV hwInv2RMin(inv2R_.first, setup_->tmttBaseInv2R(), setup_->tmttWidthInv2R(), true);
    const TTBV hwInv2RMax(inv2R_.second, setup_->tmttBaseInv2R(), setup_->tmttWidthInv2R(), true);
    TTBV hwSectorPhis(0, setup_->numSectorsPhi());
    for (int sectorPhi = 0; sectorPhi < setup_->numSectorsPhi(); sectorPhi++)
      hwSectorPhis[sectorPhi] = sectorsPhi[region * setup_->numSectorsPhi() + sectorPhi];
    // assemble final bitset
    return Frame(hwGap.str() + hwValid.str() + hwR.str() + hwPhi.str() + hwZ.str() + hwLayer.str() +
                 hwSectorPhis.str() + hwSectorEtaMin.str() + hwSectorEtaMax.str() + hwInv2RMin.str() +
                 hwInv2RMax.str());
  }

}  // namespace trackerDTC