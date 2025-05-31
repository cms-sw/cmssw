#include "L1Trigger/TrackerDTC/interface/Stub.h"

#include <cmath>
#include <iterator>
#include <algorithm>
#include <utility>

namespace trackerDTC {

  Stub::Stub(const trackerTFP::DataFormats* dataFormats, const tt::SensorModule* sm, const TTStubRef& ttStubRef)
      : setup_(dataFormats->setup()),
        dataFormats_(dataFormats),
        layerEncoding_(nullptr),
        sm_(sm),
        ttStubRef_(ttStubRef),
        valid_(true),
        regions_(0, setup_->numOverlappingRegions()) {
    const trackerTFP::DataFormat& dfR = dataFormats_->format(trackerTFP::Variable::r, trackerTFP::Process::dtc);
    const trackerTFP::DataFormat& dfPhi = dataFormats_->format(trackerTFP::Variable::phi, trackerTFP::Process::dtc);
    const trackerTFP::DataFormat& dfZ = dataFormats_->format(trackerTFP::Variable::z, trackerTFP::Process::dtc);
    const trackerTFP::DataFormat& dfInv2R = dataFormats_->format(trackerTFP::Variable::inv2R, trackerTFP::Process::ht);
    // get stub local coordinates
    const MeasurementPoint& mp = ttStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();
    // convert to uniformed local coordinates
    // column number in pitch units
    col_ = std::floor(std::pow(-1, sm_->signCol()) * (mp.y() - sm_->numColumns() / 2) / setup_->baseCol());
    // row number in half pitch units
    row_ = std::floor(std::pow(-1, sm_->signRow()) * (mp.x() - sm_->numRows() / 2) / setup_->baseRow());
    // bend number in quarter pitch units
    bend_ = std::floor(std::pow(-1, sm_->signBend()) * (ttStubRef->bendBE()) / setup_->baseBend());
    // reduced row number for look up
    rowLUT_ = std::floor(static_cast<double>(row_) / std::pow(2., setup_->widthRow() - setup_->dtcWidthRowLUT()));
    // sub row number inside reduced row number
    rowSub_ = row_ - (rowLUT_ + .5) * std::pow(2, setup_->widthRow() - setup_->dtcWidthRowLUT());
    // convert local to global coordinates
    const double y = (col_ + .5) * setup_->baseCol() * sm_->pitchCol();
    // radius of a column of strips/pixel in cm
    d_ = sm_->r() + y * sm_->sinTilt();
    // stub z in cm
    z_ = dfZ.digi(sm_->z() + y * sm_->cosTilt());
    const double x = (rowLUT_ + .5) * setup_->baseRow() * setup_->dtcNumMergedRows() * sm_->pitchRow();
    // stub r wrt chosen RofPhi in cm
    r_ = dfR.digi(std::sqrt(d_ * d_ + x * x) - setup_->chosenRofPhi());
    const double x0 = rowLUT_ * setup_->baseRow() * setup_->dtcNumMergedRows() * sm_->pitchRow();
    const double x1 = (rowLUT_ + 1) * setup_->baseRow() * setup_->dtcNumMergedRows() * sm_->pitchRow();
    const double phi0 = sm_->phi() + std::atan2(x0, d_);
    const double phi1 = sm_->phi() + std::atan2(x1, d_);
    const double c = (phi0 + phi1) / 2.;
    const double m = (phi1 - phi0) / setup_->dtcNumMergedRows();
    // intercept of linearized stub phi in rad
    c_ = digi(c, dfPhi.base() / 2.);
    // slope of linearized stub phi in rad / strip
    m_ = digi(m, setup_->dtcBaseM());
    // stub phi w.r.t. detector region centre in rad
    phi_ = dfPhi.digi(c_ + rowSub_ * m_);
    // assaign stub to processing regions
    // radial (cylindrical) component of sensor separation
    const double dr = sm_->sep() / (sm_->cosTilt() - sm_->sinTilt() * z_ / d_);
    // converts bend into inv2R in 1/cm
    const double inv2ROverBend = sm_->pitchRow() / dr / d_;
    // inv2R in 1/cm
    const double inv2R = -bend_ * setup_->baseBend() * inv2ROverBend;
    // inv2R uncertainty in 1/cm
    const double dInv2R = setup_->bendCut() * inv2ROverBend;
    inv2R_.first = dfInv2R.digi(inv2R - dInv2R);
    inv2R_.second = dfInv2R.digi(inv2R + dInv2R);
    const double maxInv2R = dfInv2R.range() / 2.;
    // cut on pt
    if (inv2R_.first > maxInv2R || inv2R_.second < -maxInv2R)
      valid_ = false;
    else {
      inv2R_.first = std::max(inv2R_.first, -maxInv2R);
      inv2R_.second = std::min(inv2R_.second, maxInv2R);
    }
    // range of stub extrapolated phi to radius chosenRofPhi in rad
    phiT_.first = phi_ - r_ * inv2R_.first;
    phiT_.second = phi_ - r_ * inv2R_.second;
    if (phiT_.first > phiT_.second)
      std::swap(phiT_.first, phiT_.second);
    if (phiT_.first < 0.)
      regions_.set(0);
    if (phiT_.second >= 0.)
      regions_.set(1);
  }

  Stub::Stub(const tt::Setup* setup,
             const trackerTFP::DataFormats* dataFormats,
             const LayerEncoding* layerEncoding,
             const tt::SensorModule* sm,
             const TTStubRef& ttStubRef)
      : setup_(setup), dataFormats_(dataFormats), layerEncoding_(layerEncoding), sm_(sm), ttStubRef_(ttStubRef) {
    const Stub stub(dataFormats, sm, ttStubRef);
    bend_ = stub.bend_;
    valid_ = stub.valid_;
    row_ = stub.row_;
    col_ = stub.col_;
    r_ = stub.r_;
    phi_ = stub.phi_;
    z_ = stub.z_;
    phiT_ = stub.phiT_;
    inv2R_ = stub.inv2R_;
    regions_ = stub.regions_;
    // apply "eta" cut
    const trackerTFP::DataFormat& dfZT = dataFormats->format(trackerTFP::Variable::zT, trackerTFP::Process::gp);
    const double r = r_ + setup->chosenRofPhi();
    const double ratioRZ = setup->chosenRofZ() / r;
    // extrapolated z at radius T assuming z0=0
    const double zT = z_ * ratioRZ;
    // extrapolated z0 window at radius T
    const double dZT = setup->beamWindowZ() * std::abs(1. - ratioRZ);
    zT_ = {zT - dZT, zT + dZT};
    if (std::abs(zT) > dfZT.range() / 2. + dZT)
      valid_ = false;
    // apply data format specific manipulations
    if (!setup_->useHybrid())
      return;
    // stub r w.r.t. an offset in cm
    r_ -= sm->offsetR() - setup->chosenRofPhi();
    // stub z w.r.t. an offset in cm
    z_ -= sm->offsetZ();
    if (sm->type() == tt::SensorModule::Disk2S) {
      // encoded r
      r_ = sm->encodedR() + (sm->side() ? -col_ : (col_ + sm->numColumns() / 2));
      r_ = (r_ + 0.5) * setup->hybridBaseR(sm->type());
    }
    // encode bend
    const std::vector<double>& encodingBend = setup->encodingBend(sm->windowSize(), sm->psModule());
    const auto pos = std::find(encodingBend.begin(), encodingBend.end(), std::abs(ttStubRef->bendBE()));
    const int uBend = std::distance(encodingBend.begin(), pos);
    bend_ = std::pow(-1, std::signbit(bend_)) * uBend;
  }

  // returns bit accurate representation of Stub
  tt::FrameStub Stub::frame(int region) const {
    return make_pair(ttStubRef_, setup_->useHybrid() ? formatHybrid(region) : formatTMTT(region));
  }

  // truncates double precision to f/w integer equivalent
  double Stub::digi(double value, double precision) const {
    return (std::floor(value / precision + 1.e-12) + .5) * precision;
  }

  // returns 64 bit stub in hybrid data format
  tt::Frame Stub::formatHybrid(int region) const {
    const tt::SensorModule::Type type = sm_->type();
    // layer encoding
    const int decodedLayerId = layerEncoding_->decode(sm_);
    // stub phi w.r.t. processing region border in rad
    double phi = phi_ - (region - .5) * setup_->baseRegion() + setup_->hybridRangePhi() / 2.;
    // convert stub variables into bit vectors
    const bool twosR = type == tt::SensorModule::BarrelPS || type == tt::SensorModule::Barrel2S;
    const TTBV hwR(r_, setup_->hybridBaseR(type), setup_->hybridWidthR(type), twosR);
    const TTBV hwPhi(phi, setup_->hybridBasePhi(type), setup_->hybridWidthPhi(type));
    const TTBV hwZ(z_, setup_->hybridBaseZ(type), setup_->hybridWidthZ(type), true);
    const TTBV hwAlpha(row_, setup_->hybridBaseAlpha(type), setup_->hybridWidthAlpha(type), true);
    const TTBV hwBend(bend_, setup_->hybridWidthBend(type), true);
    const TTBV hwLayer(decodedLayerId, setup_->hybridWidthLayerId());
    const TTBV hwGap(0, setup_->hybridNumUnusedBits(type));
    const TTBV hwValid(1, 1);
    // assemble final bitset
    return tt::Frame(hwGap.str() + hwR.str() + hwZ.str() + hwPhi.str() + hwAlpha.str() + hwBend.str() + hwLayer.str() +
                     hwValid.str());
  }

  tt::Frame Stub::formatTMTT(int region) const {
    const trackerTFP::DataFormat& dfInv2R = dataFormats_->format(trackerTFP::Variable::inv2R, trackerTFP::Process::ht);
    const trackerTFP::DataFormat& dfPhiT = dataFormats_->format(trackerTFP::Variable::phiT, trackerTFP::Process::gp);
    const trackerTFP::DataFormat& dfZT = dataFormats_->format(trackerTFP::Variable::zT, trackerTFP::Process::gp);
    const double offset = (region - .5) * dfPhiT.range();
    const double r = r_;
    const double phi = phi_ - offset;
    const double z = z_;
    const int indexLayerId = setup_->indexLayerId(ttStubRef_);
    TTBV layer(indexLayerId, dataFormats_->width(trackerTFP::Variable::layer, trackerTFP::Process::dtc));
    if (sm_->barrel()) {
      layer.set(4);
      if (sm_->tilted())
        layer.set(3);
    } else if (sm_->psModule())
      layer.set(3);
    int phiTMin = std::max(dfPhiT.integer(phiT_.first - offset), -setup_->gpNumBinsPhiT() / 2);
    int phiTMax = std::min(dfPhiT.integer(phiT_.second - offset), setup_->gpNumBinsPhiT() / 2 - 1);
    if (phiTMin > setup_->gpNumBinsPhiT() / 2 - 1)
      phiTMin = setup_->gpNumBinsPhiT() / 2 - 1;
    if (phiTMax < -setup_->gpNumBinsPhiT() / 2)
      phiTMax = -setup_->gpNumBinsPhiT() / 2;
    const int zTMin = std::max(dfZT.integer(zT_.first), -setup_->gpNumBinsZT() / 2);
    const int zTMax = std::min(dfZT.integer(zT_.second), setup_->gpNumBinsZT() / 2 - 1);
    const int inv2RMin = std::max(dfInv2R.integer(inv2R_.first), -setup_->htNumBinsInv2R() / 2);
    const int inv2RMax = std::min(dfInv2R.integer(inv2R_.second), setup_->htNumBinsInv2R() / 2 - 1);
    const trackerTFP::StubDTC stub(
        ttStubRef_, dataFormats_, r, phi, z, layer, phiTMin, phiTMax, zTMin, zTMax, inv2RMin, inv2RMax);
    return stub.frame().second;
  }

}  // namespace trackerDTC
