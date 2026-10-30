#include "L1Trigger/TrackerDTC/interface/StubDTC.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <cmath>
#include <string>

namespace trackerDTC {

  StubDTC::StubDTC(const StubGL& stubGL, int overlap) : stubGL_(&stubGL), valid_(true) {
    const StubFE* stubFE = stubGL.stubFE();
    const Setup* setup = stubFE->setup();
    const SensorModule* sm = stubFE->sm();
    const SensorModule::Type type = sm->type();
    const bool twosR = type == SensorModule::BarrelPS || type == SensorModule::Barrel2S;
    const bool noAlpha = type != SensorModule::Disk2S;
    const bool nd = type == SensorModule::Disk2S || type == SensorModule::DiskPS;
    // base transform of coordinates
    double r = tt::redigi(stubGL.r(), setup->glBaseR(), setup->stubBaseR(), setup->widthDSPbu());
    double phi = tt::redigi(stubGL.phi(), setup->glBasePhi(), setup->stubBasePhi(), setup->widthDSPbu());
    double z = tt::redigi(stubGL.z(), setup->glBaseZ(), setup->stubBaseZ(), setup->widthDSPbu());
    // stub z w.r.t. an offset in cm
    z -= tt::digiR(sm->offsetZ(), setup->stubBaseZ());
    // stub r w.r.t. an offset in cm
    r -= tt::digiR(sm->offsetR(), setup->stubBaseR());
    // encoded r
    if (type == SensorModule::Disk2S)
      r = sm->offsetR() - stubGL.stubFE()->cic() + 1;
    // encode bend
    const int bend = sm->signBend() ? stubFE->bend() : -stubFE->bend();
    // encode layer
    const int layer = sm->encodedLayer();
    // stub phi w.r.t. processing region border in rad
    phi = tt::digiR(phi, setup->stubBasePhi()) -
          tt::digiR((overlap - .5) * setup->regRangePhiT() - setup->stubRangePhi() / 2., setup->stubBasePhi());
    // move to type specific bases
    r = tt::digiR(r, setup->stubBaseR(type));
    phi = tt::digiR(phi, setup->stubBasePhi(type));
    z = tt::digiR(z, setup->stubBaseZ(type));
    // kill stubs outside phi range
    if (phi < 0 || phi >= std::pow(2., setup->stubWidthPhi(type)) * setup->stubBasePhi(type))
      valid_ = false;
    if (!valid_)
      return;
    // convert stub variables into bits
    const double alpha = stubGL.fec() + stubGL.row();
    std::string s;
    s += TTBV(0, setup->stubNumUnusedBits(type)).str();
    s += nd ? (sm->side() ? "0" : "1") : "";
    s += TTBV(r, setup->stubBaseR(type), setup->stubWidthR(type), twosR).str();
    s += TTBV(z, setup->stubBaseZ(type), setup->stubWidthZ(type), true).str();
    s += TTBV(phi, setup->stubBasePhi(type), setup->stubWidthPhi(type)).str();
    s += noAlpha ? "" : TTBV(alpha, setup->stubBaseAlpha(type), setup->stubWidthAlpha(type), true).str();
    s += TTBV(bend, setup->stubWidthBend(type), true).str();
    s += TTBV(layer, setup->stubWidthLayerId()).str();
    s += "1";
    // assemble final bitset
    frame_.second = tt::Frame(s);
    frame_.first = stubFE->ttStubRef();
  }

}  // namespace trackerDTC
