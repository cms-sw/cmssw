#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace l1t;
MuonStub::MuonStub()
    : etaRegion_(0),
      phiRegion_(0),
      depthRegion_(0),
      coord1_(0),
      coord2_(0),
      id_(0),
      quality_(-1),
      bxNum_(17),
      eta1_(0),
      eta2_(0),
      etaQuality_(-1),
      type_(0) {}

MuonStub::MuonStub(int etaRegion,
                   int phiRegion,
                   int depthRegion,
                   uint tfLayer,
                   int coord1,
                   int coord2,
                   int id,
                   int bx,
                   int quality,
                   int eta1,
                   int eta2,
                   int etaQuality,
                   int type)
    : etaRegion_(etaRegion),
      phiRegion_(phiRegion),
      depthRegion_(depthRegion),
      tfLayer_(tfLayer),
      coord1_(coord1),
      coord2_(coord2),
      id_(id),
      quality_(quality),
      bxNum_(bx),
      eta1_(eta1),
      eta2_(eta2),
      etaQuality_(etaQuality),
      type_(type) {}

MuonStub::~MuonStub() {}

bool MuonStub::operator==(const MuonStub& id) const {
  if (etaRegion_ != id.etaRegion_)
    return false;
  if (phiRegion_ != id.phiRegion_)
    return false;
  if (depthRegion_ != id.depthRegion_)
    return false;
  if (id_ != id.id_)
    return false;
  if (coord1_ != id.coord1_)
    return false;
  if (coord2_ != id.coord2_)
    return false;
  if (quality_ != id.quality_)
    return false;
  if (bxNum_ != id.bxNum_)
    return false;
  if (eta1_ != id.eta1_)
    return false;
  if (eta2_ != id.eta2_)
    return false;
  if (etaQuality_ != id.etaQuality_)
    return false;
  if (type_ != id.type_)
    return false;
  return true;
}

//
// output stream operator for phi track segments
//

void MuonStub::print() const {
  LogDebug("MuonStub") << " MuonStub : BX=" << bxNum_ << " etaRegion=" << etaRegion_ << " phiRegion=" << phiRegion_
                       << " depth=" << depthRegion_ << " ID=" << id_ << " coord1=" << coord1_ << " coord2=" << coord2_
                       << " quality=" << quality_ << " eta1=" << eta1_ << " eta2=" << eta2_
                       << " etaQuality=" << etaQuality_ << " type=" << type_;
}

const Phase2L1GMT::wordtype MuonStub::hybridStubWord() const {
  Phase2L1GMT::wordtype w = 0;
  int bstart = 0;
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, 1, 1); //valid bit is always on in emulator because invalid stubs are never created
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, quality_, Phase2L1GMT::BITSSTUBPHIQUALITY);
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, etaQuality_, Phase2L1GMT::BITSSTUBETAQUALITY);
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, bxNum_, Phase2L1GMT::BITSMUONBX);
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, coord1_, Phase2L1GMT::BITSSTUBCOORD);
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, coord2_, Phase2L1GMT::BITSSTUBCOORD);
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, eta1_, Phase2L1GMT::BITSSTUBETA);
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, eta2_, Phase2L1GMT::BITSSTUBETA);
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, 0, Phase2L1GMT::BITSSTUBTIME); //placeholder for time
  int addr = address();
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, addr, Phase2L1GMT::BITSSTUBID);
  bstart = Phase2L1GMT::wordconcat<Phase2L1GMT::wordtype>(w, bstart, tfLayer_, 3);
  return w;
}

void MuonStub::printHybridStub(std::string module="MuonStub", uint spaces=0, bool label=true) const {
  std::string lab = "";
  lab.append(spaces, ' ');
  if (label)
    lab.append("hybrid stub:    ");
  int addr = address();
  edm::LogInfo(module) << lab
                       << "quality = " << quality_ << ",    "
                       << "etaQuality = " << etaQuality_ << ",    "
                       << "bxNum = " << bxNum_ << ",    "
                       << "coord1 = " << offline_coord1_ << " (" << coord1_ << ")" << ",    "
                       << "coord2 = " << offline_coord2_ << " (" << coord2_ << ")" << ",    "
                       << "eta1 = " << offline_eta1_ << " (" << eta1_ << ")" << ",    "
                       << "eta2 = " << offline_eta2_ << " (" << eta2_ << ")" << ","
                       << "\n" << std::setfill(' ') << std::setw(lab.length()) << " "
                       << "time = " << 0 << ",    " //placeholder for time
                       << "address = " << addr << " (id = " << id_ << ")" << ",    "
                       << "tfLayer = " << tfLayer_
		       << std::flush;
}

void MuonStub::printHybridStubWord(std::string module="MuonStub", uint spaces=0, bool label=true) const {
  std::string lab = "";
  lab.append(spaces, ' ');
  if (label)
    lab.append("hybrid stub word = ");
  Phase2L1GMT::wordtype w = MuonStub::hybridStubWord();
  edm::LogInfo(module) << lab << std::setfill('0') << std::setw(16) << std::hex << w.to_uint64() << std::flush;
}

