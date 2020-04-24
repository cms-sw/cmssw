#include "DataFormats/MuonReco/interface/MuonSimInfo.h"

using namespace reco;

MuonSimInfo::MuonSimInfo():
  primaryClass(MuonSimType::Unknown),
  extendedClass(ExtendedMuonSimType::ExtUnknown),
  flavour(0),
  pdgId(0),
  g4processType(0),
  motherPdgId(0),
  motherFlavour(0),
  motherStatus(0),
  grandMotherPdgId(0),
  grandMotherFlavour(0),
  heaviestMotherFlavour(0),
  tpId(-1),
  tpEvent(999),
  tpBX(999),
  charge(0),
  tpAssoQuality(-1)
{
}
