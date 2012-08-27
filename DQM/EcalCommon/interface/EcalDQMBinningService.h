#ifndef EcalDQMBinningService_H
#define EcalDQMBinningService_H

#include <map>
#include <stdint.h>
#include <string>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

// Keeps a map between channel ids and DQM histogram bins
// The map is booked ad-hoc

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class Run;
  class EventSetup;
}
class DQMStore;
class EcalElectronicsId;

class EcalDQMBinningService {
 public:
  enum ObjectType {
    kEB,
    kEE,
    kEEm,
    kEEp,
    kSM,
    kEBSM,
    kEESM,
    kSMMEM,
    kEBSMMEM,
    kEESMMEM,
    kEcal,
    kMEM,
    kEBMEM,
    kEEMEM,
    kEcal2P,
    kEcal3P,
    kChannel,
    nObjType
  };

  enum BinningType {
    kCrystal,
    kTriggerTower,
    kSuperCrystal,
    kTCC,
    kDCC,
    kProjEta,
    kProjPhi,
    kUser,
    kReport,
    kTrend,
    nBinType
  };

  enum Constants {
    nPlotType = kEcal + 1,
    nPresetBinnings = kProjPhi + 1,

    nDCC = 54,
    nEBDCC = 36,
    nEEDCC = 18,
    nDCCMEM = 44,
    nEEDCCMEM = 8,

    nTCC = 108,
    kEEmTCCLow = 0, kEEmTCCHigh = 35,
    kEEpTCCLow = 72, kEEpTCCHigh = 107,
    kEBTCCLow = 36, kEBTCCHigh = 71,

    nEBSMEta = 85,
    nEBSMPhi = 20,
    nEESMX = 40,
    nEESMY = 40,
    nEESMXExt = 45, // for EE+-02&08
    nEESMBinsExt = nEESMXExt * nEESMY,


    nEBEtaBins = 34,
    nEEEtaBins = 20,
    nPhiBins = 36
  };

  struct AxisSpecs {
    int nbins;
    double low, high;
    double* edges;
    std::string title;
    AxisSpecs() : nbins(0), edges(0) {}
    AxisSpecs(AxisSpecs const& _specs) :
      nbins(_specs.nbins), low(_specs.low), high(_specs.high), edges(0), title(_specs.title)
    {
      if(_specs.edges){
	edges = new double[nbins + 1];
	for(int i(0); i <= nbins; i++) edges[i] = _specs.edges[i];
      }
    }
    AxisSpecs& operator=(AxisSpecs const& _rhs)
    {
      if(edges){ delete [] edges; edges = 0; }
      nbins = _rhs.nbins; low = _rhs.low; high = _rhs.high; title = _rhs.title;
      if(_rhs.edges){
	edges = new double[nbins + 1];
	for(int i(0); i <= nbins; i++) edges[i] = _rhs.edges[i];
      }
      return *this;
    }
    ~AxisSpecs() { if(edges) delete [] edges; }
  };

  EcalDQMBinningService(const edm::ParameterSet&, edm::ActivityRegistry&);
  ~EcalDQMBinningService();

  void postBeginRun(const edm::Run&, const edm::EventSetup&);
  void postEndRun(const edm::Run&, const edm::EventSetup&);

  AxisSpecs getBinning(ObjectType, BinningType, bool, int, unsigned) const;

  int findBin1D(ObjectType, BinningType, DetId const&) const;
  int findBin1D(ObjectType, BinningType, EcalElectronicsId const&) const;
  int findBin1D(ObjectType, BinningType, unsigned) const;

  int findBin2D(ObjectType, BinningType, DetId const&) const;
  int findBin2D(ObjectType, BinningType, EcalElectronicsId const&) const;

  unsigned findPlot(ObjectType, const DetId&) const;
  unsigned findPlot(ObjectType, const EcalElectronicsId&) const;
  unsigned findPlot(ObjectType, unsigned, BinningType _btype = kDCC) const;

  ObjectType getObject(ObjectType, unsigned) const;

  unsigned getNObjects(ObjectType) const;

  bool isValidIdBin(ObjectType, BinningType, unsigned, int) const;

  // used for EE binnings
  int xlow(int) const;
  int ylow(int) const;

  std::string channelName(uint32_t, BinningType _btype = kDCC) const;
  
  uint32_t idFromName(std::string const&) const;
  uint32_t idFromBin(ObjectType, BinningType, unsigned, int) const;

  AxisSpecs const* formAxis(edm::ParameterSet const&) const;

  ObjectType getObjectType(std::string const&) const;
  BinningType getBinningType(std::string const&) const;

 private:
  AxisSpecs getBinningEB_(BinningType, bool, int) const;
  AxisSpecs getBinningEE_(BinningType, bool, int, int) const;
  AxisSpecs getBinningSM_(BinningType, bool, unsigned, int) const;
  AxisSpecs getBinningSMMEM_(BinningType, bool, unsigned, int) const;
  AxisSpecs getBinningEcal_(BinningType, bool, int) const;
  AxisSpecs getBinningMEM_(BinningType, bool, int, int) const;

  int findBinCrystal_(ObjectType, const DetId&, int = -1) const;
  int findBinTriggerTower_(ObjectType, const DetId&) const;
  int findBinSuperCrystal_(ObjectType, const DetId&, int = -1) const;
  int findBinCrystal_(ObjectType _otype, const EcalElectronicsId& _id) const { return findBinCrystal_(_otype, ecaldqm::getElectronicsMap()->getDetId(_id)); }
  int findBinSuperCrystal_(ObjectType, const EcalElectronicsId&) const;

  int verbosity_;
};

inline
int
EcalDQMBinningService::xlow(int _iSM) const
{
  using namespace ecaldqm;

  switch(_iSM){
  case kEEm01: case kEEp01: return 15;
  case kEEm02: case kEEp02: return 0;
  case kEEm03: case kEEp03: return 0;
  case kEEm04: case kEEp04: return 5;
  case kEEm05: case kEEp05: return 30;
  case kEEm06: case kEEp06: return 55;
  case kEEm07: case kEEp07: return 60;
  case kEEm08: case kEEp08: return 55;
  case kEEm09: case kEEp09: return 45;
  default: break;
  }

  if(_iSM >= kEBmLow && _iSM <= kEBpHigh) return 0;

  return 0;
}

inline
int
EcalDQMBinningService::ylow(int _iSM) const
{
  using namespace ecaldqm;

  switch(_iSM){
  case kEEm01: case kEEp01: case kEEm09: case kEEp09: return 60;
  case kEEm02: case kEEp02: case kEEm08: case kEEp08: return 50;
  case kEEm03: case kEEp03: case kEEm07: case kEEp07: return 25;
  case kEEm04: case kEEp04: case kEEm06: case kEEp06: return 5;
  case kEEm05: case kEEp05: return 0;
  default: break;
  }

  if(_iSM >= kEBmLow && _iSM <= kEBmHigh) return ((_iSM - kEBmLow) % 18) * 20;
  if(_iSM >= kEBpLow && _iSM <= kEBpHigh) return (-1 - ((_iSM - kEBpLow) % 18)) * 20;

  return 0;
}

inline
unsigned
EcalDQMBinningService::getNObjects(ObjectType _otype) const
{
  switch(_otype){
  case kSM:
    return nDCC;
  case kEBSM:
    return nEBDCC;
  case kEESM:
    return nEEDCC;
  case kSMMEM:
    return nDCCMEM;
  case kEBSMMEM:
    return nEBDCC;
  case kEESMMEM:
    return nEEDCCMEM;
  case kEcal2P:
    return 2;
  case kEcal3P:
    return 3;
  default:
    return 1;
  }
}

inline
EcalDQMBinningService::ObjectType
EcalDQMBinningService::getObject(ObjectType _otype, unsigned _iObj) const
{
  if(_otype == kEcal3P) {
    switch(_iObj){
    case 0: return kEEm;
    case 1: return kEB;
    case 2: return kEEp;
    default: return nObjType;
    }
  }
  else if(_otype == kEcal2P){
    switch(_iObj){
    case 0: return kEE;
    case 1: return kEB;
    default: return nObjType;
    }
  }
  else
    return _otype;
}

#endif
