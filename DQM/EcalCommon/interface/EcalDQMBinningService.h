#ifndef EcalDQMBinningService_H
#define EcalDQMBinningService_H

#include <map>
#include <stdint.h>
#include <string>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

// functions that do not use mappings should be made static

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
    kEE2P,
    kMEM2P,
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
    AxisSpecs() : nbins(0), low(0.), high(0.), edges(0) {}
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
  int findBin2D(ObjectType, BinningType, unsigned) const;

  unsigned findPlot(ObjectType, const DetId&) const;
  unsigned findPlot(ObjectType, const EcalElectronicsId&) const;
  unsigned findPlot(ObjectType, unsigned, BinningType _btype = kDCC) const;

  static ObjectType getObject(ObjectType, unsigned);

  static unsigned getNObjects(ObjectType);

  bool isValidIdBin(ObjectType, BinningType, unsigned, int) const;

  // used for EE binnings
  static int xlow(int);
  static int ylow(int);

  std::string channelName(uint32_t, BinningType _btype = kDCC) const;
  
  uint32_t idFromName(std::string const&) const;
  uint32_t idFromBin(ObjectType, BinningType, unsigned, int) const;

  static AxisSpecs const* formAxis(edm::ParameterSet const&);

  static ObjectType getObjectType(std::string const&);
  static BinningType getBinningType(std::string const&);

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

#endif
