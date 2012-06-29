#ifndef EcalDQMBinningService_H
#define EcalDQMBinningService_H

#include <map>
#include <stdint.h>
#include <string>
#include <vector>

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/DetId/interface/DetId.h"

// Keeps a map between channel ids and DQM histogram bins
// The map is booked ad-hoc

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class Run;
  class EventSetup;
}
class CaloGeometry;
class DQMStore;
class EcalElectronicsId;

class EcalDQMBinningService {
 public:
  enum ObjectType {
    kEB,
    kEBMEM,
    kEE,
    kEEm,
    kEEp,
    kEEMEM,
    kSM,
    kSMMEM,
    kEcal,
    kEcal2P,
    kEcal3P,
    kEcalMEM2P,
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
    nDCCMEM = 44,

    nTCC = 108,
    kEEmTCCLow = 0, kEEmTCCHigh = 35,
    kEEpTCCLow = 72, kEEpTCCHigh = 107,
    kEBTCCLow = 36, kEBTCCHigh = 71,

    nEBSMEta = 85,
    nEBSMPhi = 20,
    nEBSMBins = nEBSMEta * nEBSMPhi,
    nEESMX = 40,
    nEESMY = 40,
    nEESMBins = nEESMX * nEESMY,
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
    AxisSpecs() : nbins(1), edges(0) {}
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

  std::vector<AxisSpecs> getBinning(ObjectType, BinningType, bool _isMap = true, unsigned _objOffset = 0) const;

  // takes care of "key translations" as well - okey and bkey can have changed after the function returns
  const std::vector<int>* getBinMap(ObjectType&, BinningType&) const;

  std::pair<unsigned, std::vector<int> > findBins(ObjectType, BinningType, const DetId&) const;
  std::pair<unsigned, std::vector<int> > findBins(ObjectType, BinningType, const EcalElectronicsId&) const;
  std::pair<unsigned, std::vector<int> > findBins(ObjectType, BinningType, unsigned) const;
  // EcalElectronicsId version returns at most one bin

  std::pair<unsigned, std::vector<int> > findBinsNoMap(ObjectType, BinningType, const DetId&) const;
  std::pair<unsigned, std::vector<int> > findBinsNoMap(ObjectType, BinningType, const EcalElectronicsId&) const;

  int getBin(ObjectType, BinningType, unsigned) const;

  unsigned findOffset(ObjectType, const DetId&) const;
  unsigned findOffset(ObjectType, const EcalElectronicsId&) const;
  unsigned findOffset(ObjectType, BinningType, unsigned) const;

  ObjectType objectFromOffset(ObjectType, unsigned) const;

  int smOffsetBins(ObjectType, BinningType, unsigned) const;

  // used for EE binnings
  int xlow(int) const;
  int ylow(int) const;

  std::string channelName(uint32_t, BinningType _btype = kDCC) const;

  uint32_t idFromName(std::string const&) const;

 private:
  std::vector<AxisSpecs> getBinningEB_(BinningType, bool) const;
  std::vector<AxisSpecs> getBinningEBMEM_(BinningType, bool) const;
  std::vector<AxisSpecs> getBinningEE_(BinningType, bool, int) const;
  std::vector<AxisSpecs> getBinningEEMEM_(BinningType, bool) const;
  std::vector<AxisSpecs> getBinningSM_(BinningType, bool, unsigned) const;
  std::vector<AxisSpecs> getBinningSMMEM_(BinningType, bool, unsigned) const;
  std::vector<AxisSpecs> getBinningEcal_(BinningType, bool) const;

  const std::vector<int>* getBinMapEB_(BinningType) const;
  const std::vector<int>* getBinMapEBMEM_(BinningType) const;
  const std::vector<int>* getBinMapEE_(BinningType, int) const;
  const std::vector<int>* getBinMapEEMEM_(BinningType) const;
  const std::vector<int>* getBinMapSM_(BinningType) const;
  const std::vector<int>* getBinMapSMMEM_(BinningType) const;
  const std::vector<int>* getBinMapEcal_(BinningType) const;

  void findBinsCrystal_(const DetId&, ObjectType, const std::vector<int>&, std::vector<int>&) const;
  void findBinsTriggerTower_(const DetId&, ObjectType, const std::vector<int>&, std::vector<int>&) const;
  void findBinsSuperCrystal_(const DetId&, ObjectType, const std::vector<int>&, std::vector<int>&) const;
  void findBinsDCC_(const DetId&, ObjectType, const std::vector<int>&, std::vector<int>&) const;
  void findBinsTCC_(const DetId&, ObjectType, const std::vector<int>&, std::vector<int>&) const;
  void findBinsProjEta_(const DetId&, ObjectType, const std::vector<int>&, std::vector<int>&) const;
  void findBinsProjPhi_(const DetId&, ObjectType, const std::vector<int>&, std::vector<int>&) const;

  // need to be mutable to allow runtime booking
  mutable std::vector<int> binMaps_[nPlotType][nPresetBinnings];
  /*
    Following dictionaries are created during postBeginRun:
    (kEB, kCrystal) (kEB, kSuperCrystal) (kEB, kDCC) (kEB, kTCC) (kEBMEM, kCrystal)
    (kEE, kCrystal) (kEE, kSuperCrystal) (kEE, kDCC) (kEE, kTCC) (kEEMEM, kCrystal)
    (kEEm, kCrystal) (kEEm, kSuperCrystal) (kEEm, kDCC) (kEEm, kTCC)
    (kEEp, kCrystal) (kEEp, kSuperCrystal) (kEEp, kDCC) (kEEp, kTCC)
    (kSM, kCrystal) (kSM, kSuperCrystal) (kSM, kTriggerTower) (kSMMEM, kCrystal)
  */

  mutable uint32_t cacheId_;
  mutable ObjectType cacheOtype_;
  mutable BinningType cacheBtype_;
  mutable std::pair<unsigned, std::vector<int> > cache_;

  const double etaBound_;

  const CaloGeometry* geometry_;

  bool initialized_;

  int verbosity_;

};

#endif
