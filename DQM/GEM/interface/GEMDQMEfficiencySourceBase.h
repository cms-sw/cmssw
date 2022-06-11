#ifndef DQM_GEM_GEMDQMEfficiencySourceBase_h
#define DQM_GEM_GEMDQMEfficiencySourceBase_h

/** \class GEMDQMEfficiencySourceBase
 * 
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "DataFormats/GEMDigi/interface/GEMVFATStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMOHStatusCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GEMDQMEfficiencySourceBase : public DQMEDAnalyzer {
public:
  using MEMap = std::map<GEMDetId, MonitorElement*>;

  explicit GEMDQMEfficiencySourceBase(const edm::ParameterSet&);

  // Re: region / St: station, La: layer, Ch: chamber parity, Et: eta partition
  inline GEMDetId getReStKey(const int, const int);
  inline GEMDetId getReStKey(const GEMDetId&);
  inline GEMDetId getReStLaKey(const GEMDetId&);
  inline GEMDetId getReStEtKey(const GEMDetId&);
  inline GEMDetId getReStLaChKey(const GEMDetId&);
  inline GEMDetId getKey(const GEMDetId&);  // == getReStLaChEtKey

  std::string nameNumerator(const std::string&);

  MonitorElement* bookNumerator1D(DQMStore::IBooker&, MonitorElement*);
  MonitorElement* bookNumerator2D(DQMStore::IBooker&, MonitorElement*);

  std::tuple<bool, int, int> getChamberRange(const GEMStation*);
  std::tuple<bool, int, int> getEtaPartitionRange(const GEMStation*);

  MonitorElement* bookChamber(DQMStore::IBooker&, const TString&, const TString&, const GEMStation*);
  MonitorElement* bookChamberEtaPartition(DQMStore::IBooker&, const TString&, const TString&, const GEMStation*);

  bool skipGEMStation(const int);

  bool maskChamberWithError(const GEMDetId& chamber_id, const GEMOHStatusCollection*, const GEMVFATStatusCollection*);

  bool hasMEKey(const MEMap&, const GEMDetId&);
  void fillME(MEMap&, const GEMDetId&, const double);
  void fillME(MEMap&, const GEMDetId&, const double, const double);

  double clampWithAxis(const double, const TAxis* axis);
  void fillMEWithinLimits(MonitorElement*, const double);
  void fillMEWithinLimits(MonitorElement*, const double, const double);
  void fillMEWithinLimits(MEMap&, const GEMDetId&, const double);
  void fillMEWithinLimits(MEMap&, const GEMDetId&, const double, const double);

  template <typename T>
  bool checkRefs(const std::vector<T*>&);

  const edm::EDGetTokenT<GEMOHStatusCollection> kGEMOHStatusCollectionToken_;
  const edm::EDGetTokenT<GEMVFATStatusCollection> kGEMVFATStatusCollectionToken_;

  const bool kMonitorGE11_;
  const bool kMonitorGE21_;
  const bool kMonitorGE0_;
  const bool kMaskChamberWithError_;
  const std::string kLogCategory_;
};

template <typename T>
bool GEMDQMEfficiencySourceBase::checkRefs(const std::vector<T*>& refs) {
  if (refs.empty())
    return false;
  for (T* each : refs) {
    if (each == nullptr) {
      return false;
    }
  }
  return true;
}

inline GEMDetId GEMDQMEfficiencySourceBase::getReStKey(const int region, const int station) {
  // region, ring, station, layer, chamber, ieta
  return GEMDetId{region, 1, station, 0, 0, 0};
}

inline GEMDetId GEMDQMEfficiencySourceBase::getReStKey(const GEMDetId& id) {
  return getReStKey(id.region(), id.station());
}

inline GEMDetId GEMDQMEfficiencySourceBase::getReStLaKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), id.layer(), 0, 0};
}

inline GEMDetId GEMDQMEfficiencySourceBase::getReStEtKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), 0, 0, id.ieta()};
}

inline GEMDetId GEMDQMEfficiencySourceBase::getReStLaChKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), id.layer(), id.chamber() % 2, 0};
}

inline GEMDetId GEMDQMEfficiencySourceBase::getKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), id.layer(), id.chamber() % 2, id.ieta()};
}

#endif  // DQM_GEM_GEMDQMEfficiencySourceBase_h
