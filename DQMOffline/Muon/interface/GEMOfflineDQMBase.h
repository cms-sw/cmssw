#ifndef DQMOffline_Muon_GEMOfflineDQMBase_h
#define DQMOffline_Muon_GEMOfflineDQMBase_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GEMOfflineDQMBase : public DQMEDAnalyzer {
public:
  using MEMap = std::map<GEMDetId, dqm::impl::MonitorElement*>;

  explicit GEMOfflineDQMBase(const edm::ParameterSet&);

  inline int getVFATNumber(const int, const int, const int);
  inline int getVFATNumberByStrip(const int, const int, const int);
  inline int getMaxVFAT(const int);
  inline int getDetOccXBin(const int, const int, const int);

  // Re: region / St: station, La: layer, Ch: chamber parity, Et: eta partition
  inline GEMDetId getReStKey(const int, const int);
  inline GEMDetId getReStKey(const GEMDetId&);
  inline GEMDetId getReStLaKey(const GEMDetId&);
  inline GEMDetId getReStEtKey(const GEMDetId&);
  inline GEMDetId getReStLaChKey(const GEMDetId&);
  inline GEMDetId getKey(const GEMDetId&);  // == getReStLaChEtKey

  int getDetOccXBin(const GEMDetId&, const edm::ESHandle<GEMGeometry>&);
  void setDetLabelsVFAT(MonitorElement*, const GEMStation*);
  void setDetLabelsEta(MonitorElement*, const GEMStation*);
  int getNumEtaPartitions(const GEMStation*);  // the number of eta partitions per GEMChamber
  void fillME(MEMap& me_map, const GEMDetId& key, const float x);
  void fillME(MEMap& me_map, const GEMDetId& key, const float x, const float y);

  template <typename T>
  inline bool checkRefs(const std::vector<T*>&);

  inline TString getSuffixName(Int_t);
  inline TString getSuffixName(Int_t, Int_t);
  inline TString getSuffixName(Int_t, Int_t, Int_t);
  inline TString getSuffixName(Int_t, Int_t, Int_t, Int_t);

  inline TString getSuffixTitle(Int_t);
  inline TString getSuffixTitle(Int_t, Int_t);
  inline TString getSuffixTitle(Int_t, Int_t, Int_t);
  inline TString getSuffixTitle(Int_t, Int_t, Int_t, Int_t);

  //
  std::string log_category_;
};

inline int GEMOfflineDQMBase::getMaxVFAT(const int station) {
  if (station == 0)
    return GEMeMap::maxVFatGE0_;
  else if (station == 1)
    return GEMeMap::maxVFatGE11_;
  else if (station == 2)
    return GEMeMap::maxVFatGE21_;
  else
    return -1;
}

inline int GEMOfflineDQMBase::getVFATNumber(const int station, const int ieta, const int vfat_phi) {
  const int max_vfat = getMaxVFAT(station);
  return max_vfat * (ieta - 1) + vfat_phi;
}

inline int GEMOfflineDQMBase::getVFATNumberByStrip(const int station, const int ieta, const int strip) {
  const int vfat_phi = (strip % GEMeMap::maxChan_) ? strip / GEMeMap::maxChan_ + 1 : strip / GEMeMap::maxChan_;
  return getVFATNumber(station, ieta, vfat_phi);
}

inline int GEMOfflineDQMBase::getDetOccXBin(const int chamber, const int layer, const int n_chambers) {
  return n_chambers * (chamber - 1) + layer;
}

template <typename T>
inline bool GEMOfflineDQMBase::checkRefs(const std::vector<T*>& refs) {
  if (refs.empty())
    return false;
  if (refs.front() == nullptr)
    return false;
  return true;
}

inline GEMDetId GEMOfflineDQMBase::getReStKey(const int region, const int station) {
  // region, ring, station, layer, chamber, ieta
  return GEMDetId{region, 1, station, 0, 0, 0};
}

inline GEMDetId GEMOfflineDQMBase::getReStKey(const GEMDetId& id) { return getReStKey(id.region(), id.station()); }

inline GEMDetId GEMOfflineDQMBase::getReStLaKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), id.layer(), 0, 0};
}

inline GEMDetId GEMOfflineDQMBase::getReStEtKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), 0, 0, id.roll()};
}

inline GEMDetId GEMOfflineDQMBase::getReStLaChKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), id.layer(), id.chamber() % 2, 0};
}

inline GEMDetId GEMOfflineDQMBase::getKey(const GEMDetId& id) {
  return GEMDetId{id.region(), 1, id.station(), id.layer(), id.chamber() % 2, id.roll()};
}

inline TString GEMOfflineDQMBase::getSuffixName(Int_t region_id) { return TString::Format("_Re%+d", region_id); }

inline TString GEMOfflineDQMBase::getSuffixName(Int_t region_id, Int_t station_id) {
  return TString::Format("_GE%+.2d", region_id * (station_id * 10 + 1));
}

inline TString GEMOfflineDQMBase::getSuffixName(Int_t region_id, Int_t station_id, Int_t layer_id) {
  return TString::Format("_GE%+.2d_L%d", region_id * (station_id * 10 + 1), layer_id);
}

inline TString GEMOfflineDQMBase::getSuffixName(Int_t region_id, Int_t station_id, Int_t layer_id, Int_t ieta) {
  return TString::Format("_GE%+.2d_L%d_R%d", region_id * (station_id * 10 + 1), layer_id, ieta);
}

inline TString GEMOfflineDQMBase::getSuffixTitle(Int_t region_id) { return TString::Format(" Region %+d", region_id); }

TString GEMOfflineDQMBase::getSuffixTitle(Int_t region_id, Int_t station_id) {
  return TString::Format(" GE%+.2d", region_id * (station_id * 10 + 1));
}

TString GEMOfflineDQMBase::getSuffixTitle(Int_t region_id, Int_t station_id, Int_t layer_id) {
  return TString::Format(" GE%+.2d Layer %d", region_id * (station_id * 10 + 1), layer_id);
}

TString GEMOfflineDQMBase::getSuffixTitle(Int_t region_id, Int_t station_id, Int_t layer_id, Int_t ieta) {
  return TString::Format(" GE%+.2d Layer %d Roll %d", region_id * (station_id * 10 + 1), layer_id, ieta);
}

#endif  // DQMOffline_Muon_GEMOfflineDQMBase_h
