#ifndef DQMOffline_Muon_GEMOfflineDQMBase_h
#define DQMOffline_Muon_GEMOfflineDQMBase_h

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class GEMOfflineDQMBase : public DQMEDAnalyzer {
public:
  explicit GEMOfflineDQMBase(const edm::ParameterSet&);

  typedef std::tuple<int, int> MEMapKey1;
  typedef std::tuple<int, int, bool> MEMapKey2;
  typedef std::tuple<int, int, bool, int> MEMapKey3;
  typedef std::map<MEMapKey1, MonitorElement*> MEMap1;
  typedef std::map<MEMapKey2, MonitorElement*> MEMap2;
  typedef std::map<MEMapKey3, MonitorElement*> MEMap3;

  inline int getVFATNumber(const int, const int, const int);
  inline int getVFATNumberByStrip(const int, const int, const int);
  inline int getMaxVFAT(const int);
  inline int getDetOccXBin(const int, const int, const int);

  int getDetOccXBin(const GEMDetId&, const edm::ESHandle<GEMGeometry>&);
  void setDetLabelsVFAT(MonitorElement*, const GEMStation*);
  void setDetLabelsEta(MonitorElement*, const GEMStation*);

  template <typename T>
  inline bool checkRefs(const std::vector<T*>&);

  std::string log_category_;

  class BookingHelper {
  public:
    BookingHelper(DQMStore::IBooker& ibooker, const TString& name_suffix, const TString& title_suffix)
        : ibooker_(&ibooker), name_suffix_(name_suffix), title_suffix_(title_suffix) {}

    ~BookingHelper() {}

    MonitorElement* book1D(TString name,
                           TString title,
                           int nbinsx,
                           double xlow,
                           double xup,
                           TString x_title = "",
                           TString y_title = "Entries") {
      name += name_suffix_;
      title += title_suffix_ + ";" + x_title + ";" + y_title;
      return ibooker_->book1D(name, title, nbinsx, xlow, xup);
    }

    MonitorElement* book1D(TString name,
                           TString title,
                           std::vector<double>& x_binning,
                           TString x_title = "",
                           TString y_title = "Entries") {
      name += name_suffix_;
      title += title_suffix_ + ";" + x_title + ";" + y_title;
      TH1F* h_obj = new TH1F(name, title, x_binning.size() - 1, &x_binning[0]);
      return ibooker_->book1D(name, h_obj);
    }

    MonitorElement* book2D(TString name,
                           TString title,
                           int nbinsx,
                           double xlow,
                           double xup,
                           int nbinsy,
                           double ylow,
                           double yup,
                           TString x_title = "",
                           TString y_title = "") {
      name += name_suffix_;
      title += title_suffix_ + ";" + x_title + ";" + y_title;
      return ibooker_->book2D(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
    }

  private:
    DQMStore::IBooker* ibooker_;
    const TString name_suffix_;
    const TString title_suffix_;
  };  // BookingHelper
};

inline int GEMOfflineDQMBase::getMaxVFAT(const int station) {
  if (station == 1)
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
};

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

#endif  // DQMOffline_Muon_GEMOfflineDQMBase_h
