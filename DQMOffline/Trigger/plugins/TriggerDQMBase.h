#ifndef DQMOffline_Trigger_TriggerDQMBase_h
#define DQMOffline_Trigger_TriggerDQMBase_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class TriggerDQMBase {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  TriggerDQMBase() = default;
  virtual ~TriggerDQMBase() = default;

  struct MEbinning {
    uint nbins;
    double xmin;
    double xmax;
  };

  class ObjME {
  public:
    ObjME() {}
    virtual ~ObjME() {}

    MonitorElement* numerator = nullptr;
    MonitorElement* denominator = nullptr;

    template <typename... Args>
    void fill(const bool pass_num, Args... args);
  };

  void setMETitle(ObjME& me, const std::string& titleX, const std::string& titleY);

  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              const uint nbins,
              const double xmin,
              const double xmax,
              const bool bookDen = true);
  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              const std::vector<double>& binningX,
              const bool bookDen = true);
  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              const uint nbinsX,
              const double xmin,
              const double xmax,
              const double ymin,
              const double ymax,
              const bool bookDen = true);
  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              const uint nbinsX,
              const double xmin,
              const double xmax,
              const uint nbinsY,
              const double ymin,
              const double ymax,
              const bool bookDen = true);
  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              const std::vector<double>& binningX,
              const std::vector<double>& binningY,
              const bool bookDen = true);

  static void fillHistoPSetDescription(edm::ParameterSetDescription& pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription& pset);

  static MEbinning getHistoPSet(const edm::ParameterSet& pset);
  static MEbinning getHistoLSPSet(const edm::ParameterSet& pset);
};

template <typename... Args>
void TriggerDQMBase::ObjME::fill(const bool fill_num, Args... args) {
  if (denominator) {
    denominator->Fill(args...);
  }

  if (fill_num and numerator) {
    numerator->Fill(args...);
  }
}

#endif  // DQMOffline_Trigger_TriggerDQMBase_h
