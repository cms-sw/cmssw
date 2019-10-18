#ifndef DQMOffline_Trigger_TriggerDQMBase_h
#define DQMOffline_Trigger_TriggerDQMBase_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class TriggerDQMBase {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  TriggerDQMBase() = default;
  virtual ~TriggerDQMBase() = default;

  struct MEbinning {
    unsigned nbins;
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

  static void fillHistoPSetDescription(edm::ParameterSetDescription& pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription& pset);
  static MEbinning getHistoPSet(const edm::ParameterSet& pset);
  static MEbinning getHistoLSPSet(const edm::ParameterSet& pset);

  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              unsigned nbins,
              double xmin,
              double xmax);
  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              unsigned nbinsX,
              double xmin,
              double xmax,
              double ymin,
              double ymax);
  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              unsigned nbinsX,
              double xmin,
              double xmax,
              unsigned nbinsY,
              double ymin,
              double ymax);
  void bookME(DQMStore::IBooker&,
              ObjME& me,
              const std::string& histname,
              const std::string& histtitle,
              const std::vector<double>& binningX,
              const std::vector<double>& binningY);
  void setMETitle(ObjME& me, const std::string& titleX, const std::string& titleY);

protected:
private:
};  //class

template <typename... Args>
void TriggerDQMBase::ObjME::fill(const bool fill_num, Args... args) {
  if (denominator) {
    denominator->Fill(args...);
  }

  if (fill_num and numerator) {
    numerator->Fill(args...);
  }
}

#endif  //DQMOffline_Trigger_TriggerDQMBase_h
