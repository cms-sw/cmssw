#ifndef DQMOffline_Trigger_TriggerDQMBase_H
#define DQMOffline_Trigger_TriggerDQMBase_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class TriggerDQMBase
{
 public:
  TriggerDQMBase(){};  
  virtual ~TriggerDQMBase(){};

  struct MEbinning {
    unsigned nbins;
    double xmin;
    double xmax;
  };
  
  struct ObjME {
    MonitorElement* numerator;
    MonitorElement* denominator;
    inline void clear(){
      numerator = nullptr;
      denominator = nullptr;
    };
  };

  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);
  static MEbinning getHistoPSet    (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

 protected:
  void bookME(DQMStore::IBooker &, ObjME& me, const std::string& histname, const std::string& histtitle, unsigned nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, ObjME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, ObjME& me, const std::string& histname, const std::string& histtitle, unsigned nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, ObjME& me, const std::string& histname, const std::string& histtitle, unsigned nbinsX, double xmin, double xmax, unsigned nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, ObjME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setMETitle(ObjME& me, const std::string& titleX, const std::string& titleY);

 private:

};//class

#endif //DQMOffline_Trigger_TriggerDQMBase_H
