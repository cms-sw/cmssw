#ifndef DQMOffline_Trigger_GENERICDQM_H
#define DQMOffline_Trigger_GENERICDQM_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};

struct OBJME {
  MonitorElement* numerator;
  MonitorElement* denominator;
  inline void clear(){
    numerator = nullptr;
    denominator = nullptr;
  };
};

class GENERICDQM
{
 public:
  GENERICDQM(){};  
  virtual ~GENERICDQM(){};
  static void fillHistoPSetDescription(edm::ParameterSetDescription & pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription & pset);
  static MEbinning getHistoPSet    (edm::ParameterSet pset);
  static MEbinning getHistoLSPSet  (edm::ParameterSet pset);

 protected:
  void bookME(DQMStore::IBooker &, OBJME& me, const std::string& histname, const std::string& histtitle, int nbins, double xmin, double xmax);
  void bookME(DQMStore::IBooker &, OBJME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX);
  void bookME(DQMStore::IBooker &, OBJME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, OBJME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax);
  void bookME(DQMStore::IBooker &, OBJME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY);
  void setMETitle(OBJME& me, std::string titleX, std::string titleY);

 private:

};//class

#endif //DQMOffline_Trigger_GENERICDQM_H
