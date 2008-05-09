#ifndef HistoProviderDQM_H
#define HistoProviderDQM_H


class DQMStore;
class MonitorElement;

#include <string>
#include "TString.h"

class HistoProviderDQM  {
 public:
  HistoProviderDQM(std::string prefix, std::string label);
  virtual ~HistoProviderDQM(){}
  void show();

  virtual MonitorElement* book1D       (const TString &name,
                      const TString &title,
                      int nchX, double lowX, double highX) ;
  
  virtual MonitorElement* book1D       (const TString &name,
                      const TString &title,
                      int nchX, float *xbinsize) ;
  void setDir(std::string);

  virtual MonitorElement * access(const TString &name);

 private:
  DQMStore * dqmStore_;
  std::string label_;
};                                                                                                                                                                           
#endif
