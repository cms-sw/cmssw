#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


HistoProviderDQM::HistoProviderDQM(const std::string& prefix, const std::string& label, DQMStore::IBooker & ibook): ibook_(ibook){
  //  get the store
  label_ =prefix+"/"+label;
  setDir(label_);
}

void HistoProviderDQM::show(){
  //ibook_.showDirStructure();
  std::cout<<"No showDirStructure() avaialble with DQMStore::IBooker: obsolete method!"<<std::endl;
}


void HistoProviderDQM::setDir(const std::string& name){
  ibook_.setCurrentFolder(name);
}

MonitorElement* HistoProviderDQM::book1D(const std::string &name,
                                const std::string &title,
                                const int& nchX, const double& lowX, const double& highX) {
  return (ibook_.book1D (name, title, nchX,lowX,highX));

}


MonitorElement* HistoProviderDQM::book1D (const std::string &name,
                                 const std::string &title,
                                 const int& nchX, float *xbinsize){
  return (ibook_.book1D (name, title,nchX, xbinsize));
}        

MonitorElement* HistoProviderDQM::book2D(const std::string &name,
                                const std::string &title,
                                const int& nchX, const double& lowX, const double& highX,
                                const int& nchY, const double& lowY, const double& highY) {
  return (ibook_.book2D (name, title, nchX,lowX,highX, nchY, lowY, highY));

}


MonitorElement* HistoProviderDQM::book2D (const std::string &name,
                                 const std::string &title,
                                 const int& nchX, float *xbinsize,
                                 const int& nchY, float *ybinsize){
  return (ibook_.book2D (name, title,nchX, xbinsize, nchY, ybinsize));
}
        
MonitorElement* HistoProviderDQM::bookProfile(const std::string &name,
                                const std::string &title,
                                int nchX, double lowX, double highX,
                                int nchY, double lowY, double highY) {
  return (ibook_.bookProfile (name,title, nchX,lowX,highX, nchY, lowY, highY));

}

MonitorElement * HistoProviderDQM::access(const std::string &name){
  //return   ibook_.get(label_+"/"+name);   
  std::cout<<"'get' forbidden with DQMStore::IBooker: obsolete method!"<<std::endl;
  return 0;
}

