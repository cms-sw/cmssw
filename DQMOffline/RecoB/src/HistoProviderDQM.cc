#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


HistoProviderDQM::HistoProviderDQM(std::string prefix, std::string label){
  //  get the store
  dqmStore_ = edm::Service<DQMStore>().operator->();
  label_ =prefix+"/"+label;
  setDir(label_);
}

void HistoProviderDQM::show(){
  dqmStore_->showDirStructure();
}


void HistoProviderDQM::setDir(std::string name){
  dqmStore_->setCurrentFolder(name);
}

MonitorElement* HistoProviderDQM::book1D(const TString &name,
                                const TString &title,
                                int nchX, double lowX, double highX) {
  return (dqmStore_->book1D ((const char *)name,(const char *) title, nchX,lowX,highX));

}


MonitorElement* HistoProviderDQM::book1D (const TString &name,
                                 const TString &title,
                                 int nchX, float *xbinsize){
  return (dqmStore_->book1D ((const char *)name, (const char *)title,nchX, xbinsize));
}        

MonitorElement* HistoProviderDQM::book2D(const TString &name,
                                const TString &title,
                                int nchX, double lowX, double highX,
                                int nchY, double lowY, double highY) {
  return (dqmStore_->book2D ((const char *)name,(const char *) title, nchX,lowX,highX, nchY, lowY, highY));

}


MonitorElement* HistoProviderDQM::book2D (const TString &name,
                                 const TString &title,
                                 int nchX, float *xbinsize,
                                 int nchY, float *ybinsize){
  return (dqmStore_->book2D ((const char *)name, (const char *)title,nchX, xbinsize, nchY, ybinsize));
}
        
MonitorElement * HistoProviderDQM::access(const TString &name){
return   dqmStore_->get((const char *)(label_+"/"+name));   
}

