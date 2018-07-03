#ifndef DQMOFFLINE_TRIGGER_EGHLTMONELEMMGRHEP
#define DQMOFFLINE_TRIGGER_EGHLTMONELEMMGRHEP


#include "DQMOffline/Trigger/interface/EgHLTMonElemManager.h"

namespace egHLT {
  template<class T,typename varType> class MonElemMgrHEP : public MonElemManagerBase<T>{
  private:
    MonElemManager<T,varType> hep17_;
    MonElemManager<T,varType> hem17_;
    
  public: 
    MonElemMgrHEP(DQMStore::IBooker &iBooker, const std::string& name,const std::string& title,int nrBins,float min,float max,varType (T::*varFunc)()const):
      hep17_(iBooker, name+"_hep17","hep "+title,nrBins,min,max,varFunc),
      hem17_(iBooker, name+"_hem17","hem "+title,nrBins,min,max,varFunc){}
    
    ~MonElemMgrHEP() override{}
    
    void fill(const T& obj,float weight) override;
    
  };
  
  template<class T,typename varType> void MonElemMgrHEP<T,varType>::fill(const T& obj,float weight)
  {
    if(obj.detEta()<3.0 && obj.detEta()>1.3 && obj.phi()< -0.52 && obj.phi()>-0.87) hep17_.fill(obj,weight);
    if(obj.detEta()>-3.0 && obj.detEta()<-1.3 && obj.phi()< -0.52 && obj.phi() >-0.87) hem17_.fill(obj,weight);
  }
  




  template<class T,typename varTypeX,typename varTypeY> class MonElemMgr2DHEP : public MonElemManagerBase<T>{
    
  private:
    MonElemManager2D<T,varTypeX,varTypeY> hep17_;
    MonElemManager2D<T,varTypeX,varTypeY> hem17_;
    
  public:
    MonElemMgr2DHEP(DQMStore::IBooker &iBooker, const std::string& name,const std::string& title,int nrBinsX,double xMin,double xMax,int nrBinsY,double yMin,double yMax,
		     varTypeX (T::*varFuncX)()const,varTypeY (T::*varFuncY)()const):
      hep17_(iBooker, name+"_hep17","Hep17 "+title,nrBinsX,xMin,xMax,nrBinsY,yMin,yMax,varFuncX,varFuncY),
      hem17_(iBooker, name+"_hem17","Hem17 "+title,nrBinsX,xMin,xMax,nrBinsY,yMin,yMax,varFuncX,varFuncY){}
    
    ~MonElemMgr2DHEP(){}
    
    void fill(const T& obj,float weight);
    
  };
  
  template<class T,typename varTypeX,typename varTypeY> void MonElemMgr2DHEP<T,varTypeX,varTypeY>::fill(const T& obj,float weight)
  {
    if(obj.detEta()<3.0 && obj.detEta()>1.3 && obj.phi()< -0.52 && obj.phi()>-0.87) hep17_.fill(obj,weight);
    if(obj.detEta()>-3.0 && obj.detEta()<-1.3 && obj.phi()< -0.52 && obj.phi() >-0.87) hem17_.fill(obj,weight);
  }
}
#endif
  
