#ifndef DQMOFFLINE_TRIGGER_EGHLTMONELEMWITHCUTEBEE
#define DQMOFFLINE_TRIGGER_EGHLTMONELEMWITHCUTEBEE

//struct: MonElemWithEBEE (Monitor Element Manger With Cut Barrel and Endcap)
//
//author: Sam Harper (July 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim:  a monitor element which seperates transparently the objects into barrel and endcap
//
//implimentation: class simply has two MonElemWithCuts, one for endcap electrons, one for barrel electrons
//                and fills them approprately. It assumes that the class passed in has a detEta function 
//                and uses 1.5 as the barrel,endcap descriminate
//       
//      

#include "DQMOffline/Trigger/interface/EgHLTMonElemWithCut.h"

namespace egHLT {
  template<class T,typename varType> class MonElemWithCutEBEE : public MonElemWithCutBase<T>{
  private:
    MonElemWithCut<T,varType> barrel_;
    MonElemWithCut<T,varType> endcap_;
    
  public: 
    MonElemWithCutEBEE(DQMStore::IBooker &iBooker,const std::string& name,const std::string& title,int nrBins,float min,float max,
		       varType (T::*varFunc)()const):
      barrel_(iBooker,name+"_eb","Barrel "+title,nrBins,min,max,varFunc,NULL),
      endcap_(iBooker,name+"_ee","Endcap "+title,nrBins,min,max,varFunc,NULL){}
    
    MonElemWithCutEBEE(DQMStore::IBooker &iBooker,const std::string& name,const std::string& title,int nrBins,float min,float max,
		       varType (T::*varFunc)()const,const EgHLTDQMCut<T>* cut):
      barrel_(iBooker,name+"_eb","Barrel "+title,nrBins,min,max,varFunc,cut),
      endcap_(iBooker,name+"_ee","Endcap "+title,nrBins,min,max,varFunc,cut ? cut->clone() : NULL){}
    ~MonElemWithCutEBEE(){}
    
    void fill(const T& obj,const OffEvt& evt,float weight);
    
  };
}

template<class T,typename varType> void egHLT::MonElemWithCutEBEE<T,varType>::fill(const T& obj,const OffEvt& evt,float weight)
{
  if(fabs(obj.detEta())<1.5) barrel_.fill(obj,evt,weight);
  else endcap_.fill(obj,evt,weight);
}


#endif
