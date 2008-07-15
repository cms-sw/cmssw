#ifndef DQMOFFLINE_TRIGGER_MONELEMMANAGER
#define DQMOFFLINE_TRIGGER_MONELEMMANAGER


//class: MonElemManager, short for MonitorElementManager (note not MonEleManager as Ele might be confused for electron
//
//author: Sam Harper (June 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim: to make MonitorElement objects "fire and forget"
//     specifically it allows to you just add the MonitorElement to a vector containing all
//     your monitor elements to be filled at a certain location so it will be automatically filled
//     at that location and read out
//     it does this by allowing you to specify the function pointer to the member variable you wish to fill at
//     at the time of declaration
//
//implimentation: currently experimental and limited to 1D histograms but will expand later
//                each object, Photon, GsfElectron, is a seperate (templated) class which means
//                that seperate vectors of MonElemManagers are needed for each type
//                however each type has a base class which the various types (int,float,double) 
//                of variable inherit from so dont need seperate vectors


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

template<class T> class MonElemManagerBase {

 private:
  MonitorElement *monElem_; //we own this (or do we, currently I have decided we dont) FIXME


 //disabling copying and assignment as I havnt figured out how I want them to work yet
 //incidently we cant copy a MonitorElement anyway at the moment (and we prob dont want to)
 private:
  MonElemManagerBase(const MonElemManagerBase& rhs){}
  MonElemManagerBase& operator=(const MonElemManagerBase& rhs){return *this;}
 public:
  MonElemManagerBase(std::string name,std::string title,int nrBins,double xMin,double xMax);
  virtual ~MonElemManagerBase();

  MonitorElement* monElem(){return monElem_;}
  const MonitorElement* monElem()const{return monElem_;}
  
  virtual void fill(const T& obj,float weight)=0;


};

template <class T> MonElemManagerBase<T>::MonElemManagerBase(std::string name,std::string title,int nrBins,double xMin,double xMax):
  monElem_(NULL)
{
  DQMStore* dbe = edm::Service<DQMStore>().operator->();
  monElem_ =dbe->book1D(name,title,nrBins,xMin,xMax);
}
  
 
template <class T> MonElemManagerBase<T>::~MonElemManagerBase()
{
  // delete monElem_;
}

//fills the MonitorElement with a member function of class T returning type varType
//warning only valid for 1D hist monitor elements currently
template<class T,typename varType> class MonElemManager : public MonElemManagerBase<T> {
 private:

  varType (T::*varFunc_)()const;


  //disabling copying and assignment as I havnt figured out how I want them to work yet
 private:
  MonElemManager(const MonElemManager& rhs){}
  MonElemManager& operator=(const MonElemManager& rhs){return *this;}
  
 public:
  MonElemManager(std::string name,std::string title,int nrBins,double xMin,double xMax,
		 varType (T::*varFunc)()const):
    MonElemManagerBase<T>(name,title,nrBins,xMin,xMax),
    varFunc_(varFunc){}
  ~MonElemManager();


  void fill(const T& obj,float weight);


};


template<class T,typename varType> void MonElemManager<T,varType>::fill(const T& obj,float weight)
{
  MonElemManagerBase<T>::monElem()->Fill((obj.*varFunc_)(),weight);
}

template<class T,typename varType> MonElemManager<T,varType>::~MonElemManager()
{
 
}

#endif
