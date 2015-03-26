#ifndef DQMOFFLINE_TRIGGER_EGHLTMONELEMMANAGER
#define DQMOFFLINE_TRIGGER_EGHLTMONELEMMANAGER


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

namespace egHLT {
  template<class T> class MonElemManagerBase {
    
  public:
    MonElemManagerBase(){}
    virtual ~MonElemManagerBase(){}
    
    virtual void fill(const T& obj,float weight)=0;
    
  };

  //this was the orginal base class but then I made a change where I wanted multiple MonElems wraped into a single Manager (ie endcap barrel) so a new base class was designed with interface only
  template<class T> class MonElemManagerHist : public MonElemManagerBase<T> {
    
  private:
    MonitorElement *monElem_; //we own this (or do we, currently I have decided we dont) FIXME
    
    //disabling copying and assignment as I havnt figured out how I want them to work yet
    //incidently we cant copy a MonitorElement anyway at the moment (and we prob dont want to)
  private:
    MonElemManagerHist(const MonElemManagerHist& rhs){}
    MonElemManagerHist& operator=(const MonElemManagerHist& rhs){return *this;}
  public:
    MonElemManagerHist(DQMStore::IBooker &iBooker, std::string name,std::string title,int nrBins,double xMin,double xMax);
    MonElemManagerHist(DQMStore::IBooker &iBooker, std::string name,std::string title,int nrBinsX,double xMin,double xMax,int nrBinsY,double yMin,double yMax);
    virtual ~MonElemManagerHist();
    
    MonitorElement* monElem(){return monElem_;}
    const MonitorElement* monElem()const{return monElem_;}
    
    virtual void fill(const T& obj,float weight)=0;
    
    
  };
  
  template <class T> MonElemManagerHist<T>::MonElemManagerHist(DQMStore::IBooker &iBooker, std::string name,std::string title,int nrBins,double xMin,double xMax):
    monElem_(NULL)
  {
    monElem_ = iBooker.book1D(name,title,nrBins,xMin,xMax);
  }
  
  template <class T> MonElemManagerHist<T>::MonElemManagerHist(DQMStore::IBooker &iBooker, std::string name,std::string title,
							       int nrBinsX,double xMin,double xMax,
							       int nrBinsY,double yMin,double yMax):
    monElem_(NULL)
  {
    monElem_ = iBooker.book2D(name,title,nrBinsX,xMin,xMax,nrBinsY,yMin,yMax);
  }
  
  
  template <class T> MonElemManagerHist<T>::~MonElemManagerHist()
  {
    // delete monElem_;
  }
  
  //fills the MonitorElement with a member function of class T returning type varType
  template<class T,typename varType> class MonElemManager : public MonElemManagerHist<T> {
  private:
    
    varType (T::*varFunc_)()const;
    
    
    //disabling copying and assignment as I havnt figured out how I want them to work yet
  private:
  MonElemManager(const MonElemManager& rhs){}
    MonElemManager& operator=(const MonElemManager& rhs){return *this;}
    
  public:
    MonElemManager(DQMStore::IBooker &iBooker, std::string name,std::string title,int nrBins,double xMin,double xMax,
		   varType (T::*varFunc)()const):
      MonElemManagerHist<T>(iBooker, name,title,nrBins,xMin,xMax),
      varFunc_(varFunc){}
    ~MonElemManager();
    
    
    void fill(const T& obj,float weight);

  };

  
  template<class T,typename varType> void MonElemManager<T,varType>::fill(const T& obj,float weight)
  {
    MonElemManagerHist<T>::monElem()->Fill((obj.*varFunc_)(),weight);
  }
  
  template<class T,typename varType> MonElemManager<T,varType>::~MonElemManager()
  {
    
  }
  
  
  //fills a 2D monitor element with member functions of T returning varType1 and varType2 
  template<class T,typename varTypeX,typename varTypeY=varTypeX> class MonElemManager2D : public MonElemManagerHist<T> {
  private:
    
    varTypeX (T::*varFuncX_)()const;
    varTypeY (T::*varFuncY_)()const;
    
    //disabling copying and assignment as I havnt figured out how I want them to work yet
  private:
    MonElemManager2D(const MonElemManager2D& rhs){}
    MonElemManager2D& operator=(const MonElemManager2D& rhs){return *this;}
 
  public:
    MonElemManager2D(DQMStore::IBooker &iBooker, std::string name,std::string title,int nrBinsX,double xMin,double xMax,int nrBinsY,double yMin,double yMax,
		     varTypeX (T::*varFuncX)()const,varTypeY (T::*varFuncY)()const):
      MonElemManagerHist<T>(iBooker, name,title,nrBinsX,xMin,xMax,nrBinsY,yMin,yMax),
      varFuncX_(varFuncX),varFuncY_(varFuncY){}
    ~MonElemManager2D();
    
    
    void fill(const T& obj,float weight);
    
    
  };
  
  template<class T,typename varTypeX,typename varTypeY> void MonElemManager2D<T,varTypeX,varTypeY>::fill(const T& obj,float weight)
  {
    MonElemManagerHist<T>::monElem()->Fill((obj.*varFuncX_)(),(obj.*varFuncY_)(),weight);
  }
  
  template<class T,typename varTypeX,typename varTypeY> MonElemManager2D<T,varTypeX,varTypeY>::~MonElemManager2D()
  {
    
  }
}


#endif
