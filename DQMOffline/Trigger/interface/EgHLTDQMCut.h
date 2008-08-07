#ifndef DQMOFFLINE_TRIGGER_EGHLTDQMCUT
#define DQMOFFLINE_TRIGGER_EGHLTDQMCUT 

//class: EgHLTDQMCut
//
//author: Sam Harper (June 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim: to allow the user to place a cut on the electron using it or the event
//
//implimentation:


//this is a pure virtual struct which defines the interface to the cut objects
template<class T> struct EgHLTDQMCut { 
  public:
  EgHLTDQMCut(){}
  virtual ~EgHLTDQMCut(){}
  virtual bool pass(const T& obj,const EgHLTOffData& evtData)=0;
};
  
template<class T> struct EgHLTDQMVarCut : public EgHLTDQMCut<T> {
  private:
  int cutsToPass_; //the cuts whose eff we are measuring
  int (T::*cutCodeFunc_)()const;

  public:
  EgHLTDQMVarCut(int cutsToPass,int (T::*cutCodeFunc)()const):cutsToPass_(cutsToPass),cutCodeFunc_(cutCodeFunc){}
  ~EgHLTDQMVarCut(){}

  bool pass(const T& obj,const EgHLTOffData& evtData);
};

//to understand this you need to know about
//1) templates
//2) inheritance (sort of)
//3) function pointers
//4) bitwise operations
//All it does is get the bitword corresponding to the cuts the electron failed and the mask the bits which correspond to cuts we dont care about and then see if any bits are still set
template<class T> bool EgHLTDQMVarCut<T>::pass(const T& obj,const EgHLTOffData& evtData)
{
  if(((obj.*cutCodeFunc_)() & cutsToPass_)==0) return true;
  else return false;
}


#endif
