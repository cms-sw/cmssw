#ifndef DQMOnline_Trigger_HLTDQMHist_h
#define DQMOnline_Trigger_HLTDQMHist_h

//********************************************************************************
//
// Description:
//   A histogram together with the necessary information to fill it when pass an
//   object and minimal selection cuts. These selection cuts are intended to be
//   simple selections on kinematic variables. There are two levels of these 
//   selection cuts, global (which apply to all histograms in the group and passed
//   by the calling function) and   local (which are stored with the histogram
//   and are specific to that histogram. Global selection cuts on the variable being 
//   plotted are ignored, for example Et cuts are removed for turn on plots
//   
// Implimentation:
//   std::function holds the function which generates the vs variable
//   the name of the vs variable is also stored so we can determine if we should not apply
//   a given selection cut
//   selection is done by VarRangeCutColl
//   
// Author : Sam Harper , RAL, May 2017
//
//***********************************************************************************

#include "DQMOffline/Trigger/interface/FunctionDefs.h"

//our base class for our histograms
//takes an object, edm::Event,edm::EventSetup and fills the histogram
//with the predetermined variable (or varaibles) 
template <typename ObjType> 
class HLTDQMHist {
public:
  HLTDQMHist()=default;
  virtual ~HLTDQMHist()=default;
  virtual void fill(const ObjType& objType,const edm::Event& event,
		    const edm::EventSetup& setup,const VarRangeCutColl<ObjType>& rangeCuts)=0;
};


//this class is a specific implimentation of a HLTDQMHist
//it has the value with which to fill the histogram 
//and the histogram itself
//we do not own the histogram
template <typename ObjType,typename ValType> 
class HLTDQMHist1D : public HLTDQMHist<ObjType> {
public:
  HLTDQMHist1D(TH1* hist,std::string  varName,
	       std::function<ValType(const ObjType&)> func,
	       VarRangeCutColl<ObjType>  rangeCuts):
    var_(std::move(func)),varName_(std::move(varName)),localRangeCuts_(std::move(rangeCuts)),hist_(hist){}

  void fill(const ObjType& obj,const edm::Event& event,
	    const edm::EventSetup& setup,const VarRangeCutColl<ObjType>& globalRangeCuts)override{
    //local range cuts are specific to a histogram so dont ignore variables 
    //like global ones (all local cuts should be approprate to the histogram in question
    if(globalRangeCuts(obj,varName_) && localRangeCuts_(obj)){  
      hist_->Fill(var_(obj));
    }
  }
private:
  std::function<ValType(const ObjType&)> var_;
  std::string varName_;
  VarRangeCutColl<ObjType> localRangeCuts_;
  TH1* hist_; //we do not own this
};

template <typename ObjType,typename XValType,typename YValType=XValType> 
class HLTDQMHist2D : public HLTDQMHist<ObjType> {
public:
  HLTDQMHist2D(TH2* hist,std::string  xVarName,std::string  yVarName,
	       std::function<XValType(const ObjType&)> xFunc,
	       std::function<YValType(const ObjType&)> yFunc,
	       VarRangeCutColl<ObjType>  rangeCuts):
    xVar_(std::move(xFunc)),yVar_(std::move(yFunc)),
    xVarName_(std::move(xVarName)),yVarName_(std::move(yVarName)),
    localRangeCuts_(std::move(rangeCuts)),hist_(hist){}
  
  void fill(const ObjType& obj,const edm::Event& event,
	    const edm::EventSetup& setup,const VarRangeCutColl<ObjType>& globalRangeCuts)override{
    //local range cuts are specific to a histogram so dont ignore variables 
    //like global ones (all local cuts should be approprate to the histogram in question
    if(globalRangeCuts(obj,std::vector<std::string>{xVarName_,yVarName_}) &&
       localRangeCuts_(obj)){ 
      hist_->Fill(xVar_(obj),yVar_(obj));
    }
  }
private:
  std::function<XValType(const ObjType&)> xVar_;
  std::function<YValType(const ObjType&)> yVar_;
  std::string xVarName_;
  std::string yVarName_;
  VarRangeCutColl<ObjType> localRangeCuts_;
  TH2* hist_; //we do not own this
};

#endif
