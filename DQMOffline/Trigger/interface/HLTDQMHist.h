#ifndef DQMOffline_Trigger_HLTDQMHist_h
#define DQMOffline_Trigger_HLTDQMHist_h

//********************************************************************************
//
// Description:
//   A MonitorElement together with the necessary information to fill it when pass an
//   object and minimal selection cuts. These selection cuts are intended to be
//   simple selections on kinematic variables. There are two levels of these
//   selection cuts, global (which apply to all MonitorElements in the group and passed
//   by the calling function) and local (which are stored with the MonitorElement
//   and are specific to that MonitorElement). Global selection cuts on the variable being
//   plotted are ignored, for example Et cuts are removed for turn on plots
//
// Implementation:
//   std::function holds the function which generates the vs variable
//   the name of the vs variable is also stored so we can determine if we should not apply
//   a given selection cut
//   selection is done by VarRangeCutColl
//
// Author : Sam Harper , RAL, May 2017
//
//***********************************************************************************

#include "DQMOffline/Trigger/interface/FunctionDefs.h"
#include "DQMOffline/Trigger/interface/VarRangeCutColl.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

//our base class for our MonitorElements
//takes an object, edm::Event,edm::EventSetup and fills the MonitorElement
//with the predetermined variable (or variables)
template <typename ObjType>
class HLTDQMHist {
public:
  HLTDQMHist() = default;
  virtual ~HLTDQMHist() = default;
  virtual void fill(const ObjType& objType,
                    const edm::Event& event,
                    const edm::EventSetup& setup,
                    const VarRangeCutColl<ObjType>& rangeCuts) = 0;
};

//this class is a specific implimentation of a HLTDQMHist
//it has the value with which to fill the MonitorElement
//and the MonitorElement itself
//we do not own the MonitorElement
template <typename ObjType, typename ValType>
class HLTDQMHist1D : public HLTDQMHist<ObjType> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;

  HLTDQMHist1D(MonitorElement* me_ptr,
               std::string varName,
               std::function<ValType(const ObjType&)> func,
               VarRangeCutColl<ObjType> rangeCuts)
      : var_(std::move(func)), varName_(std::move(varName)), localRangeCuts_(std::move(rangeCuts)), me_ptr_(me_ptr) {}

  void fill(const ObjType& obj,
            const edm::Event& event,
            const edm::EventSetup& setup,
            const VarRangeCutColl<ObjType>& globalRangeCuts) override {
    if (me_ptr_ == nullptr)
      return;
    //local range cuts are specific to a MonitorElement so dont ignore variables
    //like global ones (all local cuts should be appropriate to the MonitorElement in question
    if (globalRangeCuts(obj, varName_) && localRangeCuts_(obj)) {
      me_ptr_->Fill(var_(obj));
    }
  }

private:
  std::function<ValType(const ObjType&)> var_;
  std::string varName_;
  VarRangeCutColl<ObjType> localRangeCuts_;
  MonitorElement* const me_ptr_;  // we do not own this
};

template <typename ObjType, typename XValType, typename YValType = XValType>
class HLTDQMHist2D : public HLTDQMHist<ObjType> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;

  HLTDQMHist2D(MonitorElement* me_ptr,
               std::string xVarName,
               std::string yVarName,
               std::function<XValType(const ObjType&)> xFunc,
               std::function<YValType(const ObjType&)> yFunc,
               VarRangeCutColl<ObjType> rangeCuts)
      : xVar_(std::move(xFunc)),
        yVar_(std::move(yFunc)),
        xVarName_(std::move(xVarName)),
        yVarName_(std::move(yVarName)),
        localRangeCuts_(std::move(rangeCuts)),
        me_ptr_(me_ptr) {}

  void fill(const ObjType& obj,
            const edm::Event& event,
            const edm::EventSetup& setup,
            const VarRangeCutColl<ObjType>& globalRangeCuts) override {
    if (me_ptr_ == nullptr)
      return;
    //local range cuts are specific to a MonitorElement so dont ignore variables
    //like global ones (all local cuts should be appropriate to the MonitorElement in question
    if (globalRangeCuts(obj, std::vector<std::string>{xVarName_, yVarName_}) && localRangeCuts_(obj)) {
      me_ptr_->Fill(xVar_(obj), yVar_(obj));
    }
  }

private:
  std::function<XValType(const ObjType&)> xVar_;
  std::function<YValType(const ObjType&)> yVar_;
  std::string xVarName_;
  std::string yVarName_;
  VarRangeCutColl<ObjType> localRangeCuts_;
  MonitorElement* const me_ptr_;  // we do not own this
};

#endif  // DQMOffline_Trigger_HLTDQMHist_h
