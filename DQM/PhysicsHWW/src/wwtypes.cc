#include <iostream>
#include <cmath>

#include "DQM/PhysicsHWW/interface/wwtypes.h"
using namespace std;

namespace HWWFunctions {

  const char* HypothesisTypeName(HypothesisType type){
    switch (type){
    case MM:
      return "mm";
      break;
    case ME:
      return "me";
      break;
    case EM:
      return "em";
      break;
    case EE:
      return "ee";
      break;
    case ALL:
      return "all";
      break;
    }
    edm::LogError("InvalidInput") << "ERROR: unknown hypothesis name is requested: " << type;
    return "";
  }

  const char* HypothesisTypeName(unsigned int type){
    return HypothesisTypeName( HypothesisType(type) );
  }

  HypothesisType getHypothesisType( HypTypeInNtuples type ){
    switch (type){
    case ElEl:
      return EE;
      break;
    case MuMu:
      return MM;
      break;
    case ElMu: 
      return EM;
      break;
    case MuEl:
      return ME;
      break;
    }
    return EM; //to stop warnings
  }

  HypothesisType getHypothesisType( unsigned int type ){
    return getHypothesisType( HypTypeInNtuples(type) );
  }

  HypothesisType getHypothesisTypeNew(HWW& hww, unsigned int i_hyp ){
    if (abs(hww.hyp_lt_id().at(i_hyp))==11 && abs(hww.hyp_ll_id().at(i_hyp))==11) {
      return EE;
    } else if (abs(hww.hyp_lt_id().at(i_hyp))==13 && abs(hww.hyp_ll_id().at(i_hyp))==13) {
      return MM;
    } else {
      if (hww.hyp_lt_p4().at(i_hyp).pt()>hww.hyp_ll_p4().at(i_hyp).pt()) {
        if (abs(hww.hyp_lt_id().at(i_hyp))==11) return EM;
        else return ME;
      } else {
        if (abs(hww.hyp_lt_id().at(i_hyp))==11) return ME;
        else return EM;
      }
    }
    //return MM;
  }

}
