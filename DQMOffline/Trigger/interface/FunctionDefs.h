#ifndef DQMOffline_Trigger_FunctionDefs_h
#define DQMOffline_Trigger_FunctionDefs_h

//***************************************************************************
//
// Description: 
//   These are the functions we wish to access via strings
//   The function returns a std::function holding the function 
//   Have to admit, I'm not happy with the implimentation but fine
//   no time to do something better
//
//   There are various issues. The main is the awkward specialisations
//   needed for the different types. Its a nasty hack. Probably
//   can do it cleaner with ROOT dictionaries
//
// Useage:
//   1) define any extra functions you need at the top (seed scEtaFunc as example)
//   2) generic functions applicable to all normal objects are set in 
//      getUnaryFuncFloat (if they are floats, other types will need seperate
//      functions which can be done with this as example
//   3) object specific functions are done with getUnaryFuncExtraFloat
//      by specialising that function approprately for the object
//   4) user should simply call getUnaryFuncFloat()
//
// Author: Sam Harper (RAL) , 2017
//
//***************************************************************************

#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <functional>

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

namespace hltdqm {
  //here we define needed functions that otherwise dont exist
  template<typename ObjType> float scEtaFunc(const ObjType& obj){return obj.superCluster()->eta();}
 
   
  //additional functions specific to a given type (specialised later on)
  template<typename ObjType>
  std::function<float(const ObjType&)> getUnaryFuncExtraFloat(const std::string& varName){
    std::function<float(const ObjType&)> varFunc;
    return varFunc;
  }

  //the generic function to call
  template<typename ObjType>
  std::function<float(const ObjType&)> getUnaryFuncFloat(const std::string& varName){  
    std::function<float(const ObjType&)> varFunc;
    if(varName=="et") varFunc = &ObjType::et;
    else if(varName=="pt") varFunc = &ObjType::pt;
    else if(varName=="eta") varFunc = &ObjType::eta;
    else if(varName=="phi") varFunc = &ObjType::phi;
    else varFunc = getUnaryFuncExtraFloat<ObjType>(varName);
    //check if we never set varFunc and throw an error for anything but an empty input string
    if(!varFunc && !varName.empty()){
      throw cms::Exception("ConfigError") <<"var "<<varName<<" not recognised "<<__FILE__<<","<<__LINE__<<std::endl;
    }
    return varFunc;
  }

  template<>
  std::function<float(const reco::GsfElectron&)> getUnaryFuncExtraFloat<reco::GsfElectron>(const std::string& varName);
  template<>
  std::function<float(const reco::Photon&)> getUnaryFuncExtraFloat<reco::Photon>(const std::string& varName);

}


#endif
