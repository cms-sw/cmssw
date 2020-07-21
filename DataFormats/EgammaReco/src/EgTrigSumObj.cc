#include "DataFormats/EgammaReco/interface/EgTrigSumObj.h"
 
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
// namespace reco{
//   class RecoEcalCandidate{
//   public:
//     float et()const{return 0.;}
//     float pt()const{return 0.;}
//     float eta()const{return 0.;}
//     float phi()const{return 0.;}
//     float energy()const{return 0.;}
//     reco::SuperClusterRef superCluster()const{return reco::SuperClusterRef();}
//   };
// }
  //#include <iostream>
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

reco::EgTrigSumObj::EgTrigSumObj(float energy,float pt,float eta,float phi):
  energy_(energy),
  pt_(pt),
  eta_(eta),
  phi_(phi),
  hasPixelMatch_(false)
{

}

reco::EgTrigSumObj::EgTrigSumObj(const reco::RecoEcalCandidate& ecalCand):
  energy_(ecalCand.energy()),
  pt_(ecalCand.pt()),
  eta_(ecalCand.eta()),
  phi_(ecalCand.phi()),
  hasPixelMatch_(false),
  superCluster_(ecalCand.superCluster())
{

}

void reco::EgTrigSumObj::setSeeds(const reco::ElectronSeedRefVector&& seeds)
{
  seeds_ = seeds;
  hasPixelMatch_ = false;
  for(const auto& seed : seeds_){
    if(!seed->hitInfo().empty()){
      hasPixelMatch_ = true;
      break;
    }
  }
}

bool reco::EgTrigSumObj::hasVar(const std::string& varName)const
{
  return vars_.find(varName)!=vars_.end();
}

float reco::EgTrigSumObj::var(const std::string& varName,const bool raiseExcept)const
{
  auto varIt = vars_.find(varName);
  if(varIt!=vars_.end()) return varIt->second;
  else if(raiseExcept){
    cms::Exception ex("AttributeError");
    ex <<" error variable "<<varName<<" is not present, variables present are "<<varNamesStr();
    throw ex;
  }else{
    return std::numeric_limits<float>::max();
  }
}

std::vector<std::string> reco::EgTrigSumObj::varNames()const
{
  std::vector<std::string> names;
  for(const auto & var : vars_){
    names.push_back(var.first);
  }
  std::sort(names.begin(),names.end());
  return names;
}

std::string reco::EgTrigSumObj::varNamesStr()const
{
  std::string retVal;
  auto names = varNames();
  for(const auto& name : names){
    if(!retVal.empty()) retVal+=" ";
    retVal +=name;
  }
  return retVal;
}

void reco::EgTrigSumObj::setVar(std::string name,float value,bool overwrite)
{
  auto res = vars_.emplace(std::make_pair(std::move(name),value));
  if(!res.second){ //insertion failed as already exists
    if(overwrite){
      res.first->second = value;
    }else{
      throw cms::Exception("VarError") << "error, value "<<res.first->first<<" already exists with value "<<res.first->second<<" and overwrite is set to false"<<std::endl;
    }
  }
}

