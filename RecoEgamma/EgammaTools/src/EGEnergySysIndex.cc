#include "RecoEgamma/EgammaTools/interface/EGEnergySysIndex.h"

namespace{
  std::array<std::string,EGEnergySysIndex::kNrSysErrs> makeEGEnergySysNames(){
    std::array<std::string,EGEnergySysIndex::kNrSysErrs> names;
    names[EGEnergySysIndex::kScaleUp] = "energyScaleUp"; 
    names[EGEnergySysIndex::kScaleDown] = "energyScaleDown"; 
    names[EGEnergySysIndex::kScaleStatUp] = "energyScaleStatUp"; 
    names[EGEnergySysIndex::kScaleStatDown] = "energyScaleStatDown"; 
    names[EGEnergySysIndex::kScaleSystUp] = "energyScaleSystUp"; 
    names[EGEnergySysIndex::kScaleSystDown] = "energyScaleSystDown"; 
    names[EGEnergySysIndex::kScaleGainUp] = "energyScaleGainUp"; 
    names[EGEnergySysIndex::kScaleGainDown] = "energyScaleGainDown"; 
    names[EGEnergySysIndex::kSmearUp] = "energySigmaUp"; 
    names[EGEnergySysIndex::kSmearDown] = "energySigmaDown"; 
    names[EGEnergySysIndex::kSmearRhoUp] = "energySigmaRhoUp"; 
    names[EGEnergySysIndex::kSmearRhoDown] = "energySigmaRhoDown"; 
    names[EGEnergySysIndex::kSmearPhiUp] = "energySigmaPhiUp"; 
    names[EGEnergySysIndex::kSmearPhiDown] = "energySigmaPhiDown"; 
    names[EGEnergySysIndex::kScaleValue] = "energyScaleValue"; 
    names[EGEnergySysIndex::kSmearValue] = "energySigmaValue"; 
    names[EGEnergySysIndex::kSmearNrSigma] = "energySmearNrSigma"; 
    names[EGEnergySysIndex::kEcalPreCorr] = "ecalEnergyPreCorr"; 
    names[EGEnergySysIndex::kEcalErrPreCorr] = "ecalEnergyErrPreCorr"; 
    names[EGEnergySysIndex::kEcalPostCorr] = "ecalEnergyPostCorr"; 
    names[EGEnergySysIndex::kEcalErrPostCorr] = "ecalEnergyErrPostCorr"; 
    names[EGEnergySysIndex::kEcalTrkPreCorr] = "ecalTrkEnergyPreCorr"; 
    names[EGEnergySysIndex::kEcalTrkErrPreCorr] = "ecalTrkEnergyErrPreCorr";  
    names[EGEnergySysIndex::kEcalTrkPostCorr] = "ecalTrkEnergyPostCorr"; 
    names[EGEnergySysIndex::kEcalTrkErrPostCorr] = "ecalTrkEnergyErrPostCorr";
    return names;
  }
}


const std::array<std::string,EGEnergySysIndex::kNrSysErrs> 
EGEnergySysIndex::names_ = makeEGEnergySysNames();

