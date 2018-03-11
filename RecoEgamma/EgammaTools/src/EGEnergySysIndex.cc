#include "RecoEgamma/EgammaTools/interface/EGEnergySysIndex.h"

namespace{
  std::array<std::string,EGEnergySysIndex::kNrSysErrs> makeEGEnergySysNames(){
    std::array<std::string,EGEnergySysIndex::kNrSysErrs> names;
    names[EGEnergySysIndex::kScaleUp] = "EGMscaleUpUncertainty"; 
    names[EGEnergySysIndex::kScaleDown] = "EGMscaleDownUncertainty"; 
    names[EGEnergySysIndex::kScaleStatUp] = "EGMscaleStatUpUncertainty"; 
    names[EGEnergySysIndex::kScaleStatDown] = "EGMscaleStatDownUncertainty"; 
    names[EGEnergySysIndex::kScaleSystUp] = "EGMscaleSystUpUncertainty"; 
    names[EGEnergySysIndex::kScaleSystDown] = "EGMscaleSystDownUncertainty"; 
    names[EGEnergySysIndex::kScaleGainUp] = "EGMscaleGainUpUncertainty"; 
    names[EGEnergySysIndex::kScaleGainDown] = "EGMscaleGainDownUncertainty"; 
    names[EGEnergySysIndex::kSmearUp] = "EGMresolutionUpUncertainty"; 
    names[EGEnergySysIndex::kSmearDown] = "EGMresolutionDownUncertainty"; 
    names[EGEnergySysIndex::kSmearRhoUp] = "EGMresolutionRhoUpUncertainty"; 
    names[EGEnergySysIndex::kSmearRhoDown] = "EGMresolutionRhoDownUncertainty"; 
    names[EGEnergySysIndex::kSmearPhiUp] = "EGMresolutionPhiUpUncertainty"; 
    names[EGEnergySysIndex::kSmearPhiDown] = "EGMresolutionPhiDownUncertainty"; 
    names[EGEnergySysIndex::kScaleValue] = "EGMscale"; 
    names[EGEnergySysIndex::kSmearValue] = "EGMsmear"; 
    names[EGEnergySysIndex::kSmearNrSigma] = "EGMsmearNrSigma"; 
    names[EGEnergySysIndex::kEcalPreCorr] = "EGMecalEnergyPreCorr"; 
    names[EGEnergySysIndex::kEcalErrPreCorr] = "EGMecalEnergyErrPreCorr"; 
    names[EGEnergySysIndex::kEcalPostCorr] = "EGMecalEnergy"; 
    names[EGEnergySysIndex::kEcalErrPostCorr] = "EGMecalEnergyErr"; 
    names[EGEnergySysIndex::kEcalTrkPreCorr] = "EGMecalTrkEnergyPreCorr"; 
    names[EGEnergySysIndex::kEcalTrkErrPreCorr] = "EGMecalTrkEnergyErrPreCorr";  
    names[EGEnergySysIndex::kEcalTrkPostCorr] = "EGMecalTrkEnergy"; 
    names[EGEnergySysIndex::kEcalTrkErrPostCorr] = "EGMecalTrkEnergyErr";
    return names;
  }
}


const std::array<std::string,EGEnergySysIndex::kNrSysErrs> 
EGEnergySysIndex::names_ = makeEGEnergySysNames();

