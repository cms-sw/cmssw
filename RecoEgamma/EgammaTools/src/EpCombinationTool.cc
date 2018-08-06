#include "RecoEgamma/EgammaTools/interface/EpCombinationTool.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include <cmath>
#include <vector>
#include <vdt/vdtMath.h>


EpCombinationTool::EpCombinationTool(const edm::ParameterSet& iConfig):
  ecalTrkEnergyRegress_(iConfig.getParameter<edm::ParameterSet>("ecalTrkRegressionConfig")),
  ecalTrkEnergyRegressUncert_(iConfig.getParameter<edm::ParameterSet>("ecalTrkRegressionUncertConfig")),
  maxEcalEnergyForComb_(iConfig.getParameter<double>("maxEcalEnergyForComb")),
  minEOverPForComb_(iConfig.getParameter<double>("minEOverPForComb")),
  maxEPDiffInSigmaForComb_(iConfig.getParameter<double>("maxEPDiffInSigmaForComb")),
  maxRelTrkMomErrForComb_(iConfig.getParameter<double>("maxRelTrkMomErrForComb"))
{

}

edm::ParameterSetDescription EpCombinationTool::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.add<edm::ParameterSetDescription>("ecalTrkRegressionConfig",EgammaRegressionContainer::makePSetDescription());
  desc.add<edm::ParameterSetDescription>("ecalTrkRegressionUncertConfig",EgammaRegressionContainer::makePSetDescription());
  desc.add<double>("maxEcalEnergyForComb",200.);
  desc.add<double>("minEOverPForComb",0.025);
  desc.add<double>("maxEPDiffInSigmaForComb",15.);
  desc.add<double>("maxRelTrkMomErrForComb",10.);
  return desc;
}


void EpCombinationTool::setEventContent(const edm::EventSetup& iSetup)
{
  ecalTrkEnergyRegress_.setEventContent(iSetup);
  ecalTrkEnergyRegressUncert_.setEventContent(iSetup);
}

std::pair<float, float> EpCombinationTool::combine(const reco::GsfElectron& ele)const
{
  return combine(ele,ele.correctedEcalEnergyError());
}

//when doing the E/p combination, its very important to ensure the ecalEnergyErr
//that the regression is trained on is used, not the actual ecalEnergyErr of the electron
//these differ when you correct the ecalEnergyErr by smearing value needed to get data/MC to agree 
std::pair<float, float> EpCombinationTool::combine(const reco::GsfElectron& ele,const float corrEcalEnergyErr)const
{
  const float scRawEnergy = ele.superCluster()->rawEnergy(); 
  const float esEnergy = ele.superCluster()->preshowerEnergy();
  

  const float corrEcalEnergy = ele.correctedEcalEnergy();
  const float ecalMean = ele.correctedEcalEnergy() / (scRawEnergy+esEnergy);
  const float ecalSigma =  corrEcalEnergyErr / corrEcalEnergy;

  auto gsfTrk = ele.gsfTrack();

  const float trkP = gsfTrk->pMode();
  const float trkEta = gsfTrk->etaMode();
  const float trkPhi = gsfTrk->phiMode();
  const float trkPErr = std::abs(gsfTrk->qoverpModeError())*trkP*trkP; 
  const float eOverP = corrEcalEnergy/trkP;
  const float fbrem = ele.fbrem();

  if(corrEcalEnergy < maxEcalEnergyForComb_ &&
     eOverP > minEOverPForComb_ && 
     std::abs(corrEcalEnergy - trkP) < maxEPDiffInSigmaForComb_*std::sqrt(trkPErr*trkPErr+corrEcalEnergyErr*corrEcalEnergyErr) && 
     trkPErr < maxRelTrkMomErrForComb_*trkP) { 

    std::array<float, 9> eval;  
    eval[0] = corrEcalEnergy;
    eval[1] = ecalSigma/ecalMean;
    eval[2] = trkPErr/trkP;
    eval[3] = eOverP;
    eval[4] = ele.ecalDrivenSeed();
    eval[5] = ele.full5x5_showerShape().r9;
    eval[6] = fbrem;
    eval[7] = trkEta; 
    eval[8] = trkPhi;  

    const float preCombinationEt = corrEcalEnergy/std::cosh(trkEta);
    float mean  = ecalTrkEnergyRegress_(preCombinationEt,ele.isEB(),ele.nSaturatedXtals()!=0,eval.data());
    float sigma  = ecalTrkEnergyRegressUncert_(preCombinationEt,ele.isEB(),ele.nSaturatedXtals()!=0,eval.data());
    // Final correction
    // A negative energy means that the correction went
    // outside the boundaries of the training. In this case uses raw.
    // The resolution estimation, on the other hand should be ok.
    if (mean < 0.) mean = 1.0;
    
    //why this differs from the defination of corrEcalEnergyErr (it misses the mean) is not clear to me
    //still this is a direct port from EGExtraInfoModifierFromDB, potential bugs and all
    const float ecalSigmaTimesRawEnergy = ecalSigma*(scRawEnergy+esEnergy);
    const float rawCombEnergy = ( corrEcalEnergy*trkPErr*trkPErr +
				  trkP*ecalSigmaTimesRawEnergy*ecalSigmaTimesRawEnergy ) /
                                ( trkPErr*trkPErr +
				  ecalSigmaTimesRawEnergy*ecalSigmaTimesRawEnergy );

    return std::make_pair(mean*rawCombEnergy,sigma*rawCombEnergy);
  }else{
    return std::make_pair(corrEcalEnergy, corrEcalEnergyErr);
  }
}
