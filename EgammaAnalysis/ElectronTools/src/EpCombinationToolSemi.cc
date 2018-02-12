#include "EgammaAnalysis/ElectronTools/interface/EpCombinationToolSemi.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include <math.h>
#include <vector>
#include <iostream>
#include <vdt/vdtMath.h>

using namespace std;

/*****************************************************************/
EpCombinationToolSemi::EpCombinationToolSemi()
/*****************************************************************/
{}



/*****************************************************************/
EpCombinationToolSemi::~EpCombinationToolSemi()
/*****************************************************************/
{}


bool EpCombinationToolSemi::init(const std::vector<const GBRForestD*> forest) 
{

  // magic numbers
  meanlimlow  = -1.0;
  meanlimhigh = 3.0;
  meanoffset  = meanlimlow + 0.5*(meanlimhigh-meanlimlow);
  meanscale   = 0.5*(meanlimhigh-meanlimlow);

  sigmalimlow  = 0.0002;
  sigmalimhigh = 0.5;
  sigmaoffset  = sigmalimlow + 0.5*(sigmalimhigh-sigmalimlow);
  sigmascale   = 0.5*(sigmalimhigh-sigmalimlow);  
 
  // more magic numbers
  lowEnergyThr  = 50.;
  highEnergyThr = 200.;
  eOverPThr     = 0.025;
  epDiffSigThr  = 15.;
  epSigThr      = 10.;
 
  m_forest = forest;
  m_ownForest = true;
  return true;

}


/*****************************************************************/
std::pair<float, float> EpCombinationToolSemi::combine(reco::GsfElectron& electron) const
/*****************************************************************/
{

  float combinedEnergy = electron.correctedEcalEnergy();
  float combinedEnergyError = electron.correctedEcalEnergyError();

  if(!m_ownForest)
    {
      cout<<"ERROR: The combination tool is not initialized\n I will not do anything\n";
      return std::make_pair(combinedEnergy, combinedEnergyError);
    }

  auto el_track = electron.gsfTrack();

  const float trkMomentum      = el_track->pMode();
  const float trkEta           = el_track->etaMode();
  const float trkPhi           = el_track->phiMode();
  const float eOverP           = combinedEnergy/trkMomentum;
  const float fbrem            = electron.fbrem();

  const float ptMode       = el_track->ptMode();
  const float ptModeErrror = el_track->ptModeError();
  const float etaModeError = el_track->etaModeError();
  const float pModeError   = sqrt(ptModeErrror*ptModeErrror*cosh(trkEta)*cosh(trkEta) + ptMode*ptMode*sinh(trkEta)*sinh(trkEta)*etaModeError*etaModeError);

  const float trkMomentumError = pModeError;

  unsigned int coridx = 0;

  if (combinedEnergy < highEnergyThr &&
      eOverP > eOverPThr && 
      std::abs(combinedEnergy - trkMomentum) < epDiffSigThr*std::sqrt(trkMomentumError*trkMomentumError+combinedEnergyError*combinedEnergyError) && 
      trkMomentumError < epSigThr*trkMomentum) { 

    bool iseb = electron.isEB();
    float raw_pt = combinedEnergy/cosh(trkEta);

    if (iseb && raw_pt < lowEnergyThr)
      coridx = 0;
    else if (iseb && raw_pt >= lowEnergyThr)
      coridx = 1;
    else if (!iseb && raw_pt < lowEnergyThr)
      coridx = 2;
    else if (!iseb && raw_pt >= lowEnergyThr)
      coridx = 3;

    std::array<float, 9> eval;  
    eval[0] = combinedEnergy;
    eval[1] = combinedEnergyError/combinedEnergy;
    eval[2] = trkMomentumError/trkMomentum;
    eval[3] = eOverP;
    eval[4] = electron.ecalDrivenSeed();
    eval[5] = electron.full5x5_showerShape().r9;
    eval[6] = fbrem;
    eval[7] = trkEta; 
    eval[8] = trkPhi; 

    float rawcomb = ( combinedEnergy*trkMomentumError*trkMomentumError + trkMomentum*combinedEnergyError*combinedEnergyError ) / ( trkMomentumError*trkMomentumError + combinedEnergyError*combinedEnergyError );

    float rawmean_trk = m_forest[coridx]->GetResponse(eval.data());
    float rawsigma_trk = m_forest[coridx+4]->GetResponse(eval.data());
    
    float mean_trk = meanoffset + meanscale*vdt::fast_sin(rawmean_trk);
    float sigma_trk = sigmaoffset + sigmascale*vdt::fast_sin(rawsigma_trk);

    // Final correction
    // A negative energy means that the correction went
    // outside the boundaries of the training. In this case uses raw.
    // The resolution estimation, on the other hand should be ok.
    if (mean_trk < 0.) mean_trk = 1.0;
    
    combinedEnergy = mean_trk*rawcomb;
    combinedEnergyError = sigma_trk*rawcomb;

  }

  return std::make_pair(combinedEnergy, combinedEnergyError);
}
