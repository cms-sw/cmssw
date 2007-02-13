#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/TrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "TrackingTools/GsfTools/interface/GSUtilities.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university 
// 12/2005
// Stephanie's parametrisation
// 02/2006
//adapted to CMSSW by U.Berthon,LLR Paliaseau,  dec 2006
//===================================================================


void ElectronMomentumCorrector::correct(reco::PixelMatchGsfElectron &electron, TrajectoryStateOnSurface & vtxTsos) {

  if (electron.isMomentumCorrected()) return;

  //steph third sigma
  float parEB[5][3] = {
       { 2.46e-02,  1.97e-01, 5.23e-03},          //golden
       { 9.99e-07,  2.80e-01, 5.69e-03},          //big brem
       { 9.37e-07,  2.32e-01, 5.82e-03},          // narrow
       { 7.30e-02,  1.95e-01, 1.30e-02},          // showering
       { 9.25e-06,  2.84e-01, 8.77e-03}           // nominal --> crack
  }; 

  float parEE[5][3] = {
       { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // golden
       { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // big brem = golden
       { 1.25841e-01, 7.01145e-01, 2.81884e-11},  // narrow = golden
       { 1.63634e-01, 1.11307e+00, 3.64770e-03},  // showering
       {         .02,         .15,        .005}   // nominal --> crack
  };

  errorEnergy_ = 999.;
  errorTrackMomentum_ = 999.;

  int elClass = electron.classification();
  float scEnergy = electron.caloEnergy();

  // first calculate error on energy
  if (elClass == -1) return;   // electron without class, do nothing

  else if (elClass < 50) { // barrel
    errorEnergy_ =  scEnergy * energyError(scEnergy,parEB[int(elClass/10)]);
  } 
  
  else if (elClass >=100) { //endcap
    errorEnergy_ =  scEnergy * energyError(scEnergy,parEE[int(elClass/10)-10]);
  }  
    
  else edm::LogWarning("") <<"Electron without category!!";


  // then retrieve error on track momentum
  //  float trackMomentum  =  electron.getGsfTrack()->impactPointModeMomentum().mag();
  float trackMomentum  = electron.trackMomentumAtVtx().R();
  // momentum error rescaling
  //  std::vector<TrajectoryStateOnSurface> vtx_loc_comp = electron.getGsfTrack()->impactPointState().components();
  std::vector<TrajectoryStateOnSurface> vtx_loc_comp = vtxTsos.components();

  int vtx_loc_oo = 0;	
  float vtx_loc_weight[100];
  float vtx_loc_qop[100];
  float vtx_loc_qop_err[100];
  float vtx_loc_dxdz[100];
  float vtx_loc_dxdz_err[100];
  float vtx_loc_dydz[100];
  float vtx_loc_dydz_err[100];
  float vtx_loc_locx[100];
  float vtx_loc_locx_err[100];
  float vtx_loc_locy[100];
  float vtx_loc_locy_err[100];

  for (std::vector<TrajectoryStateOnSurface>::const_iterator it_vtx_locComp = vtx_loc_comp.begin(); it_vtx_locComp!= vtx_loc_comp.end(); it_vtx_locComp++)
    {
      vtx_loc_weight[vtx_loc_oo]   =  it_vtx_locComp->weight();
      vtx_loc_qop[vtx_loc_oo]      =  it_vtx_locComp->localParameters().vector()[0];      
      vtx_loc_qop_err[vtx_loc_oo]  =  sqrt(it_vtx_locComp->localError().matrix()[0][0]);
      vtx_loc_dxdz[vtx_loc_oo]     =  it_vtx_locComp->localParameters().vector()[1];       
      vtx_loc_dxdz_err[vtx_loc_oo] =  sqrt(it_vtx_locComp->localError().matrix()[1][1]);
      vtx_loc_dydz[vtx_loc_oo]     =  it_vtx_locComp->localParameters().vector()[2];    
      vtx_loc_dydz_err[vtx_loc_oo] =  sqrt(it_vtx_locComp->localError().matrix()[2][2]);
      vtx_loc_locx[vtx_loc_oo]     =  it_vtx_locComp->localParameters().vector()[3];    
      vtx_loc_locx_err[vtx_loc_oo] =  sqrt(it_vtx_locComp->localError().matrix()[3][3]);
      vtx_loc_locy[vtx_loc_oo]     =  it_vtx_locComp->localParameters().vector()[4];    
      vtx_loc_locy_err[vtx_loc_oo] =  sqrt(it_vtx_locComp->localError().matrix()[4][4]); 
      vtx_loc_oo++;   
    }

  GSUtilities my_vtx_loc_GSFUtilitiesQoP(vtx_loc_comp.size(), 
					       vtx_loc_weight, vtx_loc_qop,
					       vtx_loc_qop_err);
  float vtx_loc_most_prob_errorQoP = my_vtx_loc_GSFUtilitiesQoP.errorMode();
//  errorTrackMomentum_ = trackMomentum*trackMomentum*vtx_loc_most_prob_errorQoP*2*2.48980e-01*trackMomentum/25.; 
  errorTrackMomentum_ = trackMomentum*trackMomentum*vtx_loc_most_prob_errorQoP*0.4173; 

  // calculate E/p and corresponding error	
  float eOverP = scEnergy / trackMomentum;
  float errorEOverP = sqrt(
		   (errorEnergy_/trackMomentum)*(errorEnergy_/trackMomentum) + 
		   (scEnergy*errorTrackMomentum_/trackMomentum/trackMomentum)*
		   (scEnergy*errorTrackMomentum_/trackMomentum/trackMomentum));

  // combination

  float finalMomentum = (scEnergy/errorEnergy_/errorEnergy_ + trackMomentum/errorTrackMomentum_/errorTrackMomentum_) /
                       (1/errorEnergy_/errorEnergy_ + 1/errorTrackMomentum_/errorTrackMomentum_);

//  if ( eOverP - 1 > 2*errorEOverP ) finalMomentum = scEnergy;

  if ( eOverP - 1 > 2*errorEOverP ){
    if (int(elClass/10)!=3 && int(elClass/10)!=13 && elClass!=40){
      if (scEnergy<15) finalMomentum = trackMomentum;
      else finalMomentum = scEnergy;
    }
    else finalMomentum = scEnergy;
  }

  if ( eOverP < 1-2*errorEOverP ) {
    if (scEnergy<15) finalMomentum = trackMomentum;
    else finalMomentum = scEnergy;
  }
    
  //  HepLorentzVector oldMomentum = electron.fourMomentum();
  //  newMomentum_ = HepLorentzVector(
    math::XYZTLorentzVector oldMomentum = electron.p4();
    newMomentum_ = math::XYZTLorentzVector(
			      oldMomentum.x()*finalMomentum/oldMomentum.t(),
			      oldMomentum.y()*finalMomentum/oldMomentum.t(),
			      oldMomentum.z()*finalMomentum/oldMomentum.t(),
			      finalMomentum);

    //  electron.correctElectronFourMomentum(this);
    electron.correctElectronFourMomentum(newMomentum_, errorEnergy_ ,errorTrackMomentum_);

}

float ElectronMomentumCorrector::energyError(float E, float *par) const{
  return sqrt( pow(par[0]/sqrt(E),2) + pow(par[1]/E,2) + pow(par[2],2) );
}
