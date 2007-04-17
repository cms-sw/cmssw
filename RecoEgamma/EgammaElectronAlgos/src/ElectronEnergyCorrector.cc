#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"
//#include "ElectronPhoton/ElectronReco/interface/PRecElectron.h"
//#include "ElectronPhoton/EgammaPreshower/interface/EgammaEndcapCluster.h"
//#include <CLHEP/Vector/LorentzVector.h>

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university 
//         Ivica Puljak - FESB, Split 
// 12/2005
// updated f(eta) corrections from Ivica
// CC 02/2006
// ported to CMSSW by U. Berthon, dec 2006
//===================================================================


void ElectronEnergyCorrector::correct(reco::PixelMatchGsfElectron &electron) {

  if (!electron.isEnergyScaleCorrected()) {
    setNewEnergy(electron);
    //    electron.correctElectronEnergyScale(this);
    electron.correctElectronEnergyScale(newEnergy_);
  }

}

void ElectronEnergyCorrector::setNewEnergy(const reco::PixelMatchGsfElectron &electron) {

  int elClass = electron.classification();
  double scEta = electron.caloPosition().eta();
  double scEnergy = electron.superCluster()->energy();
  newEnergy_ = 0.;

  // decide whether electron is in barrel of endcap
  if (elClass == -1) {
    // electron without class, do nothing
    return;

  } else if (elClass < 50) {
    // barrel

    if (int(elClass/10) == 0 ||  int(elClass/10) == 1 || int(elClass/10) == 2) {
        newEnergy_ = scEnergy/fEtaBarrelGood(scEta);

    } else if (int(elClass/10) == 3) {
        newEnergy_ = scEnergy/fEtaBarrelBad(scEta);

    } else {
	// cracks, no f(eta) correction;
        newEnergy_ = scEnergy;
    }

  } else {
    // endcap, if in crack do nothing, otherwise just correct for eta effect

    //    const EgammaEndcapCluster *ec = dynamic_cast<const EgammaEndcapCluster*>(electron.getSuperCluster());
    //    double ePreshower = ec->energyUncorrected() - ec->energyUncorrected_E();
    //FIXME!!!!!
    // here we apply the correction to the corrected scEnergy and add no preshower energy
    //what should be done once we can get the preshower enregy from the supercluster is
    //subtract the preshower energy from scenergy, apply the correction and add preshower enrgy afterwards
    //the error should be small
    if (int(elClass/10) != 13) {
      //      newEnergy_ = scEnergy/fEtaEndcapGood(scEta)+ePreshower;
      newEnergy_ = scEnergy/fEtaEndcapGood(scEta);

    } else { 
      //      newEnergy_ = scEnergy/fEtaEndcapBad(scEta)+ePreshower;
      newEnergy_ = scEnergy/fEtaEndcapBad(scEta);
    }

  }

}


double ElectronEnergyCorrector::fEtaBarrelBad(double scEta) const{
  
  // f(eta) for the class = 30 (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  9.99063e-01; 
  float p1 = -2.63341e-02; 
  float p2 =  5.16054e-02; 
  float p3 = -4.95976e-02; 
  float p4 =  3.62304e-03; 

  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;  

}
  
double ElectronEnergyCorrector::fEtaEndcapGood(double scEta) const{

  // f(eta) for the first 3 classes (100, 110 and 120) 
  // Ivica's new corrections 01/06
  float p0 =        -8.51093e-01; 
  float p1 =         3.54266e+00;
  float p2 =        -2.59288e+00;
  float p3 =         8.58945e-01;
  float p4 =        -1.07844e-01; 

  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x; 

}

double ElectronEnergyCorrector::fEtaEndcapBad(double scEta) const{
  
  // f(eta) for the class = 130-134 
  // Ivica's new corrections 01/06
  float p0 =        -4.25221e+00; 
  float p1 =         1.01936e+01;
  float p2 =        -7.48247e+00;
  float p3 =         2.45520e+00;
  float p4 =        -3.02872e-01;

  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x;  

}
  
double ElectronEnergyCorrector::fEtaBarrelGood(double scEta) const{

  // f(eta) for the first 3 classes (0, 10 and 20) (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  1.00149e+00; 
  float p1 = -2.06622e-03; 
  float p2 = -1.08793e-02; 
  float p3 =  1.54392e-02; 
  float p4 = -1.02056e-02; 

  double x  = (double) fabs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x; 

}
  



