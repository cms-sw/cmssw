
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronEnergyCorrector.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/EnergyUncertaintyElectronSpecific.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "TMath.h"


/****************************************************************************
 *
 * Classification based eta corrections for the ecal cluster energy
 *
 * \author Federico Ferri - INFN Milano, Bicocca university
 * \author Ivica Puljak - FESB, Split
 * \author Stephanie Baffioni - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 *
 * \version $Id: ElectronEnergyCorrector.cc,v 1.16 2012/02/22 17:36:44 chamont Exp $
 *
 ****************************************************************************/

void ElectronEnergyCorrector::classBasedParameterizationUncertainty( reco::GsfElectron & electron )
 {
  EnergyUncertaintyElectronSpecific uncertainty ;
  double energyError = 999. ;
  double ecalEnergy = electron.correctedEcalEnergy() ;
  double eleEta = electron.superCluster()->eta() ;
  double brem = electron.superCluster()->etaWidth()/electron.superCluster()->phiWidth() ;
  energyError =  uncertainty.computeElectronEnergyUncertainty(electron.classification(),eleEta,brem,ecalEnergy) ;
  electron.setCorrectedEcalEnergyError(energyError) ;
 }

float energyError( float E, float * par )
 { return sqrt( pow(par[0]/sqrt(E),2) + pow(par[1]/E,2) + pow(par[2],2) ) ; }

void ElectronEnergyCorrector::simpleParameterizationUncertainty( reco::GsfElectron & electron )
 {
  double error = 999. ;
  double ecalEnergy = electron.correctedEcalEnergy() ;

  if (electron.isEB())
   {
    float parEB[3] = { 5.24e-02,  2.01e-01, 1.00e-02} ;
    error =  ecalEnergy*energyError(ecalEnergy,parEB) ;
   }
  else if (electron.isEE())
   {
    float parEE[3] = { 1.46e-01, 9.21e-01, 1.94e-03} ;
    error =  ecalEnergy*energyError(ecalEnergy,parEE) ;
   }
  else
   { edm::LogWarning("ElectronEnergyCorrector::simpleParameterizationUncertainty")<<"nor barrel neither endcap electron !" ; }

  electron.setCorrectedEcalEnergyError(error) ;
 }

void ElectronEnergyCorrector::classBasedParameterizationEnergy
 ( reco::GsfElectron & electron, const reco::BeamSpot & bs )
 {
  if (electron.isEcalEnergyCorrected())
   {
	  edm::LogWarning("ElectronEnergyCorrector::classBasedParameterizationEnergy")<<"already done" ;
	  return ;
   }

  reco::GsfElectron::Classification elClass = electron.classification() ;
  if ( (elClass <= reco::GsfElectron::UNKNOWN) ||
	     (elClass>reco::GsfElectron::GAP) )
   {
	  edm::LogWarning("ElectronEnergyCorrector::classBasedParameterizationEnergy")<<"unexpected classification" ;
	  return ;
   }

  // new corrections from N. Chanon et al., taken from EcalClusterCorrectionObjectSpecific.cc
  float corr = 1.;
  float corr2 = 1.;
  float energy = electron.superCluster()->energy() ;
  float newEnergy = energy;

  //int subdet = electron.superCluster()->seed()->hitsAndFractions()[0].first.subdetId();

  if (electron.isEB())
   {
    float cetacorr = fEta(electron.superCluster()->rawEnergy(), electron.superCluster()->eta(), 0)/electron.superCluster()->rawEnergy();
    energy = electron.superCluster()->rawEnergy()*cetacorr; //previously in CMSSW
    //energy = superCluster.rawEnergy()*fEta(e5x5, superCluster.seed()->eta(), 0)/e5x5;
   }
  else if (electron.isEE())
   {
    energy = electron.superCluster()->rawEnergy()+electron.superCluster()->preshowerEnergy();
   }
  else
   { edm::LogWarning("ElectronEnergyCorrector::classBasedParameterizationEnergy")<<"nor barrel neither endcap electron !" ; }

  corr = fBremEta(electron.superCluster()->phiWidth()/electron.superCluster()->etaWidth(), electron.superCluster()->eta(), 0,elClass);

  float et = energy*TMath::Sin(2*TMath::ATan(TMath::Exp(-electron.superCluster()->eta())))/corr;

  if (electron.isEB()) { corr2 = corr * fEt(et, 0,elClass) ; }
  else if (electron.isEE()) { corr2 = corr * fEnergy(energy/corr, 1,elClass) ; }
  else { edm::LogWarning("ElectronEnergyCorrector::classBasedParameterizationEnergy")<<"nor barrel neither endcap electron !" ; }

  newEnergy = energy/corr2;

  // cracks
  double crackcor = 1. ;
  for ( reco::CaloCluster_iterator
        cIt = electron.superCluster()->clustersBegin() ;
        cIt != electron.superCluster()->clustersEnd() ;
        ++cIt )
   {
    const reco::CaloClusterPtr cc = *cIt ;
    crackcor *=
     ( electron.superCluster()->rawEnergy()
       + cc->energy()*(crackCorrectionFunction_->getValue(*cc)-1.) )
     / electron.superCluster()->rawEnergy() ;
   }
  newEnergy *= crackcor ;

  // register final value
  electron.setCorrectedEcalEnergy(newEnergy) ;

 }

// main correction function
// new corrections: taken from EcalClusterCorrectionObjectSpecific.cc (N. Chanon et al.)
// this is to prepare for class based corrections, for the time being the parameters are the same as for the SC corrections
// code fully duplicated here, to be improved; electron means algorithm==0 and mode==0

float ElectronEnergyCorrector::fEta(float energy, float eta, int algorithm) const
{

  // corrections for electrons
  if (algorithm!=0) {
    edm::LogWarning("ElectronEnergyCorrector::fEta")<<"algorithm should be 0 for electrons !" ;
    return energy;
  }
  //std::cout << "fEta function" << std::endl;

  // this correction is setup only for EB
  if ( algorithm != 0 ) return energy;

  float ieta = fabs(eta)*(5/0.087);
  // bypass the DB reading for the time being
  //float p0 = (params_->params())[0];  // should be 40.2198
  //float p1 = (params_->params())[1];  // should be -3.03103e-6
  float p0 = 40.2198 ;
  float p1 = -3.03103e-6 ;

  //std::cout << "ieta=" << ieta << std::endl;

  float correctedEnergy = energy;
  if ( ieta < p0 ) correctedEnergy = energy;
  else             correctedEnergy = energy/(1.0 + p1*(ieta-p0)*(ieta-p0));
  //std::cout << "ECEC fEta = " << correctedEnergy << std::endl;
  return correctedEnergy;

 }

float ElectronEnergyCorrector::fBremEta
 ( float sigmaPhiSigmaEta, float eta, int algorithm, reco::GsfElectron::Classification cl ) const
 {
  // corrections for electrons
  if (algorithm!=0)
   {
    edm::LogWarning("ElectronEnergyCorrector::fBremEta")<<"algorithm should be 0 for electrons !" ;
    return 1. ;
   }

  const float etaCrackMin = 1.44 ;
  const float etaCrackMax = 1.56 ;

  //STD
  const int nBinsEta = 14 ;
  float leftEta[nBinsEta] = { 0.02, 0.25, 0.46, 0.81, 0.91, 1.01, 1.16, etaCrackMax, 1.653, 1.8, 2.0, 2.2, 2.3, 2.4 } ;
  float rightEta[nBinsEta] = { 0.25, 0.42, 0.77, 0.91, 1.01, 1.13, etaCrackMin, 1.653, 1.8, 2.0, 2.2, 2.3, 2.4, 2.5 } ;

  // eta = 0
  if ( TMath::Abs(eta) < leftEta[0] ) { eta = 0.02 ; }

  // outside acceptance
  if ( TMath::Abs(eta) >= rightEta[nBinsEta-1] ) { eta = 2.49 ; } //if (DBG) std::cout << " WARNING [applyScCorrections]: TMath::Abs(eta)  >=  rightEta[nBinsEta-1] " << std::endl;}

  int tmpEta = -1 ;
  for (int iEta = 0; iEta < nBinsEta; ++iEta)
   {
    if ( leftEta[iEta] <= TMath::Abs(eta) && TMath::Abs(eta) <rightEta[iEta] )
     { tmpEta = iEta ; }
   }

  float xcorr[nBinsEta][reco::GsfElectron::GAP+1] =
   {
    { 1.00227,  1.00227,  1.00227,  1.00227,  1.00227  },
    { 1.00252,  1.00252,  1.00252,  1.00252,  1.00252  },
    { 1.00225,  1.00225,  1.00225,  1.00225,  1.00225  },
    { 1.00159,  1.00159,  1.00159,  1.00159,  1.00159  },
    { 0.999475, 0.999475, 0.999475, 0.999475, 0.999475 },
    { 0.997203, 0.997203, 0.997203, 0.997203, 0.997203 },
    { 0.993886, 0.993886, 0.993886, 0.993886, 0.993886 },
    { 0.971262, 0.971262, 0.971262, 0.971262, 0.971262 },
    { 0.975922, 0.975922, 0.975922, 0.975922, 0.975922 },
    { 0.979087, 0.979087, 0.979087, 0.979087, 0.979087 },
    { 0.98495,  0.98495,  0.98495,  0.98495,  0.98495  },
    { 0.98781,  0.98781,  0.98781,  0.98781,  0.98781  },
    { 0.989546, 0.989546, 0.989546, 0.989546, 0.989546 },
    { 0.989638, 0.989638, 0.989638, 0.989638, 0.989638 }
   } ;

  float par0[nBinsEta][reco::GsfElectron::GAP+1] =
   {
     { 0.987737, 0.987737, 0.97824,  1.01909,   1.00718 },
     { 0.992735, 0.992735, 1.00651,  1.01909,   1.00713 },
     { 0.995447, 0.995447, 0.991318, 1.0184,    1.00641 },
     { 1.00328,  1.00328,  1.00758,  0.989159,  1.00761 },
     { 0.993716, 0.993716, 0.979244, 1.05678,   1.00682 },
     { 1.00796,  1.00796,  1.11168,  1.01785,   1.0073  },
     { 0.978547, 0.978547, 0.782987, 0.996132,  1.00462 },
     { 0.972798, 0.972798, 0.972798, 0.972798, 0.972798 },
     { 0.981672, 0.981672, 0.981672, 0.981672, 0.981672 },
     { 0.98251 , 0.98251 , 0.98251 , 0.98251 , 0.98251  },
     { 0.986123, 0.986123, 0.986123, 0.986123, 0.986123 },
     { 0.990124, 0.990124, 0.990124, 0.990124, 0.990124 },
     { 0.990187, 0.990187, 0.990187, 0.990187, 0.990187 },
     { 0.99372 , 0.99372 , 0.99372 , 0.99372 , 0.99372  }
   } ;

  float par1[nBinsEta][reco::GsfElectron::GAP+1] =
   {
     { 0.0211232,  0.0211232,     0.0200981,    -0.0165033,  -0.00187886 },
     { 0.0132634,  0.0132634,     -0.00404441,  -0.0165033,  -0.00227574 },
     { 0.00914973, 0.00914973,    0.0121044,    -0.015315,   -0.00259935 },
     { -0.00185513, -0.00185513,  -0.00358261,  0.00854764,  -0.00433692 },
     { 0.00680977, 0.00680977,    0.0229892,    -0.0709498,  -0.00551324 },
     { -0.0103503, -0.0103503,    -0.0954333,   -0.0218312,  -0.00799669 },
     { 0.0198188,  0.0198188,     0.188263,     -0.00748124, -0.00870057 },
     { -0.000771577, -0.000771577, -0.000771577, -0.000771577, -0.000771577 },
     { -0.00202028 , -0.00202028 , -0.00202028 , -0.00202028 , -0.00202028  },
     { 0.00441308  , 0.00441308  , 0.00441308  , 0.00441308  , 0.00441308   },
     { 0.00832913  , 0.00832913  , 0.00832913  , 0.00832913  , 0.00832913   },
     { 0.00742879  , 0.00742879  , 0.00742879  , 0.00742879  , 0.00742879   },
     { 0.0094608   , 0.0094608   , 0.0094608   , 0.0094608   , 0.0094608    },
     { 0.00560406  , 0.00560406  , 0.00560406  , 0.00560406  , 0.00560406   }
   } ;

  float par2[nBinsEta][reco::GsfElectron::GAP+1] =
   {
     { -0.00653788, -0.00653788,  -0.00408167,   0.0032172,     0 },
     { -0.00405187, -0.00405187,  0.000669508,   0.0032172,     0 },
     { -0.0028242,  -0.0028242,   -0.00381642,   0.00241369,    0 },
     { -0.000491565,-0.000491565, -0.000215741,  -0.0026585,    0 },
     { -0.00235393, -0.00235393,  -0.00896144,   0.019303,      0 },
     { 0.000584956, 0.000584956,  0.0179373,     0.00291216,    0 },
     { -0.00717285, -0.00717285,  -0.0447474,    -0.000420271,  0 },
     { -0.00276696, -0.00276696, -0.00276696, -0.00276696, -0.00276696 },
     { -0.00471028, -0.00471028, -0.00471028, -0.00471028, -0.00471028 },
     { -0.00809139, -0.00809139, -0.00809139, -0.00809139, -0.00809139 },
     { -0.00944584, -0.00944584, -0.00944584, -0.00944584, -0.00944584 },
     { -0.00960462, -0.00960462, -0.00960462, -0.00960462, -0.00960462 },
     { -0.010172  , -0.010172  , -0.010172  , -0.010172  , -0.010172   },
     { -0.00943169, -0.00943169, -0.00943169, -0.00943169, -0.00943169 }
   } ;

  float sigmaPhiSigmaEtaMin[reco::GsfElectron::GAP+1] = { 0.8, 0.8, 0.8, 0.8, 0.8 } ;
  float sigmaPhiSigmaEtaMax[reco::GsfElectron::GAP+1] = { 5., 5., 5., 5., 5. } ;
  float sigmaPhiSigmaEtaFit[reco::GsfElectron::GAP+1] = { 1.2, 1.2, 1.2, 1.2, 1.2 } ;

  // extra protections
  // fix sigmaPhiSigmaEta boundaries
  if ( sigmaPhiSigmaEta < sigmaPhiSigmaEtaMin[cl] )
   { sigmaPhiSigmaEta = sigmaPhiSigmaEtaMin[cl] ; }
  if ( sigmaPhiSigmaEta > sigmaPhiSigmaEtaMax[cl]  )
   { sigmaPhiSigmaEta = sigmaPhiSigmaEtaMax[cl] ; }

  // In eta cracks/gaps
  if ( tmpEta == -1 ) // need to interpolate
   {
    float tmpInter = 1 ;
    for ( int iEta = 0 ; iEta < nBinsEta-1 ; ++iEta )
     {
      if ( rightEta[iEta] <= TMath::Abs(eta) && TMath::Abs(eta) <leftEta[iEta+1] )
       {
        if ( sigmaPhiSigmaEta >= sigmaPhiSigmaEtaFit[cl] )
         {
	        tmpInter =
	         ( par0[iEta][cl] +
	           sigmaPhiSigmaEta*par1[iEta][cl] +
	           sigmaPhiSigmaEta*sigmaPhiSigmaEta*par2[iEta][cl] +
             par0[iEta+1][cl] +
             sigmaPhiSigmaEta*par1[iEta+1][cl] +
             sigmaPhiSigmaEta*sigmaPhiSigmaEta*par2[iEta+1][cl] ) / 2. ;
         }
	      else tmpInter = (xcorr[iEta][cl] + xcorr[iEta+1][cl])/2. ;
       }
     }
    return tmpInter ;
   }

  if (sigmaPhiSigmaEta >= sigmaPhiSigmaEtaFit[cl])
   { return par0[tmpEta][cl] + sigmaPhiSigmaEta*par1[tmpEta][cl] + sigmaPhiSigmaEta*sigmaPhiSigmaEta*par2[tmpEta][cl] ; }
  else
   { return xcorr[tmpEta][cl] ; }

  return 1. ;
 }

float ElectronEnergyCorrector::fEt(float ET, int algorithm, reco::GsfElectron::Classification cl ) const
 {
  if (algorithm==0) //Electrons EB
   {
    float par[reco::GsfElectron::GAP+1][5] =
     {
       { 0.974507, 1.16569, -0.000884133, 0.161423, -125.356 },
       { 0.974507, 1.16569, -0.000884133, 0.161423, -125.356 },
       { 0.96449, 0.991457, 0.000237869, 0.159983, -4.38755 },
       { 0.97956, 0.883959, 0.000782834, -0.106388, -124.394 },
       { 0.97213, 0.999528, 5.61192e-06, 0.0143269, -17.1776 }
     } ;
    if ( ET > 200 ) { ET =200 ; }
    if ( ET < 5 ) { return 1. ; }
    if ( 5 <= ET && ET < 10 ) { return par[cl][0] ; }
    if ( 10 <= ET && ET <= 200 ) { return (par[cl][1]  + ET*par[cl][2])*(1- par[cl][3]*exp(ET/par[cl][4])) ; }
   }
  else if (algorithm==1) //Electrons EE
   {
    float par[reco::GsfElectron::GAP+1][5] =
     {
       { 0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461 },
       { 0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461 },
       { 0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461 },
       { 0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461 },
       { 0.930081, 0.996683, 3.54079e-05, 0.0460187, -23.2461 }
     };
    if ( ET > 200 ) { ET = 200 ; }
    if ( ET < 5 ) { return 1. ; }
    if ( 5 <= ET && ET < 10 ) { return par[cl][0] ; }
    if ( 10 <= ET && ET <= 200 ) { return ( par[cl][1]  + ET*par[cl][2])*(1-par[cl][3]*exp(ET/par[cl][4])) ; }
   }
  else
   { edm::LogWarning("ElectronEnergyCorrector::fEt")<<"algorithm should be 0 or 1 for electrons !" ; }
  return 1. ;
 }

float ElectronEnergyCorrector::fEnergy(float E, int algorithm, reco::GsfElectron::Classification cl ) const
 {
  if (algorithm==0) // Electrons EB
   { return 1. ; }
  else if (algorithm==1) // Electrons EE
   {
    float par0[reco::GsfElectron::GAP+1] = { 400, 400, 400, 400, 400 } ;
    float par1[reco::GsfElectron::GAP+1] = { 0.982475, 0.982475, 0.982475, 0.982475, 0.982475 } ;
    float par2[reco::GsfElectron::GAP+1] = { 4.95413e-05, 4.95413e-05, 4.95413e-05, 4.95413e-05, 4.95413e-05 } ;
    float par3[reco::GsfElectron::GAP+1] = { 0.16886, 0.16886, 0.16886, 0.16886, 0.16886 } ;
    float par4[reco::GsfElectron::GAP+1] = { -30.1517, -30.1517, -30.1517, -30.1517, -30.1517 } ;

    if ( E > par0[cl] ) { E = par0[cl] ; }
    if ( E < 0 ) { return 1. ; }
    if ( 0 <= E && E <= par0[cl] ) { return (par1[cl] + E*par2[cl] )*(1- par3[cl]*exp(E/par4[cl] )) ; }
   }
  else
   { edm::LogWarning("ElectronEnergyCorrector::fEnergy")<<"algorithm should be 0 or 1 for electrons !" ; }
  return 1.;
 }


//==========================================================================
//
//==========================================================================

double ElectronEnergyCorrector::fEtaBarrelGood( double scEta ) const
 {
  // f(eta) for the first 3 classes (0, 10 and 20) (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  1.00149e+00 ;
  float p1 = -2.06622e-03 ;
  float p2 = -1.08793e-02 ;
  float p3 =  1.54392e-02 ;
  float p4 = -1.02056e-02 ;
  double x  = (double) std::abs(scEta) ;
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x ;
 }

double ElectronEnergyCorrector::fEtaBarrelBad(double scEta) const
 {
  // f(eta) for the class = 30 (estimated from 1Mevt single e sample)
  // Ivica's new corrections 01/06
  float p0 =  9.99063e-01;
  float p1 = -2.63341e-02;
  float p2 =  5.16054e-02;
  float p3 = -4.95976e-02;
  float p4 =  3.62304e-03;
  double x  = (double) std::abs(scEta) ;
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x ;
 }

double ElectronEnergyCorrector::fEtaEndcapGood( double scEta ) const
 {
  // f(eta) for the first 3 classes (100, 110 and 120)
  // Ivica's new corrections 01/06
  float p0 = -8.51093e-01 ;
  float p1 =  3.54266e+00 ;
  float p2 = -2.59288e+00 ;
  float p3 = 8.58945e-01 ;
  float p4 = -1.07844e-01 ;
  double x  = (double) std::abs(scEta) ;
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x ;
 }

double ElectronEnergyCorrector::fEtaEndcapBad( double scEta ) const
 {
  // f(eta) for the class = 130-134
  // Ivica's new corrections 01/06
  float p0 =        -4.25221e+00 ;
  float p1 =         1.01936e+01 ;
  float p2 =        -7.48247e+00 ;
  float p3 =         2.45520e+00 ;
  float p4 =        -3.02872e-01 ;
  double x  = (double) std::abs(scEta);
  return p0 + p1*x + p2*x*x + p3*x*x*x + p4*x*x*x*x ;
 }

