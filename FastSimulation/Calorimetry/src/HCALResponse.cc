//FastSimulation headers
#include "FastSimulation/Calorimetry/interface/HCALResponse.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

// CMSSW Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <vector>
#include <math.h>

#define debug 0

using namespace edm;

HCALResponse::HCALResponse(const edm::ParameterSet& pset,
			   const RandomEngine* engine) :
  random(engine)
{
//values for "old" response parameterizations
//--------------------------------------------------------------------
  RespPar[HCAL][0][0] = pset.getParameter<double>("HadronBarrelResolution_Stochastic");
  RespPar[HCAL][0][1] = pset.getParameter<double>("HadronBarrelResolution_Constant");
  RespPar[HCAL][0][2] = pset.getParameter<double>("HadronBarrelResolution_Noise");

  RespPar[HCAL][1][0] = pset.getParameter<double>("HadronEndcapResolution_Stochastic");
  RespPar[HCAL][1][1] = pset.getParameter<double>("HadronEndcapResolution_Constant");
  RespPar[HCAL][1][2] = pset.getParameter<double>("HadronEndcapResolution_Noise");

  RespPar[VFCAL][0][0] = pset.getParameter<double>("HadronForwardResolution_Stochastic");
  RespPar[VFCAL][0][1] = pset.getParameter<double>("HadronForwardResolution_Constant");
  RespPar[VFCAL][0][2] = pset.getParameter<double>("HadronForwardResolution_Noise");

  RespPar[VFCAL][1][0] = pset.getParameter<double>("ElectronForwardResolution_Stochastic");
  RespPar[VFCAL][1][1] = pset.getParameter<double>("ElectronForwardResolution_Constant");
  RespPar[VFCAL][1][2] = pset.getParameter<double>("ElectronForwardResolution_Noise");

  eResponseScale[0] = pset.getParameter<double>("eResponseScaleHB");  
  eResponseScale[1] = pset.getParameter<double>("eResponseScaleHE");
  eResponseScale[2] = pset.getParameter<double>("eResponseScaleHF");

  eResponsePlateau[0] = pset.getParameter<double>("eResponsePlateauHB");
  eResponsePlateau[1] = pset.getParameter<double>("eResponsePlateauHE");
  eResponsePlateau[2] = pset.getParameter<double>("eResponsePlateauHF");

  eResponseExponent = pset.getParameter<double>("eResponseExponent");
  eResponseCoefficient = pset.getParameter<double>("eResponseCoefficient");
  eResponseCorrection = pset.getParameter<double>("eResponseCorrection");

  // If need - add a small energy to each hadron ...
  eBias = pset.getParameter<double>("energyBias");
  
//pion parameters
//--------------------------------------------------------------------
  phase2Upgrade = pset.getParameter<bool>("phase2Upgrade");
  etaStep = pset.getParameter<double>("etaStep");
  maxHDe = pset.getParameter<int>("maxHDe");
  maxHDeta = pset.getParameter<int>("maxHDeta");
  eGridHD = pset.getParameter<std::vector<double> >("eGridHD");
  
  // additional tuning factor to correct the response
  barrelCorrection = pset.getParameter<std::vector<double> >("barrelCorrection");
  endcapCorrection = pset.getParameter<std::vector<double> >("endcapCorrection");
  forwardCorrectionEnergyDependent = pset.getParameter<std::vector<double> >("forwardCorrectionEnergyDependent");
  forwardCorrectionEtaDependent = pset.getParameter<std::vector<double> >("forwardCorrectionEtaDependent");
  
  // MEAN energy response for (1) all (2) MIP in ECAL (3) non-MIP in ECAL
  std::vector<double> _meanHD = pset.getParameter<std::vector<double> >("meanHD");
  std::vector<double> _meanHD_mip = pset.getParameter<std::vector<double> >("meanHD_mip");
  std::vector<double> _meanHD_nomip = pset.getParameter<std::vector<double> >("meanHD_nomip");

  // SIGMAS (from RMS)
  std::vector<double> _sigmaHD = pset.getParameter<std::vector<double> >("sigmaHD");
  std::vector<double> _sigmaHD_mip = pset.getParameter<std::vector<double> >("sigmaHD_mip");
  std::vector<double> _sigmaHD_nomip = pset.getParameter<std::vector<double> >("sigmaHD_nomip");
  
  //fill in 2D vectors
  for(int i = 0; i < maxHDe; i++){
    std::vector<double> m1, m2, m3, s1, s2, s3;
    for(int j = 0; j < maxHDeta; j++){
	  m1.push_back(_meanHD[i*maxHDeta + j]);
	  m2.push_back(_meanHD_mip[i*maxHDeta + j]);
	  m3.push_back(_meanHD_nomip[i*maxHDeta + j]);
	  s1.push_back(_sigmaHD[i*maxHDeta + j]);
	  s2.push_back(_sigmaHD_mip[i*maxHDeta + j]);
	  s3.push_back(_sigmaHD_nomip[i*maxHDeta + j]);
	}
	meanHD.push_back(m1);
	meanHD_mip.push_back(m2);
	meanHD_nomip.push_back(m3);
	sigmaHD.push_back(s1);
	sigmaHD_mip.push_back(s2);
	sigmaHD_nomip.push_back(s3);
  }
  
// MUON probability histos for bin size = 0.25 GeV (0-10 GeV, 40 bins)
//--------------------------------------------------------------------
  muStep  = pset.getParameter<double>("muStep");
  maxMUe = pset.getParameter<int>("maxMUe");
  maxMUeta = pset.getParameter<int>("maxMUeta");
  maxMUbin = pset.getParameter<int>("maxMUbin");
  eGridMU = pset.getParameter<std::vector<double> >("eGridMU");
  etaGridMU = pset.getParameter<std::vector<double> >("etaGridMU");
  std::vector<double> _responseMU = pset.getParameter<std::vector<double> >("responseMU");
  
  //fill in 3D vector
  //(complementary cumulative distribution functions, from normalized response distributions)
  for(int i = 0; i < maxMUe; i++){
    std::vector<std::vector<double> > mu1;
    for(int j = 0; j < maxMUeta; j++){
	  std::vector<double> mu2;
	  for(int k = 0; k < maxMUbin; k++){
	    mu2.push_back(_responseMU[i*maxMUeta*maxMUbin + j*maxMUbin + k]);
		
		if(debug) {
	    //cout.width(6);
	    LogInfo("FastCalorimetry") << " responseMU " << i << " " << j << " " << k  << " = " 
				      << _responseMU[i*maxMUeta*maxMUbin + j*maxMUbin + k] << std::endl;
	    }
	  }
	  mu1.push_back(mu2);
	}
	responseMU.push_back(mu1);
  }

// values for EM response in HF
//--------------------------------------------------------------------
  maxEMe = pset.getParameter<int>("maxEMe");
  maxEMeta = pset.getParameter<int>("maxEMeta");
  respFactorEM = pset.getParameter<double>("respFactorEM");
  eGridEM = pset.getParameter<std::vector<double> >("eGridEM");
 
  // e-gamma mean response and sigma in HF 
  std::vector<double> _meanEM = pset.getParameter<std::vector<double> >("meanEM");
  std::vector<double> _sigmaEM = pset.getParameter<std::vector<double> >("sigmaEM");

  //fill in 2D vectors
  for(int i = 0; i < maxEMe; i++){
    std::vector<double> m_tmp;
	std::vector<double> s_tmp;
    for(int j = 0; j < maxEMeta; j++){
	  m_tmp.push_back(_meanEM[i*maxEMeta + j]);
	  s_tmp.push_back(_sigmaEM[i*maxEMeta + j]);
	}
	meanEM.push_back(m_tmp);
	sigmaEM.push_back(s_tmp);
  }
  
// Normalize the response and sigmas to the corresponding energies
//--------------------------------------------------------------------
  for(int i = 0; i<maxHDe;  i++) {
    for(int j = 0; j<maxHDeta; j++) {
      double factor     = 1.0;
      double factor_s   = 1.0;

      if( j < 16)             factor = barrelCorrection[i];  // special HB
      if( j < 30 && j >= 16)  factor = endcapCorrection[i];  // special HE
      if( j >= 30)            factor = forwardCorrectionEnergyDependent[i]*forwardCorrectionEtaDependent[j-30]; // special HF
	  
      meanHD[i][j]        =  factor * meanHD[i][j]  / eGridHD[i];
      sigmaHD[i][j]       =  factor_s * sigmaHD[i][j] / eGridHD[i];

      meanHD_mip[i][j]    =  factor * meanHD_mip[i][j]  / eGridHD[i];
      sigmaHD_mip[i][j]   =  factor_s * sigmaHD_mip[i][j] / eGridHD[i];

      meanHD_nomip[i][j]  =  factor * meanHD_nomip[i][j]  / eGridHD[i];
      sigmaHD_nomip[i][j] =  factor_s * sigmaHD_nomip[i][j] / eGridHD[i];

    }
  }

  for(int i = 0; i<maxEMe;  i++) {
    for(int j = 0; j<maxEMeta; j++) {
      meanEM[i][j]  = respFactorEM * meanEM[i][j] / eGridEM[i];
      sigmaEM[i][j] = respFactorEM * sigmaEM[i][j] / eGridEM[i];
    }
  }


}

std::pair<double,double> 
HCALResponse::responseHCAL(int mip, double energy, double eta, int hit, int partype){
  //one of these functions sets mean and sigma
  if(phase2Upgrade)	responseHCALUpgrade(mip,energy,eta,hit,partype);
  else responseHCALStandard(mip,energy,eta,hit,partype);

  // debugging
  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::responseHCAL, partype = " <<  partype 
				<< " E, eta = " << energy << " " << eta  
				<< "  mean & sigma = " << mean   << " " << sigma << std::endl;
  }  
  
  //then this function returns them
  return std::pair<double,double>(mean,sigma);
}

 
void 
HCALResponse::responseHCALStandard(int mip, double energy, double eta, int hit, int partype)
{
  int ieta = abs((int)(eta / etaStep)) ;
  int ie = -1;

  mean  = 0.;
  sigma = 0.;

  // e/gamma in HF
  if(partype == 0) {
    ieta -= 30;  // HF starts at ieta=30 till ieta=51 
                 // but resp.vector from index=0 through 20
    if(ieta >= maxEMeta ) ieta = maxEMeta-1;
    else if(ieta < 0) ieta = 0;
 
    for (int i = 0; i < maxEMe; i++) {
      if(energy < eGridEM[i])  {
	    if(i == 0) ie = 0;       
        else  ie = i-1;
        break;
      }
    }
    if(ie == -1) ie = maxEMe - 2;  
    interEM(energy, ie, ieta);
  }
  
  // hadrons
  else if(partype == 1) {
      if(ieta >= maxHDeta) ieta = maxHDeta-1;
      
      if(ieta < 0 ) ieta = 0;
      for (int i = 0; i < maxHDe; i++) {
	    if(energy < eGridHD[i])  {
	      if(i == 0) ie = 0;     // less than minimal - back extrapolation with the 1st interval
	      else  ie = i-1;
	      break;
	    }	
      }
      if(ie == -1) ie = maxHDe - 2;     // more than maximum - extrapolation with last interv.
      
      interHD(mip, energy, ie, ieta);
	  
	  // finally apply energy scale correction
      mean  *= eResponseCorrection;
      mean  += eBias;
      sigma *= eResponseCorrection;
  }

  
  // muons
  else if(partype == 2) { 
    
    ieta = maxMUeta;
    for(int i = 0; i < maxMUeta; i++) {
      if(fabs(eta) < etaGridMU[i]) {
	    ieta = i;  
	    break;
      }      
    }
    if(ieta < 0) ieta = 0;
	
    if(ieta < maxMUeta) {  // HB-HE
      
      for (int i = 0; i < maxMUe; i++) {
	    if(energy < eGridMU[i])  {
	      if(i == 0) ie = 0;     // less than minimal - back extrapolation with the first interval
	      else  ie = i-1;
	      break;
	    }
      }
	  if(ie == -1) ie = maxMUe - 2;     // more than maximum - extrapolation using the last interval
	  
      interMU(energy, ie, ieta);
	  
	  if(mean > energy) mean = energy;  
    }
  }

}

void 
HCALResponse::responseHCALUpgrade(int mip, double energy, double eta, int hit, int partype)
{
  int ieta = abs((int)(eta / etaStep)) ;
  int ie = -1;

  mean  = 0.;
  sigma = 0.;

  // e/gamma in HF
  if(partype == 0) {
    ieta -= 30;  // HF starts at ieta=30 till ieta=51 
                 // but resp.vector from index=0 through 20
    if(ieta >= maxEMeta ) ieta = maxEMeta-1;
    else if(ieta < 0) ieta = 0;
 
    for (int i = 0; i < maxEMe; i++) {
      if(energy < eGridEM[i])  {
	    if(i == 0) ie = 0;       
        else  ie = i-1;
        break;
      }
    }
    if(ie == -1) ie = maxEMe - 2;  
    interEM(energy, ie, ieta);
  }
  
  // hadrons
  else if(partype == 1) {
      if(ieta >= maxHDeta) ieta = maxHDeta-1;
      
      if(ieta < 0 ) ieta = 0;
      for (int i = 0; i < maxHDe; i++) {
	    if(energy < eGridHD[i])  {
	      if(i == 0) ie = 0;     // less than minimal - back extrapolation with the 1st interval
	      else  ie = i-1;
	      break;
	    }	
      }
      if(ie == -1) ie = maxHDe - 2;     // more than maximum - extrapolation with last interv.
      
	  //in endcap, use parameters from standalone
	  if(hit==hcendcap) interHD(2, energy, ie, 0); //ignore mip, no eta segmentation
	  //not in endcap, use "old" versions
	  else {
	     mean = getHCALEnergyResponse(energy, hit);
	     sigma = getHCALEnergyResolution(energy, hit);		  
	  }
	  
	  // finally apply energy scale correction
      mean  *= eResponseCorrection;
      mean  += eBias;
      sigma *= eResponseCorrection;
  }

  
  // muons
  else if(partype == 2) { 
    
    ieta = maxMUeta;
    for(int i = 0; i < maxMUeta; i++) {
      if(fabs(eta) < etaGridMU[i]) {
	    ieta = i;  
	    break;
      }      
    }
    if(ieta < 0) ieta = 0;
	
    if(ieta < maxMUeta) {  // HB-HE
      
      for (int i = 0; i < maxMUe; i++) {
	    if(energy < eGridMU[i])  {
	      if(i == 0) ie = 0;     // less than minimal - back extrapolation with the first interval
	      else  ie = i-1;
	      break;
	    }
      }
	  if(ie == -1) ie = maxMUe - 2;     // more than maximum - extrapolation using the last interval  

	  //in endcap, use parameters from standalone - no eta segmentation
	  if(hit==hcendcap) interMU(energy, ie, 0);
	  //not in endcap, just use average peak of old muon histos
	  else {
	    mean = 10*muStep;
	    sigma = 0;
	  }
	  
	  if(mean > energy) mean = energy;  
    }
  }

}

void HCALResponse::interMU(double e, int ie, int ieta)
{

  double x = random->flatShoot();

  int bin1 = maxMUbin;
  for(int i = 0; i < maxMUbin; i++) {
    if(x > responseMU[ie][ieta][i]) {
      bin1 = i-1;
      break;
    }
  }
  int bin2 = maxMUbin;
  for(int i = 0; i < maxMUbin; i++) {
    if(x > responseMU[ie+1][ieta][i]) {
      bin2 = i-1;
      break;
    }
  }
   
  double x1 = eGridMU[ie];
  double x2 = eGridMU[ie+1];
  double y1 = (bin1 + random->flatShoot()) * muStep;   
  double y2 = (bin2 + random->flatShoot()) * muStep;   

  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interMU  " << std::endl
				<< " x, x1-x2, y1-y2 = " 
				<< e << ", " << x1 <<"-" << x2 << " " << y1 <<"-" << y2 << std::endl; 
  
  }


  mean  = y1 + (y2-y1) * (e - x1)/(x2 - x1);
  sigma = 0.;

  if(debug) {
    //cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interMU " << std::endl
				<< " e, ie, ieta = " << e << " " << ie << " " << ieta << std::endl
				<< " response  = " << mean << std::endl; 
  }

}

void HCALResponse::interHD(int mip, double e, int ie, int ieta)
{

  double y1, y2;

  double x1 = eGridHD[ie];
  double x2 = eGridHD[ie+1];

  if(mip == 2) {           // mip doesn't matter
    y1 = meanHD[ie][ieta]; 
    y2 = meanHD[ie+1][ieta]; 
  }
  else {
    if(mip == 0) {         // not mip
      y1 = meanHD_nomip[ie][ieta]; 
      y2 = meanHD_nomip[ie+1][ieta]; 
    }
    else {                 // mip in ECAL
      y1 = meanHD_mip[ie][ieta]; 
      y2 = meanHD_mip[ie+1][ieta]; 
    }
  }

  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interHD mean " << std::endl
				<< " x, x1-x2, y1-y2 = " 
				<< e << ", " << x1 <<"-" << x2 << " "
                                << y1 <<"-" << y2 << std::endl;  
  }
  
  mean =  e * (y1 + (y2 - y1) * (e - x1)/(x2 - x1));      
  

  if(mip == 2) {           // mip doesn't matter
    y1 = sigmaHD[ie][ieta]; 
    y2 = sigmaHD[ie+1][ieta]; 
  }
  else {
    if(mip == 0) {         // not mip
      y1 = sigmaHD_nomip[ie][ieta]; 
      y2 = sigmaHD_nomip[ie+1][ieta]; 
    }
    else {                 // mip in ECAL
      y1 = sigmaHD_mip[ie][ieta]; 
      y2 = sigmaHD_mip[ie+1][ieta]; 
    }
  }
  
  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interHD sigma" << std::endl
				<< " x, x1-x2, y1-y2 = " 
				<< e << ", " << x1 <<"-" << x2 << " " << y1 <<"-" << y2 << std::endl; 
  
  }
 
  sigma = e * (y1 + (y2 - y1) * (e - x1)/(x2 - x1));      


  if(debug) {
    //cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interHD " << std::endl
				<< " e, ie, ieta = " << e << " " << ie << " " << ieta << std::endl
				<< " mean, sigma  = " << mean << " " << sigma << std::endl ;
  }

}


void HCALResponse::interEM(double e, int ie, int ieta)
{ 
  double y1 = meanEM[ie][ieta]; 
  double y2 = meanEM[ie+1][ieta]; 
  double x1 = eGridEM[ie];
  double x2 = eGridEM[ie+1];
  
  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interEM mean " << std::endl
				<< " x, x1-x2, y1-y2 = " 
				<< e << ", " << x1 <<"-" << x2 << " " << y1 <<"-" << y2 << std::endl; 
  
  }

  mean =  e * (y1 + (y2 - y1) * (e - x1)/(x2 - x1));      
  
  y1 = sigmaEM[ie][ieta]; 
  y2 = sigmaEM[ie+1][ieta]; 
  
  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interEM sigma" << std::endl
				<< " x, x1-x2, y1-y2 = " 
				<< e << ", " << x1 <<"-" << x2 << " " << y1 <<"-" << y2 << std::endl; 
  
  }

  sigma = e * (y1 + (y2 - y1) * (e - x1)/(x2 - x1));      
}

// Old parametrization for hadrons
double HCALResponse::getHCALEnergyResolution(double e, int hit){
  
  if(hit==hcforward) 
    return e *sqrt( RespPar[VFCAL][1][0]*RespPar[VFCAL][1][0] / e + 
		    RespPar[VFCAL][1][1]*RespPar[VFCAL][1][1] );
  else
    return  e * sqrt( RespPar[HCAL][hit][0]*RespPar[HCAL][hit][0]/(e)
		      + RespPar[HCAL][hit][1]*RespPar[HCAL][hit][1]);   

}

// Old parameterization of the calo response to hadrons
double HCALResponse::getHCALEnergyResponse(double e, int hit){

  double s = eResponseScale[hit];
  double n = eResponseExponent;
  double p = eResponsePlateau[hit];
  double c = eResponseCoefficient;

  double response = e * p / (1+c*exp(n * log(s/e)));

  if(response<0.) response = 0.;

  return response;
}

// old parameterization of the HF response to electrons
double HCALResponse::getHFEnergyResolution(double EGen)
{
  return EGen *sqrt( RespPar[VFCAL][0][0]*RespPar[VFCAL][0][0] / EGen + 
		     RespPar[VFCAL][0][1]*RespPar[VFCAL][0][1] );
}  