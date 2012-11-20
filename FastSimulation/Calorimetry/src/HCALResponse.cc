//FastSimulation headers
#include "FastSimulation/Calorimetry/interface/HCALResponse.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

// CMSSW Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <vector>
#include <math.h>

using namespace edm;

HCALResponse::HCALResponse(const edm::ParameterSet& pset,
			   const RandomEngine* engine) :
  random(engine)
{
  //switches
  debug = pset.getParameter<bool>("debug");
  usemip = pset.getParameter<bool>("usemip");

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
  etaStep = pset.getParameter<double>("etaStep");
  maxHDe = pset.getParameter<int>("maxHDe");
  eGridHD = pset.getParameter<std::vector<double> >("eGridHD");
  
  //region eta indices calculated from eta values
  maxHDeta = abs((int)(pset.getParameter<double>("maxHDeta") / etaStep)) + 1; //add 1 because this is the max index
  barrelHDeta = abs((int)(pset.getParameter<double>("barrelHDeta") / etaStep));
  endcapHDeta = abs((int)(pset.getParameter<double>("endcapHDeta") / etaStep));
  forwardHDeta = abs((int)(pset.getParameter<double>("forwardHDeta") / etaStep));
  int maxHDetas[] = {endcapHDeta - barrelHDeta, forwardHDeta - endcapHDeta, maxHDeta - forwardHDeta}; //eta ranges
  
  // additional tuning factor to correct the response
  useAdHocCorrections_ = pset.getParameter<bool>("useAdHocCorrections");
  barrelCorrection = pset.getParameter<std::vector<double> >("barrelCorrection");
  endcapCorrection = pset.getParameter<std::vector<double> >("endcapCorrection");
  forwardCorrectionEnergyDependent = pset.getParameter<std::vector<double> >("forwardCorrectionEnergyDependent");
  forwardCorrectionEtaDependent = pset.getParameter<std::vector<double> >("forwardCorrectionEtaDependent");
  
  // MEAN energy response for: all, MIP in ECAL, non-MIP in ECAL
  std::vector<double> _meanHD[3] = {pset.getParameter<std::vector<double> >("meanHDBarrel"),pset.getParameter<std::vector<double> >("meanHDEndcap"),pset.getParameter<std::vector<double> >("meanHDForward")};
  std::vector<double> _meanHD_mip[3] = {pset.getParameter<std::vector<double> >("meanHDBarrel_mip"),pset.getParameter<std::vector<double> >("meanHDEndcap_mip"),pset.getParameter<std::vector<double> >("meanHDForward_mip")};
  std::vector<double> _meanHD_nomip[3] = {pset.getParameter<std::vector<double> >("meanHDBarrel_nomip"),pset.getParameter<std::vector<double> >("meanHDEndcap_nomip"),pset.getParameter<std::vector<double> >("meanHDForward_nomip")};

  // SIGMAS (from RMS)
  std::vector<double> _sigmaHD[3] = {pset.getParameter<std::vector<double> >("sigmaHDBarrel"),pset.getParameter<std::vector<double> >("sigmaHDEndcap"),pset.getParameter<std::vector<double> >("sigmaHDForward")};
  std::vector<double> _sigmaHD_mip[3] = {pset.getParameter<std::vector<double> >("sigmaHDBarrel_mip"),pset.getParameter<std::vector<double> >("sigmaHDEndcap_mip"),pset.getParameter<std::vector<double> >("sigmaHDForward_mip")};
  std::vector<double> _sigmaHD_nomip[3] = {pset.getParameter<std::vector<double> >("sigmaHDBarrel_nomip"),pset.getParameter<std::vector<double> >("sigmaHDEndcap_nomip"),pset.getParameter<std::vector<double> >("sigmaHDForward_nomip")};
  
  //initialize 2D vectors
  meanHD = std::vector<std::vector<double> >(maxHDe,std::vector<double>(maxHDeta,0));
  meanHD_mip = std::vector<std::vector<double> >(maxHDe,std::vector<double>(maxHDeta,0));
  meanHD_nomip = std::vector<std::vector<double> >(maxHDe,std::vector<double>(maxHDeta,0));
  sigmaHD = std::vector<std::vector<double> >(maxHDe,std::vector<double>(maxHDeta,0));
  sigmaHD_mip = std::vector<std::vector<double> >(maxHDe,std::vector<double>(maxHDeta,0));
  sigmaHD_nomip = std::vector<std::vector<double> >(maxHDe,std::vector<double>(maxHDeta,0));
  
  //fill in 2D vectors
  int loc, eta_loc;
  loc = eta_loc = -1;
  for(int i = 0; i < maxHDe; i++){
    for(int j = 0; j < maxHDeta; j++){
	  //check location - barrel, endcap, or forward
	  if(j==barrelHDeta) {loc = 0; eta_loc = barrelHDeta;}
	  else if(j==endcapHDeta) {loc = 1; eta_loc = endcapHDeta;}
	  else if(j==forwardHDeta) {loc = 2; eta_loc = forwardHDeta;}
	
	  meanHD[i][j] = _meanHD[loc][i*maxHDetas[loc] + j - eta_loc];
	  meanHD_mip[i][j] = _meanHD_mip[loc][i*maxHDetas[loc] + j - eta_loc];
	  meanHD_nomip[i][j] = _meanHD_nomip[loc][i*maxHDetas[loc] + j - eta_loc];
	  sigmaHD[i][j] = _sigmaHD[loc][i*maxHDetas[loc] + j - eta_loc];
	  sigmaHD_mip[i][j] = _sigmaHD_mip[loc][i*maxHDetas[loc] + j - eta_loc];
	  sigmaHD_nomip[i][j] = _sigmaHD_nomip[loc][i*maxHDetas[loc] + j - eta_loc];
	}
  }
  
// MUON probability histos for bin size = 0.25 GeV (0-10 GeV, 40 bins)
//--------------------------------------------------------------------
  muStep  = pset.getParameter<double>("muStep");
  maxMUe = pset.getParameter<int>("maxMUe");
  maxMUeta = pset.getParameter<int>("maxMUeta");
  maxMUbin = pset.getParameter<int>("maxMUbin");
  eGridMU = pset.getParameter<std::vector<double> >("eGridMU");
  etaGridMU = pset.getParameter<std::vector<double> >("etaGridMU");
  std::vector<double> _responseMU[2] = {pset.getParameter<std::vector<double> >("responseMUBarrel"),pset.getParameter<std::vector<double> >("responseMUEndcap")};
  
  //get muon region eta indices from the eta grid
  double _barrelMUeta = pset.getParameter<double>("barrelMUeta");
  double _endcapMUeta = pset.getParameter<double>("endcapMUeta");
  barrelMUeta = endcapMUeta = maxMUeta;
  for(int i = 0; i < maxMUeta; i++) {
    if(fabs(_barrelMUeta) <= etaGridMU[i]) { barrelMUeta = i; break; }      
  }
  for(int i = 0; i < maxMUeta; i++) {
    if(fabs(_endcapMUeta) <= etaGridMU[i]) { endcapMUeta = i; break; }      
  }
  int maxMUetas[] = {endcapMUeta - barrelMUeta, maxMUeta - endcapMUeta};
  
  //initialize 3D vector
  responseMU = std::vector<std::vector<std::vector<double> > >(maxMUe,std::vector<std::vector<double> >(maxMUeta,std::vector<double>(maxMUbin,0)));
  
  //fill in 3D vector
  //(complementary cumulative distribution functions, from normalized response distributions)
  loc = eta_loc = -1;
  for(int i = 0; i < maxMUe; i++){
    for(int j = 0; j < maxMUeta; j++){
	  //check location - barrel, endcap, or forward
	  if(j==barrelMUeta) {loc = 0; eta_loc = barrelMUeta;}
	  else if(j==endcapMUeta) {loc = 1; eta_loc = endcapMUeta;}
	  
	  for(int k = 0; k < maxMUbin; k++){
	    responseMU[i][j][k] = _responseMU[loc][i*maxMUetas[loc]*maxMUbin + (j-eta_loc)*maxMUbin + k];
		
		if(debug) {
	    //cout.width(6);
	    LogInfo("FastCalorimetry") << " responseMU " << i << " " << j << " " << k  << " = " 
				      << responseMU[i][j][k] << std::endl;
	    }
	  }
	}
  }

// values for EM response in HF
//--------------------------------------------------------------------
  maxEMe = pset.getParameter<int>("maxEMe");
  maxEMeta = maxHDetas[2];
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

      if (useAdHocCorrections_) {// these correction factors make no sense when the FullDigitizer is used, and when working in Upgrades scenarios
	if( j < endcapHDeta)        factor = barrelCorrection[i];  // special HB
	else if( j < forwardHDeta)  factor = endcapCorrection[i];  // special HE
	else                        factor = forwardCorrectionEnergyDependent[i]*forwardCorrectionEtaDependent[j-forwardHDeta]; // special HF
      } else factor = 1.;	  

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
HCALResponse::responseHCAL(int _mip, double energy, double eta, int partype){
  int ieta = abs((int)(eta / etaStep)) ;
  int ie = -1;

  int mip;
  if(usemip) mip = _mip;
  else mip = 2; //ignore mip, use only overall (mip + nomip) parameters
  
  mean  = 0.;
  sigma = 0.;

  // e/gamma in HF
  if(partype == 0) {
    ieta -= forwardHDeta;  // HF starts at ieta=30 till ieta=51 
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

  // debugging
  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::responseHCAL, partype = " <<  partype 
				<< " E, eta = " << energy << " " << eta  
				<< "  mean & sigma = " << mean   << " " << sigma << std::endl;
  }  
  
  return std::pair<double,double>(mean,sigma);
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
