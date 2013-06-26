//FastSimulation headers
#include "FastSimulation/Calorimetry/interface/HCALResponse.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/Utilities/interface/DoubleCrystalBallGenerator.h"

// CMSSW Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <vector>
#include <string>
#include <math.h>

using namespace edm;

HCALResponse::HCALResponse(const edm::ParameterSet& pset, const RandomEngine* engine) :  random(engine), cball(random) {
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
  
//pion parameters
//--------------------------------------------------------------------
  //energy values
  maxHDe[0] = pset.getParameter<int>("maxHBe");
  maxHDe[1] = pset.getParameter<int>("maxHEe");
  maxHDe[2] = pset.getParameter<int>("maxHFe");
  eGridHD[0] = pset.getParameter<vec1>("eGridHB");
  eGridHD[1] = pset.getParameter<vec1>("eGridHE");
  eGridHD[2] = pset.getParameter<vec1>("eGridHF");
  
  //region eta indices calculated from eta values
  etaStep = pset.getParameter<double>("etaStep");
  //eta boundaries
  HDeta[0] = abs((int)(pset.getParameter<double>("HBeta") / etaStep));
  HDeta[1] = abs((int)(pset.getParameter<double>("HEeta") / etaStep));
  HDeta[2] = abs((int)(pset.getParameter<double>("HFeta") / etaStep));
  HDeta[3] = abs((int)(pset.getParameter<double>("maxHDeta") / etaStep)); //add 1 because this is the max index
  //eta ranges
  maxHDetas[0] = HDeta[1] - HDeta[0];
  maxHDetas[1] = HDeta[2] - HDeta[1];
  maxHDetas[2] = HDeta[3] - HDeta[2];
  
  //parameter info
  nPar = pset.getParameter<int>("nPar");
  parNames = pset.getParameter<std::vector<std::string> >("parNames");
  std::string detNames[] = {"_HB","_HE","_HF"};
  std::string mipNames[] = {"_mip","_nomip",""};
  
  //setup parameters (5D vector)
  parameters = vec5(nPar,vec4(3,vec3(3)));
  for(int p = 0; p < nPar; p++){ //loop over parameters
    for(int m = 0; m < 3; m++){ //loop over mip, nomip, total
	  for(int d = 0; d < 3; d++){ //loop over dets: HB, HE, HF
	    //get from python
		std::string pname = parNames[p] + detNames[d] + mipNames[m];
		vec1 tmp = pset.getParameter<vec1>(pname);
	  
	    //resize vector for energy range of det d
	    parameters[p][m][d].resize(maxHDe[d]);
		
		for(int i = 0; i < maxHDe[d]; i++){ //loop over energy for det d
		  //resize vector for eta range of det d
		  parameters[p][m][d][i].resize(maxHDetas[d]);
		  
		  for(int j = 0; j < maxHDetas[d]; j++){ //loop over eta for det d
		    //fill in parameters vector from python
			parameters[p][m][d][i][j] = tmp[i*maxHDetas[d] + j];
		  }
		}
	  }
	}
  }

// MUON probability histos for bin size = 0.25 GeV (0-10 GeV, 40 bins)
//--------------------------------------------------------------------
  muStep  = pset.getParameter<double>("muStep");
  maxMUe = pset.getParameter<int>("maxMUe");
  maxMUeta = pset.getParameter<int>("maxMUeta");
  maxMUbin = pset.getParameter<int>("maxMUbin");
  eGridMU = pset.getParameter<vec1>("eGridMU");
  etaGridMU = pset.getParameter<vec1>("etaGridMU");
  vec1 _responseMU[2] = {pset.getParameter<vec1>("responseMUBarrel"),pset.getParameter<vec1>("responseMUEndcap")};
  
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
  responseMU = vec3(maxMUe,vec2(maxMUeta,vec1(maxMUbin,0)));
  
  //fill in 3D vector
  //(complementary cumulative distribution functions, from normalized response distributions)
  int loc, eta_loc;
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
  eGridEM = pset.getParameter<vec1>("eGridEM");
 
  // e-gamma mean response and sigma in HF 
  vec1 _meanEM = pset.getParameter<vec1>("meanEM");
  vec1 _sigmaEM = pset.getParameter<vec1>("sigmaEM");

  //fill in 2D vectors (w/ correction factor applied)
  meanEM = vec2(maxEMe,vec1(maxEMeta,0));
  sigmaEM = vec2(maxEMe,vec1(maxEMeta,0));
  for(int i = 0; i < maxEMe; i++){
    for(int j = 0; j < maxEMeta; j++){
	  meanEM[i][j] = respFactorEM * _meanEM[i*maxEMeta + j];
	  sigmaEM[i][j] = respFactorEM * _sigmaEM[i*maxEMeta + j];
	}
  }

}

 
double HCALResponse::responseHCAL(int _mip, double energy, double eta, int partype){
  int ieta = abs((int)(eta / etaStep)) ;
  int ie = -1;

  int mip;
  if(usemip) mip = _mip;
  else mip = 2; //ignore mip, use only overall (mip + nomip) parameters

  double mean = 0;
  
  // e/gamma in HF
  if(partype == 0) {
    //check eta
    ieta -= HDeta[2];  // HF starts at ieta=30 till ieta=51 
                 // but resp.vector from index=0 through 20
    if(ieta >= maxEMeta ) ieta = maxEMeta-1;
    else if(ieta < 0) ieta = 0;
 
    //find energy range
    for (int i = 0; i < maxEMe; i++) {
      if(energy < eGridEM[i])  {
	    if(i == 0) ie = 0; // less than minimal - back extrapolation with the 1st interval  
        else  ie = i-1;
        break;
      }
    }
    if(ie == -1) ie = maxEMe - 2; // more than maximum - extrapolation with last interval
	
	//do smearing
    mean = interEM(energy, ie, ieta);
  }
  
  // hadrons
  else if(partype == 1) {
      //check eta and det
	  int det = getDet(ieta);
	  int deta = ieta - HDeta[det];
      if(deta >= maxHDetas[det]) deta = maxHDetas[det] - 1;
      else if(deta < 0 ) deta = 0;
	  
	  //find energy range
      for (int i = 0; i < maxHDe[det]; i++) {
	    if(energy < eGridHD[det][i])  {
	      if(i == 0) ie = 0; // less than minimal - back extrapolation with the 1st interval
	      else  ie = i-1;
	      break;
	    }	
      }
      if(ie == -1) ie = maxHDe[det] - 2; // more than maximum - extrapolation with last interval
      
	  //do smearing
      mean = interHD(mip, energy, ie, deta, det);
  }

  
  // muons
  else if(partype == 2) { 
    //check eta
    ieta = maxMUeta;
    for(int i = 0; i < maxMUeta; i++) {
      if(fabs(eta) < etaGridMU[i]) {
	    ieta = i;  
	    break;
      }      
    }
    if(ieta < 0) ieta = 0;
	
    if(ieta < maxMUeta) {  // HB-HE
	  //find energy range
      for (int i = 0; i < maxMUe; i++) {
	    if(energy < eGridMU[i])  {
	      if(i == 0) ie = 0; // less than minimal - back extrapolation with the 1st interval
	      else  ie = i-1;
	      break;
	    }
      }
	  if(ie == -1) ie = maxMUe - 2; // more than maximum - extrapolation with last interval
	  
	  //do smearing
      mean = interMU(energy, ie, ieta);
	  if(mean > energy) mean = energy;
    }
  }

  // debugging
  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::responseHCAL, partype = " <<  partype 
				<< " E, eta = " << energy << " " << eta  
				<< "  mean = " << mean << std::endl;
  }  
  
  return mean;
}

double HCALResponse::interMU(double e, int ie, int ieta) {
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

  //linear interpolation
  double mean = (y1*(x2-e) + y2*(e-x1))/(x2-x1);

  if(debug) {
    //cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interMU " << std::endl
				<< " e, ie, ieta = " << e << " " << ie << " " << ieta << std::endl
				<< " response  = " << mean << std::endl; 
  }

  return mean;
}

double HCALResponse::interHD(int mip, double e, int ie, int ieta, int det) {
  double y1, y2;

  double x1 = eGridHD[det][ie];
  double x2 = eGridHD[det][ie+1];
  
  vec1 pars(nPar,0);

  //calculate all parameters
  for(int p = 0; p < nPar; p++){
	y1 = parameters[p][mip][det][ie][ieta];
	y2 = parameters[p][mip][det][ie+1][ieta];

	//par-specific checks
	double custom = 0;
	bool use_custom = false;
	
	//do not let mu or sigma get extrapolated below zero for low energies
	//especially important for HF since extrapolation is used for E < 15 GeV
	if((p==0 || p==1) && e < x1){
		double tmp = (y1*x2-y2*x1)/(x2-x1); //extrapolate down to e=0
		if(tmp<0) { //require mu,sigma > 0 for E > 0
			custom = y1*e/x1;
			use_custom = true;
		}
	}
	//tail parameters have lower bounds - never extrapolate down
	else if((p==2 || p==3 || p==4 || p==5)){
		if(e < x1 && y1 < y2){
			custom = y1;
			use_custom = true;
		}
		else if(e > x2 && y2 < y1){
			custom = y2;
			use_custom = true;
		}
	}
	
	//linear interpolation
	if(use_custom) pars[p] = custom;
	else pars[p] = (y1*(x2-e) + y2*(e-x1))/(x2-x1);
  }
  
  //random smearing
  double mean = 0;
  if(nPar==6) mean = cballShootNoNegative(pars[0],pars[1],pars[2],pars[3],pars[4],pars[5]);
  else if(nPar==2) mean = gaussShootNoNegative(pars[0],pars[1]); //gaussian fallback

  return mean;
}


double HCALResponse::interEM(double e, int ie, int ieta) {
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

  //linear interpolation
  double mean = (y1*(x2-e) + y2*(e-x1))/(x2-x1);     
  
  y1 = sigmaEM[ie][ieta]; 
  y2 = sigmaEM[ie+1][ieta]; 
  
  if(debug) {
    //  cout.width(6);
    LogInfo("FastCalorimetry") << std::endl
				<< " HCALResponse::interEM sigma" << std::endl
				<< " x, x1-x2, y1-y2 = " 
				<< e << ", " << x1 <<"-" << x2 << " " << y1 <<"-" << y2 << std::endl;
  }

  //linear interpolation
  double sigma = (y1*(x2-e) + y2*(e-x1))/(x2-x1);
  
  //random smearing
  double rndm = gaussShootNoNegative(mean,sigma);
  
  return rndm;
}

// Old parameterization of the calo response to hadrons
double HCALResponse::getHCALEnergyResponse(double e, int hit){
  //response
  double s = eResponseScale[hit];
  double n = eResponseExponent;
  double p = eResponsePlateau[hit];
  double c = eResponseCoefficient;

  double response = e * p / (1+c*exp(n * log(s/e)));

  if(response<0.) response = 0.;

  //resolution
  double resolution;
  if(hit==hcforward) 
    resolution = e *sqrt( RespPar[VFCAL][1][0]*RespPar[VFCAL][1][0] / e + RespPar[VFCAL][1][1]*RespPar[VFCAL][1][1] );
  else
    resolution = e * sqrt( RespPar[HCAL][hit][0]*RespPar[HCAL][hit][0]/(e) + RespPar[HCAL][hit][1]*RespPar[HCAL][hit][1] );   
  
  //random smearing
  double rndm = gaussShootNoNegative(response,resolution);
  
  return rndm;
}

//find subdet and eta offset
int HCALResponse::getDet(int ieta){
	int d;
	for(d = 0; d < 2; d++){
		if(ieta < HDeta[d+1]){
			break;
		}
	}
	return d;
}

// Remove (most) hits with negative energies
double HCALResponse::gaussShootNoNegative(double e, double sigma) {
  double out = random->gaussShoot(e,sigma);
  if (e >= 0.) {
    while (out < 0.) out = random->gaussShoot(e,sigma);
  }
  //else give up on re-trying, otherwise too much time can be lost before emeas comes out positive

  return out;
}

// Remove (most) hits with negative energies
double HCALResponse::cballShootNoNegative(double mu, double sigma, double aL, double nL, double aR, double nR) {
  double out = cball.shoot(mu,sigma,aL,nL,aR,nR);
  if (mu >= 0.) {
    while (out < 0.) out = cball.shoot(mu,sigma,aL,nL,aR,nR);
  }
  //else give up on re-trying, otherwise too much time can be lost before emeas comes out positive

  return out;
}