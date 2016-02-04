#ifndef HCALResponse_h
#define HCALResponse_h

/** \file FastSimulation/Calorimetry/interface/HCALResponse.h. 
 *
 *  Helper class to contain the HCAL response to hadrons and e/gamma
 * 
 */

#include <map>

#define maxHDe    15   // energy points for hadrons  
#define maxHDeta  51   // eta    points for hadrons
#define maxEMe     6   // energy points  e/gamma in HF
#define maxEMeta  21   // eta points     e/gamma in HF
#define maxMUe     4   // energy points for muons
#define maxMUeta   6   
#define maxMUbin  40   // binning oh muon histograms  

typedef std::pair<double,double> response;
enum part{hcbarrel=0, hcendcap=1, hcforward=2};
enum type{ECAL=0, HCAL=1, VFCAL=2};

class RandomEngine;

namespace edm { 
  class ParameterSet;
}

class HCALResponse
{
public:
  HCALResponse(const edm::ParameterSet& pset,
	       const RandomEngine* engine);
  ~HCALResponse(){;} 

  // Get the response in the for of pair 
  // parameters:  energy, eta, e/gamma = 0, hadron = 1, mu = 2, mip: 0/1/2
  // mip = 2 means "mean" response regardless actual mip
  response responseHCAL(int mip, double energy, double eta, int partype);

  // legacy methods using simple furmulae
  double getHCALEnergyResponse   (double e, int hit);
  double getHCALEnergyResolution (double e, int hit);
  double getHFEnergyResolution   (double EGen);
  
private:

  // calculates interpolated-extrapolated reponse (mean and sigma, see below)
  // for hadrons and e/gamma (the latter in HF specifically)
  void interHD(int mip, double e, int ie, int ieta);
  void interEM(double e, int ie, int ieta); 
  void interMU(double e, int ie, int ieta); 

  ///Default values for resolution parametrisation:
  ///stochastic, constant and noise.
  //in the barrel and in the endcap
  //in the ECAL, HCAL, VFCAL
  double RespPar[3][2][3];

  ///HCAL response parameters
  double eResponseScale[3];
  double eResponsePlateau[3];
  double eResponseExponent;
  double eResponseCoefficient;
  double eResponseCorrection;
  double eBias;

  // just eta step of the tabulated data
  double etaStep, muStep;
  // mean and sigma
  double mean, sigma; 

  // Tabulated energy, et/pt and eta points
  double eGridHD  [maxHDe];
  double eGridEM  [maxEMe];
  double eGridMU  [maxMUe];
  double etaGridMU[maxMUeta];

  // Tabulated response and mean for hadrons normalized to the energy
  double meanHD      [maxHDe][maxHDeta], sigmaHD      [maxHDe][maxHDeta];
  double meanHD_mip  [maxHDe][maxHDeta], sigmaHD_mip  [maxHDe][maxHDeta];
  double meanHD_nomip[maxHDe][maxHDeta], sigmaHD_nomip[maxHDe][maxHDeta];
  
  // Tabulated response and mean for e/gamma in HF specifically (normalized) 
  double meanEM[maxEMe][maxEMeta], sigmaEM[maxEMe][maxEMeta];

  // muon histos 
  double responseMU[maxMUe][maxMUeta][maxMUbin]; 

  // Famos random engine
  const RandomEngine* random;

};
#endif

