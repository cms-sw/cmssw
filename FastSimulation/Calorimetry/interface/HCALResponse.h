#ifndef HCALResponse_h
#define HCALResponse_h

/** \file FastSimulation/Calorimetry/interface/HCALResponse.h. 
 *
 *  Helper class to contain the HCAL response to hadrons and e/gamma
 * 
 */

#include <map>
#include <vector>

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
  response responseHCAL(int _mip, double energy, double eta, int partype);

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

  //debugging and mip toggles
  bool debug, usemip;
  
  //Default values for resolution parametrisation:
  //stochastic, constant and noise.
  //in the barrel and in the endcap
  //in the ECAL, HCAL, VFCAL
  double RespPar[3][2][3];

  //HCAL response parameters
  double eResponseScale[3];
  double eResponsePlateau[3];
  double eResponseExponent;
  double eResponseCoefficient;
  double eResponseCorrection;
  double eBias;
  
  //correction factors
  bool useAdHocCorrections_;
  std::vector<double> barrelCorrection;
  std::vector<double> endcapCorrection;
  std::vector<double> forwardCorrectionEnergyDependent;
  std::vector<double> forwardCorrectionEtaDependent;

  //max values
  int maxHDe, maxHDeta, maxMUe, maxMUeta, maxMUbin, maxEMe, maxEMeta;

  // eta step for eta index calc
  double etaStep;
  // eta index for different regions
  int  barrelHDeta, endcapHDeta, forwardHDeta, barrelMUeta, endcapMUeta;
  // energy step of the tabulated muon data
  double muStep;
  // correction factor for HF EM
  double respFactorEM;
  // mean and sigma
  double mean, sigma; 

  // Tabulated energy, et/pt and eta points
  std::vector<double> eGridHD;
  std::vector<double> eGridEM;
  std::vector<double> eGridMU;
  std::vector<double> etaGridMU;

  // Tabulated response and mean for hadrons normalized to the energy
  // indices: meanHD[energy][eta]
  std::vector<std::vector<double> > meanHD, sigmaHD;
  std::vector<std::vector<double> > meanHD_mip, sigmaHD_mip;
  std::vector<std::vector<double> > meanHD_nomip, sigmaHD_nomip;
  
  // Tabulated response and mean for e/gamma in HF specifically (normalized)
  // indices: meanEM[energy][eta]
  std::vector<std::vector<double> > meanEM, sigmaEM;

  // muon histos
  // indices: responseMU[energy][eta][bin]
  std::vector<std::vector<std::vector<double> > > responseMU; 

  // Famos random engine
  const RandomEngine* random;

};
#endif

