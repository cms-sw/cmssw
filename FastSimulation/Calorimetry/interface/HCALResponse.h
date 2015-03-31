#ifndef HCALResponse_h
#define HCALResponse_h

/** \file FastSimulation/Calorimetry/interface/HCALResponse.h. 
 *
 *  Helper class to contain the HCAL response to hadrons and e/gamma
 * 
 */

#include "FastSimulation/Utilities/interface/DoubleCrystalBallGenerator.h"
#include <vector>
#include <string>

//define multidimensional vector types
typedef std::vector<double> vec1;
typedef std::vector<vec1>   vec2;
typedef std::vector<vec2>   vec3;
typedef std::vector<vec3>   vec4;
typedef std::vector<vec4>   vec5;
enum part{hcbarrel=0, hcendcap=1, hcforward=2};
enum type{ECAL=0, HCAL=1, VFCAL=2};

class RandomEngineAndDistribution;

namespace edm { 
  class ParameterSet;
}

class HCALResponse
{
public:
  HCALResponse(const edm::ParameterSet& pset);
  ~HCALResponse(){ } 

  // Get the response smearing factor
  // for  e/gamma = 0, hadron = 1, mu = 2, mip: 0/1/2
  // mip = 2 means "mean" response regardless actual mip
  double responseHCAL(int _mip, double energy, double eta, int partype, RandomEngineAndDistribution const*);

  //Get the energy and eta dependent mip fraction
   double getMIPfraction(double energy, double eta);

  // legacy methods using simple formulae
  double getHCALEnergyResponse(double e, int hit, RandomEngineAndDistribution const*);

  // correct HF response for SL
  void correctHF(double e, int type);
  vec1 & getCorrHF() {return corrHF;}   
  
private:

  // calculates interpolated-extrapolated response smearing factors
  // for hadrons, muons, and e/gamma (the last in HF specifically)
  double interHD(int mip, double e, int ie, int ieta, int det, RandomEngineAndDistribution const*);
  double interEM(double e, int ie, int ieta, RandomEngineAndDistribution const*);
  double interMU(double e, int ie, int ieta, RandomEngineAndDistribution const*);
  
  //random shooting functions w/ protection from negative energies
  double gaussShootNoNegative(double e, double sigma,RandomEngineAndDistribution const*);
  double cballShootNoNegative(double mu, double sigma, double aL, double nL, double aR, double nR,RandomEngineAndDistribution const*);
  double PoissonShootNoNegative(double e, double sigma,RandomEngineAndDistribution const*);

  //find subdet
  int getDet(int ieta);
  
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

  //max values
  int maxMUe, maxMUeta, maxMUbin, maxEMe, maxEMeta;
  int maxHDe[4];
  // eta step for eta index calc
  double etaStep;
  // eta index for different regions
  int HDeta[4], maxHDetas[3], barrelMUeta, endcapMUeta;
  // energy step of the tabulated muon data
  double muStep;
  // correction factor for HF EM
  double respFactorEM;

  // Tabulated energy, et/pt and eta points
  vec1 eGridHD[4];
  vec1 eGridEM;
  vec1 eGridMU;
  vec1 etaGridMU;

  // Tabulated response and mean for hadrons normalized to the energy
  // indices: parameters[par][mip][det][energy][eta]
  int nPar;
  std::vector<std::string> parNames;
  vec5 parameters;
  
  // Tabulated response and mean for e/gamma in HF specifically (normalized)
  // indices: meanEM[energy][eta]
  vec2 meanEM, sigmaEM;

  // muon histos
  // indices: responseMU[energy][eta][bin]
  vec3 responseMU; 
  vec3 mipfraction;
  vec3 PoissonParameters;

  // crystal ball generator
  DoubleCrystalBallGenerator cball;

  // HF correction for SL
  int maxEta, maxEne;
  vec1 energyHF;
  vec1 corrHFg, corrHFh;
  vec1 corrHF;
};
#endif

