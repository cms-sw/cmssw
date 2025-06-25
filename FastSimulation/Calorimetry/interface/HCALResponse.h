#ifndef FastSimulation_Calorimetry_HCALResponse_h
#define FastSimulation_Calorimetry_HCALResponse_h

/** \file FastSimulation/Calorimetry/interface/HCALResponse.h. 
 *
 *  Helper class to contain the HCAL response to hadrons and e/gamma
 * 
 */

#include "FastSimulation/Utilities/interface/DoubleCrystalBallGenerator.h"
#include <vector>
#include <string>
#include <utility>

//define multidimensional vector types
typedef std::vector<double> vec1;
typedef std::vector<vec1> vec2;
typedef std::vector<vec2> vec3;
typedef std::vector<vec3> vec4;
typedef std::vector<vec4> vec5;
enum part { hcbarrel = 0, hcendcap = 1, hcforward = 2 };
enum type { ECAL = 0, HCAL = 1, VFCAL = 2 };

class RandomEngineAndDistribution;

namespace edm {
  class ParameterSet;
}

class HCALResponse {
public:
  HCALResponse(const edm::ParameterSet& pset);
  ~HCALResponse() {}

  // Get the response smearing factor
  // for  e/gamma = 0, hadron = 1, mu = 2, mip: 0/1/2
  // mip = 2 means "mean" response regardless actual mip
  double responseHCAL(int _mip, double energy, double eta, int partype, RandomEngineAndDistribution const*) const;

  //Get the energy and eta dependent mip fraction
  double getMIPfraction(double energy, double eta) const;

  // legacy methods using simple formulae
  double getHCALEnergyResponse(double e, int hit, RandomEngineAndDistribution const*) const;

  // correct HF response for SL
  std::pair<vec1,vec1> correctHF(double e, int type) const;

private:
  // calculates interpolated-extrapolated response smearing factors
  // for hadrons, muons, and e/gamma (the last in HF specifically)
  double interHD(int mip, double e, int ie, int ieta, int det, RandomEngineAndDistribution const*) const;
  double interEM(double e, int ie, int ieta, RandomEngineAndDistribution const*) const;
  double interMU(double e, int ie, int ieta, RandomEngineAndDistribution const*) const;

  //random shooting functions w/ protection from negative energies
  double gaussShootNoNegative(double e, double sigma, RandomEngineAndDistribution const*) const;
  double cballShootNoNegative(
      double mu, double sigma, double aL, double nL, double aR, double nR, RandomEngineAndDistribution const*) const;
  double PoissonShootNoNegative(double e, double sigma, RandomEngineAndDistribution const*) const;

  //find subdet
  int getDet(int ieta) const;

  //debugging and mip toggles
  bool debug_, usemip_;

  //Default values for resolution parametrisation:
  //stochastic, constant and noise.
  //in the barrel and in the endcap
  //in the ECAL, HCAL, VFCAL
  double respPar_[3][2][3];

  //HCAL response parameters
  double eResponseScale_[3];
  double eResponsePlateau_[3];
  double eResponseExponent_;
  double eResponseCoefficient_;

  //max values
  int maxMUe_, maxMUeta_, maxMUbin_, maxEMe_, maxEMeta_;
  int maxHDe_[4];
  // eta step for eta index calc
  double etaStep_;
  // eta index for different regions
  int etaHD_[4], maxEtasHD_[3], barrelMUeta_, endcapMUeta_;
  // energy step of the tabulated muon data
  double muStep_;
  // correction factor for HF EM
  double respFactorEM_;

  // Tabulated energy, et/pt and eta points
  vec1 eGridHD_[4];
  vec1 eGridEM_;
  vec1 eGridMU_;
  vec1 etaGridMU_;

  // Tabulated response and mean for hadrons normalized to the energy
  // indices: parameters[par][mip][det][energy][eta]
  int nPar_;
  vec5 parameters_;

  // Tabulated response and mean for e/gamma in HF specifically (normalized)
  // indices: meanEM[energy][eta]
  vec2 meanEM_, sigmaEM_;

  // muon histos
  // indices: responseMU[energy][eta][bin]
  vec3 responseMU_;
  vec3 mipfraction_;
  vec3 poissonParameters_;

  // crystal ball generator
  DoubleCrystalBallGenerator cball_;

  // HF correction for SL
  int maxEta_, maxEne_;
  vec1 energyHF_;
  vec2 corrHFgEm_, corrHFgHad_;
  vec2 corrHFhEm_, corrHFhHad_;
};
#endif
