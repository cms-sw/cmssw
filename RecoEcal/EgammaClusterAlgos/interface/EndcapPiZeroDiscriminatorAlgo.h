#ifndef RecoEcal_EgammaClusterAlgos_EndcapPiZeroDiscriminatorAlgo_h
#define RecoEcal_EgammaClusterAlgos_EndcapPiZeroDiscriminatorAlgo_h

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"

// C/C++ headers
#include <string>
#include <vector>
#include <map>

// authors A. Kyriakis , D. Maletic

class EndcapPiZeroDiscriminatorAlgo {
public:
  typedef std::map<DetId, EcalRecHit> RecHitsMap;

  EndcapPiZeroDiscriminatorAlgo() : preshStripEnergyCut_(0.), preshSeededNstr_(5) {}

  EndcapPiZeroDiscriminatorAlgo(double stripEnergyCut, int nStripCut, const std::string& path);

  std::vector<float> findPreshVector(ESDetId strip, RecHitsMap* rechits_map, CaloSubdetectorTopology* topology_p);

  void findPi0Road(ESDetId strip, EcalPreshowerNavigator& theESNav, int plane, std::vector<ESDetId>& vout);

  bool goodPi0Strip(RecHitsMap::iterator candidate_it, ESDetId lastID);

  bool calculateNNInputVariables(
      std::vector<float>& vph1, std::vector<float>& vph2, float pS1_max, float pS9_max, float pS25_max, int EScorr);

  void calculateBarrelNNInputVariables(float et,
                                       double s1,
                                       double s9,
                                       double s25,
                                       double m2,
                                       double cee,
                                       double cep,
                                       double cpp,
                                       double s4,
                                       double s6,
                                       double ratio,
                                       double xcog,
                                       double ycog);

  float GetNNOutput(float EE_Et);

  float GetBarrelNNOutput(float EB_Et);

  std::vector<float> const& get_input_vector() const { return input_var; }

private:
  void readWeightFile(const char* WFile, int& Layers, int& Indim, int& Hidden, int& Outdim);
  float getNNoutput(int sel_wfile, int Layers, int Indim, int Hidden, int Outdim, int barrelstart) const;
  float Activation_fun(float SUM) const;

  double preshStripEnergyCut_;
  int preshSeededNstr_;
  int debugLevel_;

  int EE_Layers, EE_Indim, EE_Hidden, EE_Outdim;
  int EB_Layers, EB_Indim, EB_Hidden, EB_Outdim;

  std::vector<float> I_H_Weight_all;
  std::vector<float> H_O_Weight_all;
  std::vector<float> H_Thresh_all;
  std::vector<float> O_Thresh_all;

  std::vector<float> input_var;
  //   float input_var[25]; // array with the 25 variables to be used as input in NN
};
#endif
