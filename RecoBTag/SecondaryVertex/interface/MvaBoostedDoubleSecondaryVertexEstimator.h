#ifndef RecoBTag_SecondaryVertex_MvaBoostedDoubleSecondaryVertexEstimator_h
#define RecoBTag_SecondaryVertex_MvaBoostedDoubleSecondaryVertexEstimator_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class MvaBoostedDoubleSecondaryVertexEstimator {

  public:
    MvaBoostedDoubleSecondaryVertexEstimator(const std::string &);
    ~MvaBoostedDoubleSecondaryVertexEstimator();

   float mvaValue(float, float, float, float, float, float, float, float, float, float);

  private:

    TMVA::Reader* TMVAReader;

    float mva_PFLepton_ptrel, mva_z_ratio, mva_tau_dot, mva_SV_mass_0, mva_SV_EnergyRatio_0,
          mva_SV_EnergyRatio_1, mva_PFLepton_IP2D, mva_tau21, mva_nSL, mva_jetNTracksEtaRel;
    float mva_massGroomed, mva_flavour, mva_nbHadrons, mva_ptGroomed, mva_etaGroomed;
};

#endif // RecoBTag_SecondaryVertex_MvaBoostedDoubleSecondaryVertexEstimator_h

