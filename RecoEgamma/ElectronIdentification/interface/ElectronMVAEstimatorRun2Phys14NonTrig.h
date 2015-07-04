#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Phys14NonTrig_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Phys14NonTrig_H

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include <vector>
#include <string>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class ElectronMVAEstimatorRun2Phys14NonTrig : public AnyMVAEstimatorRun2Base{
  
 public:

  // Define here the number and the meaning of the categories
  // for this specific MVA
  const int nCategories = 6;
  enum mvaCategories {
    UNDEFINED = -1,
    CAT_EB1_PT5to10  = 0,
    CAT_EB2_PT5to10  = 1,
    CAT_EE_PT5to10   = 2,
    CAT_EB1_PT10plus = 3,
    CAT_EB2_PT10plus = 4,
    CAT_EE_PT10plus  = 5
  };

  // Define the struct that contains all necessary for MVA variables
  struct AllVariables {
    float kfhits;
    // Pure ECAL -> shower shapes
    float see;
    float spp;
    float OneMinusE1x5E5x5;
    float R9;
    float etawidth;
    float phiwidth;
    float HoE;
    // Endcap only variables
    float PreShowerOverRaw;
    //Pure tracking variables
    float kfchi2;
    float gsfchi2;
    // Energy matching
    float fbrem;
    float EoP;
    float eleEoPout;
    float IoEmIoP;
    // Geometrical matchings
    float deta;
    float dphi;
    float detacalo;
    // Spectator variables  
    float pt;
    float isBarrel;
    float isEndcap;
    float SCeta;
  };
  
  // Constructor and destructor
  ElectronMVAEstimatorRun2Phys14NonTrig(const edm::ParameterSet& conf);
  ~ElectronMVAEstimatorRun2Phys14NonTrig();

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& particle);
 
  // Utility functions
  TMVA::Reader *createSingleReader(const int iCategory, const edm::FileInPath &weightFile);

  inline int getNCategories(){return nCategories;};
  bool isEndcapCategory( int category );
  const inline std::string getName(){return name_;};

  // Functions that should work on both pat and reco electrons
  // (use the fact that pat::Electron inherits from reco::GsfElectron)
  void fillMVAVariables(const edm::Ptr<reco::Candidate>& particle);
  int findCategory( const edm::Ptr<reco::Candidate>& particle);
  // The function below ensures that the variables passed to MVA are 
  // within reasonable bounds
  void constrainMVAVariables();
  
 private:

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string name_ = "ElectronMVAEstimatorRun2Phys14NonTrig";

  // Data members
  std::vector< std::unique_ptr<TMVA::Reader> > _tmvaReaders;

  // All variables needed by this MVA
  std::string _MethodName;
  AllVariables _allMVAVars;
  
};

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
		  ElectronMVAEstimatorRun2Phys14NonTrig,
		  "ElectronMVAEstimatorRun2Phys14NonTrig");

#endif
