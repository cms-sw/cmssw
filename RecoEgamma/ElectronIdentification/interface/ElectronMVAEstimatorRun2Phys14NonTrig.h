#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Phys14NonTrig_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Phys14NonTrig_H

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include <vector>
#include <string>
#include <memory>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodBDT.h"

class ElectronMVAEstimatorRun2Phys14NonTrig : public AnyMVAEstimatorRun2Base{
  
 public:

  // Define here the number and the meaning of the categories
  // for this specific MVA
  static constexpr int nCategories = 6;
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
    float kfhits; // 0
    // Pure ECAL -> shower shapes
    float see; // 1
    float spp; // 2
    float OneMinusE1x5E5x5; // 3
    float R9; // 4
    float etawidth; // 5
    float phiwidth; // 6
    float HoE; // 7
    // Endcap only variables
    float PreShowerOverRaw; // 8
    //Pure tracking variables
    float kfchi2; // 9 
    float gsfchi2; // 10
    // Energy matching
    float fbrem; // 11
    float EoP; // 12
    float eleEoPout; // 13
    float IoEmIoP; // 14
    // Geometrical matchings
    float deta; // 15
    float dphi; // 16
    float detacalo; // 17 
    // Spectator variables  
    float pt; // 18
    float isBarrel; // 19
    float isEndcap; // 20
    float SCeta; // 21
  };
  
  // Constructor and destructor
  ElectronMVAEstimatorRun2Phys14NonTrig(const edm::ParameterSet& conf);
  ~ElectronMVAEstimatorRun2Phys14NonTrig();

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event& evt) const;
 
  // Utility functions
  std::unique_ptr<const GBRForest> createSingleReader(const int iCategory, const edm::FileInPath &weightFile) ;
  
  virtual int getNCategories() const override final { return nCategories; }
  bool isEndcapCategory( int category ) const;
  virtual const std::string& getName() const override final { return _name; } 
  virtual const std::string& getTag() const override final { return _tag; }
  
  // Functions that should work on both pat and reco electrons
  // (use the fact that pat::Electron inherits from reco::GsfElectron)
  std::vector<float> fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event&) const;
  int findCategory(const edm::Ptr<reco::Candidate>& particle) const;
  // The function below ensures that the variables passed to MVA are 
  // within reasonable bounds
  void constrainMVAVariables(AllVariables& vars) const;
  
 private:

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string _name = "ElectronMVAEstimatorRun2Phys14NonTrig";
  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  std::string _tag;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > _gbrForests;

  // All variables needed by this MVA
  std::string _MethodName;
  AllVariables _allMVAVars;
  
};

#endif
