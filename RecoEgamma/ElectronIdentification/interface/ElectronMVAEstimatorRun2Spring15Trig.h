#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Spring15Trig_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Spring15Trig_H

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include <vector>
#include <string>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class ElectronMVAEstimatorRun2Spring15Trig : public AnyMVAEstimatorRun2Base{
  
 public:

  // Define here the number and the meaning of the categories
  // for this specific MVA
  const int nCategories = 3;
  enum mvaCategories {
    UNDEFINED = -1,
    CAT_EB1  = 0,
    CAT_EB2  = 1,
    CAT_EE   = 2
  };

  // Define the struct that contains all necessary for MVA variables
  // Note: all variables have to be floats for TMVA Reader, even if 
  // the training was done with ints.
  struct AllVariables {
    // Pure ECAL -> shower shapes
    float see; // 0
    float spp; // 1
    float OneMinusE1x5E5x5; // 2 
    float R9; // 3
    float etawidth; // 4
    float phiwidth; // 5
    float HoE; // 6

    // Endcap only variables
    float PreShowerOverRaw; // 7  

    //Pure tracking variables
    float kfhits; // 8
    float kfchi2; // 9
    float gsfchi2; // 10
    // Energy matching
    float fbrem; // 11

    float gsfhits; // 12
    float expectedMissingInnerHits; // 13
    float convVtxFitProbability; // 14

    float EoP; // 15
    float eleEoPout; // 16
    float IoEmIoP; // 17
    // Geometrical matchings
    float deta; // 18
    float dphi; // 19
    float detacalo; // 20 

    // Spectator variables  
    //   ... none in this version ...

    // Other variables. These ones are not needed for this version
    // of MVA, but kept in this data structure for convenience.
    float pt; // 21
    float SCeta; //22

    
  };
  
  // Constructor and destructor
  ElectronMVAEstimatorRun2Spring15Trig(const edm::ParameterSet& conf);
  ~ElectronMVAEstimatorRun2Spring15Trig();

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event&) const override;
 
  // Utility functions
  std::unique_ptr<const GBRForest> createSingleReader(const int iCategory, 
                                                      const edm::FileInPath &weightFile);

  virtual int getNCategories() const override { return nCategories; }
  bool isEndcapCategory( int category ) const;
  virtual const std::string& getName() const override final { return _name; }
  virtual const std::string& getTag() const override final { return _tag; }

  // Functions that should work on both pat and reco electrons
  // (use the fact that pat::Electron inherits from reco::GsfElectron)
  std::vector<float> fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event&) const override;
  int findCategory( const edm::Ptr<reco::Candidate>& particle) const override;
  // The function below ensures that the variables passed to MVA are 
  // within reasonable bounds
  void constrainMVAVariables(AllVariables&) const;

  // Call this function once after the constructor to declare
  // the needed event content pieces to the framework
  void setConsumes(edm::ConsumesCollector&&) const override final;
  // Call this function once per event to retrieve all needed
  // event content pices
  
 private:

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string _name = "ElectronMVAEstimatorRun2Spring15Trig";
  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  const std::string _tag;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > _gbrForests;

  // All variables needed by this MVA
  const std::string _MethodName;
  AllVariables _allMVAVars;

  //
  // Declare all tokens that will be needed to retrieve misc
  // data from the event content required by this MVA
  //
  const edm::InputTag _beamSpotLabel;
  // Conversions in AOD and miniAOD have different names
  const edm::InputTag _conversionsLabelAOD;
  const edm::InputTag _conversionsLabelMiniAOD;
  
  
};

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
		  ElectronMVAEstimatorRun2Spring15Trig,
		  "ElectronMVAEstimatorRun2Spring15Trig");

#endif
