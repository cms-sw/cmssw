#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Spring15Trig25ns_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Spring15Trig25ns_H

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include <vector>
#include <string>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class ElectronMVAEstimatorRun2Spring15Trig25ns : public AnyMVAEstimatorRun2Base{
  
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
    float kfhits;
    float kfchi2;
    float gsfchi2;
    // Energy matching
    float fbrem;

    int gsfhits;
    int expectedMissingInnerHits;
    float convVtxFitProbability;

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
    //
    float eClass;               
    float pfRelIso;             
    float expectedInnerHits;
    float vtxconv;   
    float mcEventWeight;
    float mcCBMatchingCategory;
    
  };
  
  // Constructor and destructor
  ElectronMVAEstimatorRun2Spring15Trig25ns(const edm::ParameterSet& conf);
  ~ElectronMVAEstimatorRun2Spring15Trig25ns();

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& particle);
 
  // Utility functions
  TMVA::Reader *createSingleReader(const int iCategory, const edm::FileInPath &weightFile);

  inline int getNCategories(){return nCategories;};
  bool isEndcapCategory( int category );
  const inline std::string getName(){return _name;};
  const inline std::string getTag(){return _tag;};

  // Functions that should work on both pat and reco electrons
  // (use the fact that pat::Electron inherits from reco::GsfElectron)
  void fillMVAVariables(const edm::Ptr<reco::Candidate>& particle);
  int findCategory( const edm::Ptr<reco::Candidate>& particle);
  // The function below ensures that the variables passed to MVA are 
  // within reasonable bounds
  void constrainMVAVariables();

  // Call this function once after the constructor to declare
  // the needed event content pieces to the framework
  void setConsumes(edm::ConsumesCollector&&) override;
  // Call this function once per event to retrieve all needed
  // event content pices
  void getEventContent(const edm::Event& iEvent) override;
  
 private:

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string _name = "ElectronMVAEstimatorRun2Spring15Trig25ns";
  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  std::string _tag;

  // Data members
  std::vector< std::unique_ptr<TMVA::Reader> > _tmvaReaders;

  // All variables needed by this MVA
  std::string _MethodName;
  AllVariables _allMVAVars;

  //
  // Declare all tokens that will be needed to retrieve misc
  // data from the event content required by this MVA
  //
  edm::EDGetTokenT<reco::BeamSpot> _beamSpotToken;
  // Conversions in AOD and miniAOD have different names
  edm::EDGetTokenT<reco::ConversionCollection> _conversionsTokenAOD;
  edm::EDGetTokenT<reco::ConversionCollection> _conversionsTokenMiniAOD;
  // 
  // Declare all value maps corresponding to the above tokens
  //
  edm::Handle<reco::BeamSpot> _theBeamSpot;
  edm::Handle<reco::ConversionCollection> _conversions;
  
};

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
		  ElectronMVAEstimatorRun2Spring15Trig25ns,
		  "ElectronMVAEstimatorRun2Spring15Trig25ns");

#endif
