#ifndef RecoEgamma_PhotonIdentification_PhotonMVAEstimatorRun2Spring15NonTrig_H
#define RecoEgamma_PhotonIdentification_PhotonMVAEstimatorRun2Spring15NonTrig_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include <vector>
#include <string>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class PhotonMVAEstimatorRun2Spring15NonTrig : public AnyMVAEstimatorRun2Base{
  
 public:

  // Define here the number and the meaning of the categories
  // for this specific MVA
  const int nCategories = 2;
  enum mvaCategories {
    UNDEFINED = -1,
    CAT_EB  = 0,
    CAT_EE  = 1
  };

  // Define the struct that contains all necessary for MVA variables
  struct AllVariables {
    
    float varPhi;
    float varR9;
    float varSieie;
    float varSieip;
    float varE1x3overE5x5;
    float varE2x2overE5x5;
    float varE2x5overE5x5;
    float varSCEta;
    float varRawE;
    float varSCEtaWidth;
    float varSCPhiWidth;
    float varESEnOverRawE; // for endcap MVA only
    float varESEffSigmaRR; // for endcap MVA only
    // Pile-up
    float varRho;
    // Isolations
    float varPhoIsoRaw;
    float varChIsoRaw;
    float varWorstChRaw;
    // Spectators
    float varPt;
    float varEta;
  };
  
  // Constructor and destructor
  PhotonMVAEstimatorRun2Spring15NonTrig(const edm::ParameterSet& conf);
  ~PhotonMVAEstimatorRun2Spring15NonTrig();

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event&) const;
 
  // Utility functions
  std::unique_ptr<const GBRForest> createSingleReader(const int iCategory, const edm::FileInPath &weightFile);
  
  virtual int getNCategories() const { return nCategories; }
  bool isEndcapCategory( int category ) const;
  virtual const std::string& getName() const override final { return _name; }
  virtual const std::string& getTag() const override final { return _tag; }
  
  // Functions that should work on both pat and reco electrons
  // (use the fact that pat::Electron inherits from reco::GsfElectron)
  std::vector<float> fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event& iEvent) const override;
  int findCategory( const edm::Ptr<reco::Candidate>& particle ) const override;
  // The function below ensures that the variables passed to MVA are 
  // within reasonable bounds
  void constrainMVAVariables(AllVariables&) const;

  // Call this function once after the constructor to declare
  // the needed event content pieces to the framework
  void setConsumes(edm::ConsumesCollector&&) const override;
  // Call this function once per event to retrieve all needed
  // event content pices
  //void getEventContent(const edm::Event& iEvent) override;

  
 private:

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string _name = "PhotonMVAEstimatorRun2Spring15NonTrig";

  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  std::string _tag;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > _gbrForests;

  // All variables needed by this MVA
  const std::string _MethodName;
  AllVariables _allMVAVars;
  
  // This MVA implementation relies on several ValueMap objects
  // produced upstream. 

  //
  // Declare all tokens that will be needed to retrieve misc
  // data from the event content required by this MVA
  //
  const edm::InputTag _full5x5SigmaIEtaIEtaMapLabel; 
  const edm::InputTag _full5x5SigmaIEtaIPhiMapLabel; 
  const edm::InputTag _full5x5E1x3MapLabel; 
  const edm::InputTag _full5x5E2x2MapLabel; 
  const edm::InputTag _full5x5E2x5MaxMapLabel; 
  const edm::InputTag _full5x5E5x5MapLabel; 
  const edm::InputTag _esEffSigmaRRMapLabel; 
  //
  const edm::InputTag _phoChargedIsolationLabel; 
  const edm::InputTag _phoPhotonIsolationLabel; 
  const edm::InputTag _phoWorstChargedIsolationLabel; 
  const edm::InputTag _rhoLabel;
};

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
		  PhotonMVAEstimatorRun2Spring15NonTrig,
		  "PhotonMVAEstimatorRun2Spring15NonTrig");

#endif
