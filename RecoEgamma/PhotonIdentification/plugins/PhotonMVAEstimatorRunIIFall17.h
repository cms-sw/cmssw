#ifndef RecoEgamma_PhotonIdentification_PhotonMVAEstimatorRunIIFall17_H
#define RecoEgamma_PhotonIdentification_PhotonMVAEstimatorRunIIFall17_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"

#include <vector>
#include <string>

#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/DataLoader.h"

class PhotonMVAEstimatorRunIIFall17 : public AnyMVAEstimatorRun2Base{
  
 public:

  // Define here the number and the meaning of the categories
  // for this specific MVA
  const uint nCategories = 2;
  enum mvaCategories {
    UNDEFINED = -1,
    CAT_EB  = 0,
    CAT_EE  = 1
  };

  // Define the struct that contains all necessary for MVA variables
  struct AllVariables {

    float varRawE;
    float varR9;
    float varSieie;
    float varSCEtaWidth;
    float varSCPhiWidth;
    float varSieip;
    float varE2x2overE5x5;
    float varSCEta;
    float varESEnOverRawE; // for endcap MVA only
    float varESEffSigmaRR; // for endcap MVA only
    // Pile-up
    float varRho;
    // Isolations
    float varPhoIsoRaw;
    float varChIsoRaw;
    float varWorstChRaw;

  };
  
  // Constructor and destructor
  PhotonMVAEstimatorRunIIFall17(const edm::ParameterSet& conf);
  ~PhotonMVAEstimatorRunIIFall17();

  // Calculation of the MVA value
  float mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event&) const;
 
  // Utility functions
  virtual int getNCategories() const { return nCategories; }
  bool isEndcapCategory( int category ) const;
  virtual const std::string& getName() const override final { return name_; }
  virtual const std::string& getTag() const override final { return tag_; }
  
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


  
 private:

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.
  const std::string name_ = "PhotonMVAEstimatorRunIIFall17";

  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  std::string tag_;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > gbrForests_;

  // All variables needed by this MVA
  const std::string methodName_;
  AllVariables allMVAVars_;
  
  // This MVA implementation relies on several ValueMap objects
  // produced upstream. 

  //
  // Declare all tokens that will be needed to retrieve misc
  // data from the event content required by this MVA
  //
  const edm::InputTag phoChargedIsolationLabel_; 
  const edm::InputTag phoPhotonIsolationLabel_; 
  const edm::InputTag phoWorstChargedIsolationLabel_; 
  const edm::InputTag rhoLabel_;

  // Other objects needed by the MVA
  EffectiveAreas effectiveAreas_;
  std::vector<double> phoIsoPtScalingCoeff_;

};

#endif
