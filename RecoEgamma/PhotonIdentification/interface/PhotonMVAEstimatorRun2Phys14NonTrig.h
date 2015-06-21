#ifndef RecoEgamma_PhotonIdentification_PhotonMVAEstimatorRun2Phys14NonTrig_H
#define RecoEgamma_PhotonIdentification_PhotonMVAEstimatorRun2Phys14NonTrig_H

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include <vector>
#include <string>
#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class PhotonMVAEstimatorRun2Phys14NonTrig : public AnyMVAEstimatorRun2Base{
  
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
  PhotonMVAEstimatorRun2Phys14NonTrig(const edm::ParameterSet& conf);
  ~PhotonMVAEstimatorRun2Phys14NonTrig();

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
  const std::string name_ = "PhotonMVAEstimatorRun2Phys14NonTrig";

  // Data members
  std::vector< std::unique_ptr<TMVA::Reader> > _tmvaReaders;

  // All variables needed by this MVA
  std::string _MethodName;
  AllVariables _allMVAVars;
  
  // This MVA implementation relies on several ValueMap objects
  // produced upstream. 

  //
  // Declare all tokens that will be needed to retrieve misc
  // data from the event content required by this MVA
  //
  edm::EDGetTokenT<edm::ValueMap<float> > _full5x5SigmaIEtaIEtaMapToken; 
  edm::EDGetTokenT<edm::ValueMap<float> > _full5x5SigmaIEtaIPhiMapToken; 
  edm::EDGetTokenT<edm::ValueMap<float> > _full5x5E1x3MapToken; 
  edm::EDGetTokenT<edm::ValueMap<float> > _full5x5E2x2MapToken; 
  edm::EDGetTokenT<edm::ValueMap<float> > _full5x5E2x5MaxMapToken; 
  edm::EDGetTokenT<edm::ValueMap<float> > _full5x5E5x5MapToken; 
  edm::EDGetTokenT<edm::ValueMap<float> > _esEffSigmaRRMapToken; 
  //
  edm::EDGetTokenT<edm::ValueMap<float> > _phoChargedIsolationToken; 
  edm::EDGetTokenT<edm::ValueMap<float> > _phoPhotonIsolationToken; 
  edm::EDGetTokenT<edm::ValueMap<float> > _phoWorstChargedIsolationToken; 

  // 
  // Declare all value maps corresponding to the above tokens
  //
  edm::Handle<edm::ValueMap<float> > _full5x5SigmaIEtaIEtaMap;
  edm::Handle<edm::ValueMap<float> > _full5x5SigmaIEtaIPhiMap;
  edm::Handle<edm::ValueMap<float> > _full5x5E1x3Map;
  edm::Handle<edm::ValueMap<float> > _full5x5E2x2Map;
  edm::Handle<edm::ValueMap<float> > _full5x5E2x5MaxMap;
  edm::Handle<edm::ValueMap<float> > _full5x5E5x5Map;
  edm::Handle<edm::ValueMap<float> > _esEffSigmaRRMap;
  //
  edm::Handle<edm::ValueMap<float> > _phoChargedIsolationMap;
  edm::Handle<edm::ValueMap<float> > _phoPhotonIsolationMap;
  edm::Handle<edm::ValueMap<float> > _phoWorstChargedIsolationMap;

  // Rho will be pulled from the event content
  edm::EDGetTokenT<double> _rhoToken;
  edm::Handle<double> _rho;

};

DEFINE_EDM_PLUGIN(AnyMVAEstimatorRun2Factory,
		  PhotonMVAEstimatorRun2Phys14NonTrig,
		  "PhotonMVAEstimatorRun2Phys14NonTrig");

#endif
