#ifndef ElectronMVAEstimatorRun2Fall17_h
#define ElectronMVAEstimatorRun2Fall17_h

#include <vector>
#include <string>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"
#include "RecoEgamma/EgammaTools/interface/GBRForestTools.h"

class ElectronMVAEstimatorRun2Fall17 : public AnyMVAEstimatorRun2Base{

 public:

  // Constructor and destructor
  ElectronMVAEstimatorRun2Fall17(const edm::ParameterSet& conf, bool withIso);
  ~ElectronMVAEstimatorRun2Fall17() override;

  // Calculation of the MVA value (VID accessor)
  float mvaValue( const edm::Ptr<reco::Candidate>& particle, const edm::Event&) const override;
  // Calculation of the MVA value (fwlite-compatible accessor)
  float mvaValue( const reco::GsfElectron * particle, const edm::EventBase &) const ;
  // Calculation of the MVA value (bare version)
  float mvaValue( const int iCategory, const std::vector<float> & vars) const ;

  // Utility functions
  int getNCategories() const override { return nCategories_; }
  bool isEndcapCategory( int category ) const;
  const std::string& getName() const final { return name_; }
  const std::string& getTag() const final { return tag_; }

  // Functions that should work on both pat and reco electrons
  // (use the fact that pat::Electron inherits from reco::GsfElectron)
  std::vector<float> fillMVAVariables(const edm::Ptr<reco::Candidate>& particle, const edm::Event&) const override;
  std::vector<float> fillMVAVariables( const reco::GsfElectron * particle, const edm::Handle<reco::ConversionCollection> conversions, const reco::BeamSpot *beamSpot, const edm::Handle<double> rho) const;
  int findCategory( const edm::Ptr<reco::Candidate>& particle) const override;
  int findCategory( const reco::GsfElectron * particle) const ;
  // The function below ensures that the variables passed to MVA are
  // within reasonable bounds
  void constrainMVAVariables(std::vector<float>&) const;

  // Call this function once after the constructor to declare
  // the needed event content pieces to the framework
  void setConsumes(edm::ConsumesCollector&&) const override;

 protected:

  // Define here the number and the meaning of the categories
  // for this specific MVA
  const int nCategories_ = 6;
  const int nVar_ = 22;
  enum MVACategories_ {
    UNDEFINED = -1,
    CAT_EB1_PT5to10  = 0,
    CAT_EB2_PT5to10  = 1,
    CAT_EE_PT5to10   = 2,
    CAT_EB1_PT10plus = 3,
    CAT_EB2_PT10plus = 4,
    CAT_EE_PT10plus  = 5
  };

  // MVA tag. This is an additional string variable to distinguish
  // instances of the estimator of this class configured with different
  // weight files.
  const std::string tag_;

  // MVA name. This is a unique name for this MVA implementation.
  // It will be used as part of ValueMap names.
  // For simplicity, keep it set to the class name.

  const std::string name_;

  // Data members
  std::vector< std::unique_ptr<const GBRForest> > gbrForests_;

  const std::string methodName_;

  //
  // Declare all labels and tokens that will be needed to retrieve misc
  // data from the event content required by this MVA
  //
  const edm::InputTag beamSpotLabel_;
  const edm::InputTag conversionsLabelAOD_;
  const edm::InputTag conversionsLabelMiniAOD_;
  const edm::InputTag rhoLabel_;

  //edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  //edm::EDGetTokenT<reco::ConversionCollection> conversionsTokenAOD_;
  //edm::EDGetTokenT<reco::ConversionCollection> conversionsTokenMiniAOD_;
  //edm::EDGetTokenT<double> rhoToken_;

  bool withIso_;
};

#endif
