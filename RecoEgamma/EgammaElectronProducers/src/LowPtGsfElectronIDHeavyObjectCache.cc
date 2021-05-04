#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "RecoEgamma/EgammaElectronProducers/interface/LowPtGsfElectronIDHeavyObjectCache.h"
#include <string>

namespace lowptgsfeleid {

  ////////////////////////////////////////////////////////////////////////////////
  //
  std::vector<float> Features::get() {
    std::vector<float> output = { 
      rho_,
      ele_pt_,
      sc_eta_,
      shape_full5x5_sigmaIetaIeta_,
      shape_full5x5_sigmaIphiIphi_,
      shape_full5x5_circularity_,
      shape_full5x5_r9_,
      sc_etaWidth_,
      sc_phiWidth_,
      shape_full5x5_HoverE_,
      trk_nhits_,
      trk_chi2red_,
      gsf_chi2red_,
      brem_frac_,
      gsf_nhits_,
      match_SC_EoverP_,
      match_eclu_EoverP_,
      match_SC_dEta_,
      match_SC_dPhi_,
      match_seed_dEta_,
      sc_E_,
      trk_p_,
    };
    return output;
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  //
  void Features::set( const reco::GsfElectron& ele, double rho ) {

    // KF tracks
    if ( ele.core().isNonnull() ) {
      reco::TrackRef trk = ele.core()->ctfTrack(); //@@ is this what we want?!
      if ( trk.isNonnull() ) {
	trk_p_ = float(trk->p());
	trk_nhits_ = float(trk->found());
	trk_chi2red_ = float(trk->normalizedChi2());
      }
    }

    // GSF tracks
    if ( ele.core().isNonnull() ) {
      reco::GsfTrackRef gsf = ele.core()->gsfTrack();
      if ( gsf.isNonnull() ) {
	gsf_nhits_ = gsf->found();
	gsf_chi2red_ = gsf->normalizedChi2();
      }
    }

    // Super clusters
    if ( ele.core().isNonnull() ) {
      reco::SuperClusterRef sc = ele.core()->superCluster();
      if ( sc.isNonnull() ) {
	sc_E_ = sc->energy();
	sc_eta_ = sc->eta();
	sc_etaWidth_ = sc->etaWidth();
	sc_phiWidth_ = sc->phiWidth();
      }
    }

    // Track-cluster matching
    match_seed_dEta_ = ele.deltaEtaSeedClusterTrackAtCalo();
    match_eclu_EoverP_ = (1./ele.ecalEnergy()) - (1./ele.p());
    match_SC_EoverP_ = ele.eSuperClusterOverP();
    match_SC_dEta_ = ele.deltaEtaSuperClusterTrackAtVtx();
    match_SC_dPhi_ = ele.deltaPhiSuperClusterTrackAtVtx();

    // Shower shape vars
    shape_full5x5_sigmaIetaIeta_ = ele.full5x5_sigmaIetaIeta();
    shape_full5x5_sigmaIphiIphi_ = ele.full5x5_sigmaIphiIphi();
    shape_full5x5_HoverE_    = ele.full5x5_hcalOverEcal();
    shape_full5x5_r9_ = ele.full5x5_r9();
    shape_full5x5_circularity_   = 1. - ele.full5x5_e1x5() / ele.full5x5_e5x5();

    // Misc
    rho_ = rho;
    brem_frac_ = ele.fbrem();
    ele_pt_ = ele.pt();
    
  };

  ////////////////////////////////////////////////////////////////////////////////
  //
  HeavyObjectCache::HeavyObjectCache( const edm::ParameterSet& conf ) {
    for ( auto& name : conf.getParameter< std::vector<std::string> >("ModelNames") ) 
      {
	names_.push_back(name);
      }
    for ( auto& weights : conf.getParameter< std::vector<std::string> >("ModelWeights") ) 
      {
	models_.push_back(createGBRForest(edm::FileInPath(weights)));
      }
    for ( auto& thresh : conf.getParameter< std::vector<double> >("ModelThresholds") )
      {
	thresholds_.push_back(thresh);
      }
    if ( names_.size() != models_.size() ) {
      throw cms::Exception("Incorrect configuration")
	<< "'ModelNames' size (" << names_.size()
	<< ") != 'ModelWeights' size (" << models_.size()
	<< ").\n";
    }
    if ( models_.size() != thresholds_.size() ) {
      throw cms::Exception("Incorrect configuration")
	<< "'ModelWeights' size (" << models_.size()
	<< ") != 'ModelThresholds' size (" << thresholds_.size()
	<< ").\n";
    }
    
  }

  ////////////////////////////////////////////////////////////////////////////////
  //
  double HeavyObjectCache::eval( const std::string& name,
				 const reco::GsfElectronRef& ele,
				 double rho ) const
  {
    std::vector<std::string>::const_iterator iter = std::find( names_.begin(), 
							       names_.end(), 
							       name );
    if ( iter != names_.end() ) {
      int index = std::distance(names_.begin(),iter);
      Features features;
      features.set(ele,rho);
      std::vector<float> inputs = features.get();
      return models_.at(index)->GetResponse( inputs.data() );
    } else {
      throw cms::Exception("Unknown model name")
	<< "'Name given: '" << name
	<< "'. Check against configuration file.\n";
    }
    return 0.;
  }

} // namespace lowptgsfeleid
