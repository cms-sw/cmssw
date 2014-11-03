#ifndef RecoMET_METPUSubtraction_PFMEtSignInterfaceBase_h
#define RecoMET_METPUSubtraction_PFMEtSignInterfaceBase_h

/** \class PFMEtSignInterfaceBase
 *
 * Auxiliary class interfacing the TauAnalysis software to 
 *  RecoMET/METAlgorithms/interface/significanceAlgo.h 
 * for computing (PF)MEt significance
 * (see CMS AN-10/400 for description of the (PF)MEt significance computation)
 *
 * \author Christian Veelken, LLR
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoMET/METAlgorithms/interface/SignAlgoResolutions.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/SigInputObj.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TrackReco/interface/Track.h"


#include <TFile.h>
#include <TH2.h>


class PFMEtSignInterfaceBase
{
 public:

  PFMEtSignInterfaceBase(const edm::ParameterSet&);
  ~PFMEtSignInterfaceBase();

  template <typename T>
  metsig::SigInputObj compResolution(const T* particle) const
  {
    double pt   = particle->pt();
    double eta  = particle->eta();
    double phi  = particle->phi();
    
    if ( dynamic_cast<const reco::GsfElectron*>(particle) != 0 ||
	 dynamic_cast<const pat::Electron*>(particle) != 0 ) {
      std::string particleType = "electron";
      // WARNING: SignAlgoResolutions::PFtype2 needs to be kept in sync with reco::PFCandidate::e !!
      double dpt  = pfMEtResolution_->eval(metsig::PFtype2, metsig::ET,  pt, phi, eta);
      double dphi = pfMEtResolution_->eval(metsig::PFtype2, metsig::PHI, pt, phi, eta);
      //std::cout << "electron: pt = " << pt << ", eta = " << eta << ", phi = " << phi 
      //          << " --> dpt = " << dpt << ", dphi = " << dphi << std::endl;
      return metsig::SigInputObj(particleType, pt, phi, dpt, dphi);
    } else if ( dynamic_cast<const reco::Photon*>(particle) != 0 ||
		dynamic_cast<const pat::Photon*>(particle) != 0 ) {
      // CV: assume resolutions for photons to be identical to electron resolutions
      std::string particleType = "electron";
      // WARNING: SignAlgoResolutions::PFtype2 needs to be kept in sync with reco::PFCandidate::e !!
      double dpt  = pfMEtResolution_->eval(metsig::PFtype2, metsig::ET,  pt, phi, eta);
      double dphi = pfMEtResolution_->eval(metsig::PFtype2, metsig::PHI, pt, phi, eta);
      //std::cout << "photon: pt = " << pt << ", eta = " << eta << ", phi = " << phi 
      //          << " --> dpt = " << dpt << ", dphi = " << dphi << std::endl;
      return metsig::SigInputObj(particleType, pt, phi, dpt, dphi);
    } else if ( dynamic_cast<const reco::Muon*>(particle) != 0 ||
		dynamic_cast<const pat::Muon*>(particle) != 0 ) {
      std::string particleType = "muon";
      double dpt, dphi;
      const reco::Track* muonTrack = nullptr;
    if ( dynamic_cast<const pat::Muon*>(particle) != 0 ) {
      const pat::Muon* muon = dynamic_cast<const pat::Muon*>(particle);
      if ( muon->track().isNonnull() && muon->track().isAvailable() ) muonTrack = muon->track().get();
    } else if ( dynamic_cast<const reco::Muon*>(particle) != 0 ) {
      const reco::Muon* muon = dynamic_cast<const reco::Muon*>(particle);
      if ( muon->track().isNonnull() && muon->track().isAvailable() ) muonTrack = muon->track().get();
    } else assert(0);
    if ( muonTrack ) {
      dpt  = muonTrack->ptError();
      dphi = pt*muonTrack->phiError(); // CV: pt*dphi is indeed correct
    } else {
      // WARNING: SignAlgoResolutions::PFtype3 needs to be kept in sync with reco::PFCandidate::mu !!
      dpt  = pfMEtResolution_->eval(metsig::PFtype3, metsig::ET,  pt, phi, eta);
      dphi = pfMEtResolution_->eval(metsig::PFtype3, metsig::PHI, pt, phi, eta);
    }
    //std::cout << "muon: pt = " << pt << ", eta = " << eta << ", phi = " << phi 
    //	  << " --> dpt = " << dpt << ", dphi = " << dphi << std::endl;
    return metsig::SigInputObj(particleType, pt, phi, dpt, dphi);
    } else if ( dynamic_cast<const reco::PFTau*>(particle) != 0 ||
		dynamic_cast<const pat::Tau*>(particle) != 0 ) {
      // CV: use PFJet resolutions for PFTaus for now...
      //    (until PFTau specific resolutions are available)
      if ( dynamic_cast<const pat::Tau*>(particle) != 0 ) {
	const pat::Tau* pfTau = dynamic_cast<const pat::Tau*>(particle);
	//std::cout << "tau: pt = " << pt << ", eta = " << eta << ", phi = " << phi << std::endl;
	return pfMEtResolution_->evalPFJet(pfTau->pfJetRef().get());
      } else if ( dynamic_cast<const reco::PFTau*>(particle) != 0  ) {
	const reco::PFTau* pfTau = dynamic_cast<const reco::PFTau*>(particle);
	//std::cout << "tau: pt = " << pt << ", eta = " << eta << ", phi = " << phi << std::endl;
	return pfMEtResolution_->evalPFJet(pfTau->jetRef().get());
      } else assert(0);
    } else if ( dynamic_cast<const reco::PFJet*>(particle) != 0 ||
		dynamic_cast<const pat::Jet*>(particle) != 0 ) {
      metsig::SigInputObj pfJetResolution;
      if ( dynamic_cast<const reco::PFJet*>(particle) != 0 ) {
	const reco::PFJet* pfJet = dynamic_cast<const reco::PFJet*>(particle);
	pfJetResolution = pfMEtResolution_->evalPFJet(pfJet);
      } else if ( dynamic_cast<const pat::Jet*>(particle) != 0 ) {
	const pat::Jet* jet = dynamic_cast<const pat::Jet*>(particle);
	if ( jet->isPFJet() ) {
	  reco::PFJet pfJet(jet->p4(), jet->vertex(), jet->pfSpecific(), jet->getJetConstituents());
	  pfJetResolution = pfMEtResolution_->evalPFJet(&pfJet);
	} else throw cms::Exception("addPFMEtSignObjects")
	    << "PAT jet not of PF-type !!\n";
      } else assert(0);
      //std::cout << "pfJet: pt = " << pt << ", eta = " << eta << ", phi = " << phi << std::endl;
      // CV: apply additional jet energy resolution corrections
      //     not included in (PF)MEt significance algorithm yet
      //    (cf. CMS AN-11/400 vs. CMS AN-11/330)
      if ( lut_ && pt > 10. ) {
	double x = std::abs(eta);
	double y = pt;
	if ( x > lut_->GetXaxis()->GetXmin() && x < lut_->GetXaxis()->GetXmax() &&
	     y > lut_->GetYaxis()->GetXmin() && y < lut_->GetYaxis()->GetXmax() ) {
	  int binIndex = lut_->FindBin(x, y);
	  double addJERcorrFactor = lut_->GetBinContent(binIndex);
	  //std::cout << " addJERcorrFactor = " << addJERcorrFactor << std::endl;
	  pfJetResolution.set(pfJetResolution.get_type(),
			      pfJetResolution.get_energy(),
			      pfJetResolution.get_phi(),
			      addJERcorrFactor*pfJetResolution.get_sigma_e(),
			      pfJetResolution.get_sigma_tan());
	}
      }
      return pfJetResolution;
    } else if ( dynamic_cast<const reco::PFCandidate*>(particle) != 0 ) {
      const reco::PFCandidate* pfCandidate = dynamic_cast<const reco::PFCandidate*>(particle);
      //std::cout << "pfCandidate: pt = " << pt << ", eta = " << eta << ", phi = " << phi << std::endl;
      return pfMEtResolution_->evalPF(pfCandidate);
    } else throw cms::Exception("addPFMEtSignObjects")
	<< "Invalid type of particle:"
	<< " valid types = { reco::GsfElectron/pat::Electron, reco::Photon/pat::Photon, reco::Muon/pat::Muon, reco::PFTau/pat::Tau," 
	<< " reco::PFJet/pat::Jet, reco::PFCandidate } !!\n";
  }

  template <typename T>
  std::vector<metsig::SigInputObj> compResolution(const std::list<T*>& particles) const
  {
    LogDebug("compResolution")
      << " particles: entries = " << particles.size() << std::endl;
    
    std::vector<metsig::SigInputObj> pfMEtSignObjects;
    addPFMEtSignObjects(pfMEtSignObjects, particles);
    
    return pfMEtSignObjects;
  }

  reco::METCovMatrix operator()(const std::vector<metsig::SigInputObj>&) const;

  template <typename T>
    reco::METCovMatrix operator()(const std::list<T*>& particles) const
  {
    std::vector<metsig::SigInputObj> pfMEtSignObjects = compResolution(particles);
    return this->operator()(pfMEtSignObjects);
  }

 protected:

  template<typename T>
  void addPFMEtSignObjects(std::vector<metsig::SigInputObj>& metSignObjects, 
			   const std::list<T*>& particles) const
  {
    for ( typename std::list<T*>::const_iterator particle = particles.begin();
	  particle != particles.end(); ++particle ) {
      metSignObjects.push_back(this->compResolution(*particle));
    }
  }

 private:

  metsig::SignAlgoResolutions* pfMEtResolution_;

  // CV: look-up table for additional jet energy resolution corrections
  //     not included in (PF)MEt significance algorithm yet
  //    (cf. CMS AN-11/400 vs. CMS AN-11/330)
  TFile* inputFile_;
  TH2* lut_;

  int verbosity_;
};

#endif
