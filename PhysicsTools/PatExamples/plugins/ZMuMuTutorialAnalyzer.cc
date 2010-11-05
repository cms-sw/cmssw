/* \class ZMuMuTutorialAnalyzer
 *
 * Z->mu+mu- tutorial analysis.
 * To be run over a dimuon skim collections
 * 
 * Produces mass spectra histograms for
 * ZMuMu candidates satisfying cuts
 * 
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "TH1.h"
#include <iostream>
#include <iterator>
using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;

class ZMuMuTutorialAnalyzer : public edm::EDAnalyzer {
public:
  ZMuMuTutorialAnalyzer(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endJob();
  
  InputTag dimuons_;
  float isocut_, etacut_, ptcut_, minZmass_, maxZmass_;
  float dRConeVeto_, ptThresholdVeto_, dRIsolationCone_;
  TH1F * h_muFromZ_pt_;
  TH1F * h_muFromZ_eta_;
  TH1F * h_zMuMuAll_mass_;
  TH1F * h_zMuMu_mass_;
  TH1F * h_zMuMuMatched_mass_;
  TH1F * h_zMuMuMC_mass_;
};

ZMuMuTutorialAnalyzer::ZMuMuTutorialAnalyzer(const edm::ParameterSet& pset) : 
  dimuons_( pset.getParameter<InputTag>( "src" ) ),
  isocut_( pset.getParameter<double>( "isocut" ) ),
  etacut_( pset.getParameter<double>( "etacut" ) ),
  ptcut_( pset.getParameter<double>( "ptcut" ) ),
  minZmass_( pset.getParameter<double>( "minZmass" )),
  maxZmass_( pset.getParameter<double>( "maxZmass" )),
  dRConeVeto_( pset.getParameter<double>( "dRConeVeto" )),
  ptThresholdVeto_( pset.getParameter<double>( "ptThresholdVeto" )),
  dRIsolationCone_( pset.getParameter<double>( "dRIsolationCone" ))
 {
  edm::Service<TFileService> fs;
  h_muFromZ_pt_ = fs->make<TH1F>( "muPt", "mu pT(GeV/c)", 200, 0., 200. );
  h_muFromZ_eta_ = fs->make<TH1F>( "muEta", "mu eta", 100, -2.5, 2.5 );
  h_zMuMuAll_mass_ = fs->make<TH1F>( "ZMuMuAllmass", "ZMuMu mass(GeV)", 200, 0., 200. );
  h_zMuMu_mass_ = fs->make<TH1F>( "ZMuMumass", "ZMuMu mass(GeV)", 200, 0., 200. );
  h_zMuMuMatched_mass_ = fs->make<TH1F>( "ZMuMuMatchedmass", "ZMuMu mass(GeV)", 200, 0., 200. );
  h_zMuMuMC_mass_ = fs->make<TH1F>( "ZMuMuMCmass", "ZMuMu mass(GeV)", 200, 0., 200. );
}

void ZMuMuTutorialAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  Handle<CandidateView> dimuons;
  event.getByLabel(dimuons_, dimuons);

  size_t nDimuons = dimuons->size();

  int goodZMuMu(0);

  // looping on dimuons candidates
  for( size_t i = 0; i < nDimuons; i++ ) {
    const Candidate & dimuonCand = (*dimuons)[ i ];
    h_zMuMuAll_mass_->Fill( dimuonCand.mass() );
    // accessing the daughters of the dimuon candidate
    const Candidate * lep0 = dimuonCand.daughter( 0 );
    const Candidate * lep1 = dimuonCand.daughter( 1 );
    // needed to access specific methods of pat::Muon
    const pat::Muon & muonDau0 = dynamic_cast<const pat::Muon &>(*lep0->masterClone());
    const pat::Muon & muonDau1 = dynamic_cast<const pat::Muon &>(*lep1->masterClone());

    // considering only pairs of global muons
    if( muonDau0.isGlobalMuon() && muonDau1.isGlobalMuon() ) {
      // Selection

      // the charge of the dimuon pair must be 0
      if( dimuonCand.charge() != 0 )
	continue;

      float ptMu0 = lep0->pt();  
      float ptMu1 = lep1->pt();  
      float etaMu0 = lep0->eta();  
      float etaMu1 = lep1->eta();  


      // kinematical and acceptance cut
      if( (ptMu0 < ptcut_) || (ptMu1 < ptcut_) )
	continue;
      if( (fabs(etaMu0) > etacut_) || (fabs(etaMu1) > etacut_) )
	continue;

      // Accessing the pre-computed user-defined tracker isolation
      // Note: if you want the POG-defined isolation you must use muonDau0.trackIso();
      float trackerIsoMu0 = muonDau0.userIsolation(pat::TrackIso);

      // Another possibility: compute tracker isolation from IsoDeposit
      // Accessing tracker IsoDeposits collection around the muon
      const pat::IsoDeposit * trackIsodeposit = muonDau1.trackIsoDeposit();
      Direction muDir = Direction(etaMu1, lep1->phi());
      // Defining vetos to compute isolation
      IsoDeposit::AbsVetos vetos_mu;
      // Veto cone around muon direction
      vetos_mu.push_back(new ConeVeto( muDir, dRConeVeto_ ));
      // pT threshold of deposits
      vetos_mu.push_back(new ThresholdVeto( ptThresholdVeto_ ));
      // Computing tracker isolation: sum of IsoDeposits (= pT sum) within a cone
      float trackerIsoMu1 = trackIsodeposit->sumWithin(dRIsolationCone_, vetos_mu);
      
      // isolation cut
      if( trackerIsoMu0 > isocut_ || trackerIsoMu1 > isocut_ )
	continue;

      // mass cut
      if( (dimuonCand.mass()<minZmass_) || (dimuonCand.mass()>maxZmass_) )
	continue;

      goodZMuMu++;
      h_zMuMu_mass_->Fill( dimuonCand.mass() );
      h_muFromZ_pt_->Fill( ptMu0 );
      h_muFromZ_pt_->Fill( ptMu1 );
      h_muFromZ_eta_->Fill( etaMu0 );
      h_muFromZ_eta_->Fill( etaMu1 );

      bool isMCMatchedZMuMu = false;

      //  Looking at MCtruth
      // final state muons have status = 1
      // these muons have status = 3 muons as parents
      // The decay tree is the following:
      // Z(status 3) -> Z(status 2) mu(status 3) mu(status 3)
      // mu(status 3) -> mu(status 1)
      // or
      // mu(status 3) -> mu(status 1) gamma(status 1)
      GenParticleRef mc0 = muonDau0.genParticleRef();
      GenParticleRef mc1 = muonDau1.genParticleRef();
      if(mc0.isNonnull() && mc1.isNonnull()) {
	if( abs(mc0->pdgId()) == 13 && abs(mc1->pdgId()) == 13 && 
	    (mc0->numberOfMothers() > 0) && (mc1->numberOfMothers() > 0) ) {
	  GenParticleRef moth0 = mc0->motherRef()->motherRef(); 
	  GenParticleRef moth1 = mc1->motherRef()->motherRef();
	  if(moth0.isNonnull() && moth1.isNonnull()){
	    if( (moth0 == moth1) && (moth0->pdgId() == 23) && (moth0->numberOfDaughters() == 3) ){
	      isMCMatchedZMuMu = true;
	      h_zMuMuMatched_mass_->Fill( dimuonCand.mass() );
	      h_zMuMuMC_mass_->Fill( moth0->mass() );
	    }
	  }
	}
      } 

    }
  }
    
}

   
void ZMuMuTutorialAnalyzer::endJob() {
}
  
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuTutorialAnalyzer);
  
