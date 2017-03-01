#include "PFCandWithSuperClusterExtractor.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

using namespace edm;
using namespace reco;

PFCandWithSuperClusterExtractor::PFCandWithSuperClusterExtractor( const ParameterSet& par, edm::ConsumesCollector && iC ) :
  thePFCandToken(iC.consumes<PFCandidateCollection>(par.getParameter<edm::InputTag>("inputCandView"))),
  theDepositLabel(par.getUntrackedParameter<std::string>("DepositLabel")),
  theVetoSuperClusterMatch(par.getParameter<bool>("SCMatch_Veto")),
  theMissHitVetoSuperClusterMatch(par.getParameter<bool>("MissHitSCMatch_Veto")),
  theDiff_r(par.getParameter<double>("Diff_r")),
  theDiff_z(par.getParameter<double>("Diff_z")),
  theDR_Max(par.getParameter<double>("DR_Max")),
  theDR_Veto(par.getParameter<double>("DR_Veto"))
{
  //  std::cout << " Loading PFCandWithSuperClusterExtractor "  << std::endl;
}
/*
reco::IsoDeposit::Vetos PFCandWithSuperClusterExtractor::vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & cand) const
{
  reco::isodeposit::Direction dir(cand.eta(),cand.phi());
  return reco::IsoDeposit::Vetos(1,veto(dir));
}
*/

reco::IsoDeposit::Veto PFCandWithSuperClusterExtractor::veto(const reco::IsoDeposit::Direction & dir) const
{
  reco::IsoDeposit::Veto result;
  result.vetoDir = dir;
  result.dR = theDR_Veto;
  return result;
}


IsoDeposit PFCandWithSuperClusterExtractor::depositFromObject(const Event & event, const EventSetup & eventSetup, const Photon & cand) const
{
    reco::isodeposit::Direction candDir(cand.eta(), cand.phi());
    IsoDeposit deposit(candDir );
    deposit.setVeto( veto(candDir) );
    deposit.addCandEnergy(cand.pt());

    Handle< PFCandidateCollection > PFCandH;
    event.getByToken(thePFCandToken, PFCandH);

    double eta = cand.eta(), phi = cand.phi();
    reco::Particle::Point vtx = cand.vertex();
    for (PFCandidateCollection::const_iterator it = PFCandH->begin(), ed = PFCandH->end(); it != ed; ++it) {
      double dR = deltaR(it->eta(), it->phi(), eta, phi);
      // veto SC
      if (theVetoSuperClusterMatch && cand.superCluster().isNonnull() && it->superClusterRef().isNonnull() && cand.superCluster() == it->superClusterRef()) continue;
      if ( (dR < theDR_Max) && (dR > theDR_Veto) &&
	   (std::abs(it->vz() - cand.vz()) < theDiff_z) &&
	   ((it->vertex() - vtx).Rho() < theDiff_r)) {
	// ok
	reco::isodeposit::Direction dirTrk(it->eta(), it->phi());
	deposit.addDeposit(dirTrk, it->pt());
      }
    }

    return deposit;
}


IsoDeposit PFCandWithSuperClusterExtractor::depositFromObject(const Event & event, const EventSetup & eventSetup, const GsfElectron & cand) const
{
    reco::isodeposit::Direction candDir(cand.eta(), cand.phi());
    IsoDeposit deposit(candDir );
    deposit.setVeto( veto(candDir) );
    deposit.addCandEnergy(cand.pt());

    Handle< PFCandidateCollection > PFCandH;
    event.getByToken(thePFCandToken, PFCandH);

    double eta = cand.eta(), phi = cand.phi();
    reco::Particle::Point vtx = cand.vertex();
    for (PFCandidateCollection::const_iterator it = PFCandH->begin(), ed = PFCandH->end(); it != ed; ++it) {
        double dR = deltaR(it->eta(), it->phi(), eta, phi);
        // If MissHits>0 (possibly reconstructed as a photon in the PF in this case, kill the the photon if sharing the same SC)
        if (cand.gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS) > 0 
                && theMissHitVetoSuperClusterMatch && it->mva_nothing_gamma() > 0.99 
                && cand.superCluster().isNonnull() && it->superClusterRef().isNonnull() 
                && cand.superCluster() == it->superClusterRef()){
            continue;
        }
        if ((dR < theDR_Max) && (dR > theDR_Veto) 
                && (std::abs(it->vz() - cand.vz()) < theDiff_z) 
                && ((it->vertex() - vtx).Rho() < theDiff_r)) {
            // ok
            reco::isodeposit::Direction dirTrk(it->eta(), it->phi());
            deposit.addDeposit(dirTrk, it->pt());
        }
    }

    return deposit;
}



IsoDeposit PFCandWithSuperClusterExtractor::depositFromObject(const Event & event, const EventSetup & eventSetup, const Track & cand) const
{
    reco::isodeposit::Direction candDir(cand.eta(), cand.phi());
    IsoDeposit deposit(candDir );
    deposit.setVeto( veto(candDir) );
    deposit.addCandEnergy(cand.pt());
    Handle< PFCandidateCollection > PFCandH;
    event.getByToken(thePFCandToken, PFCandH);

    double eta = cand.eta(), phi = cand.phi();
    reco::Particle::Point vtx = cand.vertex();
    for (PFCandidateCollection::const_iterator it = PFCandH->begin(), ed = PFCandH->end(); it != ed; ++it) {
        double dR = deltaR(it->eta(), it->phi(), eta, phi);

        if ( (dR < theDR_Max) && (dR > theDR_Veto) &&
                (std::abs(it->vz() - cand.vz()) < theDiff_z) &&
                ((it->vertex() - vtx).Rho() < theDiff_r)) {
            // ok
            reco::isodeposit::Direction dirTrk(it->eta(), it->phi());
            deposit.addDeposit(dirTrk, it->pt());
        }
    }

    return deposit;
}


IsoDeposit PFCandWithSuperClusterExtractor::depositFromObject(const Event & event, const EventSetup & eventSetup, const PFCandidate & cand) const
{
    reco::isodeposit::Direction candDir(cand.eta(), cand.phi());
    IsoDeposit deposit(candDir );
    deposit.setVeto( veto(candDir) );
    deposit.addCandEnergy(cand.pt());
    Handle< PFCandidateCollection > PFCandH;
    event.getByToken(thePFCandToken, PFCandH);

    double eta = cand.eta(), phi = cand.phi();
    reco::Particle::Point vtx = cand.vertex();
    for (PFCandidateCollection::const_iterator it = PFCandH->begin(), ed = PFCandH->end(); it != ed; ++it) {
      // veto SC
      if (theVetoSuperClusterMatch && cand.superClusterRef().isNonnull() && it->superClusterRef().isNonnull() && cand.superClusterRef() == it->superClusterRef()) continue;
      double dR = deltaR(it->eta(), it->phi(), eta, phi);

      if ( (dR < theDR_Max) && (dR > theDR_Veto) &&
	   (std::abs(it->vz() - cand.vz()) < theDiff_z) &&
	   ((it->vertex() - vtx).Rho() < theDiff_r)) {
	// ok
	reco::isodeposit::Direction dirTrk(it->eta(), it->phi());
	deposit.addDeposit(dirTrk, it->pt());
      }
    }

    return deposit;
}



#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "PFCandWithSuperClusterExtractor.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, PFCandWithSuperClusterExtractor, "PFCandWithSuperClusterExtractor");
