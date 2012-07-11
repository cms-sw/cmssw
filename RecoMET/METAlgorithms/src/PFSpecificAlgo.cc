/*
class: PFSpecificAlgo.cc
description:  MET made from Particle Flow candidates
authors: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
  date: 10/27/08
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
#include "RecoMET/METAlgorithms/interface/significanceAlgo.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
using namespace reco;
using namespace std;

//--------------------------------------------------------------------------------------
// This algorithm adds Particle Flow specific global event information to the MET object
//--------------------------------------------------------------------------------------

reco::PFMET PFSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > PFCandidates, CommonMETData met)
{
  // Instantiate the container to hold the PF specific information
  SpecificPFMETData specific;
  // Initialize the container
  specific.NeutralEMFraction = 0.0;
  specific.NeutralHadFraction = 0.0;
  specific.ChargedEMFraction = 0.0;
  specific.ChargedHadFraction = 0.0;
  specific.MuonFraction = 0.0;
  specific.Type6Fraction = 0.0;
  specific.Type7Fraction = 0.0;

  if(!PFCandidates->size()) // if no Particle Flow candidates in the event
  {
    const LorentzVector p4( met.mex, met.mey, 0.0, met.met);
    const Point vtx(0.0, 0.0, 0.0 );
    PFMET specificPFMET( specific, met.sumet, p4, vtx);
    return specificPFMET;
  } 

  double NeutralEMEt = 0.0;
  double NeutralHadEt = 0.0;
  double ChargedEMEt = 0.0;
  double ChargedHadEt = 0.0;
  double MuonEt = 0.0;
  double type6Et = 0.0;
  double type7Et = 0.0;


  pfsignalgo_.useOriginalPtrs(PFCandidates.id());
  
  for( edm::View<reco::Candidate>::const_iterator iParticle = (PFCandidates.product())->begin() ; iParticle != (PFCandidates.product())->end() ; ++iParticle )
  {   
    const Candidate* candidate = &(*iParticle);
    if (candidate) {
      //const PFCandidate* pfCandidate = static_cast<const PFCandidate*> (candidate);
      const PFCandidate* pfCandidate = dynamic_cast<const PFCandidate*> (candidate);
      if (pfCandidate)
      {
	//cout << pfCandidate->et() << "     "
	//   << pfCandidate->hcalEnergy() << "    "
	//   << pfCandidate->ecalEnergy() << endl;
	//std::cout << "pfCandidate->particleId() = " << pfCandidate->particleId() << std::endl;
	const double theta = iParticle->theta();
	const double e     = iParticle->energy();
	const double et    = e*sin(theta);
	if(alsocalcsig){
	    reco::CandidatePtr dau(PFCandidates, iParticle - PFCandidates->begin());
	    if(dau.isNonnull () && dau.isAvailable()){
		reco::PFCandidatePtr pf(dau.id(), pfCandidate, dau.key());
		pfsignalgo_.addPFCandidate(pf);
	    }
	  //metSigInputVector.push_back(resolutions_.evalPF(pfCandidate));
	}

	if (pfCandidate->particleId() == 1) ChargedHadEt += et;
	if (pfCandidate->particleId() == 2) ChargedEMEt += et;
	if (pfCandidate->particleId() == 3) MuonEt += et;
	if (pfCandidate->particleId() == 4) NeutralEMEt += et;
	if (pfCandidate->particleId() == 5) NeutralHadEt += et;
	if (pfCandidate->particleId() == 6) type6Et += et;
	if (pfCandidate->particleId() == 7) type7Et += et;
      }
    } 
  }

  const double Et_total=NeutralEMEt+NeutralHadEt+ChargedEMEt+ChargedHadEt+MuonEt+type6Et+type7Et;

  if (Et_total!=0.0)
  {
    specific.NeutralEMFraction = NeutralEMEt/Et_total;
    specific.NeutralHadFraction = NeutralHadEt/Et_total;
    specific.ChargedEMFraction = ChargedEMEt/Et_total;
    specific.ChargedHadFraction = ChargedHadEt/Et_total;
    specific.MuonFraction = MuonEt/Et_total;
    specific.Type6Fraction = type6Et/Et_total;
    specific.Type7Fraction = type7Et/Et_total;
  }
  
  const LorentzVector p4(met.mex , met.mey, 0.0, met.met);
  const Point vtx(0.0,0.0,0.0);
  PFMET specificPFMET( specific, met.sumet, p4, vtx );

  specificPFMET.setSignificanceMatrix(pfsignalgo_.getSignifMatrix());

  return specificPFMET;
}

void PFSpecificAlgo::runSignificance(metsig::SignAlgoResolutions &resolutions, edm::Handle<edm::View<reco::PFJet> > jets)
{
  alsocalcsig=true;
  pfsignalgo_.setResolutions( &resolutions );
  pfsignalgo_.addPFJets(jets);
}
