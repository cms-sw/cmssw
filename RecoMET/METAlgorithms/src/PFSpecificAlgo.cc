/*
class: PFSpecificAlgo.cc
description:  MET made from Particle Flow candidates
authors: R. Remington (UF), R. Cavanaugh (UIC/Fermilab)
  date: 10/27/08
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/PFSpecificAlgo.h"
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
  specific.NeutralEMEtFraction = 0.0;
  specific.NeutralHadEtFraction = 0.0;
  specific.ChargedEMEtFraction = 0.0;
  specific.ChargedHadEtFraction = 0.0;
  specific.MuonEtFraction = 0.0;
  specific.Type6EtFraction = 0.0;
  specific.Type7EtFraction = 0.0;


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
  
  for( edm::View<reco::Candidate>::const_iterator iParticle = (PFCandidates.product())->begin() ; iParticle != (PFCandidates.product())->end() ; ++iParticle )
  {   
    const Candidate* candidate = &(*iParticle);
    if (candidate) {
      const PFCandidate* pfCandidate = static_cast<const PFCandidate*> (candidate);
      //const PFCandidate* pfCandidate = dynamic_cast<const PFCandidate*> (candidate);
      if (pfCandidate)
      {
	//cout << pfCandidate->et() << "     "
	//   << pfCandidate->hcalEnergy() << "    "
	//   << pfCandidate->ecalEnergy() << endl;
	//std::cout << "pfCandidate->particleId() = " << pfCandidate->particleId() << std::endl;
	const double theta = iParticle->theta();
	const double e     = iParticle->energy();
	const double et    = e*sin(theta);
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
    specific.NeutralEMEtFraction = NeutralEMEt/Et_total;
    specific.NeutralHadEtFraction = NeutralHadEt/Et_total;
    specific.ChargedEMEtFraction = ChargedEMEt/Et_total;
    specific.ChargedHadEtFraction = ChargedHadEt/Et_total;
    specific.MuonEtFraction = MuonEt/Et_total;
    specific.Type6EtFraction = type6Et/Et_total;
    specific.Type7EtFraction = type7Et/Et_total;
  }

  const LorentzVector p4(met.mex , met.mey, 0.0, met.met);
  const Point vtx(0.0,0.0,0.0);
  PFMET specificPFMET( specific, met.sumet, p4, vtx );
  return specificPFMET;
}
