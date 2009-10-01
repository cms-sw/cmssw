#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"
#include "TMath.h"
using namespace reco;
using namespace std;

//-------------------------------------------------------------------------
// This algorithm adds calorimeter specific global event information to 
// the MET object which may be useful/needed for MET Data Quality Monitoring
// and MET cleaning.  This list is not exhaustive and additional 
// information will be added in the future. 
//-------------------------------------
//reco::GenMET GenSpecificAlgo::addInfo(const CandidateCollection *particles, CommonMETData met)
reco::GenMET GenSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > particles, CommonMETData met, bool onlyFiducial)
{ 
  // Instantiate the container to hold the calorimeter specific information
  SpecificGenMETData specific;

  // Initialise the container 
  specific.NeutralEMEtFraction     = 0.0;
  specific.NeutralHadEtFraction    = 0.0;
  specific.ChargedEMEtFraction     = 0.0;
  specific.ChargedHadEtFraction    = 0.0;
  specific.MuonEtFraction          = 0.0;
  specific.InvisibleEtFraction     = 0.0;

  // tally Et contribution from undecayed genparticles that don't fall into the following classifications (for normalization purposes)
  double Et_unclassified = 0.;
  
  for( edm::View<reco::Candidate>::const_iterator iParticle = (particles.product())->begin() ; iParticle != (particles.product())->end() ; iParticle++ )
    {
      //Fiducial requirement
      if( onlyFiducial )
	{
	  if( TMath::Abs(iParticle->eta()) > 5.0  )
	    {
	      // Add back the Et that was subtracted in the calculation of MET from METAlgo.cc
	      // The CMS detector won't see this Et
	      met.mex += iParticle->et()*TMath::Cos( iParticle->phi() );
	      met.mey += iParticle->et()*TMath::Sin( iParticle->phi() ); 
	      continue;
	    }
	}

      int pdgId = TMath::Abs( iParticle->pdgId() ) ;
      
      switch (pdgId) {
      case 22 : // photon
	specific.NeutralEMEtFraction += iParticle->et();
	break;
      case 11 : // e
	specific.ChargedEMEtFraction += iParticle->et();
	break;
      case 130 : // K_long
      case 310 : // K_short
      case 3122: // Lambda
      case 2112: // n
      case 3222: // Neutral Cascade
	specific.NeutralHadEtFraction += iParticle->et();
	break;
      case 211 : // pi
      case 321 : // K+/K-
      case 2212: // p
      case 3312: // Cascade -
      case 3112: // Sigma -
      case 3322: // Sigma + 
      case 3334: // Omega -	
	specific.ChargedHadEtFraction += iParticle->et();
	break;
      case 13 : //muon
	specific.MuonEtFraction += iParticle->et();
	break;
      case 12 : // e_nu
      case 14 : // mu_nu
      case 16 : // tau_nu
      case 1000022 : // Neutral ~Chi_0 
      case 1000012 :  // LH ~e_nu  
      case 1000014 :  // LH ~mu_nu
      case 1000016 :  // LH ~tau_nu
      case 1000039 :  // ~G
      case 4000012 :  // excited e_nu
      case 39 :  //  G
	specific.InvisibleEtFraction   += iParticle->et();
	break;
      default : 
	Et_unclassified += iParticle->et();
	cout << "PdgId : "<< iParticle->pdgId() << "    " << iParticle->status() << "  does not fall into a category " << endl;
      }
    }

  double Et_Total = specific.NeutralEMEtFraction + specific.NeutralHadEtFraction + specific.ChargedEMEtFraction + 
    specific.ChargedHadEtFraction + specific.MuonEtFraction + specific.InvisibleEtFraction + Et_unclassified;
  
  //Normalize
  if( Et_Total ) 
    {
      specific.NeutralEMEtFraction /= Et_Total;
      specific.NeutralHadEtFraction /= Et_Total;
      specific.ChargedEMEtFraction /= Et_Total;
      specific.ChargedHadEtFraction /= Et_Total;
      specific.MuonEtFraction /= Et_Total;
      specific.InvisibleEtFraction /= Et_Total;
    }

  // Instantiate containers for the MET candidate and initialise them with
  // the MET information in "met" (of type CommonMETData)
  const LorentzVector p4( met.mex, met.mey, 0.0, met.met );
  const Point vtx( 0.0, 0.0, 0.0 );
  // Create and return an object of type GenMET, which is a MET object with 
  // the extra calorimeter specfic information added
  GenMET specificmet( specific, met.sumet, p4, vtx );
  return specificmet;
}
//-------------------------------------------------------------------------
