#include "DataFormats/Math/interface/LorentzVector.h"
#include "RecoMET/METAlgorithms/interface/GenSpecificAlgo.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
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
reco::GenMET GenSpecificAlgo::addInfo(edm::Handle<edm::View<Candidate> > particles, CommonMETData *met, double globalThreshold, bool onlyFiducial, bool usePt)
{ 

  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;

  // Instantiate the container to hold the calorimeter specific information
  SpecificGenMETData specific = SpecificGenMETData();

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

  
      //  if "onlyFiducial" is true take only particles within fudical volume AND use pT instead of et if "usePt" is true
      if(!onlyFiducial || (onlyFiducial && TMath::Abs(iParticle->eta()) < 5.0)) // 
	{
	  if(!usePt)
	    {
	      if( iParticle->et() > globalThreshold  )
		{
		  double phi   = iParticle->phi();
		  double theta = iParticle->theta();
		  double e     = iParticle->energy();
		  double et    = e*sin(theta);
		  sum_ez += e*cos(theta);
		  sum_et += et;
		  sum_ex += et*cos(phi);
		  sum_ey += et*sin(phi);
		}
	    }
      
	  if(usePt)
	    {
	      if( iParticle->pt() > globalThreshold  )
		{
		  double phi   = iParticle->phi();
		  double et    = iParticle->pt(); // change et-->pt
		  sum_ez += iParticle->pz(); // change ez-->pz
		  sum_et += et;
		  sum_ex += et*cos(phi);
		  sum_ey += et*sin(phi);
		}
	    }
	}

      

      int pdgId = TMath::Abs( iParticle->pdgId() ) ;
      
      switch (pdgId) {

	  case 22 : // photon
            if(usePt){
	      specific.NeutralEMEtFraction += iParticle->pt();
            }else{
	      specific.NeutralEMEtFraction += iParticle->et();
	    }
	    break;
	  case 11 : // e
	    if(usePt){
	    specific.ChargedEMEtFraction += iParticle->pt();
            }else{
            specific.ChargedEMEtFraction += iParticle->et();
	    }
	    break;
	  case 130 : // K_long
	  case 310 : // K_short
	  case 3122: // Lambda
	  case 2112: // n
	  case 3222: // Neutral Cascade
            if(usePt){
	      specific.NeutralHadEtFraction += iParticle->pt();
            }else{
	      specific.NeutralHadEtFraction += iParticle->et();
	    }
	    break;
	  case 211 : // pi
	  case 321 : // K+/K-
	  case 2212: // p
	  case 3312: // Cascade -
	  case 3112: // Sigma -
	  case 3322: // Sigma + 
	  case 3334: // Omega -	
            if(usePt){
	      specific.ChargedHadEtFraction += iParticle->pt();
            }else{
	      specific.ChargedHadEtFraction += iParticle->et();
            }
	    break;
	  case 13 : //muon
            if(usePt){
	      specific.MuonEtFraction += iParticle->pt();
            }else{
              specific.MuonEtFraction += iParticle->et();
	    }
	    break;
	  case 12 : // e_nu
	  case 14 : // mu_nu
	  case 16 : // tau_nu
	  case 1000022 : // Neutral ~Chi_0 
	  case 1000012 :  // LH ~e_nu  
	  case 1000014 :  // LH ~mu_nu
	  case 1000016 :  // LH ~tau_nu
	  case 2000012 :  // RH ~e_nu  
	  case 2000014 :  // RH ~mu_nu
	  case 2000016 :  // RH ~tau_nu
	  case 39 :       //  G
	  case 1000039 :  // ~G
	  case 5100039 :  // KK G
	  case 4000012 :  // excited e_nu
	  case 4000014 :  // excited mu_nu 
	  case 4000016 :  // excited tau_nu 
	  case 9900012 :  // Maj e_nu
	  case 9900014 :  // Maj mu_nu 
	  case 9900016 :  // Maj tau_nu 
            if(usePt){
	    specific.InvisibleEtFraction   += iParticle->pt();
            }else {
	      specific.InvisibleEtFraction   += iParticle->et();
            }
	    break;
	  default : 
            if(usePt){
	    Et_unclassified += iParticle->pt();
            }else{
	      Et_unclassified += iParticle->et(); 
	    }
	    //	cout << "PdgId : "<< iParticle->pdgId() << "    " << iParticle->status() << "  does not fall into a category " << endl;
	 
     
      }
    }
  
  met->mex   = -sum_ex;
  met->mey   = -sum_ey;
  met->mez   = -sum_ez;
  met->met   = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  // cout << "MET = " << met->met << endl;
  met->sumet = sum_et;
  met->phi   = atan2( -sum_ey, -sum_ex );

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
  const LorentzVector p4( met->mex, met->mey, met->mez, met->met );
  const Point vtx( 0.0, 0.0, 0.0 );
  // Create and return an object of type GenMET, which is a MET object with 
  // the extra calorimeter specfic information added
  GenMET specificmet( specific, met->sumet, p4, vtx );
  return specificmet;
}
//-------------------------------------------------------------------------
