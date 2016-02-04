#include "CommonTools/ParticleFlow/interface/PFIsoDepositAlgo.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;
using namespace reco;
using namespace math;
using namespace pf2pat;

PFIsoDepositAlgo::PFIsoDepositAlgo(const edm::ParameterSet& iConfig): 
  verbose_ ( iConfig.getUntrackedParameter<bool>("verbose",false) )

{}



PFIsoDepositAlgo::~PFIsoDepositAlgo() { }


const PFIsoDepositAlgo::IsoDeposits&  
PFIsoDepositAlgo::produce( const ParticleCollection& toBeIsolated,
			   const ParticleCollection& forIsolation) {
  

  isoDeposits_.clear();
  isoDeposits_.reserve( toBeIsolated.size() );

  for( unsigned i=0; i<toBeIsolated.size(); i++ ) {
    const reco::PFCandidate& toBeIso = toBeIsolated[i];

    if(verbose_ ) 
      cout<<"to be isolated: "<<toBeIso<<endl;

    isoDeposits_.push_back( buildIsoDeposit( toBeIso, forIsolation ) ); 
  }
  

  if(verbose_) {
    cout<<"PFIsoDepositAlgo "<<endl;
  }

  return isoDeposits_;
}


IsoDeposit PFIsoDepositAlgo::buildIsoDeposit( const Particle& particle, 
					      const ParticleCollection& forIsolation ) const {
  

  reco::isodeposit::Direction pfDir(particle.eta(), 
				    particle.phi());
//   reco::IsoDeposit::Veto veto;
//   veto.vetoDir = pfDir;
//   veto.dR = 0.05; 

  IsoDeposit isoDep( pfDir );
  
  for( unsigned i=0; i<forIsolation.size(); i++ ) {
    
    const reco::PFCandidate& pfc = forIsolation[i];

    // need to remove "particle"!

    if( sameParticle( particle, pfc ) ) continue; 


    XYZTLorentzVector pvi(pfc.p4());
    reco::isodeposit::Direction dirPfc(pfc.eta(), pfc.phi());
    double dR = pfDir.deltaR(dirPfc);

    //COLIN make a parameter
    double maxDeltaRForIsoDep_ = 1;
    if(dR > maxDeltaRForIsoDep_) {
      //      if( verbose_ ) cout<<"OUT OF CONE"<<endl;
      continue;
    }
    //    else if(verbose_) cout<<endl;

    if(verbose_ ) 
      cout<<"\t"<<pfc<<endl;

    double pt = pvi.Pt();
    isoDep.addDeposit(dirPfc, pt); 
  }

  return isoDep;
}


bool PFIsoDepositAlgo::sameParticle( const Particle& particle1,
				     const Particle& particle2 ) const {

  double smallNumber = 1e-15;
  
  if( particle1.particleId() != particle2.particleId() ) return false;
  else if( fabs( particle1.energy() - particle2.energy() ) > smallNumber ) return false;
  else if( fabs( particle1.eta() - particle2.eta() ) > smallNumber ) return false;
  else if( fabs( particle1.eta() - particle2.eta() ) > smallNumber ) return false;
  else return true; 
  
}
