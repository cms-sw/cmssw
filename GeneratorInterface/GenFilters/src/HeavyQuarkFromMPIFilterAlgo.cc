#include "GeneratorInterface/GenFilters/interface/HeavyQuarkFromMPIFilterAlgo.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "PhysicsTools/HepMCCandAlgos/interface/GenParticlesHelper.h"

using namespace edm;
using namespace std;


HeavyQuarkFromMPIFilterAlgo::HeavyQuarkFromMPIFilterAlgo(const edm::ParameterSet& iConfig) { 
  HeavyQuarkFlavour=(int)iConfig.getParameter<int>("HQFlavour");
  genParSource_=iConfig.getParameter<edm::InputTag>("genParSource");
}

HeavyQuarkFromMPIFilterAlgo::~HeavyQuarkFromMPIFilterAlgo() {
}


//look for status==1 e coming from b or c hadron
//there is an eT threshold on the electron (configurable)
bool HeavyQuarkFromMPIFilterAlgo::filter(const edm::Event& iEvent)  {
  bool Veto=false;
  Handle<reco::GenParticleCollection> genParsHandle;
  iEvent.getByLabel(genParSource_,genParsHandle);
  reco::GenParticleCollection genPars=*genParsHandle;
//  std::cout<<" in the filter HQF="<<HeavyQuarkFlavour<<std::endl;
  bool fromMPI=false;
  for (uint32_t ig=0;ig<genPars.size();ig++) {
    reco::GenParticle gp=genPars[ig];
    if (abs(gp.pdgId())==HeavyQuarkFlavour) {
        if(gp.status()>=30 && gp.status()<40){
//		cout<<"Found a B with status of 3X (eta="<<gp.eta()<<")==>fromMPI=true"<<endl;
		fromMPI=true;
	}
	if(gp.status()>=40 && fromMPI==false){
//		cout<<"Found a B with status >= 4X (eta="<<gp.eta()<<")==>Need to check ancestors!"<<endl;
		const reco::GenParticleRefVector& mothers = gp.motherRefVector();
//		cout<<"Note: it has "<<mothers.size()<<" mothers"<<endl;
		for( reco::GenParticleRefVector::const_iterator im = mothers.begin(); im!=mothers.end(); ++im) {
			const reco::GenParticle& part = **im;
//			cout<<"--->Going to a mother, having eta="<<part.eta()<<endl;
			if( hasMPIAncestor( &part) ){
//				cout<<"------>Found one ancestor with status of 3X (eta="<<gp.eta()<<")==>fromMPI=true"<<endl;
				fromMPI=true;
			}
		} 
	}
	if(fromMPI)Veto=true;
        else Veto=false;
    }
  }
//  cout<<"RETURN "<<Veto<<endl;
  return Veto;
}

bool HeavyQuarkFromMPIFilterAlgo::hasMPIAncestor( const reco::GenParticle* particle ) {
	if( particle->status() >=30 && particle->status()<40 ){
//		cout<<"------->Mother found with eta="<<particle->eta()<<" and status "<<particle->status()<<"==> returning true!"<<endl;
		return true;
	}
//	cout<<"------->in hasMPIAncestor, current particle has eta="<<particle->eta()<<" and status!=3X ==> checking next mothers"<<endl;
	const reco::GenParticleRefVector& mothers = particle->motherRefVector();
//	 cout<<"------>Note: it has "<<mothers.size()<<" mothers"<<endl;
	for( reco::GenParticleRefVector::const_iterator im = mothers.begin(); im!=mothers.end(); ++im) {
		const reco::GenParticle& part = **im;
		if( hasMPIAncestor( &part )){
//			cout<<"--------->Found one ancestor with status of 3X ==>returning true"<<endl;
			return true;
		}
	}
//	cout<<"------>No 30's ancestor found ==>returning false"<<endl;
	return false;
}

