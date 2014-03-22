#include "GeneratorInterface/TauolaInterface/interface/TauSpinnerCMS.h"

//MC-TESTER header files
#include "Tauola/Tauola.h"
#include "TauSpinner/tau_reweight_lib.h"
#include "TauSpinner/Tauola_wrapper.h"
#include "GeneratorInterface/TauolaInterface/interface/read_particles_from_HepMC.h"
#include "TLorentzVector.h"

#include "CLHEP/Random/RandomEngine.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"

using namespace edm;
using namespace TauSpinner;

CLHEP::HepRandomEngine* TauSpinnerCMS::fRandomEngine= nullptr;
bool                    TauSpinnerCMS::isTauSpinnerConfigure=false;

bool TauSpinnerCMS::fInitialized=false;

TauSpinnerCMS::TauSpinnerCMS( const ParameterSet& pset ) :
  EDProducer()
  ,isReco_(pset.getParameter<bool>("isReco"))
  ,isTauolaConfigured_(pset.getParameter<bool>("isTauolaConfigured" ))
  ,isLHPDFConfigured_(pset.getParameter<bool>("isLHPDFConfigured" ))
  ,LHAPDFname_(pset.getUntrackedParameter("LHAPDFname",(string)("MSTW2008nnlo90cl.LHgrid")))
  ,CMSEnergy_(pset.getParameter<double>("CMSEnergy"))//GeV
  ,gensrc_(pset.getParameter<edm::InputTag>("gensrc"))
  ,MotherPDGID_(pset.getUntrackedParameter("MotherPDGID",(int)(-1)))
  ,Ipol_(pset.getUntrackedParameter("Ipol",(int)(0)))
  ,nonSM2_(pset.getUntrackedParameter("nonSM2",(int)(0)))
  ,nonSMN_(pset.getUntrackedParameter("nonSMN",(int)(0)))
  ,roundOff_(pset.getUntrackedParameter("roundOff",(double)(0.01)))
{
  //usesResource(edm::uniqueSharedResourceName());
  usesResource(edm::SharedResourceNames::kTauola);

  produces<bool>("TauSpinnerWTisValid").setBranchAlias("TauSpinnerWTisValid");
  produces<double>("TauSpinnerWT").setBranchAlias("TauSpinnerWT");
  produces<double>("TauSpinnerWTFlip").setBranchAlias("TauSpinnerWTFlip");
  produces<double>("TauSpinnerWThplus").setBranchAlias("TauSpinnerWThplus");
  produces<double>("TauSpinnerWThminus").setBranchAlias("TauSpinnerWThminus");

  if(isReco_){
    GenParticleCollectionToken_=consumes<reco::GenParticleCollection>(gensrc_);
  }
  else{
    hepmcCollectionToken_=consumes<HepMCProduct>(gensrc_);
  }
}

void TauSpinnerCMS::initialize(){
  // Now for Tauola and TauSpinner
  if(!isTauolaConfigured_){
    Tauolapp::Tauola::setRandomGenerator(TauSpinnerCMS::flat);
    Tauolapp::Tauola::initialize();
  }
  if(!isLHPDFConfigured_){
    LHAPDF::initPDFSetByName(LHAPDFname_);
  }
  if(!isTauSpinnerConfigure){
    isTauSpinnerConfigure=true;
    bool Ipp = true;  // for pp collisions
    // Initialize TauSpinner
    //Ipol - polarization of input sample
    //nonSM2 - nonstandard model calculations
    //nonSMN
    TauSpinner::initialize_spinner(Ipp,Ipol_,nonSM2_,nonSMN_,CMSEnergy_);
  }
  fInitialized=true;
}

void TauSpinnerCMS::endLuminosityBlockProduce(edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup){}

void TauSpinnerCMS::beginJob(){

}

void TauSpinnerCMS::produce( edm::Event& e, const edm::EventSetup& iSetup){
  RandomEngineSentry<TauSpinnerCMS> randomEngineSentry(this, e.streamID());
  if(!fInitialized) initialize();

  Tauolapp::Tauola::setRandomGenerator(TauSpinnerCMS::flat);  // rest tauola++ random number incase other modules use tauola++

  double WT=1.0;
  double WTFlip=1.0;
  double polSM=-999; //range [-1,1]
  SimpleParticle X, tau, tau2;
  std::vector<SimpleParticle> tau_daughters, tau_daughters2;
  int stat(0);
  if(isReco_){
    stat=readParticlesfromReco(e,X,tau,tau2,tau_daughters,tau_daughters2);
  }
  else{
    edm::Handle<HepMCProduct> evt;
    e.getByToken(hepmcCollectionToken_, evt);
    //Get EVENT
    HepMC::GenEvent *Evt = new HepMC::GenEvent(*(evt->GetEvent()));
    stat=readParticlesFromHepMC(Evt,X,tau,tau2,tau_daughters,tau_daughters2);
  }  
  if(MotherPDGID_<0 || abs(X.pdgid())==MotherPDGID_){
    if(stat!=1){
      // Determine the weight      
      if( abs(X.pdgid())==24 ||  abs(X.pdgid())==37 ){
        TLorentzVector tau_1r(0,0,0,0);
        TLorentzVector tau_1(tau.px(),tau.py(),tau.pz(),tau.e());
        for(unsigned int i=0; i<tau_daughters.size();i++){
          tau_1r+=TLorentzVector(tau_daughters.at(i).px(),tau_daughters.at(i).py(),tau_daughters.at(i).pz(),tau_daughters.at(i).e());
        }
	if(fabs(tau_1r.M()-tau_1.M())<roundOff_){
	  WT = TauSpinner::calculateWeightFromParticlesWorHpn(X, tau, tau2, tau_daughters); // note that tau2 is tau neutrino
	  polSM=getTauSpin();
	  WTFlip=(2.0-WT)/WT;
	}
      }
      else if( X.pdgid()==25 || X.pdgid()==36 || X.pdgid()==22 || X.pdgid()==23 ){
	TLorentzVector tau_1r(0,0,0,0), tau_2r(0,0,0,0);
	TLorentzVector tau_1(tau.px(),tau.py(),tau.pz(),tau.e()), tau_2(tau2.px(),tau2.py(),tau2.pz(),tau2.e());
	for(unsigned int i=0; i<tau_daughters.size();i++){
	  tau_1r+=TLorentzVector(tau_daughters.at(i).px(),tau_daughters.at(i).py(),tau_daughters.at(i).pz(),tau_daughters.at(i).e());
	}
	for(unsigned int i=0; i<tau_daughters2.size();i++){
	  tau_2r+=TLorentzVector(tau_daughters2.at(i).px(),tau_daughters2.at(i).py(),tau_daughters2.at(i).pz(),tau_daughters2.at(i).e());
	}

        if(fabs(tau_1r.M()-tau_1.M())<roundOff_ && fabs(tau_2r.M()-tau_2.M())<roundOff_){
          WT = TauSpinner::calculateWeightFromParticlesH(X, tau, tau2, tau_daughters,tau_daughters2);
	  //std::cout << "WT " << WT << std::endl;
          polSM=getTauSpin();
          if(X.pdgid()==25 || X.pdgid()==22 || X.pdgid()==23 ){
            if(X.pdgid()==25) X.setPdgid(23);
            if( X.pdgid()==22 || X.pdgid()==23 ) X.setPdgid(25);

            double WTother=TauSpinner::calculateWeightFromParticlesH(X, tau, tau2, tau_daughters,tau_daughters2);
            WTFlip=WTother/WT;
          }
        }
      }
      else{
	cout<<"TauSpinner: WARNING: Unexpected PDG for tau mother: "<<X.pdgid()<<endl;
      }
    }
  }
  bool isValid=true;
  if(!(0<=WT && WT<10)){isValid=false; WT=1.0; WTFlip=1.0;}
  std::auto_ptr<bool> TauSpinnerWeightisValid(new bool);
  *TauSpinnerWeightisValid =isValid;
  e.put(TauSpinnerWeightisValid,"TauSpinnerWTisValid");

  // regular weight
  std::auto_ptr<double> TauSpinnerWeight(new double);
  *TauSpinnerWeight =WT;    
  e.put(TauSpinnerWeight,"TauSpinnerWT");  
  
  // flipped weight (ie Z->H or H->Z)
  std::auto_ptr<double> TauSpinnerWeightFlip(new double);
  *TauSpinnerWeightFlip =WTFlip;
  e.put(TauSpinnerWeightFlip,"TauSpinnerWTFlip");
  
  // h+ polarization
  double WThplus=WT;
  if(polSM<0.0 && polSM!=-999 && isValid) WThplus=0; 
  std::auto_ptr<double> TauSpinnerWeighthplus(new double);
  *TauSpinnerWeighthplus = WThplus;
  e.put(TauSpinnerWeighthplus,"TauSpinnerWThplus");

  // h- polarization
  double WThminus=WT;
  if(polSM>0.0&& polSM!=-999 && isValid) WThminus=0;
  std::auto_ptr<double> TauSpinnerWeighthminus(new double);
  *TauSpinnerWeighthminus = WThminus;
  e.put(TauSpinnerWeighthminus,"TauSpinnerWThminus");
  return ;
}  

void TauSpinnerCMS::endRun( const edm::Run& r, const edm::EventSetup& ){}

void TauSpinnerCMS::endJob(){}

int TauSpinnerCMS::readParticlesfromReco(edm::Event& e,SimpleParticle &X,SimpleParticle &tau,SimpleParticle &tau2, 
					 std::vector<SimpleParticle> &tau_daughters,std::vector<SimpleParticle> &tau2_daughters){
  edm::Handle<reco::GenParticleCollection> genParticles;
  e.getByToken(GenParticleCollectionToken_, genParticles);
  for(reco::GenParticleCollection::const_iterator itr = genParticles->begin(); itr!= genParticles->end(); ++itr){
    int pdgid=abs(itr->pdgId());
    if(pdgid==24 || pdgid==37 || pdgid ==25 || pdgid==36 || pdgid==22 || pdgid==23 ){
      const reco::GenParticle *hx=&(*itr);
      if(!isFirst(hx)) continue;
      GetLastSelf(hx);
      const reco::GenParticle *recotau1=NULL;
      const reco::GenParticle *recotau2=NULL;
      unsigned int ntau(0),ntauornu(0);
      for(unsigned int i=0; i<itr->numberOfDaughters(); i++){
	const reco::Candidate *dau=itr->daughter(i);
	if(abs(dau->pdgId())!=pdgid){
	  if(abs(dau->pdgId())==15 || abs(dau->pdgId())==16){
	    if(ntau==0 && abs(dau->pdgId())==15){
	      recotau1=static_cast<const reco::GenParticle*>(dau);
	      GetLastSelf(recotau1);
	      ntau++;
	    }
	    else if((ntau==1 && abs(dau->pdgId())==15) || abs(dau->pdgId())==16){
	      recotau2=static_cast<const reco::GenParticle*>(dau);
	      if(abs(dau->pdgId())==15){ntau++;GetLastSelf(recotau2);}
	    }
	    ntauornu++;
	  }
	}
      }
      if((ntau==2 && ntauornu==2) || (ntau==1 && ntauornu==2)){
	X.setPx(itr->p4().Px());
	X.setPy(itr->p4().Py());
	X.setPz(itr->p4().Pz());
	X.setE (itr->p4().E());
	X.setPdgid(itr->pdgId());
	tau.setPx(recotau1->p4().Px());
	tau.setPy(recotau1->p4().Py());
	tau.setPz(recotau1->p4().Pz());
	tau.setE (recotau1->p4().E());
        tau.setPdgid(recotau1->pdgId());
	GetRecoDaughters(recotau1,tau_daughters,recotau1->pdgId());
	tau2.setPx(recotau2->p4().Px());
        tau2.setPy(recotau2->p4().Py());
        tau2.setPz(recotau2->p4().Pz());
        tau2.setE (recotau2->p4().E());
        tau2.setPdgid(recotau2->pdgId());
	if(ntau==2)GetRecoDaughters(recotau2,tau2_daughters,recotau2->pdgId());
	return 0;
      }
    }
  }
  return 1;
}

void TauSpinnerCMS::GetLastSelf(const reco::GenParticle *Particle){
  for (unsigned int i=0; i< Particle->numberOfDaughters(); i++){
    const reco::GenParticle *dau=static_cast<const reco::GenParticle*>(Particle->daughter(i));
    if(Particle->pdgId()==dau->pdgId()){
      Particle=dau;
      GetLastSelf(Particle);  
    }
  }
}

bool TauSpinnerCMS::isFirst(const reco::GenParticle *Particle){
  for (unsigned int i=0; i< Particle->numberOfMothers(); i++){
    const reco::GenParticle *moth=static_cast<const reco::GenParticle*>(Particle->mother(i));
    if(Particle->pdgId()==moth->pdgId()){
      return false;
    }
  }
  return true;
}

void TauSpinnerCMS::GetRecoDaughters(const reco::GenParticle *Particle,std::vector<SimpleParticle> &daughters, int parentpdgid){
  if( Particle->numberOfDaughters()==0 || abs(Particle->pdgId())==111){
    SimpleParticle tp(Particle->p4().Px(), Particle->p4().Py(), Particle->p4().Pz(), Particle->p4().E(), Particle->pdgId());
    daughters.push_back(tp);
    return;
  }
  for (unsigned int i=0; i< Particle->numberOfDaughters(); i++){
    const reco::Candidate *dau=Particle->daughter(i);
    GetRecoDaughters(static_cast<const reco::GenParticle*>(dau),daughters,Particle->pdgId());
  }
}

double TauSpinnerCMS::flat()
{
  if ( !fRandomEngine ) {
    throw cms::Exception("LogicError")
      << "TauSpinnerCMS::flat: Attempt to generate random number when engine pointer is null\n"
      << "This might mean that the code was modified to generate a random number outside the\n"
      << "event and beginLuminosityBlock methods, which is not allowed.\n";
  }
  return fRandomEngine->flat();
}


DEFINE_FWK_MODULE(TauSpinnerCMS);
