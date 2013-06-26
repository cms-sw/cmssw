#include "GeneratorInterface/GenFilters/interface/BCToEFilterAlgo.h"

using namespace edm;
using namespace std;


BCToEFilterAlgo::BCToEFilterAlgo(const edm::ParameterSet& iConfig) { 

  //set constants
  FILTER_ETA_MAX_=2.5;
  eTThreshold_=(float)iConfig.getParameter<double>("eTThreshold");
  genParSource_=iConfig.getParameter<edm::InputTag>("genParSource");

}

BCToEFilterAlgo::~BCToEFilterAlgo() {
}


//look for status==1 e coming from b or c hadron
//there is an eT threshold on the electron (configurable)
bool BCToEFilterAlgo::filter(const edm::Event& iEvent)  {

  bool result=false;

  
  
  Handle<reco::GenParticleCollection> genParsHandle;
  iEvent.getByLabel(genParSource_,genParsHandle);
  reco::GenParticleCollection genPars=*genParsHandle;

  for (uint32_t ig=0;ig<genPars.size();ig++) {
    reco::GenParticle gp=genPars.at(ig);
    if (gp.status()==1 && abs(gp.pdgId())==11 && gp.et()>eTThreshold_ && fabs(gp.eta())<FILTER_ETA_MAX_) {
      if (hasBCAncestors(gp)) {
	result=true;
      }
    }
  }
  return result;

}



//does this particle have an ancestor (mother, mother of mother, etc.) that is a b or c hadron?
//attention: the GenParticle argument must have the motherRef correctly filled for this
//to work.  That is, you had better have got it out of the genParticles collection
bool BCToEFilterAlgo::hasBCAncestors(const reco::GenParticle& gp) {

  //stopping condition: this particle is a b or c hadron
  if (isBCHadron(gp)) return true;
  //stopping condition: this particle has no mothers
  if (gp.numberOfMothers()==0) return false;
  //otherwise continue
  bool retval=false;
  for (uint32_t im=0;im<gp.numberOfMothers();im++) {
    retval=retval || hasBCAncestors(*gp.motherRef(im));
  }
  return retval;
  
}

bool BCToEFilterAlgo::isBCHadron(const reco::GenParticle& gp) {
  return isBCMeson(gp) || isBCBaryon(gp);
}

bool BCToEFilterAlgo::isBCMeson(const reco::GenParticle& gp) {
  
  uint32_t pdgid=abs(gp.pdgId());
  uint32_t hundreds=pdgid%1000;
  if (hundreds>=400 && hundreds<600) {
    return true;
  } else {
    return false;
  }

}

bool BCToEFilterAlgo::isBCBaryon(const reco::GenParticle& gp) {
  
  uint32_t pdgid=abs(gp.pdgId());
  uint32_t thousands=pdgid%10000;
  if (thousands>=4000 && thousands <6000) {
    return true;
  } else {
    return false;
  }

}
