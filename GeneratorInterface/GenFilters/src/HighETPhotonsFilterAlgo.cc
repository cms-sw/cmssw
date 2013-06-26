#include "GeneratorInterface/GenFilters/interface/HighETPhotonsFilterAlgo.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace std;


HighETPhotonsFilterAlgo::HighETPhotonsFilterAlgo(const edm::ParameterSet& iConfig) { 

  //set constants
  FILTER_ETA_MAX_=2.5;
  //parameters
  sumETThreshold_=(float)iConfig.getParameter<double>("sumETThreshold");
  seedETThreshold_=(float)iConfig.getParameter<double>("seedETThreshold");
  nonPhotETMax_=(float)iConfig.getParameter<double>("nonPhotETMax");
  isoConeSize_=(float)iConfig.getParameter<double>("isoConeSize");


}

HighETPhotonsFilterAlgo::~HighETPhotonsFilterAlgo() {
}


bool HighETPhotonsFilterAlgo::filter(const edm::Event& iEvent)  {

  Handle<reco::GenParticleCollection> genParsHandle;
  iEvent.getByLabel("genParticles",genParsHandle);
  vector<reco::GenParticle> genPars=*genParsHandle;
  
  for (uint32_t ig=0;ig<genPars.size();ig++) {
    reco::GenParticle gp=genPars.at(ig);
    //look for seeds
    if (fabs(gp.pdgId())!=22) continue;
    if (gp.status()!=1) continue;
    if (gp.et()<seedETThreshold_) continue;
    //found a seed
    float photetsum=gp.et();
    float nonphotetsum=0;
    for (uint32_t jg=0;jg<genPars.size();jg++) {
      if (ig==jg) continue;
      reco::GenParticle opar=genPars.at(jg);
      if (opar.status()!=1) continue;
      float dr=deltaR(gp,opar);
      if (dr>=isoConeSize_) continue;
      if (fabs(opar.pdgId())==22) {
	photetsum+=opar.et();
      } else {
	nonphotetsum+=opar.et();
      }
    }
    //check that photon energy exceeds threshold, and isolation
    //energy is below threshold
    if (photetsum>sumETThreshold_ && nonphotetsum<nonPhotETMax_) return true;
  }
  return false;

}


