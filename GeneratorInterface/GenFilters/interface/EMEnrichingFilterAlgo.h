#ifndef EMEnrichingFilterAlgo_h
#define EMEnrichingFilterAlgo_h

/** \class EMEnrichingFilter
 *
 *  EMEnrichingFilter 
 *
 * \author J Lamb, UCSB
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


class EMEnrichingFilterAlgo {
 public:
  EMEnrichingFilterAlgo(const edm::ParameterSet&);
  ~EMEnrichingFilterAlgo();
  
  bool filter(const edm::Event& iEvent, const edm::EventSetup& iSetup);

 private:
  bool filterPhotonElectronSeed(float seedthreshold, 
				float clusterthreshold, 
				float conesize,
				float seedthresholdendcap, 
				float clusterthresholdendcap, 
				float conesizeendcap,
				const std::vector<reco::GenParticle> &genPars);

  std::vector<reco::GenParticle> applyBFieldCurv(const std::vector<reco::GenParticle> &genPars, const edm::EventSetup& iSetup);
  bool filterIsoGenPar(float etmin, float conesize,const reco::GenParticleCollection &gph,
		       const reco::GenParticleCollection &gphCurved);
  float deltaRxyAtEE(const reco::GenParticle &gp1, const reco::GenParticle &gp2);
    
		       
 private:

  //constants
  float FILTER_TKISOCUT_;
  float FILTER_CALOISOCUT_;
  float FILTER_ETA_MIN_;
  float FILTER_ETA_MAX_;
  float ECALBARRELMAXETA_;
  float ECALBARRELRADIUS_;
  float ECALENDCAPZ_;

  //parameters of the filter
  float seedThresholdBarrel_;
  float clusterThresholdBarrel_;
  float coneSizeBarrel_;
  float seedThresholdEndcap_;
  float clusterThresholdEndcap_;
  float coneSizeEndcap_;
  float isoGenParETMin_;
  float isoGenParConeSize_;
  
};
#endif
