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
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class EMEnrichingFilterAlgo {
public:
  EMEnrichingFilterAlgo(const edm::ParameterSet &, edm::ConsumesCollector &&);
  ~EMEnrichingFilterAlgo();

  bool filter(const edm::Event &iEvent, const edm::EventSetup &iSetup);

private:
  bool filterPhotonElectronSeed(float clusterthreshold,
                                float isoConeSize,
                                float hOverEMax,
                                float tkIsoMax,
                                float caloIsoMax,
                                bool requiretrackmatch,
                                const std::vector<reco::GenParticle> &genPars,
                                const std::vector<reco::GenParticle> &genParsCurved);

  std::vector<reco::GenParticle> applyBFieldCurv(const std::vector<reco::GenParticle> &genPars,
                                                 const edm::EventSetup &iSetup);
  bool filterIsoGenPar(float etmin,
                       float conesize,
                       const reco::GenParticleCollection &gph,
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

  float isoGenParETMin_;
  float isoGenParConeSize_;
  float clusterThreshold_;
  float isoConeSize_;
  float hOverEMax_;
  float tkIsoMax_;
  float caloIsoMax_;
  bool requireTrackMatch_;
  edm::InputTag genParSource_;

  edm::EDGetTokenT<reco::GenParticleCollection> genParSourceToken_;
};
#endif
