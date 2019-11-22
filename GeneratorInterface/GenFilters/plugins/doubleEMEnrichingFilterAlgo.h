#ifndef doubleEMEnrichingFilterAlgo_h
#define doubleEMEnrichingFilterAlgo_h

/** \class doubleEMEnrichingFilter
 *
 *  doubleEMEnrichingFilter 
 *
 * \author R.Arcidiacono,C.Rovelli,R.Paramatti
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

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class doubleEMEnrichingFilterAlgo {
public:
  doubleEMEnrichingFilterAlgo(const edm::ParameterSet&);
  ~doubleEMEnrichingFilterAlgo();

  bool filter(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  bool hasBCAncestors(const reco::GenParticle& gp);

private:
  int filterPhotonElectronSeed(float clusterthreshold,
                               float seedthreshold,
                               float isoConeSize,
                               float hOverEMax,
                               float tkIsoMax,
                               float caloIsoMax,
                               bool requiretrackmatch,
                               const std::vector<reco::GenParticle>& genPars,
                               const std::vector<reco::GenParticle>& genParsCurved);

  std::vector<reco::GenParticle> applyBFieldCurv(const std::vector<reco::GenParticle>& genPars,
                                                 const edm::EventSetup& iSetup);
  int filterIsoGenPar(float etmin,
                      float conesize,
                      const reco::GenParticleCollection& gph,
                      const reco::GenParticleCollection& gphCurved);
  float deltaRxyAtEE(const reco::GenParticle& gp1, const reco::GenParticle& gp2);

  bool isBCHadron(const reco::GenParticle& gp);
  bool isBCMeson(const reco::GenParticle& gp);
  bool isBCBaryon(const reco::GenParticle& gp);

private:
  //constants
  float FILTER_TKISOCUT_;
  float FILTER_CALOISOCUT_;
  float FILTER_ETA_MIN_;
  float FILTER_ETA_MAX_;
  float ECALBARRELMAXETA_;
  float ECALBARRELRADIUS_;
  float ECALENDCAPZ_;

  // parameters of the filter
  float isoGenParETMin_;
  float isoGenParConeSize_;
  float clusterThreshold_;
  float seedThreshold_;
  float isoConeSize_;
  float hOverEMax_;
  float tkIsoMax_;
  float caloIsoMax_;
  float eTThreshold_;  // from bctoe
  bool requireTrackMatch_;
  edm::InputTag genParSource_;

  // for double em object
  std::vector<reco::GenParticle> sel1seeds;
  std::vector<reco::GenParticle> sel2seeds;
  std::vector<reco::GenParticle> selBCtoEseeds;
};
#endif
