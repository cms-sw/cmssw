#ifndef EMEnrichingFilterAlgo_h
#define EMEnrichingFilterAlgo_h

/** \class EMEnrichingFilter
 *
 *  EMEnrichingFilter 
 *
 * \author J Lamb, UCSB
 *
 ************************************************************/

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <vector>

class EMEnrichingFilterAlgo {
public:
  EMEnrichingFilterAlgo(const edm::ParameterSet &, edm::ConsumesCollector &&);

  bool filter(const edm::Event &iEvent, const edm::EventSetup &iSetup) const;

private:
  bool filterPhotonElectronSeed(float clusterthreshold,
                                float isoConeSize,
                                float hOverEMax,
                                float tkIsoMax,
                                float caloIsoMax,
                                bool requiretrackmatch,
                                const std::vector<reco::GenParticle> &genPars,
                                const std::vector<reco::GenParticle> &genParsCurved) const;

  std::vector<reco::GenParticle> applyBFieldCurv(const std::vector<reco::GenParticle> &genPars,
                                                 const edm::EventSetup &iSetup) const;
  bool filterIsoGenPar(float etmin,
                       float conesize,
                       const reco::GenParticleCollection &gph,
                       const reco::GenParticleCollection &gphCurved) const;
  float deltaRxyAtEE(const reco::GenParticle &gp1, const reco::GenParticle &gp2) const;

private:
  //constants
  const float FILTER_TKISOCUT_;
  const float FILTER_CALOISOCUT_;
  const float FILTER_ETA_MIN_;
  const float FILTER_ETA_MAX_;
  const float ECALBARRELMAXETA_;
  const float ECALBARRELRADIUS_;
  const float ECALENDCAPZ_;

  //parameters of the filter
  const float isoGenParETMin_;
  const float isoGenParConeSize_;
  const float clusterThreshold_;
  const float isoConeSize_;
  const float hOverEMax_;
  const float tkIsoMax_;
  const float caloIsoMax_;
  const bool requireTrackMatch_;

  const edm::EDGetTokenT<reco::GenParticleCollection> genParSourceToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
};
#endif
