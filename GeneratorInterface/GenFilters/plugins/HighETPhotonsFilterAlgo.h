#ifndef HighETPhotonsFilterAlgo_h
#define HighETPhotonsFilterAlgo_h

/** \class HighETPhotonsFilterAlgo
 *
 *  HighETPhotonsFilterAlgo
 *  a gen-level filter that selects events that will reconstruct an _isolated_ photon
 *   only tested with high ET thresholds (aiming for 100 GeV photons)
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

class HighETPhotonsFilterAlgo {
public:
  HighETPhotonsFilterAlgo(const edm::ParameterSet&);
  ~HighETPhotonsFilterAlgo();

  bool filter(const edm::Event& iEvent);

private:
private:
  //constants:
  float FILTER_ETA_MAX_;
  //filter parameters:
  float sumETThreshold_;
  float seedETThreshold_;
  float nonPhotETMax_;
  float isoConeSize_;
};
#endif
