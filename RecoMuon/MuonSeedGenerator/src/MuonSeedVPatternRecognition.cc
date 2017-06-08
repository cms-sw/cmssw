#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVPatternRecognition.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

MuonSeedVPatternRecognition::MuonSeedVPatternRecognition(const edm::ParameterSet & pset)
: enableDTMeasurement(pset.getParameter<bool>("EnableDTMeasurement")),
  enableCSCMeasurement(pset.getParameter<bool>("EnableCSCMeasurement")),
  enableGEMMeasurement(pset.getParameter<bool>("EnableGEMMeasurement")),
  enableME0Measurement(pset.getParameter<bool>("EnableME0Measurement"))
{
  if(enableDTMeasurement)
    // the name of the DT rec hits collection
    theDTRecSegmentLabel = pset.getParameter<edm::InputTag>("DTRecSegmentLabel");

  if(enableCSCMeasurement)
    // the name of the CSC rec hits collection
    theCSCRecSegmentLabel = pset.getParameter<edm::InputTag>("CSCRecSegmentLabel");

  if(enableGEMMeasurement){
    theGEMRecSegmentLabel = pset.getParameter<edm::InputTag>("GEMRecSegmentLabel");
    theGEMRecHitLabel     = pset.getParameter<edm::InputTag>("GEMRecHitLabel");
  }
  if(enableME0Measurement)
    theME0RecSegmentLabel = pset.getParameter<edm::InputTag>("ME0RecSegmentLabel");
}

