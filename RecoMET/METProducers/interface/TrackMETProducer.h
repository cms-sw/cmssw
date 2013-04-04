#ifndef RecoMET_METProducers_TrackMETProducer_h
#define RecoMET_METProducers_TrackMETProducer_h

/** \class TrackMETProducer
 *
 * Produce track MET
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: TrackMETProducer.cc,v 1.1 2012/09/15 16:46:18 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include <string>

class TrackMETProducer : public edm::EDProducer  
{
 public:

  explicit TrackMETProducer(const edm::ParameterSet&);
  ~TrackMETProducer() {}
    
 private:

  void produce(edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  edm::InputTag src_;
  double globalThreshold_;
};

#endif



 
