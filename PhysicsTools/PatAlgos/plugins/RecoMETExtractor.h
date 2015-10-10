#ifndef PhysicsTools_PatAlgos_RecoMETExtractor_h
#define PhysicsTools_PatAlgos_RecoMETExtractor_h

/**
  \class    pat::RecoMETExtractor RecoMETExtractor.h "PhysicsTools/PatAlgos/interface/RecoMETExtractor.h"
  \brief    Retrieves the recoMET from a pat::MET

   The RecoMETExtractor produces the analysis-level pat::MET starting from
   a collection of objects of METType.

  \author   Matthieu Marionneau
  \version  $Id: RecoMETExtractor.h,v 1.0 2015/07/22 mmarionn Exp $
*/


#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/METReco/interface/MET.h"

namespace pat {

  class RecoMETExtractor : public edm::global::EDProducer<> {

    public:

    explicit RecoMETExtractor(const edm::ParameterSet& iConfig);
    ~RecoMETExtractor();

    virtual void produce(edm::StreamID streamID, edm::Event & iEvent,
			 const edm::EventSetup & iSetup) const;

  private:

    edm::EDGetTokenT<std::vector<pat::MET> > metSrcToken_;
    
    pat::MET::METCorrectionLevel corLevel_;

  };

}

#endif
