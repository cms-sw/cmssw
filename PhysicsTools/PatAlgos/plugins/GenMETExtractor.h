
#ifndef PhysicsTools_PatAlgos_GenMETExtractor_h
#define PhysicsTools_PatAlgos_GenMETExtractor_h

/**
  \class    pat::GenMETExtractor GenMETExtractor.h "PhysicsTools/PatAlgos/interface/GenMETExtractor.h"
  \brief    Retrieves the genMET from a pat::MET

   The GenMETExtractor produces the analysis-level pat::MET starting from
   a collection of objects of METType.

  \author   Matthieu Marionneau
  \version  $Id: GenMETExtractor.h,v 1.0 2015/07/22 mmarionn Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/METReco/interface/GenMET.h"

namespace pat {

  class GenMETExtractor : public edm::EDProducer {

    public:

    explicit GenMETExtractor(const edm::ParameterSet& iConfig);
    ~GenMETExtractor();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    edm::EDGetTokenT<std::vector<pat::MET> > metSrcToken_;
    
  };

}

#endif
