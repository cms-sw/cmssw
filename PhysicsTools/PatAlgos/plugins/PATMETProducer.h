//
// $Id$
//

#ifndef PhysicsTools_PatAlgos_PATMETProducer_h
#define PhysicsTools_PatAlgos_PATMETProducer_h

/**
  \class    pat::PATMETProducer PATMETProducer.h "PhysicsTools/PatAlgos/interface/PATMETProducer.h"
  \brief    Produces the pat::MET

   The PATMETProducer produces the analysis-level pat::MET starting from
   a collection of objects of METType.

  \author   Steven Lowette
  \version  $Id$
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"

#include "DataFormats/PatCandidates/interface/MET.h"


namespace pat {


  class ObjectResolutionCalc;


  class PATMETProducer : public edm::EDProducer {

    public:

      explicit PATMETProducer(const edm::ParameterSet & iConfig);
      ~PATMETProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:

      // configurables
      edm::InputTag metSrc_;
      bool          addGenMET_;
      edm::InputTag genMETSrc_;
      bool          addTrigMatch_;
      std::vector<edm::InputTag> trigPrimSrc_;
      bool          addResolutions_;
      bool          useNNReso_;
      std::string   metResoFile_;
      bool          addMuonCorr_;
      edm::InputTag muonSrc_;
      // tools
      ObjectResolutionCalc * metResoCalc_;
      GreaterByEt<MET> eTComparator_;

  };


}

#endif
