//
// $Id: PATLeptonCountFilter.h,v 1.2 2013/02/27 23:26:56 wmtan Exp $
//

#ifndef PhysicsTools_PatAlgos_PATLeptonCountFilter_h
#define PhysicsTools_PatAlgos_PATLeptonCountFilter_h

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace pat {


  class PATLeptonCountFilter : public edm::EDFilter {

    public:

      explicit PATLeptonCountFilter(const edm::ParameterSet & iConfig);
      virtual ~PATLeptonCountFilter();

    private:

      virtual bool filter(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:

      edm::InputTag electronSource_;
      edm::InputTag muonSource_;
      edm::InputTag tauSource_;
      bool          countElectrons_;
      bool          countMuons_;
      bool          countTaus_;
      unsigned int  minNumber_;
      unsigned int  maxNumber_;

  };


}

#endif
