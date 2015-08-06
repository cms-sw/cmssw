//
//

#ifndef PhysicsTools_PatAlgos_PATLeptonCountFilter_h
#define PhysicsTools_PatAlgos_PATLeptonCountFilter_h

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"


namespace pat {


  class PATLeptonCountFilter : public edm::global::EDFilter<> {

    public:

      explicit PATLeptonCountFilter(const edm::ParameterSet & iConfig);
      virtual ~PATLeptonCountFilter();

    private:

      virtual bool filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const override;

    private:

      const edm::EDGetTokenT<edm::View<Electron> > electronToken_;
      const edm::EDGetTokenT<edm::View<Muon> > muonToken_;
      const edm::EDGetTokenT<edm::View<Tau> > tauToken_;
      const bool          countElectrons_;
      const bool          countMuons_;
      const bool          countTaus_;
      const unsigned int  minNumber_;
      const unsigned int  maxNumber_;

  };


}

#endif
