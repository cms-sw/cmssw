//
//

#ifndef PhysicsTools_PatAlgos_PATLeptonCountFilter_h
#define PhysicsTools_PatAlgos_PATLeptonCountFilter_h

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"


namespace pat {


  class PATLeptonCountFilter : public edm::EDFilter {

    public:

      explicit PATLeptonCountFilter(const edm::ParameterSet & iConfig);
      virtual ~PATLeptonCountFilter();

    private:

      virtual bool filter(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:

      edm::EDGetTokenT<edm::View<Electron> > electronToken_;
      edm::EDGetTokenT<edm::View<Muon> > muonToken_;
      edm::EDGetTokenT<edm::View<Tau> > tauToken_;
      bool          countElectrons_;
      bool          countMuons_;
      bool          countTaus_;
      unsigned int  minNumber_;
      unsigned int  maxNumber_;

  };


}

#endif
