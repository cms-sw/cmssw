#ifndef PhysicsTools_PatAlgos_PATSingleVertexSelector_h
#define PhysicsTools_PatAlgos_PATSingleVertexSelector_h
//
//

/**
  \class    pat::PATSingleVertexSelector PATSingleVertexSelector.h "PhysicsTools/PatAlgos/plugins/PATSingleVertexSelector.h"
  \brief    Produces a list containing a single vertex selected by some criteria


  \author   Giovanni Petrucciani
  \version  $Id: PATSingleVertexSelector.h,v 1.5 2011/06/15 11:47:25 friis Exp $
*/

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace pat {

  class PATSingleVertexSelector : public edm::stream::EDFilter<> {

    public:

      explicit PATSingleVertexSelector(const edm::ParameterSet & iConfig);
      ~PATSingleVertexSelector();

      virtual bool filter(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:
      enum Mode { First, NearestToCand, FromCand, FromBeamSpot };
      typedef StringCutObjectSelector<reco::Vertex>    VtxSel;
      typedef StringCutObjectSelector<reco::Candidate> CandSel;

      Mode parseMode(const std::string &name) const;
      
      std::auto_ptr<std::vector<reco::Vertex> >
        filter_(Mode mode, const edm::Event & iEvent, const edm::EventSetup & iSetup);
      bool hasMode_(Mode mode) const ;
      // configurables
      std::vector<Mode> modes_; // mode + optional fallbacks
      edm::EDGetTokenT<std::vector<reco::Vertex> > verticesToken_;
      std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > > candidatesToken_;
      const VtxSel vtxPreselection_;
      const CandSel candPreselection_;
      edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
      // transient data. meaningful while 'filter()' is on the stack
      std::vector<reco::VertexRef> selVtxs_;
      reco::CandidatePtr           bestCand_;

      // flag to enable/disable EDFilter functionality:
      // if set to false, PATSingleVertexSelector selects the "one" event vertex,
      // but does not reject any events
      bool doFilterEvents_;
  };

}

#endif

