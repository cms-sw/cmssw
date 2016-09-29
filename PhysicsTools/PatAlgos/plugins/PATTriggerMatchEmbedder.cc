// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      pat::PATTriggerMatchEmbedder
//
/**
  \class    pat::PATTriggerMatchEmbedder PATTriggerMatchEmbedder.cc "PhysicsTools/PatAlgos/plugins/PATTriggerMatchEmbedder.cc"
  \brief

   .

  \author   Volker Adler
  \version  $Id: PATTriggerMatchEmbedder.cc,v 1.6 2010/09/02 17:52:47 vadler Exp $
*/
//
//


#include <vector>

#include "FWCore/Utilities/interface/transform.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"


namespace pat {

  template< class PATObjectType >
  class PATTriggerMatchEmbedder : public edm::global::EDProducer<> {

      const edm::InputTag src_;
      const edm::EDGetTokenT< edm::View< PATObjectType > > srcToken_;
      const std::vector< edm::InputTag > matches_;
      const std::vector< edm::EDGetTokenT< TriggerObjectStandAloneMatch > > matchesTokens_;

    public:

      explicit PATTriggerMatchEmbedder( const edm::ParameterSet & iConfig );
      ~PATTriggerMatchEmbedder() {};

    private:

    virtual void produce( edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const override;

  };

  typedef PATTriggerMatchEmbedder< Electron > PATTriggerMatchElectronEmbedder;
  typedef PATTriggerMatchEmbedder< Jet >      PATTriggerMatchJetEmbedder;
  typedef PATTriggerMatchEmbedder< MET >      PATTriggerMatchMETEmbedder;
  typedef PATTriggerMatchEmbedder< Muon >     PATTriggerMatchMuonEmbedder;
  typedef PATTriggerMatchEmbedder< Photon >   PATTriggerMatchPhotonEmbedder;
  typedef PATTriggerMatchEmbedder< Tau >      PATTriggerMatchTauEmbedder;

}


using namespace pat;


template< class PATObjectType >
PATTriggerMatchEmbedder< PATObjectType >::PATTriggerMatchEmbedder( const edm::ParameterSet & iConfig ) :
  src_( iConfig.getParameter< edm::InputTag >( "src" ) ),
  srcToken_( consumes< edm::View< PATObjectType > >( src_ ) ),
  matches_( iConfig.getParameter< std::vector< edm::InputTag > >( "matches" ) ),
  matchesTokens_( edm::vector_transform( matches_, [this](edm::InputTag const & tag) { return consumes< TriggerObjectStandAloneMatch >( tag ); } ) )
{
  produces< std::vector< PATObjectType > >();
}

template< class PATObjectType >
void PATTriggerMatchEmbedder< PATObjectType >::produce( edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const
{
  auto output = std::make_unique<std::vector<PATObjectType>>();

  edm::Handle< edm::View< PATObjectType > > candidates;
  iEvent.getByToken( srcToken_, candidates );
  if ( ! candidates.isValid() ) {
    edm::LogError( "missingInputSource" ) << "Input source with InputTag " << src_.encode() << " not in event.";
    return;
  }

  for ( typename edm::View< PATObjectType >::const_iterator iCand = candidates->begin(); iCand != candidates->end(); ++iCand ) {
    const unsigned index( iCand - candidates->begin() );
    PATObjectType cand( candidates->at( index ) );
    std::set< TriggerObjectStandAloneRef > cachedRefs;
    for ( size_t iMatch = 0; iMatch < matchesTokens_.size(); ++iMatch ) {
      edm::Handle< TriggerObjectStandAloneMatch > match;
      iEvent.getByToken( matchesTokens_.at( iMatch ), match );
      if ( ! match.isValid() ) {
        edm::LogError( "missingInputMatch" ) << "Input match with InputTag " << matches_.at( iMatch ).encode() << " not in event.";
        continue;
      }
      const TriggerObjectStandAloneRef trigRef( ( *match )[ candidates->refAt( index ) ] );
      if ( trigRef.isNonnull() && trigRef.isAvailable() ) {
        if ( cachedRefs.insert( trigRef ).second ) { // protection from multiple entries of the same trigger objects
          cand.addTriggerObjectMatch( *trigRef );
        }
      }
    }
    output->push_back( cand );
  }

  iEvent.put(std::move(output) );
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( PATTriggerMatchElectronEmbedder );
DEFINE_FWK_MODULE( PATTriggerMatchJetEmbedder );
DEFINE_FWK_MODULE( PATTriggerMatchMETEmbedder );
DEFINE_FWK_MODULE( PATTriggerMatchMuonEmbedder );
DEFINE_FWK_MODULE( PATTriggerMatchPhotonEmbedder );
DEFINE_FWK_MODULE( PATTriggerMatchTauEmbedder );
