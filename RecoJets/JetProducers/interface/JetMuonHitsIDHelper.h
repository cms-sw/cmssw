#ifndef RecoJets_JetProducers_interface_JetMuonHitsIDHelper_h
#define RecoJets_JetProducers_interface_JetMuonHitsIDHelper_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"


namespace reco {

  namespace helper {

    class JetMuonHitsIDHelper {

    public : 
      // construction
      JetMuonHitsIDHelper() {}
      JetMuonHitsIDHelper( edm::ParameterSet const & pset,  edm::ConsumesCollector&& iC );
      ~JetMuonHitsIDHelper() {} 

      void fillDescription(edm::ParameterSetDescription& iDesc);

      void initValues ();

      // interface
      void calculate( const edm::Event& event, const edm::EventSetup & isetup, 
		      const reco::Jet &jet, const int iDbg = 0 );

      // access
      int numberOfHits1RPC() const { return numberOfHits1RPC_;}
      int numberOfHits2RPC() const { return numberOfHits2RPC_;}
      int numberOfHits3RPC() const { return numberOfHits3RPC_;}
      int numberOfHits4RPC() const { return numberOfHits4RPC_;}
      int numberOfHitsRPC () const { return numberOfHitsRPC_ ;}
  
    private:

      edm::InputTag rpcRecHits_; // collection of rpc rechits
      bool          isRECO_;     // if this is RECO this will run, else nothing will be added

      int numberOfHits1RPC_;
      int numberOfHits2RPC_;
      int numberOfHits3RPC_;
      int numberOfHits4RPC_;
      int numberOfHitsRPC_;

      edm::EDGetTokenT<RPCRecHitCollection> input_rpchits_token_;

    };
  }
}
#endif
