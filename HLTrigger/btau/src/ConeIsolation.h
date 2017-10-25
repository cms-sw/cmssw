#ifndef RecoTauTag_ConeIsolation
#define RecoTauTag_ConeIsolation

/** \class ConeIsolation
 *  EDProducer of the tagged TauJet with the ConeIsolationAlgorithm. 
 * It returns two collections: base collection is the JetTag, and extended Collection which is the IsolatedTauTagInfo.
 * The method implemented in the IsolatedTauTagInfo class are used to compute the discriminator variable.
 * A trick is used to link the IsolatedTauTagInfo to a smart reference to the JetTag.   
 *  \author Simone Gennai
 */



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "HLTrigger/btau/src/ConeIsolationAlgorithm.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


class ConeIsolation : public edm::EDProducer {
   public:
      explicit ConeIsolation(const edm::ParameterSet&);
      ~ConeIsolation() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      void produce(edm::Event&, const edm::EventSetup&) override;

   private:
      edm::InputTag                                          jetTrackTag;
      edm::EDGetTokenT<reco::JetTracksAssociationCollection> jetTrackToken;
      edm::InputTag                                          vertexTag;
      edm::EDGetTokenT<reco::VertexCollection>               vertexToken;
      edm::InputTag                                          beamSpotTag;
      edm::EDGetTokenT<reco::BeamSpot>                       beamSpotToken;
      bool usingBeamSpot;
      ConeIsolationAlgorithm* m_algo;
};
#endif

