#ifndef RecoTauTag_ConeIsolation
#define RecoTauTag_ConeIsolation

/** \class ConeIsolation
 *  EDProducer of the tagged TauJet with the ConeIsolationAlgorithm. 
 * It returns two collections: base collection is the JetTag, and extended Collection which is the IsolatedTauTagInfo.
 * The method implemented in the IsolatedTauTagInfo class are used to compute the discriminator variable.
 * A trick is used to link the IsolatedTauTagInfo to a smart reference to the JetTag.   
 *  $Revision: 1.2 $
 *  \author Simone Gennai
 */



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "HLTrigger/btau/src/ConeIsolationAlgorithm.h"


class ConeIsolation : public edm::EDProducer {
   public:
      explicit ConeIsolation(const edm::ParameterSet&);
      ~ConeIsolation();


      virtual void produce(edm::Event&, const edm::EventSetup&);
 private:
      ConeIsolationAlgorithm* m_algo;
      edm::InputTag jetTrackSrc;
      edm::InputTag vertexSrc;
      bool usingVertex;
      edm::InputTag beamSpotProducer;
      bool usingBeamSpot;
};
#endif

