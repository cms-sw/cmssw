#ifndef RecoTauTag_InvariantMass
#define RecoTauTag_InvariantMass

/** \class InvariantMass
 *  EDProducer of the tagged TauJet with the InvariantMassAlgorithm. 
 * It returns two collections: base collection is the JetTag, and extended Collection which is the IsolatedTauTagInfo.
 * The method implemented in the IsolatedTauTagInfo class are used to compute the discriminator variable.
 * A trick is used to link the IsolatedTauTagInfo to a smart reference to the JetTag.   
 *  $Revision: 1.1 $
 *  \author Suchandra Dutta
 */



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "RecoTauTag/InvariantMass/interface/InvariantMassAlgorithm.h"


class InvariantMass : public edm::EDProducer {
   public:
  explicit InvariantMass(const edm::ParameterSet&);
  ~InvariantMass();
  
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  reco::BasicClusterRefProd getSelectedClusters(reco::BasicClusterRefProd bclus, reco::BasicClusterRefProd eclus, reco::Jet& jet);
  
  InvariantMassAlgorithm* m_algo;
  std::string jetTrackSrc;
  std::string m_ecalBClSrc;
};
#endif

