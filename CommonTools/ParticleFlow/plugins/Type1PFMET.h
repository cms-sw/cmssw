#ifndef PhysicsTools_PFCandProducer_Type1PFMET_h
#define PhysicsTools_PFCandProducer_Type1PFMET_h

/**\class Type1PFMET
\brief Computes the Type-1 corrections for pfMET. A specific version of the Type1MET class from the JetMETCorrections/Type1MET package.

\todo Unify with the Type1MET class from the JetMETCorrections/Type1MET package

\author Michal Bluj
\date   February 2009
*/

// system include files
#include <memory>
#include <string.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"



// PRODUCER CLASS DEFINITION -------------------------------------
class Type1PFMET : public edm::EDProducer
{
 public:
  explicit Type1PFMET( const edm::ParameterSet& );
  explicit Type1PFMET();
  virtual ~Type1PFMET();
  virtual void produce( edm::Event&, const edm::EventSetup& );
 private:
  edm::EDGetTokenT<reco::METCollection> tokenUncorMet;
  edm::EDGetTokenT<reco::PFJetCollection> tokenUncorJets;
  edm::EDGetTokenT<reco::JetCorrector> correctorToken;
  double jetPTthreshold;
  double jetEMfracLimit;
  double jetMufracLimit;
  void run(const reco::METCollection& uncorMET,
	   const reco::JetCorrector& corrector,
	   const reco::PFJetCollection& uncorJet,
	   double jetPTthreshold,
	   double jetEMfracLimit,
	   double jetMufracLimit,
	   reco::METCollection* corMET);
};

#endif
