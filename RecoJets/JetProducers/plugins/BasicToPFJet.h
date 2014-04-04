#ifndef UserCode_BasicToPFJet_h
#define UserCode_BasicToPFJet_h

// system include files
#include <memory>
#include <vector>
#include <sstream>

// user include files
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/JetReco/interface/CATopJetTagInfo.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
using std::cout;
using std::endl;

class BasicToPFJet : public edm::EDProducer {

 public:

  explicit BasicToPFJet(const edm::ParameterSet& PSet);
  virtual ~BasicToPFJet();
  virtual void produce(edm::Event & event, const edm::EventSetup & EventSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  edm::InputTag src_;
  const edm::EDGetTokenT<reco::BasicJetCollection> inputToken_;
};


#endif
