#ifndef RECOJETS_JETPRODUCERS_SUBJETFILTERJETPRODUCER_H
#define RECOJETS_JETPRODUCERS_SUBJETFILTERJETPRODUCER_H 1


/*
  The plugin produces the output of the Subjet/Filter jet reconstruction
  algorithm which was first proposed here: http://arXiv.org/abs/0802.2470
  
  The real work is done in RecoJets/JetAlgorithms/src/SubjetFilterAlgorithm.cc
  
  CMSSW implementation by David Lopes-Pegna           <david.lopes-pegna@cern.ch>
                      and Philipp Schieferdecker <philipp.schieferdecker@cern.ch>

  see: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSubjetFilterJetProducer

*/


#include "VirtualJetProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"
#include "RecoJets/JetAlgorithms/interface/SubjetFilterAlgorithm.h"


class SubjetFilterJetProducer : public VirtualJetProducer
{
  //
  // construction / destruction
  //
public:
  SubjetFilterJetProducer(const edm::ParameterSet& ps);
  virtual ~SubjetFilterJetProducer();
  
  
  //
  // member functions
  //
public:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
  void endJob();
  void runAlgorithm(edm::Event& iEvent, const edm::EventSetup& iSetup);
  void inputTowers();
  void output(edm::Event& iEvent,const edm::EventSetup& iSetup);
  template<class T>
  void writeCompoundJets(edm::Event& iEvent,const edm::EventSetup& iSetup);
  
  
  //
  // member data
  //
private:
  SubjetFilterAlgorithm           alg_;
  std::vector<CompoundPseudoJet>  fjCompoundJets_;
};


#endif
