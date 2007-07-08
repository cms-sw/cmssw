
// -*- C++ -*-
//
// Package:    L1Analyzer
// Class:      L1EtMissParticleShallowCloneProducer
// 
/**\class L1EtMissParticleShallowCloneProducer 

 Description: Make shallow clone of L1Extra Etmiss particle

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Jan 19 14:30:35 CET 2007
// $Id: L1EtMissParticleShallowCloneProducer.cc,v 1.1 2007/07/06 19:52:57 tapper Exp $
//
//

// user include files
#include "L1TriggerOffline/L1Analyzer/interface/L1EtMissParticleShallowCloneProducer.h"
      
// Data formats 
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"

using namespace edm;
using namespace reco;
using namespace l1extra;

L1EtMissParticleShallowCloneProducer::L1EtMissParticleShallowCloneProducer(const edm::ParameterSet& iConfig):
  m_l1EtMissSource(iConfig.getParameter<edm::InputTag>("src"))
{
   produces<CandidateCollection>();
}

L1EtMissParticleShallowCloneProducer::~L1EtMissParticleShallowCloneProducer()
{
}

void L1EtMissParticleShallowCloneProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  Handle<L1EtMissParticle> l1EtMiss;
  iEvent.getByLabel(m_l1EtMissSource, l1EtMiss);
  
  std::auto_ptr<CandidateCollection> cand(new CandidateCollection);
  RefProd<L1EtMissParticle> ref(l1EtMiss);
     
  cand->push_back(new ShallowCloneCandidate(CandidateBaseRef(ref)));
  iEvent.put(cand);   
}

