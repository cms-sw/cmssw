// -*- C++ -*-
//
// Package:    TagProbeProducer
// Class:      TagProbeProducer
// 
/**\class TagProbeProducer TagProbeProducer.cc PhysicsTools/TagProbeProducer/src/TagProbeProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

// User includes
#include "PhysicsTools/TagAndProbe/interface/TagProbePairProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include <memory>


TagProbePairProducer::TagProbePairProducer(const edm::ParameterSet& iConfig) {

   probeCollection_ = iConfig.getParameter<edm::InputTag>("ProbeCollection");
   passingProbeCollection_ = iConfig.getParameter<edm::InputTag>("PassingProbeCollection");

   produces< std::vector< std::pair <reco::CandidateBaseRef,bool> > >();
}

TagProbePairProducer::~TagProbePairProducer() {}

// ------------ method called to produce the data  ------------
void TagProbePairProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

   std::auto_ptr< std::vector< std::pair <reco::CandidateBaseRef, bool> > > ProbeCollection (new std::vector< std::pair <reco::CandidateBaseRef, bool> >); 

   if ( !iEvent.getByLabel( probeCollection_, probes ) ) {
      edm::LogWarning("TagProbe") << "Could not extract probe with input tag "
				 << probeCollection_;
   }

   if ( !iEvent.getByLabel( passingProbeCollection_, passingProbes ) ) {
      edm::LogWarning("TagProbe") << "Could not extract passing probe with input tag "
				 << passingProbeCollection_;
   }

   // Loop over Tag and associate with Probes
   if( passingProbes.isValid() && probes.isValid() ) {

      const edm::RefToBaseVector<reco::Candidate>& vprobes = probes->refVector();

      int iprobe = 0;
      edm::RefToBaseVector<reco::Candidate>::const_iterator probe = vprobes.begin();
      for( ; probe != vprobes.end(); ++probe, ++iprobe ) {
	
	bool isPassing = isPassingProbe (iprobe);
	
	ProbeCollection->push_back(std::make_pair(vprobes[iprobe], isPassing));
      }
   }

   // Finally put the tag probe collection in the event
   iEvent.put( ProbeCollection );
}

// ------------ method called once each job just before starting event loop  ------------
void 
TagProbePairProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TagProbePairProducer::endJob() {
}

void TagProbePairProducer::checkPassingProbes () const {

  edm::RefToBase<reco::Candidate> probeRef;
  edm::RefToBase<reco::Candidate> passingProbeRef;

  unsigned int numProbes = probes->size();
  unsigned int numPassingProbes = passingProbes->size();

  for (unsigned int iPassProbe = 0; iPassProbe < numPassingProbes; ++iPassProbe) { 
    for (unsigned int iProbe = 0; iProbe < numProbes; ++iProbe) {
      probeRef = probes->refAt(iProbe);
      passingProbeRef = passingProbes->refAt(iPassProbe);
      
      if (passingProbeRef == probeRef) break;
    }
    edm::LogError("TagAndProbe") << "Passing probe is not in the set of probes. Please, review your selection criteria.";
  }
}


bool TagProbePairProducer::isPassingProbe (const unsigned int iProbe) const {

  if (iProbe > probes->size()) return false;

  edm::RefToBase<reco::Candidate> probeRef = probes->refAt(iProbe);
  edm::RefToBase<reco::Candidate> passingProbeRef;

  unsigned int numPassingProbes = passingProbes->size();

  for (unsigned int iPassProbe = 0; iPassProbe < numPassingProbes; ++iPassProbe) {
    passingProbeRef = passingProbes->refAt(iPassProbe);
    if (passingProbeRef == probeRef) {
      return true;
    }
  }
  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TagProbePairProducer);
