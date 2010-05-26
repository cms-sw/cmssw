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
//
// Original Author:  Nadia Adam
//         Created:  Wed Apr 16 09:46:30 CDT 2008
// $Id: TagProbeProducer.cc,v 1.3 2009/03/24 19:32:38 ahunt Exp $
//
//


// User includes
#include "PhysicsTools/TagAndProbe/interface/TagProbeProducer.h"
#include "PhysicsTools/TagAndProbe/interface/CandidateAssociation.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Math/GenVector/VectorUtil.h"

TagProbeProducer::TagProbeProducer(const edm::ParameterSet& iConfig)
{
   tagCollection_   = iConfig.getParameter<edm::InputTag>("TagCollection");
   probeCollection_ = iConfig.getParameter<edm::InputTag>("ProbeCollection");
   passingProbeCollection_ = iConfig.getParameter<edm::InputTag>("PassingProbeCollection");

   massMinCut_      = iConfig.getUntrackedParameter<double>("MassMinCut",50.0);
   massMaxCut_      = iConfig.getUntrackedParameter<double>("MassMaxCut",120.0);
   delRMinCut_      = iConfig.getUntrackedParameter<double>("DelRMinCut",0.0);
   delRMaxCut_      = iConfig.getUntrackedParameter<double>("DelRMaxCut",10000.0);

   requireOS_       = iConfig.getUntrackedParameter<bool>("RequireOS",true);

   produces<reco::CandViewCandViewAssociation>();
}


TagProbeProducer::~TagProbeProducer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TagProbeProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   // We need the output Muon association collection to fill
   std::auto_ptr<reco::CandViewCandViewAssociation> muonTPCollection( new reco::CandViewCandViewAssociation );

   if ( !iEvent.getByLabel( tagCollection_, tags ) ) {
      edm::LogWarning("TagProbe") << "Could not extract tag muons with input tag "
				 << tagCollection_;
   }

   if ( !iEvent.getByLabel( probeCollection_, probes ) ) {
      edm::LogWarning("TagProbe") << "Could not extract probe muons with input tag "
				 << probeCollection_;
   }

   if ( !iEvent.getByLabel( passingProbeCollection_, passingProbes ) ) {
      edm::LogWarning("TagProbe") << "Could not extract passing probe muons with input tag "
				 << passingProbeCollection_;
   }

   // Loop over Tag and associate with Probes
   if( tags.isValid() && probes.isValid() )
   {
      const edm::RefToBaseVector<reco::Candidate>& vtags = tags->refVector();
      const edm::RefToBaseVector<reco::Candidate>& vprobes = probes->refVector();

      int itag = 0;
      edm::RefToBaseVector<reco::Candidate>::const_iterator tag = vtags.begin();
      for( ; tag != vtags.end(); ++tag, ++itag ) 
      {  
	 int iprobe = 0;
	 edm::RefToBaseVector<reco::Candidate>::const_iterator probe = vprobes.begin();
	 for( ; probe != vprobes.end(); ++probe, ++iprobe ) 
	 {
	    // Tag-Probe invariant mass cut
	    double invMass = ROOT::Math::VectorUtil::InvariantMass((*tag)->p4(), (*probe)->p4());
	    if( invMass < massMinCut_ ) continue;
	    if( invMass > massMaxCut_ ) continue;

	    // Tag-Probe deltaR cut
	    double delR = deltaR<double>((*tag)->eta(),(*tag)->phi(),(*probe)->eta(),(*probe)->phi());
	    if( delR < delRMinCut_ ) continue;
	    if( delR > delRMaxCut_ ) continue;

	    // Tag-Probe opposite sign
	    int sign = (*tag)->charge() * (*probe)->charge();
	    if( requireOS_ && sign > 0 ) continue;

	    bool isPassing = isPassingProbe (iprobe);
	    
            muonTPCollection->insert( vtags[itag], std::make_pair(vprobes[iprobe], isPassing) );
	 }	 
      }
   }

   // Finally put the tag probe collection in the event
   iEvent.put( muonTPCollection );
}

// ------------ method called once each job just before starting event loop  ------------
void 
TagProbeProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TagProbeProducer::endJob() {
}


bool TagProbeProducer::isPassingProbe (const unsigned int iProbe) const {

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
DEFINE_FWK_MODULE(TagProbeProducer);
