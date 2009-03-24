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
// $Id: TagProbeProducer.cc,v 1.2 2008/07/30 13:38:25 srappocc Exp $
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


//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
TagProbeProducer::TagProbeProducer(const edm::ParameterSet& iConfig)
{
   tagCollection_   = iConfig.getParameter<edm::InputTag>("TagCollection");
   probeCollection_ = iConfig.getParameter<edm::InputTag>("ProbeCollection");

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
   using namespace edm;
   using namespace reco;
   using namespace std;

   // We need the output Muon association collection to fill
   auto_ptr<CandViewCandViewAssociation> muonTPCollection( new CandViewCandViewAssociation );

   // Read in the tag muons
   Handle< CandidateView > tags;
   if ( !iEvent.getByLabel( tagCollection_, tags ) ) {
      LogWarning("TagProbe") << "Could not extract tag muons with input tag "
				 << tagCollection_;
   }

   // Read in the probe muons
   Handle< CandidateView > probes;
   if ( !iEvent.getByLabel( probeCollection_, probes ) ) {
      LogWarning("TagProbe") << "Could not extract probe muons with input tag "
				 << probeCollection_;
   }


   // Loop over Tag and associate with Probes
   if( tags.isValid() && probes.isValid() )
   {
      const RefToBaseVector<Candidate>& vtags = tags->refVector();
      const RefToBaseVector<Candidate>& vprobes = probes->refVector();

      int itag = 0;
      RefToBaseVector<Candidate>::const_iterator tag = vtags.begin();
      for( ; tag != vtags.end(); ++tag, ++itag ) 
      {  
	 int iprobe = 0;
	 RefToBaseVector<Candidate>::const_iterator probe = vprobes.begin();
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

            muonTPCollection->insert( vtags[itag], make_pair(vprobes[iprobe],invMass) );
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

//define this as a plug-in
DEFINE_FWK_MODULE(TagProbeProducer);
