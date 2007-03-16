//
// Package:         RecoTracker/TkSeedGenerator
// Class:           CandidateFromHLTFOWRProducer
// 

#include <iostream>
#include <memory>
#include <string>
#include <TMath.h>
#include <TLorentzVector.h>

#include "HLTrigger/btau/interface/CandidateFromHLTFOWRProducer.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/LorentzVector.h"
#include "Math/PxPyPzM4D.h"
 

using namespace std;
using namespace reco;

CandidateFromHLTFOWRProducer::CandidateFromHLTFOWRProducer(edm::ParameterSet const& conf) : 
  conf_(conf)
{
  edm::LogInfo ("CandidateFromHLTFOWRProducer")<<"Enter the CandidateFromHLTFOWRProducer";
  produces<TrajectorySeedCollection>();

  ptmin=conf_.getParameter<double>("ptMin");
  vertexSrc=conf_.getParameter<string>("vertexSrc");
  thePhiCorrection = conf_.getParameter<double>("phiCorrection");
  directionSrc = conf_.getParameter<edm::InputTag>("directionSource");
  candMass = conf_.getParameter<double>("candMass");
  produces<CandidateCollection>();

}

// Virtual destructor needed.
CandidateFromHLTFOWRProducer::~CandidateFromHLTFOWRProducer() { }  

// Functions that gets called by framework every event
void CandidateFromHLTFOWRProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  // get Inputs
  edm::Handle<reco::VertexCollection> vertices;
  e.getByLabel(vertexSrc,vertices);
  const reco::VertexCollection vertCollection = *(vertices.product());
  reco::VertexCollection::const_iterator ci = vertCollection.begin();
  if(vertCollection.size() == 0) return;
  originz = ci->z();
    
  std::auto_ptr<CandidateCollection> output(new CandidateCollection());  

  //Get the HLTObject direction
  edm::Handle<HLTFilterObjectWithRefs> ref;
  e.getByLabel(directionSrc,ref);

  if(ref.isValid())
  {
      for (unsigned int i=0;i<ref->size();i++)
	{
	 
	 LeafCandidate* particle = new LeafCandidate();
	 double phiCorrection = thePhiCorrection*(((ref->getParticleRef(i)).get())->charge());
	 
	 cout << phiCorrection << endl;
	 
	 GlobalVector::Cylindrical coords(((ref->getParticleRef(i)).get())->pt(),((ref->getParticleRef(i)).get())->phi()+phiCorrection,((ref->getParticleRef(i)).get())->pz());
	  
	 GlobalVector direction(coords);
	 
	 ROOT::Math::LorentzVector<ROOT::Math::PxPyPzM4D<double> > p4(direction.x(), direction.y(), direction.z(), candMass);
	 
	 math::XYZTLorentzVector p4_new(p4);
	 
	 particle->setCharge(((ref->getParticleRef(i)).get())->charge());
	 particle->setP4(p4_new);
	 particle->setVertex(math::XYZPoint(0,0,originz));
	 
	  // write output to file
	  output->push_back(particle);
	}
   }   
	cout << "found " << output->size() << " candidates" << endl;
//     LogDebug("Algorithm Performance")<<" number of seeds = "<< output->size();
    e.put(output);
	cout << "end of module" << endl;
}
