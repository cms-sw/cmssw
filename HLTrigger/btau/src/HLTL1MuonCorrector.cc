//
// Package:         HLTrigger/btau
// Class:           HLTL1MuonCorrector
// 

#include <iostream>
#include <memory>
#include <string>
#include <TMath.h>
#include <TLorentzVector.h>

#include "HLTrigger/btau/interface/HLTL1MuonCorrector.h"
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

//Constructor
HLTL1MuonCorrector::HLTL1MuonCorrector(edm::ParameterSet const& conf)
{
  edm::LogInfo ("HLTL1MuonCorrector")<<"Enter the HLTL1MuonCorrector";
  produces<TrajectorySeedCollection>();

  ptmin=conf.getParameter<double>("ptMin");
  vertexSrc=conf.getParameter<string>("vertexSrc");
  thePhiCorrection = conf.getParameter<double>("phiCorrection");
  directionSrc = conf.getParameter<edm::InputTag>("directionSource");
  candMass = conf.getParameter<double>("candMass");
  produces<CandidateCollection>();

}

// Virtual destructor needed.
HLTL1MuonCorrector::~HLTL1MuonCorrector() { }  

// Functions that gets called by framework every event
void HLTL1MuonCorrector::produce(edm::Event& e, const edm::EventSetup& es)
{
  // get Inputs
  edm::Handle<reco::VertexCollection> vertices;
  e.getByLabel(vertexSrc,vertices);
  const reco::VertexCollection vertCollection = *(vertices.product());
  reco::VertexCollection::const_iterator ci = vertCollection.begin();
  std::auto_ptr<CandidateCollection> output(new CandidateCollection());  
  
  if(vertCollection.size() != 0){
  originz = ci->z();
  //Get the HLTObject direction
  edm::Handle<HLTFilterObjectWithRefs> ref;
  e.getByLabel(directionSrc,ref);

  if(ref.isValid())
  {
      for (unsigned int i=0;i<ref->size();i++)
	{
	 
	 LeafCandidate* particle = new LeafCandidate();
	 double phiCorrection = thePhiCorrection*(((ref->getParticleRef(i)).get())->charge());
	 LogDebug("HLTL1MuonCorrector")<<" phi correction: "<< phiCorrection;
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
	}
    LogDebug("HLTL1MuonCorrector")<<" number of corrected candidates = "<< output->size();
    e.put(output);
}
