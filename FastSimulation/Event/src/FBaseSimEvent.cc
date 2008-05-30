//HepMC Headers
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"

//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//CMSSW Data Formats
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//FAMOS Headers
#include "FastSimulation/Event/interface/FBaseSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/Event/interface/BetaFuncPrimaryVertexGenerator.h"
#include "FastSimulation/Event/interface/GaussianPrimaryVertexGenerator.h"
#include "FastSimulation/Event/interface/FlatPrimaryVertexGenerator.h"
#include "FastSimulation/Event/interface/NoPrimaryVertexGenerator.h"
//#include "FastSimulation/Utilities/interface/Histos.h"

using namespace HepPDT;

// system include
#include <iostream>
#include <iomanip>
#include <map>
#include <string>

FBaseSimEvent::FBaseSimEvent(const edm::ParameterSet& kine) 
  :
  nSimTracks(0),
  nSimVertices(0),
  nGenParticles(0),
  nChargedParticleTracks(0),
  initialSize(5000),
  random(0)
{

  theVertexGenerator = new NoPrimaryVertexGenerator();
  theBeamSpot = math::XYZPoint(0.0,0.0,0.0);

  // Initialize the vectors of particles and vertices
  theGenParticles = new std::vector<HepMC::GenParticle*>(); 
  theSimTracks = new std::vector<FSimTrack>;
  theSimVertices = new std::vector<FSimVertex>;
  theChargedTracks = new std::vector<unsigned>();

  // Reserve some size to avoid mutiple copies
  /* */
  theSimTracks->resize(initialSize);
  theSimVertices->resize(initialSize);
  theGenParticles->resize(initialSize);
  theChargedTracks->resize(initialSize);
  theTrackSize = initialSize;
  theVertexSize = initialSize;
  theGenSize = initialSize;
  theChargedSize = initialSize;
  /* */

  // Initialize the Particle filter
  myFilter = new KineParticleFilter(kine);

}

FBaseSimEvent::FBaseSimEvent(const edm::ParameterSet& vtx,
			     const edm::ParameterSet& kine,
			     const RandomEngine* engine) 
  :
  nSimTracks(0),
  nSimVertices(0),
  nGenParticles(0),
  nChargedParticleTracks(0), 
  initialSize(5000),
  theVertexGenerator(0), 
  random(engine)
{

  // Initialize the vertex generator
  std::string vtxType = vtx.getParameter<std::string>("type");
  if ( vtxType == "Gaussian" ) 
    theVertexGenerator = new GaussianPrimaryVertexGenerator(vtx,random);
  else if ( vtxType == "Flat" ) 
    theVertexGenerator = new FlatPrimaryVertexGenerator(vtx,random);
  else if ( vtxType == "BetaFunc" )
    theVertexGenerator = new BetaFuncPrimaryVertexGenerator(vtx,random);
  else
    theVertexGenerator = new NoPrimaryVertexGenerator();
  // Initialize the beam spot, if not read from the DataBase
  theBeamSpot = math::XYZPoint(0.0,0.0,0.0);

  // Initialize the vectors of particles and vertices
  theGenParticles = new std::vector<HepMC::GenParticle*>(); 
  theSimTracks = new std::vector<FSimTrack>;
  theSimVertices = new std::vector<FSimVertex>;
  theChargedTracks = new std::vector<unsigned>();

  // Reserve some size to avoid mutiple copies
  /* */
  theSimTracks->resize(initialSize);
  theSimVertices->resize(initialSize);
  theGenParticles->resize(initialSize);
  theChargedTracks->resize(initialSize);
  theTrackSize = initialSize;
  theVertexSize = initialSize;
  theGenSize = initialSize;
  theChargedSize = initialSize;
  /* */

  // Initialize the Particle filter
  myFilter = new KineParticleFilter(kine);

  // Get the Famos Histos pointer
  //  myHistos = Histos::instance();

  // Initialize a few histograms
  /*
  myHistos->book("hvtx",100,-0.1,0.1);
  myHistos->book("hvty",100,-0.1,0.1);
  myHistos->book("hvtz",100,-500.,500.);
  */
}
 
FBaseSimEvent::~FBaseSimEvent(){

  // Clear the vectors
  theGenParticles->clear();
  theSimTracks->clear();
  theSimVertices->clear();
  theChargedTracks->clear();

  // Delete 
  delete theGenParticles;
  delete theSimTracks;
  delete theSimVertices;
  delete theChargedTracks;
  delete myFilter;

  //Write the histograms
  //  myHistos->put("histos.root");
  //  delete myHistos;
}

void 
FBaseSimEvent::initializePdt(const HepPDT::ParticleDataTable* aPdt) { 

  pdt = aPdt; 

}

/*
const HepPDT::ParticleDataTable*
FBaseSimEvent::theTable() const {
  return pdt;
}
*/

void
FBaseSimEvent::fill(const HepMC::GenEvent& myGenEvent) {
  
  // Clear old vectors
  clear();

  // Add the particles in the FSimEvent
  addParticles(myGenEvent);

  /*
  std::cout << "The MC truth! " << std::endl;
  printMCTruth(myGenEvent);

  std::cout << std::endl  << "The FAMOS event! " << std::endl;
  print();
  */

}

void
FBaseSimEvent::fill(const reco::GenParticleCollection& myGenParticles) {
  
  // Clear old vectors
  clear();

  // Add the particles in the FSimEvent
  addParticles(myGenParticles);

}

void
FBaseSimEvent::fill(const std::vector<SimTrack>& simTracks, 
		    const std::vector<SimVertex>& simVertices) {

  // Watch out there ! A SimVertex is in mm (stupid), 
  //            while a FSimVertex is in cm (clever).
  
  clear();

  unsigned nVtx = simVertices.size();
  unsigned nTks = simTracks.size();

  // Empty event, do nothin'
  if ( nVtx == 0 ) return;

  // Two arrays for internal use.
  std::vector<int> myVertices(nVtx,-1);
  std::vector<int> myTracks(nTks,-1);

  // create a map associating geant particle id and position in the 
  // event SimTrack vector
  
  std::map<unsigned, unsigned> geantToIndex;
  for( unsigned it=0; it<simTracks.size(); ++it ) {
    geantToIndex[ simTracks[it].trackId() ] = it;
  }  

  // Create also a map associating a SimTrack with its endVertex
  /*
  std::map<unsigned, unsigned> endVertex;
  for ( unsigned iv=0; iv<simVertices.size(); ++iv ) { 
    endVertex[ simVertices[iv].parentIndex() ] = iv;
  }
  */

  // Set the main vertex for the kine particle filter
  // SimVertices were in mm until 110_pre2
  //  HepLorentzVector primaryVertex = simVertices[0].position()/10.;
  // SImVertices are now in cm
  // Also : position is copied until SimVertex switches to Mathcore.
  //  XYZTLorentzVector primaryVertex = simVertices[0].position();
  // The next 5 lines to be then replaced by the previous line
  XYZTLorentzVector primaryVertex(simVertices[0].position().x(),
				  simVertices[0].position().y(),
				  simVertices[0].position().z(),
				  simVertices[0].position().t());
  //
  myFilter->setMainVertex(primaryVertex);
  // Add the main vertex to the list.
  addSimVertex(myFilter->vertex());
  myVertices[0] = 0;

  for( unsigned trackId=0; trackId<nTks; ++trackId ) {

    // The track
    const SimTrack& track = simTracks[trackId];
    //    std::cout << std::endl << "SimTrack " << trackId << " " << track << std::endl;

    // The origin vertex
    int vertexId = track.vertIndex();
    const SimVertex& vertex = simVertices[vertexId];
    //std::cout << "Origin vertex " << vertexId << " " << vertex << std::endl;

    // The mother track 
    int motherId = -1;
    if( !vertex.noParent() ) { // there is a parent to this vertex
      // geant id of the mother
      unsigned motherGeantId =   vertex.parentIndex(); 
      std::map<unsigned, unsigned >::iterator association  
	= geantToIndex.find( motherGeantId );
      if(association != geantToIndex.end() )
	motherId = association->second;
    }
    int originId = motherId == - 1 ? -1 : myTracks[motherId];
    //std::cout << "Origin id " << originId << std::endl;

    /*
    if ( endVertex.find(trackId) != endVertex.end() ) 
      std::cout << "End vertex id = " << endVertex[trackId] << std::endl;
    else
      std::cout << "No endVertex !!! " << std::endl;
    std::cout << "Tracker surface position " << track.trackerSurfacePosition() << std::endl;
    */

    // Add the vertex (if it does not already exist!)
    XYZTLorentzVector position(vertex.position().px(),vertex.position().py(),
			       vertex.position().pz(),vertex.position().e());
    if ( myVertices[vertexId] == -1 ) 
      // Momentum and position are copied until SimTrack and SimVertex
      // switch to Mathcore.
      //      myVertices[vertexId] = addSimVertex(vertex.position(),originId); 
      // The next line to be then replaced by the previous line
      myVertices[vertexId] = addSimVertex(position,originId); 

    // Add the track (with protection for brem'ing electrons)
    int motherType = motherId == -1 ? 0 : simTracks[motherId].type();

    if ( abs(motherType) != 11 || motherType != track.type() ) {
      // Momentum and position are copied until SimTrack and SimVertex
      // switch to Mathcore.
      //      RawParticle part(track.momentum(), vertex.position());
      // The next 3 lines to be then replaced by the previous line
      XYZTLorentzVector momentum(track.momentum().px(),track.momentum().py(),
				 track.momentum().pz(),track.momentum().e());
      RawParticle part(momentum,position);
      //
      part.setID(track.type()); 
      //std::cout << "Ctau  = " << part.PDGcTau() << std::endl;
      // Don't save tracks that have decayed immediately but for which no daughters
      // were saved (probably due to cuts on E, pT and eta)
      //  if ( part.PDGcTau() > 0.1 || endVertex.find(trackId) != endVertex.end() ) 
	myTracks[trackId] = addSimTrack(&part,myVertices[vertexId],track.genpartIndex());
	if ( track.trackerSurfacePosition().perp2() > 22500. || fabs(track.trackerSurfacePosition().z()) > 400. ) 
	  (*theSimTracks)[ myTracks[trackId] ].setTkPosition(track.trackerSurfacePosition()/10.);
	else
	  (*theSimTracks)[ myTracks[trackId] ].setTkPosition(track.trackerSurfacePosition());
	(*theSimTracks)[ myTracks[trackId] ].setTkMomentum(track.trackerSurfaceMomentum());
    } else {
      myTracks[trackId] = myTracks[motherId];
      if ( track.trackerSurfacePosition().perp2() > 22500. || fabs(track.trackerSurfacePosition().z()) > 400. ) 
	(*theSimTracks)[ myTracks[trackId] ].setTkPosition(track.trackerSurfacePosition()/10.);
      else
	(*theSimTracks)[ myTracks[trackId] ].setTkPosition(track.trackerSurfacePosition());
      (*theSimTracks)[ myTracks[trackId] ].setTkMomentum(track.trackerSurfaceMomentum());
    }
    
  }

  // Now loop over the remaining end vertices !
  for( unsigned vertexId=0; vertexId<nVtx; ++vertexId ) {

    // if the vertex is already saved, just ignore.
    if ( myVertices[vertexId] != -1 ) continue;

    // The yet unused vertex
    const SimVertex& vertex = simVertices[vertexId];

    // The mother track 
    int motherId = -1;
    if( !vertex.noParent() ) { // there is a parent to this vertex

      // geant id of the mother
      unsigned motherGeantId =   vertex.parentIndex(); 
      std::map<unsigned, unsigned >::iterator association  
	= geantToIndex.find( motherGeantId );
      if(association != geantToIndex.end() )
	motherId = association->second;
    }
    int originId = motherId == - 1 ? -1 : myTracks[motherId];

    // Add the vertex
    // Momentum and position are copied until SimTrack and SimVertex
    // switch to Mathcore.
    //    myVertices[vertexId] = addSimVertex(vertex.position(),originId);
    // The next 3 lines to be then replaced by the previous line
    XYZTLorentzVector position(vertex.position().px(),vertex.position().py(),
			       vertex.position().pz(),vertex.position().e());
    myVertices[vertexId] = addSimVertex(position,originId); 
  }

  // Finally, propagate all particles to the calorimeters
  BaseParticlePropagator myPart;
  XYZTLorentzVector mom;
  XYZTLorentzVector pos;

  // Loop over the tracks
  for( int fsimi=0; fsimi < (int)nTracks() ; ++fsimi) {

    
    FSimTrack& myTrack = track(fsimi);
    double trackerSurfaceTime = myTrack.vertex().position().t() 
                              + myTrack.momentum().e()/myTrack.momentum().pz()
                              * ( myTrack.trackerSurfacePosition().z()
				- myTrack.vertex().position().z() );
    pos = XYZTLorentzVector(myTrack.trackerSurfacePosition().x(),
			    myTrack.trackerSurfacePosition().y(),
			    myTrack.trackerSurfacePosition().z(),
			            trackerSurfaceTime);
    mom = XYZTLorentzVector(myTrack.trackerSurfaceMomentum().x(),
			    myTrack.trackerSurfaceMomentum().y(),
			    myTrack.trackerSurfaceMomentum().z(),
			    myTrack.trackerSurfaceMomentum().t());

    if ( mom.T() >  0. ) {  
      // The particle to be propagated
      myPart = BaseParticlePropagator(RawParticle(mom,pos),0.,0.,4.);
      myPart.setCharge(myTrack.charge());
      
      // Propagate to Preshower layer 1
      myPart.propagateToPreshowerLayer1(false);
      if ( myTrack.notYetToEndVertex(myPart.vertex()) && myPart.getSuccess()>0 )
	myTrack.setLayer1(myPart,myPart.getSuccess());
      
      // Propagate to Preshower Layer 2 
      myPart.propagateToPreshowerLayer2(false);
      if ( myTrack.notYetToEndVertex(myPart.vertex()) && myPart.getSuccess()>0 )
	myTrack.setLayer2(myPart,myPart.getSuccess());
      
      // Propagate to Ecal Endcap
      myPart.propagateToEcalEntrance(false);
      if ( myTrack.notYetToEndVertex(myPart.vertex()) )
	myTrack.setEcal(myPart,myPart.getSuccess());
      
      // Propagate to HCAL entrance
      myPart.propagateToHcalEntrance(false);
      if ( myTrack.notYetToEndVertex(myPart.vertex()) )
	myTrack.setHcal(myPart,myPart.getSuccess());
      
      // Propagate to VFCAL entrance
      myPart.propagateToVFcalEntrance(false);
      if ( myTrack.notYetToEndVertex(myPart.vertex()) )
	myTrack.setVFcal(myPart,myPart.getSuccess());
    }
 
  }

}


void
FBaseSimEvent::addParticles(const HepMC::GenEvent& myGenEvent) {

  /// Some internal array to work with.
  int genEventSize = myGenEvent.particles_size();
  std::vector<int> myGenVertices(genEventSize, static_cast<int>(0));

  // If no particles, no work to be done !
  if ( myGenEvent.particles_empty() ) return;

  // Are there particles in the FSimEvent already ? 
  int offset = nGenParts();

  // Primary vertex (already smeared by the SmearedVtx module)
  HepMC::GenVertex* primaryVertex = *(myGenEvent.vertices_begin());
  XYZTLorentzVector primaryVertexPosition(primaryVertex->position().x()/10.,
					  primaryVertex->position().y()/10.,
					  primaryVertex->position().z()/10.,
					  primaryVertex->position().t()/10.);

  // Smear the main vertex if needed
  // Now takes the origin from the database
  XYZTLorentzVector smearedVertex; 
  if ( primaryVertex->point3d().mag() < 1E-10 ) {
    theVertexGenerator->generate();
    smearedVertex = XYZTLorentzVector(
      theVertexGenerator->X()-theVertexGenerator->beamSpot().X()+theBeamSpot.X(),
      theVertexGenerator->Y()-theVertexGenerator->beamSpot().Y()+theBeamSpot.Y(),
      theVertexGenerator->Z()-theVertexGenerator->beamSpot().Z()+theBeamSpot.Z(),
      0.);
  }

  // Set the main vertex
  myFilter->setMainVertex(primaryVertexPosition+smearedVertex);

  // This is the smeared main vertex
  int mainVertex = addSimVertex(myFilter->vertex());

  HepMC::GenEvent::particle_const_iterator piter;
  HepMC::GenEvent::particle_const_iterator pbegin = myGenEvent.particles_begin();
  HepMC::GenEvent::particle_const_iterator pend = myGenEvent.particles_end();

  int initialBarcode = 0; 
  if ( pbegin != pend ) initialBarcode = (*pbegin)->barcode();
  // Loop on the particles of the generated event
  for ( piter = pbegin; piter != pend; ++piter ) {

    // This is the generated particle pointer - for the signal event only
    HepMC::GenParticle* p = *piter;

    if  ( !offset ) {
      (*theGenParticles)[nGenParticles++] = p;
      if ( nGenParticles/theGenSize*theGenSize == nGenParticles ) { 
	theGenSize *= 2;
	theGenParticles->resize(theGenSize);
      }

    }

    // Keep only: 
    // 1) Stable particles (watch out! New status code = 1001!)
    bool testStable = p->status()%1000==1;

    // 2) or particles with stable daughters (watch out! New status code = 1001!)
    bool testDaugh = false;
    if ( !testStable && 
	 p->status() == 2 &&
	 p->end_vertex() && 
	 p->end_vertex()->particles_out_size() ) { 
      HepMC::GenVertex::particles_out_const_iterator firstDaughterIt = 
	p->end_vertex()->particles_out_const_begin();
      HepMC::GenVertex::particles_out_const_iterator lastDaughterIt = 
	p->end_vertex()->particles_out_const_end();
      for ( ; firstDaughterIt != lastDaughterIt ; ++firstDaughterIt ) {
	HepMC::GenParticle* daugh = *firstDaughterIt;
	if ( daugh->status()%1000==1 ) {
	  testDaugh=true;
	  break;
	}
      }
    }

    // 3) or particles that fly more than one micron.
    double dist = 0.;
    if ( !testStable && !testDaugh && p->production_vertex() ) {
      XYZTLorentzVector 
	productionVertexPosition(p->production_vertex()->position().x()/10.,
				 p->production_vertex()->position().y()/10.,
				 p->production_vertex()->position().z()/10.,
				 p->production_vertex()->position().t()/10.);
      dist = (primaryVertexPosition-productionVertexPosition).Vect().Mag2();
    }
    bool testDecay = ( dist > 1e-8 ) ? true : false; 

    // Save the corresponding particle and vertices
    if ( testStable || testDaugh || testDecay ) {
      
      /*
      const HepMC::GenParticle* mother = p->production_vertex() ?
	*(p->production_vertex()->particles_in_const_begin()) : 0;
      */

      int motherBarcode = p->production_vertex() && 
	p->production_vertex()->particles_in_const_begin() !=
	p->production_vertex()->particles_in_const_end() ?
	(*(p->production_vertex()->particles_in_const_begin()))->barcode() : 0;

      int originVertex = 
	motherBarcode && myGenVertices[motherBarcode-initialBarcode] ?
	myGenVertices[motherBarcode-initialBarcode] : mainVertex;

      XYZTLorentzVector momentum(p->momentum().px(),
				 p->momentum().py(),
				 p->momentum().pz(),
				 p->momentum().e());
      RawParticle part(momentum, vertex(originVertex).position());
      part.setID(p->pdg_id());

      // Add the particle to the event and to the various lists
      
      int theTrack = testStable && p->end_vertex() ? 
	// The particle is scheduled to decay
	addSimTrack(&part,originVertex, nGenParts()-offset,p->end_vertex()) :
        // The particle is not scheduled to decay 
	addSimTrack(&part,originVertex, nGenParts()-offset);

      if ( 
	  // This one deals with particles with no end vertex
	  !p->end_vertex() ||
	  // This one deals with particles that have a pre-defined
	  // decay proper time, but have not decayed yet
	   ( testStable && p->end_vertex() ) 
	  // In both case, just don't add a end vertex in the FSimEvent 
	  ) continue; 
      
      // Add the vertex to the event and to the various lists
      XYZTLorentzVector decayVertex = 
	XYZTLorentzVector(p->end_vertex()->position().x()/10.,
			  p->end_vertex()->position().y()/10.,
			  p->end_vertex()->position().z()/10.,
			  p->end_vertex()->position().t()/10.) +
	smearedVertex;
      //	vertex(mainVertex).position();
      int theVertex = addSimVertex(decayVertex,theTrack);

      if ( theVertex != -1 ) myGenVertices[p->barcode()-initialBarcode] = theVertex;

      // There we are !
    }
  }

}

void
FBaseSimEvent::addParticles(const reco::GenParticleCollection& myGenParticles) {

  // If no particles, no work to be done !
  unsigned int nParticles = myGenParticles.size();
  nGenParticles = nParticles;

  if ( !nParticles ) return;

  /// Some internal array to work with.
  std::map<const reco::Candidate*,int> myGenVertices;

  // Are there particles in the FSimEvent already ? 
  int offset = nTracks();

  // Skip the incoming protons
  nGenParticles = 0;
  unsigned int ip = 0;
  if ( nParticles > 1 && 
       myGenParticles[0].pdgId() == 2212 &&
       myGenParticles[1].pdgId() == 2212 ) { 
    ip = 2;
    nGenParticles = 2;
  }

  // Primary vertex (already smeared by the SmearedVtx module)
  XYZTLorentzVector primaryVertex (myGenParticles[ip].vx(),
				   myGenParticles[ip].vy(),
				   myGenParticles[ip].vz(),
				   0.);

  // Smear the main vertex if needed
  XYZTLorentzVector smearedVertex;
  if ( primaryVertex.mag() < 1E-10 ) {
    theVertexGenerator->generate();
    smearedVertex = XYZTLorentzVector(
      theVertexGenerator->X()-theVertexGenerator->beamSpot().X()+theBeamSpot.X(),
      theVertexGenerator->Y()-theVertexGenerator->beamSpot().Y()+theBeamSpot.Y(),
      theVertexGenerator->Z()-theVertexGenerator->beamSpot().Z()+theBeamSpot.Z(),
      0.);
  }

  // Set the main vertex
  myFilter->setMainVertex(primaryVertex+smearedVertex);

  // This is the smeared main vertex
  int mainVertex = addSimVertex(myFilter->vertex());

  // Loop on the particles of the generated event
  for ( ; ip<nParticles; ++ip ) { 
    
    // nGenParticles = ip;
    
    nGenParticles++;
    const reco::GenParticle& p = myGenParticles[ip];

    // Keep only: 
    // 1) Stable particles
    bool testStable = p.status()%1000==1;

    // 2) or particles with stable daughters
    bool testDaugh = false;
    unsigned int nDaughters = p.numberOfDaughters();
    if ( !testStable  && 
	 //	 p.status() == 2 && 
	 nDaughters ) {  
      for ( unsigned iDaughter=0; iDaughter<nDaughters; ++iDaughter ) {
	const reco::Candidate* daughter = p.daughter(iDaughter);
	if ( daughter->status()%1000==1 ) {
	  testDaugh=true;
	  break;
	}
      }
    }

    // 3) or particles that fly more than one micron.
    double dist = 0.;
    if ( !testStable && !testDaugh ) {
      XYZTLorentzVector productionVertex(p.vx(),p.vy(),p.vz(),0.);
      dist = (primaryVertex-productionVertex).Vect().Mag2();
    }
    bool testDecay = ( dist > 1e-8 ) ? true : false; 

    // Save the corresponding particle and vertices
    if ( testStable || testDaugh || testDecay ) {
      
      const reco::Candidate* mother = p.numberOfMothers() ? p.mother(0) : 0;

      int originVertex = 
	mother &&  
	myGenVertices.find(mother) != myGenVertices.end() ? 
      	myGenVertices[mother] : mainVertex;
      
      XYZTLorentzVector momentum(p.px(),p.py(),p.pz(),p.energy());
      RawParticle part(momentum, vertex(originVertex).position());
      part.setID(p.pdgId());

      // Add the particle to the event and to the various lists
      int theTrack = addSimTrack(&part,originVertex, nGenParts()-offset);

      // It there an end vertex ?
      if ( !nDaughters ) continue; 
      const reco::Candidate* daughter = p.daughter(0);

      // Add the vertex to the event and to the various lists
      XYZTLorentzVector decayVertex = 
	XYZTLorentzVector(daughter->vx(), daughter->vy(),
			  daughter->vz(), 0.) + smearedVertex;
      int theVertex = addSimVertex(decayVertex,theTrack);

      if ( theVertex != -1 ) myGenVertices[&p] = theVertex;

      // There we are !
    }
  }

  // There is no GenParticle's in that case...
  // nGenParticles=0;

}

int 
FBaseSimEvent::addSimTrack(const RawParticle* p, int iv, int ig, 
			   const HepMC::GenVertex* ev) { 
  
  // Check that the particle is in the Famos "acceptance"
  // Keep all primaries of pile-up events, though
  if ( !myFilter->accept(p) && ig >= -1 ) return -1;

  // The new track index
  int trackId = nSimTracks++;
  if ( nSimTracks/theTrackSize*theTrackSize == nSimTracks ) {
    theTrackSize *= 2;
    theSimTracks->resize(theTrackSize);
  }

  // Attach the particle to the origin vertex, and to the mother
  vertex(iv).addDaughter(trackId);
  if ( !vertex(iv).noParent() ) {
    track(vertex(iv).parent().id()).addDaughter(trackId);

    if ( ig == -1 ) {
      int motherId = track(vertex(iv).parent().id()).genpartIndex();
      if ( motherId < -1 ) ig = motherId;
    }
  }
    
  // Some transient information for FAMOS internal use
  (*theSimTracks)[trackId] = ev ? 
    // A proper decay time is scheduled
    FSimTrack(p,iv,ig,trackId,this,
	      ev->position().t()/10.
	      * p->PDGmass()
	      / std::sqrt(p->momentum().Vect().Mag2())) : 
    // No proper decay time is scheduled
    FSimTrack(p,iv,ig,trackId,this);

  return trackId;

}

int
FBaseSimEvent::addSimVertex(const XYZTLorentzVector& v,int im) {
  
  // Check that the vertex is in the Famos "acceptance"
  if ( !myFilter->accept(v) ) return -1;

  // The number of vertices
  int vertexId = nSimVertices++;
  if ( nSimVertices/theVertexSize*theVertexSize == nSimVertices ) {
    theVertexSize *= 2;
    theSimVertices->resize(theVertexSize);
  }

  // Attach the end vertex to the particle (if accepted)
  if ( im !=-1 ) track(im).setEndVertex(vertexId);

  // Some transient information for FAMOS internal use
  (*theSimVertices)[vertexId] = FSimVertex(v,im,vertexId,this);

  return vertexId;

}

void
FBaseSimEvent::printMCTruth(const HepMC::GenEvent& myGenEvent) {
  
  std::cout << "Id  Gen Name       eta    phi     pT     E    Vtx1   " 
	    << " x      y      z   " 
	    << "Moth  Vtx2  eta   phi     R      Z   Da1  Da2 Ecal?" << std::endl;

  for ( HepMC::GenEvent::particle_const_iterator 
	  piter  = myGenEvent.particles_begin();
	  piter != myGenEvent.particles_end(); 
	++piter ) {
    
    HepMC::GenParticle* p = *piter;
     /* */
    int partId = p->pdg_id();
    std::string name;

    if ( pdt->particle(ParticleID(partId)) !=0 ) {
      name = (pdt->particle(ParticleID(partId)))->name();
    } else {
      name = "none";
    }
  
    XYZTLorentzVector momentum1(p->momentum().px(),
				p->momentum().py(),
				p->momentum().pz(),
				p->momentum().e());

    int vertexId1 = 0;

    if ( !p->production_vertex() ) continue;

    XYZVector vertex1 (p->production_vertex()->position().x()/10.,
		       p->production_vertex()->position().y()/10.,
		       p->production_vertex()->position().z()/10.);
    vertexId1 = p->production_vertex()->barcode();
    
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.setf(std::ios::right, std::ios::adjustfield);
    
    std::cout << std::setw(4) << p->barcode() << " " 
	 << name;
    
    for(unsigned int k=0;k<11-name.length() && k<12; k++) std::cout << " ";  
    
    double eta = momentum1.eta();
    if ( eta > +10. ) eta = +10.;
    if ( eta < -10. ) eta = -10.;
    std::cout << std::setw(6) << std::setprecision(2) << eta << " " 
	      << std::setw(6) << std::setprecision(2) << momentum1.phi() << " " 
	      << std::setw(7) << std::setprecision(2) << momentum1.pt() << " " 
	      << std::setw(7) << std::setprecision(2) << momentum1.e() << " " 
	      << std::setw(4) << vertexId1 << " " 
	      << std::setw(6) << std::setprecision(1) << vertex1.x() << " " 
	      << std::setw(6) << std::setprecision(1) << vertex1.y() << " " 
	      << std::setw(6) << std::setprecision(1) << vertex1.z() << " ";

    const HepMC::GenParticle* mother = 
      *(p->production_vertex()->particles_in_const_begin());

    if ( mother )
      std::cout << std::setw(4) << mother->barcode() << " ";
    else 
      std::cout << "     " ;
    
    if ( p->end_vertex() ) {  
      XYZTLorentzVector vertex2(p->end_vertex()->position().x()/10.,
				p->end_vertex()->position().y()/10.,
				p->end_vertex()->position().z()/10.,
				p->end_vertex()->position().t()/10.);
      int vertexId2 = p->end_vertex()->barcode();

      std::vector<const HepMC::GenParticle*> children;
      HepMC::GenVertex::particles_out_const_iterator firstDaughterIt = 
        p->end_vertex()->particles_out_const_begin();
      HepMC::GenVertex::particles_out_const_iterator lastDaughterIt = 
        p->end_vertex()->particles_out_const_end();
      for ( ; firstDaughterIt != lastDaughterIt ; ++firstDaughterIt ) {
	children.push_back(*firstDaughterIt);
      }      

      std::cout << std::setw(4) << vertexId2 << " "
		<< std::setw(6) << std::setprecision(2) << vertex2.eta() << " " 
		<< std::setw(6) << std::setprecision(2) << vertex2.phi() << " " 
		<< std::setw(5) << std::setprecision(1) << vertex2.pt() << " " 
		<< std::setw(6) << std::setprecision(1) << vertex2.z() << " ";
      for ( unsigned id=0; id<children.size(); ++id )
	std::cout << std::setw(4) << children[id]->barcode() << " ";
    }
    std::cout << std::endl;

  }

}

void
FBaseSimEvent::print() const {

  std::cout << "  Id  Gen Name       eta    phi     pT     E    Vtx1   " 
	    << " x      y      z   " 
	    << "Moth  Vtx2  eta   phi     R      Z   Daughters Ecal?" << std::endl;

  for( int i=0; i<(int)nTracks(); i++ ) 
    std::cout << track(i) << std::endl;
}

void 
FBaseSimEvent::clear() {

  nSimTracks = 0;
  nSimVertices = 0;
  nGenParticles = 0;
  nChargedParticleTracks = 0;

}

void 
FBaseSimEvent::addChargedTrack(int id) { 
  (*theChargedTracks)[nChargedParticleTracks++] = id;
  if ( nChargedParticleTracks/theChargedSize*theChargedSize 
       == nChargedParticleTracks ) {
    theChargedSize *= 2;
    theChargedTracks->resize(theChargedSize);
  }
}

int
FBaseSimEvent::chargedTrack(int id) const {
  if (id>=0 && id<(int)nChargedParticleTracks) 
    return (*theChargedTracks)[id]; 
  else 
    return -1;
}

/* 
const SimTrack & 
FBaseSimEvent::embdTrack(int i) const {  
  return (*theSimTracks)[i].simTrack();
}

const SimVertex & 
FBaseSimEvent::embdVertex(int i) const { 
  return (*theSimVertices)[i].simVertex();
}
*/

const HepMC::GenParticle* 
FBaseSimEvent::embdGenpart(int i) const {
  return (*theGenParticles)[i]; 
}

/*
FSimTrack&  
FBaseSimEvent::track(int id) const { 
  return (*theSimTracks)[id];
}


FSimVertex&  
FBaseSimEvent::vertex(int id) const { 
  return (*theSimVertices)[id];
}
*/
