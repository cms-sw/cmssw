//FAMOS Headers
#include "FastSimulation/Event/interface/FBaseSimEvent.h"

// system include
#include <iostream>
#include <iomanip>

//CLHEP Headers
#include "CLHEP/Random/RandGauss.h"
#include "SimGeneral/HepPDT/interface/HepPDTable.h"
#include "SimGeneral/HepPDT/interface/HepParticleData.h"

using namespace std;
using namespace edm;
using namespace HepMC;

FBaseSimEvent::FBaseSimEvent() {


  // Initialize the vectors of particles and vertices
  mySimTracks = new EmbdSimTrackContainer(); 
  mySimVertices = new EmbdSimVertexContainer(); 
  myGenParticles = new vector<GenParticle*>(); 

  // The vertex smearing (if needed)
  sigmaVerteX = 0.015;
  sigmaVerteY = 0.015;
  sigmaVerteZ = 53.0;

  // The particle Data Table
  tab = & HepPDT::theTable();
  
}
 
FBaseSimEvent::~FBaseSimEvent(){
}

void
FBaseSimEvent::fill(const HepMC::GenEvent& myGenEvent) {
  
  // Clean memory
  cout << "Nombre d'objets en memoire avant clean : " 
       << counter() <<endl;
  delete_all_vertices();
  cout << "Nombre d'objets en memoire apres clean : " 
       << counter() <<endl;

  // Clear the vectors
  mySimTracks->clear();
  mySimVertices->clear();
  myGenParticles->clear();

  // Set the event Id
  //  Id = theId;
  set_signal_process_id(myGenEvent.signal_process_id());
  set_event_number(myGenEvent.event_number());

  printMCTruth(myGenEvent);

  // Fill the event with the stable particles of the GenEvent 
  // (and their mother), in the order defined by the original 
  // particle barcodes

  // Add the particles in the FSimEvent
  addParticles(myGenEvent);

  // Check 
  for ( unsigned i=0; i<nTracks(); ++i ) cout << embdTrack(i) << endl;

  for ( unsigned i=0; i<nVertices(); ++i ) cout << embdVertex(i) << endl;
}

void
FBaseSimEvent::addParticles(const HepMC::GenEvent& myGenEvent) {

  // If no particles, no work to be done !
  if ( myGenEvent.particles_empty() ) return;

  // Are there particles in the FSimEvent already ? 
  int offset = particles_size();

  // Primary vertex
  GenVertex* primaryVertex = *(myGenEvent.vertices_begin());

  // Set the main vertex with smearing
  HepLorentzVector smearedVertex = 
     primaryVertex->point3d().mag() > 1E-10 ?
     HepLorentzVector(0.,0.,0.,0.) :
     HepLorentzVector(sigmaVerteX*RandGauss::shoot(),
		      sigmaVerteY*RandGauss::shoot(),
		      sigmaVerteZ*RandGauss::shoot(),
		      0.);
  myFilter.setMainVertex(primaryVertex->position()+smearedVertex);

  // This is the smeared main vertex
  GenVertex* mainVertex = new GenVertex(myFilter.vertex());
  addSimVertex(mainVertex);

  // Loop on the particles of the generated event
  for ( HepMC::GenEvent::particle_const_iterator 
	  piter  = myGenEvent.particles_begin();
	  piter != myGenEvent.particles_end(); 
	++piter ) {

    // This is the generated particle pointer
    GenParticle* p = *piter;
    myGenParticles->push_back(p);

    // Keep only: 
    // 1) Stable particles
    bool testStable = p->status()==1;

    // 2) or particles with stable daughters
    bool testDaugh = false;
    for ( unsigned i=0; i<p->listChildren().size(); ++i ) { 
      GenParticle* daugh = p->listChildren()[i];
      if ( daugh->status()==1 ) {
	testDaugh=true;
	break;
      }
    }

    // 3) or particles that fly more than one micron.
    double dist = p->production_vertex() ? 
       ( primaryVertex->position()
       - p->production_vertex()->position() ).vect().mag() : 0.; 
    bool testDecay = ( dist > 0.001 ) ? true : false; 

    // Save the corresponding particle and vertices
    if ( testStable || testDaugh || testDecay ) {

      // The particle is the copy of the original
      GenParticle* part = 
	new GenParticle(p->momentum(),
			p->pdg_id(),
			p->status(),
			p->flow(),
			p->polarization());

      // The origin vertex is either the primary, 
      // or the end vertex of the mother, if saved
      GenVertex* originVertex = 
	p->mother() &&  
	myGenVertices.find(p->mother()) != myGenVertices.end() ? 
	originVertex = myGenVertices[p->mother()] : mainVertex;

      // Add the particle to the event and to the various lists
      int theTrack = addSimTrack(part,originVertex,nGenParts()-1);

      // It there an end vertex ?
      if ( !p->end_vertex() ) continue; 

      // If yes, create it
      GenVertex* decayVertex = 
	new GenVertex(p->end_vertex()->position()
		      +mainVertex->position());

      // Add the vertex to the event and to the various lists
      int theVertex = addSimVertex(decayVertex, part, theTrack);

      // And record it for later use 
      if ( theVertex != -1 ) myGenVertices[p] = decayVertex;

      // There we are !


    }
  }

  printMCTruth(*this);

}

int 
FBaseSimEvent::addSimTrack(GenParticle* part, 
			   GenVertex* originVertex,
			   int ig) {
  
  // Check that the particle is in the Famos "acceptance"
  if ( !myFilter.accept(RawParticle(part)) ) return -1;

  // An increasing barcode, corresponding to the list index
  part->suggest_barcode(nTracks()+1);
  
  // Attach the particle to the origin vertex
  originVertex->add_particle_out(part);
  
  // Attach the vertex to the event (inoccuous if the vertex exists)
  // add_vertex(originVertex);
  
  // Some persistent information for the users
  mySimTracks->push_back(
     EmbdSimTrack(part->pdg_id(), part->momentum(), 
		  -originVertex->barcode()-1, ig));

  return nTracks()-1;

}

int
FBaseSimEvent::addSimVertex(GenVertex* decayVertex, 
			    GenParticle* motherParticle,
			    int it) {
  
  // Check that the vertex is in the Famos "acceptance"
  if ( !myFilter.accept(RawParticle(HepLorentzVector(),
				    decayVertex->position()))) return -1;

  // Attach the vertex to the event (inoccuous if the vertex exists)
  add_vertex(decayVertex);

  // Attach the end vertex to the particle (if accepted)
  if ( it!=-1 ) decayVertex->add_particle_in(motherParticle);

  // Some persistent information for the users
  mySimVertices->push_back(
    EmbdSimVertex(decayVertex->position().vect(),
		  decayVertex->position().e(), 
		  it));

  return nVertices()-1;

}

void
FBaseSimEvent::printMCTruth(const HepMC::GenEvent& myGenEvent) {
  
  cout << "Id  Gen Name       eta    phi     pT     E    Vtx1   " 
       << " x      y      z   " 
       << "Moth  Vtx2  eta   phi     R      Z   Da1  Da2 Ecal?" << endl;

  for ( HepMC::GenEvent::particle_const_iterator 
	  piter  = myGenEvent.particles_begin();
	  piter != myGenEvent.particles_end(); 
	++piter ) {
  //  for ( int i=1; i !=myGenEvent.particles_size(); ++i ) { 
    
    HepMC::GenParticle* p = *piter;
     /* */
     //     const std::string name = (*p)->particledata().name();
    int partId = p->pdg_id();
    std::string name;
    if (tab->getParticleData(partId) != 0) {
      name = (tab->getParticleData(partId))->name();
    } else {
      name = "none";
    }
       
    HepLorentzVector momentum1 = p->momentum();
    Hep3Vector vertex1 = p->creationVertex().vect();
    int vertexId1 = 0;
    if ( !p->production_vertex() ) continue;
    vertexId1 = p->production_vertex()->barcode();
    
    cout.setf(ios::fixed, ios::floatfield);
    cout.setf(ios::right, ios::adjustfield);
    
    cout << setw(4) << p->barcode() << " " 
	 << name;
    
    for(unsigned int k=0;k<9-name.length() && k<10; k++) cout << " ";  
    
    double eta = momentum1.eta();
    if ( eta > +10. ) eta = +10.;
    if ( eta < -10. ) eta = -10.;
    cout << setw(6) << setprecision(2) << eta << " " 
	 << setw(6) << setprecision(2) << momentum1.phi() << " " 
	 << setw(7) << setprecision(2) << momentum1.perp() << " " 
	 << setw(7) << setprecision(2) << momentum1.e() << " " 
	 << setw(4) << vertexId1 << " " 
	 << setw(6) << setprecision(1) << vertex1.x() << " " 
	 << setw(6) << setprecision(1) << vertex1.y() << " " 
	 << setw(6) << setprecision(1) << vertex1.z() << " ";
    if ( p->mother() )
      cout << setw(4) << p->mother()->barcode() << " ";
    else 
      cout << "     " ;
    
    if ( p->end_vertex() ) {  
      HepLorentzVector vertex2 = p->decayVertex();
      int vertexId2 = p->end_vertex()->barcode();
      
      cout << setw(4) << vertexId2 << " "
	   << setw(6) << setprecision(2) << vertex2.eta() << " " 
	   << setw(6) << setprecision(2) << vertex2.phi() << " " 
	   << setw(5) << setprecision(1) << vertex2.perp() << " " 
	   << setw(6) << setprecision(1) << vertex2.z() << " "
	   << setw(4) << p->beginDaughters()->barcode() << " "
	   << setw(4) << p->beginDaughters()->barcode() + 
	                 p->listChildren().size()-1 << " " ;
    }
    cout << endl;

  }

}

EmbdSimTrackContainer*
FBaseSimEvent::tracks() const { return mySimTracks; }

EmbdSimVertexContainer*
FBaseSimEvent::vertices() const { return mySimVertices; }

vector<GenParticle*>* 
FBaseSimEvent::genparts() const { return myGenParticles; }

unsigned int 
FBaseSimEvent::nTracks() const {
  return mySimTracks->size();
}


unsigned int 
FBaseSimEvent::nVertices() const { 
  return mySimVertices->size();
}

unsigned int 
FBaseSimEvent::nGenParts() const {
  return myGenParticles->size();
}

static  const EmbdSimVertex zeroVertex;
const EmbdSimVertex & 
FBaseSimEvent::embdVertex(int i) const { 
  if (i>=0 && i<=(int)mySimVertices->size()) 
    return (*mySimVertices)[i]; 
  else 
    return zeroVertex;
}

static  const EmbdSimTrack zeroTrack;
const EmbdSimTrack & 
FBaseSimEvent::embdTrack(int i) const { 
  if (i>=0 && i<=(int)mySimTracks->size()) 
    return (*mySimTracks)[i]; 
  else 
    return zeroTrack;
}

const GenParticle* 
FBaseSimEvent::embdGenpart(int i) const { 
  if (i>=0 && i<=(int)myGenParticles->size()) 
    return (*myGenParticles)[i]; 
  else 
    return 0;
}
