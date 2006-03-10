//FAMOS Headers
#include "FastSimulation/Event/interface/FBaseSimEvent.h"

// system include
#include <iostream>
#include <iomanip>

//CLHEP Headers
#include "CLHEP/Random/RandGauss.h"

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

  
  // Get the Susy particle names (ugly, but this is because the 
  // particle data table does not contain SUSY particles!)
  particleNames[1] = "d";
  particleNames[2] = "u";
  particleNames[3] = "s";
  particleNames[4] = "c";
  particleNames[5] = "b";
  particleNames[6] = "t";
  particleNames[7] = "l";
  particleNames[8] = "h";
  particleNames[-1] = "d~";
  particleNames[-2] = "u~";
  particleNames[-3] = "s~";
  particleNames[-4] = "c~";
  particleNames[-5] = "b~";
  particleNames[-6] = "t~";
  particleNames[-7] = "l~";
  particleNames[-8] = "h~";

  particleNames[11] = "e-";
  particleNames[12] = "nu_e";
  particleNames[13] = "mu-";
  particleNames[14] = "nu_mu";
  particleNames[15] = "tau-";
  particleNames[16] = "nu_tau";
  particleNames[17] = "chi-";
  particleNames[18] = "nu_chi";
  particleNames[-11] = "e+";
  particleNames[-12] = "nu_e~";
  particleNames[-13] = "mu+";
  particleNames[-14] = "nu_mu~";
  particleNames[-15] = "tau+";
  particleNames[-16] = "nu_tau~";
  particleNames[-17] = "chi+";
  particleNames[-18] = "nu_chi~";

  particleNames[21] = "g";
  particleNames[22] = "gamma";
  particleNames[23] = "Z0";
  particleNames[24] = "W+";
  particleNames[-24] = "W-";
  particleNames[25] = "H0";

  particleNames[81] = "specflav";
  particleNames[82] = "rndmflav";
  particleNames[83] = "phasespa";
  particleNames[84] = "c-hadron";
  particleNames[85] = "b-hadron";
  particleNames[86] = "t-hadron";
  particleNames[87] = "l-hadron";
  particleNames[88] = "h-hadron";
  particleNames[89] = "Wvirt";
  particleNames[90] = "diquark";
  particleNames[91] = "cluster";
  particleNames[92] = "string";
  particleNames[93] = "indep.";
  particleNames[94] = "CMshower";
  particleNames[95] = "SPHEaxis";
  particleNames[96] = "THRUaxis";
  particleNames[97] = "CLUSjet";
  particleNames[98] = "CELLjet";
  particleNames[99] = "table";

  particleNames[211] = "pi+";
  particleNames[311] = "K0";
  particleNames[321] = "K+";
  particleNames[411] = "D+";
  particleNames[421] = "D0";
  particleNames[431] = "D_s+";
  particleNames[511] = "B0";
  particleNames[521] = "B+";
  particleNames[531] = "B_s0";
  particleNames[541] = "B_c+";
  particleNames[-211] = "pi-";
  particleNames[-311] = "K0~";
  particleNames[-321] = "K-";
  particleNames[-411] = "D-";
  particleNames[-421] = "D0~";
  particleNames[-431] = "D_s-";
  particleNames[-511] = "B0~";
  particleNames[-521] = "B-";
  particleNames[-531] = "B_s0~";
  particleNames[-541] = "B_c-";

  particleNames[111] = "pi0";
  particleNames[221] = "eta";
  particleNames[331] = "eta'";
  particleNames[441] = "eta_c";
  particleNames[551] = "eta_b";
  particleNames[661] = "eta_t";
  particleNames[130] = "K_L0";
  particleNames[310] = "K_S0";

  particleNames[213] = "rho+";
  particleNames[313] = "K*0";
  particleNames[323] = "K*+";
  particleNames[413] = "D*+";
  particleNames[423] = "D*0";
  particleNames[433] = "D*+";
  particleNames[513] = "B*0";
  particleNames[523] = "B*+";
  particleNames[533] = "B*_s0";
  particleNames[543] = "B*_c+";
  particleNames[-213] = "rho-";
  particleNames[-313] = "K*0~";
  particleNames[-323] = "K*-";
  particleNames[-413] = "D*-";
  particleNames[-423] = "D*0~";
  particleNames[-433] = "D*-";
  particleNames[-513] = "B*0~";
  particleNames[-523] = "B*-";
  particleNames[-533] = "B*_s0~";
  particleNames[-543] = "B*_c-";

  particleNames[113] = "rho0";
  particleNames[223] = "omega";
  particleNames[333] = "phi";
  particleNames[443] = "J/psi";
  particleNames[553] = "Upsilon";
  particleNames[663] = "Theta";

  particleNames[2112] = "n0";
  particleNames[2212] = "p";
  particleNames[-2112] = "n0~";
  particleNames[-2212] = "p~";

  particleNames[3112] = "Sigma-";
  particleNames[3122] = "Lambda0";
  particleNames[3212] = "Sigma0";
  particleNames[3222] = "Sigma+";
  particleNames[3312] = "Xi-";
  particleNames[3322] = "Xi0";
  particleNames[-3112] = "Sigma+";
  particleNames[-3122] = "Lambda0~";
  particleNames[-3212] = "Sigma0~";
  particleNames[-3222] = "Sigma-";
  particleNames[-3312] = "Xi+";
  particleNames[-3322] = "Xi0~";
  
  particleNames[1114] = "Delta-";
  particleNames[2114] = "Delta0";
  particleNames[2214] = "Delta+";
  particleNames[2224] = "Delta++";
  particleNames[3114] = "Sigma*-";
  particleNames[-1114] = "Delta+";
  particleNames[-2114] = "Delta0~";
  particleNames[-2214] = "Delta=";
  particleNames[-2224] = "Delta--";
  particleNames[-3114] = "Sigma*+";

  particleNames[3214] = "Sigma*0";
  particleNames[3224] = "Sigma*+";
  particleNames[3314] = "Xi*-";
  particleNames[3324] = "Xi*0";
  particleNames[3334] = "Omega-";
  particleNames[-3214] = "Sigma*0~";
  particleNames[-3224] = "Sigma*-";
  particleNames[-3314] = "Xi*+";
  particleNames[-3324] = "Xi*0~";
  particleNames[-3334] = "Omega+";

  particleNames[4114] = "Sigma*_c0";
  particleNames[4214] = "Sigma*_c+";
  particleNames[4224] = "Sigma*_c++";
  particleNames[4314] = "Xi*_c0";
  particleNames[4324] = "Xi*_c+";
  particleNames[4334] = "Omega*_c0";
  particleNames[-4114] = "Sigma*_c0~";
  particleNames[-4214] = "Sigma*_c-";
  particleNames[-4224] = "Sigma*_c--";
  particleNames[-4314] = "Xi*_c0~";
  particleNames[-4324] = "Xi*_c-";
  particleNames[-4334] = "Omega*_c0~";

  particleNames[2101] = "ud_0";
  particleNames[3101] = "sd_0";
  particleNames[3201] = "su_0";
  particleNames[1103] = "dd_1";
  particleNames[2103] = "ud_1";
  particleNames[2203] = "uu_1";
  particleNames[3103] = "sd_1";
  particleNames[3203] = "su_1";
  particleNames[3303] = "ss_1";
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

  // Loop on the particles of the generated event
  int sugBar=0;
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
      // With an increasing barcode
      part->suggest_barcode(++sugBar);
      
      // The origin vertex is either the primary, 
      // or the end vertex of the mother, if saved
      GenVertex* originVertex = 
	p->mother() &&  
	myGenVertices.find(p->mother()) != myGenVertices.end() ? 
	originVertex = myGenVertices[p->mother()] : mainVertex;

      // Attach the particle to the origin vertex
      originVertex->add_particle_out(part);

      // Attach the vertex to the event
      add_vertex(originVertex);

      // Some persistent information
      mySimTracks->push_back(
        EmbdSimTrack(part->pdg_id(), part->momentum(), 
		     -originVertex->barcode()-1, nGenParts()));

      // It there an end vertex ?
      if ( !p->end_vertex() ) continue; 

      // If yes, create it
      GenVertex* decayVertex = 
	new GenVertex(p->end_vertex()->position()
		      +mainVertex->position());
      
      // Attach the end vertex to the particle
      decayVertex->add_particle_in(part);

      // And record it for later use 
      myGenVertices[p] = decayVertex;
      mySimVertices->push_back(
	EmbdSimVertex(decayVertex->position().vect(),
		      decayVertex->position().e(), 
		      nTracks()));

      // There we are !
    }
  }

  printMCTruth(*this);

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
    if ( particleNames.find(partId) != particleNames.end() ) {
      name = particleNames[partId];
    } else {
      name = "Unknown";
      cout << "Unknown particle with id = " << partId << endl;
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
