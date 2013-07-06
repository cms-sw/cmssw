#include <iostream>
#include <cmath>

#include "boost/lexical_cast.hpp"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "GeneratorInterface/HijingInterface/interface/HijingHadronizer.h"
#include "GeneratorInterface/HijingInterface/interface/HijingPythiaWrapper.h"
#include "GeneratorInterface/HijingInterface/interface/HijingWrapper.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"
#include "HepMC/SimpleVector.h"
#include "CLHEP/Random/RandomEngine.h"

static const double pi = 3.14159265358979;

using namespace edm;
using namespace std;
using namespace gen;


HijingHadronizer::HijingHadronizer(const ParameterSet &pset) :
    BaseHadronizer(pset),
    evt(0), 
    pset_(pset),
    bmax_(pset.getParameter<double>("bMax")),
    bmin_(pset.getParameter<double>("bMin")),
    efrm_(pset.getParameter<double>("comEnergy")),
    frame_(pset.getParameter<string>("frame")),
    proj_(pset.getParameter<string>("proj")),
    targ_(pset.getParameter<string>("targ")),
    iap_(pset.getParameter<int>("iap")),
    izp_(pset.getParameter<int>("izp")),
    iat_(pset.getParameter<int>("iat")),
    izt_(pset.getParameter<int>("izt")),
    phi0_(0.),
    sinphi0_(0.),
    cosphi0_(1.),
    rotate_(pset.getParameter<bool>("rotateEventPlane"))
{
  // Default constructor
  Service<RandomNumberGenerator> rng;
  hijRandomEngine = &(rng->getEngine());

}


//_____________________________________________________________________
HijingHadronizer::~HijingHadronizer()
{
  // destructor
}

//_____________________________________________________________________
void HijingHadronizer::add_heavy_ion_rec(HepMC::GenEvent *evt)
{
  // heavy ion record in the final CMSSW Event
  HepMC::HeavyIon* hi = new HepMC::HeavyIon(
    himain1.jatt,                               // Ncoll_hard/N of SubEvents
    himain1.np,                               // Npart_proj
    himain1.nt,                               // Npart_targ
    himain1.n0+himain1.n01+himain1.n10+himain1.n11, // Ncoll
    0,                                   // spectator_neutrons
    0,                                   // spectator_protons
    himain1.n01,                          // N_Nwounded_collisions
    himain1.n10,                          // Nwounded_N_collisions
    himain1.n11,                          // Nwounded_Nwounded_collisions
    //gsfs Changed from 19 to 18 (Fortran counts from 1 , not 0) 
    hiparnt.hint1[18],                   // impact_parameter in [fm]
    phi0_,                               // event_plane_angle
    0,                                   // eccentricity
    //gsfs Changed from 12 to 11 (Fortran counts from 1 , not 0) 
    hiparnt.hint1[11]                    // sigma_inel_NN
  );
  evt->set_heavy_ion(*hi);
  delete hi;
}

//___________________________________________________________________     
HepMC::GenParticle* HijingHadronizer::build_hijing(int index, int barcode)
{
   // Build particle object corresponding to index in hijing
                                                                                                                                                         
   double x0 = himain2.patt[0][index];
   double y0 = himain2.patt[1][index];

   double x = x0*cosphi0_-y0*sinphi0_;
   double y = y0*cosphi0_+x0*sinphi0_;

   // Hijing gives V0's status=4, they need to have status=1 to be decayed in geant
   // also change status=11 to status=2
   if(himain2.katt[3][index]<=10 && himain2.katt[3][index]>0) himain2.katt[3][index]=1;
   if(himain2.katt[3][index]<=20 && himain2.katt[3][index]>10) himain2.katt[3][index]=2;

   HepMC::GenParticle* p = new HepMC::GenParticle(
                                                  HepMC::FourVector(x,  // px                                                                            
                                                                    y,  // py                                                                            
                                                                    himain2.patt[2][index],  // pz                                                         
                                                                    himain2.patt[3][index]), // E                                                          
                                                  himain2.katt[0][index],// id                                                                             
                                                  himain2.katt[3][index] // status                                                          
                                                  );
   p->suggest_barcode(barcode);
   
   return p;
}

//___________________________________________________________________     
HepMC::GenVertex* HijingHadronizer::build_hijing_vertex(int i,int id)
{
  // build verteces for the hijing stored events                        
   double x0=himain2.vatt[0][i];
   double y0=himain2.vatt[1][i];
   double x = x0*cosphi0_-y0*sinphi0_;
   double y = y0*cosphi0_+x0*sinphi0_;
   double z=himain2.vatt[2][i];
   double t=himain2.vatt[3][i];

   HepMC::GenVertex* vertex = new HepMC::GenVertex(HepMC::FourVector(x,y,z,t),id);
   return vertex;
}

bool HijingHadronizer::generatePartonsAndHadronize()
{
   // generate single event
   if(rotate_) rotateEvtPlane();

   // generate a HIJING event
   
   float f_bmin = bmin_;
   float f_bmax = bmax_;
   HIJING(frame_.data(), f_bmin, f_bmax, strlen(frame_.data()));

   // event information
   HepMC::GenEvent *evt = new HepMC::GenEvent();
   get_particles(evt); 

   //   evt->set_signal_process_id(pypars.msti[0]);      // type of the process
   //   evt->set_event_scale(pypars.pari[16]);           // Q^2
   add_heavy_ion_rec(evt);

   event().reset(evt);

   return true;
}

//_____________________________________________________________________  
bool HijingHadronizer::get_particles(HepMC::GenEvent *evt )
{
      HepMC::GenVertex*  vertice;

      vector<HepMC::GenParticle*> particles;
      vector<int>                 mother_ids;
      vector<HepMC::GenVertex*>   prods;

      vertice = new HepMC::GenVertex(HepMC::FourVector(0,0,0,0),0);
      evt->add_vertex(vertice);
      if(!evt->signal_process_vertex()) evt->set_signal_process_vertex(vertice);

      const unsigned int knumpart = himain1.natt;

      for (unsigned int ipart = 0; ipart<knumpart; ipart++) {
	
	int mid = himain2.katt[2][ipart] - 1;  // careful of fortan to c++ array index

	particles.push_back(build_hijing(ipart,ipart+1));
	prods.push_back(build_hijing_vertex(ipart,0));
	mother_ids.push_back(mid);
	LogDebug("DecayChain")<<"Mother index : "<<mid;
      }	
      
      LogDebug("Hijing")<<"Number of particles in vector "<<particles.size();

      for (unsigned int ipart = 0; ipart<particles.size(); ipart++) {
	 HepMC::GenParticle* part = particles[ipart];

         int mid = mother_ids[ipart];
	 LogDebug("DecayChain")<<"Particle "<<ipart;
	 LogDebug("DecayChain")<<"Mother's ID "<<mid;
	 LogDebug("DecayChain")<<"Particle's PDG ID "<<part->pdg_id();
	 
	 // remove zero pT particles from list, protection for fastJet against pt=0 jets
	 if(part->status()==1&&sqrt(part->momentum().px()*part->momentum().px()+part->momentum().py()*part->momentum().py())==0) 
	   continue;
	 
         if(mid <= 0){
	   vertice->add_particle_out(part);
            continue;
         }

         if(mid > 0){
	    HepMC::GenParticle* mother = particles[mid];
	    LogDebug("DecayChain")<<"Mother's PDG ID "<<mother->pdg_id();
	    HepMC::GenVertex* prod_vertex = mother->end_vertex();
            if(!prod_vertex){
               prod_vertex = prods[ipart];
	       prod_vertex->add_particle_in(mother);
	       
               evt->add_vertex(prod_vertex);
               prods[ipart]=0; // mark to protect deletion                                                                                                   
	       
            }
            prod_vertex->add_particle_out(part);
         }
      }

      // cleanup vertices not assigned to evt                                                                                                            
      for (unsigned int i = 0; i<prods.size(); i++) {
         if(prods[i]) delete prods[i];
      }

   return true;
}

//_____________________________________________________________________
bool HijingHadronizer::call_hijset(double efrm, std::string frame, std::string proj, std::string targ, int iap, int izp, int iat, int izt)
{

   float ef = efrm;
  // initialize hydjet  
   HIJSET(ef,frame.data(),proj.data(),targ.data(),iap,izp,iat,izt,strlen(frame.data()),strlen(proj.data()),strlen(targ.data()));
   return true;
}

//______________________________________________________________
bool HijingHadronizer::initializeForInternalPartons(){

  //initialize pythia5

  if(0){
    std::string dumstr = "";
    call_pygive(dumstr);
  }

   // initialize hijing
   LogInfo("HIJINGinAction") << "##### Calling HIJSET(" << efrm_ << "," <<frame_<<","<<proj_<<","<<targ_<<","<<iap_<<","<<izp_<<","<<iat_<<","<<izt_<<") ####";
   call_hijset(efrm_,frame_,proj_,targ_,iap_,izp_,iat_,izt_);

   return true;

}

bool HijingHadronizer::declareStableParticles( std::vector<int> pdg )
{
   return true;
}

//________________________________________________________________                                                                    
void HijingHadronizer::rotateEvtPlane(){

   phi0_ = 2.*pi*gen::hijran_(0) - pi;
   sinphi0_ = sin(phi0_);
   cosphi0_ = cos(phi0_);
}

//________________________________________________________________ 
bool HijingHadronizer::hadronize()
{
   return false;
}

bool HijingHadronizer::decay()
{
   return true;
}
   
bool HijingHadronizer::residualDecay()
{  
   return true;
}

void HijingHadronizer::finalizeEvent(){
    return;
}

void HijingHadronizer::statistics(){
    return;
}

const char* HijingHadronizer::classname() const
{  
   return "gen::HijingHadronizer";
}

