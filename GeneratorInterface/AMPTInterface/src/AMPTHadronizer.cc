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

#include "GeneratorInterface/AMPTInterface/interface/AMPTHadronizer.h"
#include "GeneratorInterface/AMPTInterface/interface/AMPTWrapper.h"

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

CLHEP::HepRandomEngine* _amptRandomEngine;

extern "C"
{
  float gen::ranart_(int *idummy)
  {
    if(0) idummy = idummy; 
    float rannum = _amptRandomEngine->flat();
    return rannum;
  }
}

extern "C"
{
  float gen::ran1_(int *idummy)
  {
    if(0) idummy = idummy;
    return _amptRandomEngine->flat();
  }
}

AMPTHadronizer::AMPTHadronizer(const ParameterSet &pset) :
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
    amptmode_(pset.getParameter<int>("amptmode")),
    ntmax_(pset.getParameter<int>("ntmax")),
    dt_(pset.getParameter<double>("dt")),
    stringFragA_(pset.getParameter<double>("stringFragA")),
    stringFragB_(pset.getParameter<double>("stringFragB")),
    popcornmode_(pset.getParameter<bool>("popcornmode")),
    popcornpar_(pset.getParameter<double>("popcornpar")),
    shadowingmode_(pset.getParameter<bool>("shadowingmode")),
    quenchingmode_(pset.getParameter<bool>("quenchingmode")),
    quenchingpar_(pset.getParameter<double>("quenchingpar")),
    pthard_(pset.getParameter<double>("pthard")),
    mu_(pset.getParameter<double>("mu")),
    izpc_(pset.getParameter<int>("izpc")),
    alpha_(pset.getParameter<double>("alpha")),
    dpcoal_(pset.getParameter<double>("dpcoal")),
    drcoal_(pset.getParameter<double>("drcoal")),
    ks0decay_(pset.getParameter<bool>("ks0decay")),
    phidecay_(pset.getParameter<bool>("phidecay")),
    deuteronmode_(pset.getParameter<int>("deuteronmode")),          
    deuteronfactor_(pset.getParameter<int>("deuteronfactor")),
    deuteronxsec_(pset.getParameter<int>("deuteronxsec")),
    minijetpt_(pset.getParameter<double>("minijetpt")),
    maxmiss_(pset.getParameter<int>("maxmiss")),
    doInitialAndFinalRadiation_(pset.getParameter<int>("doInitialAndFinalRadiation")),
    ktkick_(pset.getParameter<int>("ktkick")),
    diquarkembedding_(pset.getParameter<int>("diquarkembedding")),
    diquarkpx_(pset.getParameter<double>("diquarkpx")),
    diquarkpy_(pset.getParameter<double>("diquarkpy")),
    diquarkx_(pset.getParameter<double>("diquarkx")),
    diquarky_(pset.getParameter<double>("diquarky")),
    phi0_(0.),
    sinphi0_(0.),
    cosphi0_(1.),
    rotate_(pset.getParameter<bool>("rotateEventPlane"))
{
  // Default constructor
  edm::Service<RandomNumberGenerator> rng;
  _amptRandomEngine = &(rng->getEngine());
}


//_____________________________________________________________________
AMPTHadronizer::~AMPTHadronizer()
{
}

//_____________________________________________________________________
void AMPTHadronizer::add_heavy_ion_rec(HepMC::GenEvent *evt)
{
  // heavy ion record in the final CMSSW Event
  HepMC::HeavyIon* hi = new HepMC::HeavyIon(
    hmain1.jatt,                               // Ncoll_hard/N of SubEvents
    hmain1.np,                               // Npart_proj
    hmain1.nt,                               // Npart_targ
    hmain1.n0+hmain1.n01+hmain1.n10+hmain1.n11,  // Ncoll
    0,                                   // spectator_neutrons
    0,                                   // spectator_protons
    hmain1.n01,                          // N_Nwounded_collisions
    hmain1.n10,                          // Nwounded_N_collisions
    hmain1.n11,                          // Nwounded_Nwounded_collisions
    hparnt.hint1[18],                   // impact_parameter in [fm]
    phi0_,                              // event_plane_angle
    0,                                   // eccentricity
    hparnt.hint1[11]                    // sigma_inel_NN
  );
  evt->set_heavy_ion(*hi);
  delete hi;
}

//___________________________________________________________________     
HepMC::GenParticle* AMPTHadronizer::build_ampt(int index, int barcode)
{
   // Build particle object corresponding to index in ampt

   float px0 = hbt.plast[index][0];
   float py0 = hbt.plast[index][1];
   float pz0 = hbt.plast[index][2];
   float m = hbt.plast[index][3];

   float px = px0*cosphi0_-py0*sinphi0_;
   float py = py0*cosphi0_+px0*sinphi0_;
   float pz = pz0;
   float e = sqrt(px*px+py*py+pz*pz+m*m); 
   int status = 1;

   HepMC::GenParticle* p = new HepMC::GenParticle(
                                                  HepMC::FourVector(px,                                                                            
                                                                    py,                                                                            
                                                                    pz,                                                         
                                                                    e),                                                           
                                                  INVFLV(hbt.lblast[index]),// id                                                                             
                                                  status // status                                                          
                                                  );

   p->suggest_barcode(barcode);
   return p;
}

//___________________________________________________________________     
HepMC::GenVertex* AMPTHadronizer::build_ampt_vertex(int i,int id)
{
   // build verteces for the ampt stored events                        
   HepMC::GenVertex* vertex = new HepMC::GenVertex(HepMC::FourVector(0,0,0,0),id);
   return vertex;
}
//_____________________________________________________________________  
bool AMPTHadronizer::generatePartonsAndHadronize()
{
   // generate single event
   if(rotate_) rotateEvtPlane();

   // generate a AMPT event
   AMPT(frame_.data(), bmin_, bmax_, strlen(frame_.data()));

   // event information
   HepMC::GenEvent *evt = new HepMC::GenEvent();
   get_particles(evt); 

   add_heavy_ion_rec(evt);

   event().reset(evt);


   return true;
}

//_____________________________________________________________________  
bool AMPTHadronizer::get_particles(HepMC::GenEvent *evt )
{
      HepMC::GenVertex*  vertice;

      vector<HepMC::GenParticle*> particles;
      vector<int>                 mother_ids;
      vector<HepMC::GenVertex*>   prods;

      vertice = new HepMC::GenVertex(HepMC::FourVector(0,0,0,0),0);
      evt->add_vertex(vertice);
      if(!evt->signal_process_vertex()) evt->set_signal_process_vertex(vertice);

      const unsigned int knumpart = hbt.nlast;
      for (unsigned int ipart = 0; ipart<knumpart; ipart++) {
         int mid = 0;
         particles.push_back(build_ampt(ipart,ipart+1));
         prods.push_back(build_ampt_vertex(ipart,0));
         mother_ids.push_back(mid);
         LogDebug("DecayChain")<<"Mother index : "<<mid;
      }
      
      LogDebug("AMPT")<<"Number of particles in vector "<<particles.size();

      for (unsigned int ipart = 0; ipart<particles.size(); ipart++) {
	 HepMC::GenParticle* part = particles[ipart];

         int mid = mother_ids[ipart];
	 LogDebug("DecayChain")<<"Particle "<<ipart;
	 LogDebug("DecayChain")<<"Mother's ID "<<mid;
	 LogDebug("DecayChain")<<"Particle's PDG ID "<<part->pdg_id();

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
bool AMPTHadronizer::call_amptset(double efrm, std::string frame, std::string proj, std::string targ, int iap, int izp, int iat, int izt)
{
  // initialize hydjet  
   AMPTSET(efrm,frame.data(),proj.data(),targ.data(),iap,izp,iat,izt,strlen(frame.data()),strlen(proj.data()),strlen(targ.data()));
	return true;
}
//______________________________________________________________________
bool AMPTHadronizer::ampt_init(const ParameterSet &pset)
{
    anim.isoft=amptmode_;
    input2.ntmax=ntmax_;
    input1.dt=dt_;
    ludat1.parj[40]=stringFragA_;
    ludat1.parj[41]=stringFragB_;
    popcorn.ipop=popcornmode_;
    ludat1.parj[4]=popcornpar_;
    hparnt.ihpr2[5]=shadowingmode_;
    hparnt.ihpr2[3]=quenchingmode_;
    hparnt.hipr1[13]=quenchingpar_;
    hparnt.hipr1[7]=pthard_;
    para2.xmu=mu_;
    anim.izpc=izpc_;
    para2.alpha=alpha_;
    coal.dpcoal=dpcoal_;
    coal.drcoal=drcoal_;
    resdcy.iksdcy=ks0decay_;
    phidcy.iphidcy=phidecay_;
    para8.idpert=deuteronmode_;
    para8.npertd=deuteronfactor_;
    para8.idxsec=deuteronxsec_;
    phidcy.pttrig=minijetpt_;
    phidcy.maxmiss=maxmiss_;
    hparnt.ihpr2[1]=doInitialAndFinalRadiation_;
    hparnt.ihpr2[4]=ktkick_;
    embed.iembed=diquarkembedding_;
    embed.pxqembd=diquarkpx_;
    embed.pyqembd=diquarkpy_;
    embed.xembd=diquarkx_;
    embed.yembd=diquarky_;

    return true;
}

//_____________________________________________________________________
bool AMPTHadronizer::initializeForInternalPartons(){

   // ampt running options
   ampt_init(pset_);

   // initialize ampt
   LogInfo("AMPTinAction") << "##### Calling AMPTSET(" << efrm_ << "," <<frame_<<","<<proj_<<","<<targ_<<","<<iap_<<","<<izp_<<","<<iat_<<","<<izt_<<") ####";

   call_amptset(efrm_,frame_,proj_,targ_,iap_,izp_,iat_,izt_);

   return true;
}

bool AMPTHadronizer::declareStableParticles( const std::vector<int>& pdg )
{
   return true;
}

//________________________________________________________________                                                                    
void AMPTHadronizer::rotateEvtPlane(){
   int zero = 0;
   double test = (double)gen::ranart_(&zero);
   phi0_ = 2.*pi*test - pi;
   sinphi0_ = sin(phi0_);
   cosphi0_ = cos(phi0_);
}

//________________________________________________________________ 
bool AMPTHadronizer::hadronize()
{
   return false;
}

bool AMPTHadronizer::decay()
{
   return true;
}
   
bool AMPTHadronizer::residualDecay()
{  
   return true;
}

void AMPTHadronizer::finalizeEvent(){
    return;
}

void AMPTHadronizer::statistics(){
    return;
}

const char* AMPTHadronizer::classname() const
{  
   return "gen::AMPTHadronizer";
}

