/*
 * $Id: HydjetSource.cc,v 1.16 2008/01/28 15:32:03 yilmaz Exp $
 *
 * Interface to the HYDJET generator, produces HepMC events
 *
 * Original Author: Camelia Mironov
 */

#include <iostream>
#include <cmath>

#include "boost/lexical_cast.hpp"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "GeneratorInterface/HydjetInterface/interface/HydjetSource.h"
#include "GeneratorInterface/HydjetInterface/interface/HydjetWrapper.h"
#include "GeneratorInterface/CommonInterface/interface/PythiaCMS.h"

#include "HepMC/PythiaWrapper6_2.h"
#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"
#include "HepMC/SimpleVector.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"


using namespace edm;
using namespace std;


HydjetSource::HydjetSource(const ParameterSet &pset, InputSourceDescription const &desc)
  : GeneratedInputSource(pset, desc), evt(0), 
    abeamtarget_(pset.getParameter<double>("aBeamTarget")),
    bfixed_(pset.getParameter<double>("bFixed")),
    bmax_(pset.getParameter<double>("bMax")),
    bmin_(pset.getParameter<double>("bMin")),
    cflag_(pset.getParameter<int>("cFlag")),
    comenergy(pset.getParameter<double>("comEnergy")),
    doradiativeenloss_(pset.getParameter<bool>("doRadiativeEnLoss")),
    docollisionalenloss_(pset.getParameter<bool>("doCollisionalEnLoss")),
    fracsoftmult_(pset.getParameter<double>("fracSoftMultiplicity")),
    hadfreeztemp_(pset.getParameter<double>("hadronFreezoutTemperature")),
    hymode_(pset.getParameter<string>("hydjetMode")),
    maxEventsToPrint_(pset.getUntrackedParameter<int>("maxEventsToPrint", 1)),
    maxlongy_(pset.getParameter<double>("maxLongitudinalRapidity")),
    maxtrany_(pset.getParameter<double>("maxTransverseRapidity")),
    nsub_(0),
    nhard_(0),
    nmultiplicity_(pset.getParameter<int>("nMultiplicity")),
    nsoft_(0),
    nquarkflavor_(pset.getParameter<int>("qgpNumQuarkFlavor")),
    pythiaPylistVerbosity_(pset.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
    qgpt0_(pset.getParameter<double>("qgpInitialTemperature")),
    qgptau0_(pset.getParameter<double>("qgpProperTimeFormation")),
    shadowingswitch_(pset.getParameter<int>("shadowingSwitch")),
    signn_(pset.getParameter<double>("sigmaInelNN"))
{
  // Default constructor

  // PYLIST Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
  LogDebug("PYLISTverbosity") << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_;

  //Max number of events printed on verbosity level 
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  LogDebug("Events2Print") << "Number of events to be printed = " << maxEventsToPrint_;

  // the input impact parameter (bxx_) is in [fm]; transform in [fm/RA] for hydjet usage
  const float ra = nuclear_radius();
  LogInfo("RAScaling")<<"Nuclear radius(RA) =  "<<ra;
  bmin_     /= ra;
  bmax_     /= ra;
  bfixed_   /= ra;

  // initialize pythia
  hyjpythia_init(pset);

  // hydjet running options
  hydjet_init(pset);

  // initialize hydjet
  LogInfo("HYDJETinAction") << "##### Calling HYINIT("<<comenergy<<","<<abeamtarget_<<","<<cflag_<<","<<bmin_<<","<<bmax_<<","<<bfixed_<<","<<nmultiplicity_<<") ####";

  call_hyinit(comenergy,abeamtarget_,cflag_,bmin_,bmax_,bfixed_,nmultiplicity_);
    
  produces<HepMCProduct>();
  produces<GenHIEvent>();
}


//_____________________________________________________________________
HydjetSource::~HydjetSource()
{
  // destructor

  call_pystat(1);

  clear();
}


//_____________________________________________________________________
void HydjetSource::add_heavy_ion_rec(HepMC::GenEvent *evt)
{
  // heavy ion record in the final CMSSW Event

  HepMC::HeavyIon* hi = new HepMC::HeavyIon(
    nsub_,                               // Ncoll_hard/N of SubEvents
    static_cast<int>(hyfpar.npart / 2),  // Npart_proj
    static_cast<int>(hyfpar.npart / 2),  // Npart_targ
    static_cast<int>(hyfpar.nbcol),      // Ncoll
    0,                                   // spectator_neutrons
    0,                                   // spectator_protons
    0,                                   // N_Nwounded_collisions
    0,                                   // Nwounded_N_collisions
    0,                                   // Nwounded_Nwounded_collisions
    hyfpar.bgen * nuclear_radius(),      // impact_parameter in [fm]
    0,                                   // event_plane_angle
    0,                                   // eccentricity
    hyjpar.sigin                         // sigma_inel_NN
  );

  evt->set_heavy_ion(*hi);
}


//________________________________________________________________
HepMC::GenParticle* HydjetSource::build_hyjet(int index, int barcode)
{
   // Build particle object corresponding to index in hyjets (soft+hard)

   HepMC::GenParticle* p = new HepMC::GenParticle(
						  HepMC::FourVector(hyjets.phj[0][index],  // px
								    hyjets.phj[1][index],  // py
								    hyjets.phj[2][index],  // pz
								    hyjets.phj[3][index]), // E
						  hyjets.khj[1][index],// id
						  hyjets.khj[0][index] // status
						  );
   p->suggest_barcode(barcode);

   return p;
}


//___________________________________________________________________
HepMC::GenVertex* HydjetSource::build_hyjet_vertex(int i,int id)
{
  // build verteces for the hyjets stored events 

   double x=hyjets.vhj[0][i];
   double y=hyjets.vhj[1][i];
   double z=hyjets.vhj[2][i];
   double t=hyjets.vhj[4][i];

   HepMC::GenVertex* vertex = new HepMC::GenVertex(HepMC::FourVector(x,y,z,t),id);
   return vertex;
}


//______________________________________________________________________
bool HydjetSource::call_pygive(const std::string& iParm ) 
{
  // Set Pythia parameters

  int numWarn = pydat1.mstu[26]; //# warnings
  int numErr = pydat1.mstu[22];// # errors
  // call the fortran routine pygive with a fortran string
  PYGIVE( iParm.c_str(), iParm.length() );  

  // if an error or warning happens it is problem
  return pydat1.mstu[26] == numWarn && pydat1.mstu[22] == numErr;   
}


//____________________________________________________________________
void HydjetSource::clear()
{

}


//_____________________________________________________________________
bool HydjetSource::get_hard_particles(HepMC::GenEvent *evt, vector<SubEvent>& subs )
{
   // Hard particles. The first nhard_ lines from hyjets array.
   // Pythia/Pyquen sub-events (sub-collisions) for a given event
   // Return T/F if success/failure
   // Create particles from lujet entries, assign them into vertices and 
   // put the vertices in the GenEvent, for each SubEvent
   // The SubEvent information is kept by storing indeces of main vertices 
   // of subevents as a vector in GenHIEvent.

   int nhard = nhard_;
   
   vector<HepMC::GenVertex*>  sub_vertices(nsub_); 

   int ihy  = 0;   
   for(int isub=0;isub<nsub_;isub++){
     
      int sub_up = (isub+1)*10000; // Upper limit in mother index, determining the range of Sub-Event
      vector<HepMC::GenParticle*> particles;
      vector<int>                 mother_ids;
      vector<HepMC::GenVertex*>   prods;

      sub_vertices[isub] = new HepMC::GenVertex(HepMC::FourVector(0,0,0,0),isub);
      evt->add_vertex(sub_vertices[isub]);
      subs.push_back(SubEvent(isub));

      while(ihy<nhard && hyjets.khj[2][ihy] < sub_up){
	
	 particles.push_back(build_hyjet(ihy,ihy+1));
	 prods.push_back(build_hyjet_vertex(ihy,isub));
	 mother_ids.push_back(hyjets.khj[2][ihy]);

	 ihy++;
      }       

      //Produce Vertices and add them to the GenEvent. Remember that GenParticles are adopted by
      //GenVertex and GenVertex is adopted by GenEvent.

      for (unsigned int i = 0; i<particles.size(); i++) {

	 HepMC::GenParticle* part = particles[i];

	 //The Fortran code is modified to preserve mother id info, by seperating the beginning 
         //mother indices of successive subevents by 10000.

	 int mid = mother_ids[i]-isub*10000-1;
	 if(mid <= 0){
	    sub_vertices[isub]->add_particle_out(part);
	    continue;
	 }

	 if(mid > 0){
	    HepMC::GenParticle* mother = particles[mid];
	    HepMC::GenVertex* prod_vertex = mother->end_vertex();
	    if(!prod_vertex){
	       prod_vertex = prods[i];
	       prod_vertex->add_particle_in(mother);
	       evt->add_vertex(prod_vertex);
	    }
	    prod_vertex->add_particle_out(part);
	 }
      }
   }
  return true;
}


//___________________________________________________________
bool HydjetSource::get_soft_particles(HepMC::GenEvent *evt, vector<SubEvent>& subs)
{
   // Soft particles. The last nsoft_ lines of hyjets
   // It corresponds to HYDRO-induced, hadrons only
   // As they have no mothers and no daughters, all are assigned to a single background vertex

   vector<HepMC::GenParticle*> hyj_entries(nsoft_);
   for (int i1 = 0; i1<nsoft_; i1++) {
      hyj_entries[i1] = build_hyjet(nhard_+i1, nhard_+i1+1);
   }
   HepMC::GenVertex* soft_vertex = new HepMC::GenVertex(HepMC::FourVector(0,0,0,0),nsub_);
   subs.push_back(SubEvent(nsub_));
   
   for ( int i2 = 0; i2<nsoft_; i2++ ) {
      soft_vertex->add_particle_out( hyj_entries[i2] ) ;
   } 
   evt->add_vertex( soft_vertex );
   
   return true;
}


//______________________________________________________________
bool HydjetSource::call_hyinit(double energy,double a, int ifb, double bmin,
			       double bmax,double bfix,int nh)
{
  // initialize hydjet  

   pydatr.mrpy[2]=1;
   HYINIT(energy,a,ifb,bmin,bmax,bfix,nh);
    return true;
}


//______________________________________________________________
bool HydjetSource::hydjet_init(const ParameterSet &pset)
{
  // set hydjet options

  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  ludatr.mrlu[0]=seed;

  // hydjet running mode mode
  // kHydroOnly --- nhsel=0 jet production off (pure HYDRO event), nhsel=0
  // kHydroJets --- nhsle=1 jet production on, jet quenching off (HYDRO+njet*PYTHIA events)
  // kHydroQJet --- nhsel=2 jet production & jet quenching on (HYDRO+njet*PYQUEN events)
  // kJetsOnly  --- nhsel=3 jet production on, jet quenching off, HYDRO off (njet*PYTHIA events)
  // kQJetsOnly --- nhsel=4 jet production & jet quenching on, HYDRO off (njet*PYQUEN events)

  if(hymode_ == "kHydroOnly") hyjpar.nhsel=0;
  else if ( hymode_ == "kHydroJets") hyjpar.nhsel=1;
  else if ( hymode_ == "kHydroQJets") hyjpar.nhsel=2;
  else if ( hymode_ == "kJetsOnly") hyjpar.nhsel=3;
  else if ( hymode_ == "kQJetsOnly") hyjpar.nhsel=4;
  else  hyjpar.nhsel=2;

  // fraction of soft hydro induced multiplicity 
  hyflow.fpart =  fracsoftmult_; 

  // hadron freez-out temperature
  hyflow.Tf   = hadfreeztemp_;

  // maximum longitudinal collective rapidity
  hyflow.ylfl = maxlongy_;
  
  // maximum transverse collective rapidity
  hyflow.ytfl = maxtrany_;  

  // shadowing on=1, off=0
  hyjpar.ishad  = shadowingswitch_;

  // set inelastic nucleon-nucleon cross section
  hyjpar.sigin  = signn_;

  // number of active quark flavors in qgp
  pyqpar.nfu    = nquarkflavor_;

  // initial temperature of QGP
  pyqpar.T0u    = qgpt0_;

  // proper time of QGP formation
  pyqpar.tau0u  = qgptau0_;

  // type of medium induced partonic energy loss
  if( doradiativeenloss_ && docollisionalenloss_ ){
    edm::LogInfo("HydjetEnLoss") << "##### Radiative AND Collisional partonic energy loss ON ####";
    pyqpar.ienglu = 0; 
  } else if ( doradiativeenloss_ ) {
    edm::LogInfo("HydjetenLoss") << "##### Only RADIATIVE partonic energy loss ON ####";
    pyqpar.ienglu = 1; 
  } else if ( docollisionalenloss_ ) {
    edm::LogInfo("HydjetEnLoss") << "##### Only COLLISIONAL partonic energy loss ON ####";
    pyqpar.ienglu = 2; 
  } else {
    edm::LogInfo("HydjetEnLoss") << "##### Radiative AND Collisional partonic energy loss ON ####";
    pyqpar.ienglu = 0; 
  }

  return true;
}


//____________________________________________________________________
bool HydjetSource::hyjpythia_init(const ParameterSet &pset)
{
  //Set PYTHIA parameters

  //random number seed
  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  ostringstream sRandomSet;
  sRandomSet << "MRPY(1)=" << seed;
  call_pygive(sRandomSet.str());

    // Set PYTHIA parameters in a single ParameterSet
  ParameterSet pythia_params = pset.getParameter<ParameterSet>("PythiaParameters") ;
  // The parameter sets to be read 
  vector<string> setNames = pythia_params.getParameter<vector<string> >("parameterSets");

    // Loop over the sets
  for ( unsigned i=0; i<setNames.size(); ++i ) {
    string mySet = setNames[i];
    
    // Read the PYTHIA parameters for each set of parameters
    vector<string> pars = pythia_params.getParameter<vector<string> >(mySet);
    
    cout << "----------------------------------------------" << endl;
    cout << "Read PYTHIA parameter set " << mySet << endl;
    cout << "----------------------------------------------" << endl;
    
    // Loop over all parameters and stop in case of mistake
    for( vector<string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar ) {
      static string sRandomValueSetting("MRPY(1)");
      if( 0 == itPar->compare(0,sRandomValueSetting.size(),sRandomValueSetting) ) {
	throw edm::Exception(edm::errors::Configuration,"PythiaError")
	  <<" Attempted to set random number using 'MRPY(1)'. NOT ALLOWED! \n Use RandomNumberGeneratorService to set the random number seed.";
      }
      if( !call_pygive(*itPar) ) {
	throw edm::Exception(edm::errors::Configuration,"PythiaError") 
	  <<"PYTHIA did not accept \""<<*itPar<<"\"";
      }
    }
  }

  return true;
}


//_____________________________________________________________________
bool HydjetSource::produce(Event & e)
{
  // generate single event

  nsoft_    = 0;
  nhard_    = 0;

  edm::LogInfo("HYDJETmode") << "##### HYDJET  nhsel = " << hyjpar.nhsel;
  edm::LogInfo("HYDJETfpart") << "##### HYDJET fpart = " << hyflow.fpart;
  edm::LogInfo("HYDJETtf") << "##### HYDJET hadron freez-out temp, Tf = " << hyflow.Tf;
  edm::LogInfo("HYDJETinTemp") << "##### HYDJET: QGP init temperature, T0 ="<<pyqpar.T0u;
  edm::LogInfo("HYDJETinTau") << "##### HYDJET: QGP formation time,tau0 ="<<pyqpar.tau0u;

  // generate a HYDJET event
  int ntry = 0;
  while(nsoft_ == 0 && nhard_ == 0){
     if(ntry > 100){
	edm::LogError("HydjetEmptyEvent") << "##### HYDJET: No Particles generated, Number of tries ="<<ntry;
	return false;
     }else{
	HYEVNT();
	nsoft_    = hyfpar.nhyd;
	nhard_    = hyfpar.npyt;
	ntry++;
     }
  }
  
  nsub_     = hyjpar.njet;

  std::vector<SubEvent> subvector;

  // event information
  HepMC::GenEvent *evt = new HepMC::GenEvent();

  if(nhard_>0) get_hard_particles(evt,subvector); 
  if(nsoft_>0) get_soft_particles(evt,subvector);

  evt->set_signal_process_id(pypars.msti[0]);      // type of the process
  evt->set_event_scale(pypars.pari[16]);           // Q^2
  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);

  add_heavy_ion_rec(evt);

  auto_ptr<HepMCProduct> bare_product(new HepMCProduct());
  bare_product->addHepMCData(evt );
  auto_ptr<GenHIEvent> genhi(new GenHIEvent(subvector,hyjpar.nhsel));
  e.put(bare_product);
  e.put(genhi);

  // print PYLIST info
  if (event() <= maxEventsToPrint_ && pythiaPylistVerbosity_)     
     call_pylist(pythiaPylistVerbosity_);      
  

  return true;
}


//________________________________________________________________
