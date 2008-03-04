/*
 * $Id: HydjetSource.cc,v 1.6 2007/05/21 14:49:06 mironov Exp $
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
    hymode_(pset.getParameter<string>("hydjetMode")),
    maxEventsToPrint_(pset.getUntrackedParameter<int>("maxEventsToPrint", 1)),
    nhard_(0),
    nmultiplicity_(pset.getParameter<int>("nMultiplicity")),
    nsoft_(0),
    ptmin_(pset.getParameter<double>("ptMin")),
    pythiaPylistVerbosity_(pset.getUntrackedParameter<int>("pythiaPylistVerbosity", 0))
{
  // Default constructor

  // the input impact parameter (bxx_) is in [fm]; transform in [fm/RA] for hydjet usage
  const float ra = nuclear_radius();
  LogDebug("RAScaling")<<"Nuclear radius(RA) =  "<<ra;
  bmin_     /= ra;
  bmax_     /= ra;
  bfixed_   /= ra;

  // PYLIST Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
  LogDebug("PYLISTverbosity") << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_;

  //Max number of events printed on verbosity level 
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  LogDebug("Events2Print") << "Number of events to be printed = " << maxEventsToPrint_;

  //initialize pythia
  hyjpythia_init(pset);

  //initialize hydjet
  hyjhydro_init(pset);

  if( hymode_ != "kHydroOnly" ) { 
    call_pyinit("CMS", "p", "p", comenergy);
  }
    
  cout<<endl;

  produces<HepMCProduct>();

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
  HepMC::HeavyIon *hi = new HepMC::HeavyIon(
    hyfpar.nbcol,                       // Ncoll_hard
    hyfpar.npart / 2,                   // Npart_proj
    hyfpar.npart / 2,                   // Npart_targ
    hyfpar.nbcol,                       // Ncoll
    -1,                                 // spectator_neutrons
    -1,                                 // spectator_protons
    -1,                                 // N_Nwounded_collisions
    -1,                                 // Nwounded_N_collisions
    -1,                                 // Nwounded_Nwounded_collisions
    hyfpar.bgen * nuclear_radius(),     // impact_parameter in [fm]
    0,                                  // event_plane_angle
    0,                                  // eccentricity
    hyipar.sigin                        // sigma_inel_NN
  );

  evt->set_heavy_ion(*hi);
}


//________________________________________________________________
HepMC::GenParticle* HydjetSource::build_particle(int index)                             
{
  // Build particle object corresponding to index in lujets.


  HepMC::GenParticle* p = new HepMC::GenParticle(
                            HepMC::FourVector(lujets.p[0][index],  // px
                                              lujets.p[1][index],  // py
                                              lujets.p[2][index],  // pz
                                              lujets.p[3][index]), // E
                            lujets.k[1][index],// id
                            lujets.k[0][index] // status
                            );
  p->suggest_barcode(index);

  return p;
}


//____________________________________________________________________
bool HydjetSource::build_vertices(int i, vector<HepMC::GenParticle*>& luj_entries,
                                  HepMC::GenEvent* evt)
{
  // Build a production vertex for a particle with index i in lujets
  // and add the vertex to the event.


  // fix: need to fix to look for the second mothers in case of flavor 
  // K(I,2)=91-94; cluster, string, indep, CMshower

  HepMC::GenParticle* pi       = luj_entries[i]; 
  HepMC::GenVertex* prod_vtx_i = pi->production_vertex();
  int mother_i                 = lujets.k[2][i];    
    
  if ( !prod_vtx_i && mother_i > 0 ) {
    prod_vtx_i = luj_entries[mother_i]->end_vertex(); //decay vertex of the mother
    if (prod_vtx_i) {
          // if the decay vertex of its mother exists
          // assign it to the particle, as the production vertex
          prod_vtx_i->add_particle_out( pi );
    } 
  }
         
  HepMC::FourVector prod_pos( lujets.v[0][mother_i],
                             lujets.v[1][mother_i],
                             lujets.v[2][mother_i],
                             lujets.v[4][mother_i]
                             ); 
  if (!prod_vtx_i && (mother_i > 0 || prod_pos != HepMC::FourVector(0,0,0,0))) {
    prod_vtx_i = new HepMC::GenVertex();
    prod_vtx_i->add_particle_out( pi );       
    evt->add_vertex( prod_vtx_i );
  }
           
  if (prod_vtx_i && prod_vtx_i->position()==HepMC::FourVector(0,0,0,0)) {
      prod_vtx_i->set_position( prod_pos );
  }
            
  //  check the consistency of the end_vertices
  if ( prod_vtx_i && mother_i > 0 ) {
    if ( !luj_entries[mother_i]->end_vertex() ) {
      // if end vtx of the  mother isn't specified, do it now
      prod_vtx_i->add_particle_in( luj_entries[mother_i] );
    } else if ( luj_entries[mother_i]->end_vertex() != prod_vtx_i ) {
      //error. the decay vtx of the mother is different from the daughter production vtx
      cerr << "HydjetSource::build_production_vertex: "<<
              "inconsistent mother/daughter produced vtx in event!" << endl;
      luj_entries[mother_i]->end_vertex()->print(cerr);
      prod_vtx_i->print(cerr);

    }
  }
  return true;
}


//_____________________________________________________________________
bool HydjetSource::call_hyjgive(const std::string& param ) 
{
  // Set Hydjet parameters

  string::size_type loc = param.find('=', 0);
  if(loc == string::npos) {
    throw edm::Exception(edm::errors::Configuration, "HydjetError")
          << " no '=' in parameter string '" << param << "'.";
  }

  string tag = param.substr(0, loc);
  string val = param.substr(loc+1);

  bool accepted = true;

  if(tag == "nhsel") {
      hyjpar.nhsel = boost::lexical_cast<int>(val);
      edm::LogInfo("HYDJETnhsel") << "nhsel = " << hyjpar.nhsel;
  } else if(tag == "ptmin") {
      hyjpar.ptmin = boost::lexical_cast<float>(val);
      edm::LogInfo("HYDJETptmin") << "ptmin = " << hyjpar.ptmin;
  } else if(tag == "fpart") {
      hyflow.fpart = boost::lexical_cast<float>(val);
      edm::LogInfo("HYDJETfpart") << "fpart = " << hyflow.fpart;
  } else if(tag == "ylfl") {
      hyflow.ylfl  = boost::lexical_cast<float>(val);
      edm::LogInfo("HYDJETylfl") << "ylfl = " << hyflow.ylfl;
  } else if(tag == "ytfl") {
      hyflow.ytfl  = boost::lexical_cast<float>(val);
      edm::LogInfo("HYDJETytfl") << "ytfl = " << hyflow.ytfl;
  } else
      accepted = false;

  return accepted;
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
bool HydjetSource::get_hydjet_particles(HepMC::GenEvent *evt)
{
  // Hard particles. The first nhard_ lines form lujets array.
  // It corresponds to hard multijet part of the event: hard 
  // pythia/pyquen sub-events (sub-collisions) for a given event
  // PYTHIA/PYQUEN-induced, initial protons and partons, final partons, 
  // strings, unstable and stable hadrons - full multijet story a la pythia-pyjets

  // Soft particles. The last nsoft_ lines of lujets
  // It corresponds to HYDRO-induced, hadrons only

  // return T/F if succes/failure

  int lujetsEntries= nhard_+nsoft_;

  // create a particle instance for each lujets entry and fill a map
  // create a vector which maps from the lujets particle index to the 
  // GenParticle address

  vector<HepMC::GenParticle*> luj_entries(lujetsEntries);
  for (int i1 = 0; i1<lujetsEntries; i1++) {     
    luj_entries[i1] = build_particle(i1);
  }

  // loop over particles again to create vertices
  for (int i2 = 0; i2<lujetsEntries; i2++) {
      build_vertices( i2,luj_entries,evt );
  } 

  // handle the case with particles comming from nowhere 
  // no mothers, no daughters
  for ( int i3 = 0; i3<lujetsEntries; i3++ ) {
    if ( !luj_entries[i3]->end_vertex() && !luj_entries[i3]->production_vertex() ) {
      HepMC::GenVertex* prod_vtx_i3 = new  HepMC::GenVertex();
      prod_vtx_i3->add_particle_out( luj_entries[i3] ) ;
      evt->add_vertex( prod_vtx_i3 );
    } 
  }

  return true;
}


//______________________________________________________________
bool HydjetSource::hyjhydro_init(const ParameterSet &pset)
{
  //initialize hydjet HYDRO part

  // hydjet mode
  // kHydroOnly --- nhsel=0 jet production off (pure HYDRO event), nhsel=0
  // kHydroJets --- nhsle=1 jet production on, jet quenching off (HYDRO+njet*PYTHIA events)
  // kHydroQJet --- nhsel=2 jet production & jet quenching on (HYDRO+njet*PYQUEN events)
  // kJetsOnly  --- nhsel=3 jet production on, jet quenching off, HYDRO off (njet*PYTHIA events)
  // kQJetsOnly --- nhsel=4 jet production & jet quenching on, HYDRO off (njet*PYQUEN events)

  // Read HYDJET parameters 
  ParameterSet hydjet_params = pset.getParameter<ParameterSet>("HydjetParameters") ;
    
  // Read the HYDJET parameters from the set
  vector<string> pars_hyj = hydjet_params.getParameter<vector<string> >("hydjet");
    
  // Loop over all parameters and stop in case of mistake
  for( vector<string>::const_iterator  itPar = pars_hyj.begin();
       itPar != pars_hyj.end(); ++itPar ) {     

    if( ! call_hyjgive(*itPar) ) {
      throw edm::Exception(edm::errors::Configuration,"HYDJET Error") 
        <<" HYDJET did not accept the following \""<<*itPar<<"\"";
    }
  }

  if(hymode_ == "kHydroOnly")       
    call_hyjgive("nhsel=0");
  else if ( hymode_ == "kHydroJets")
    call_hyjgive("nhsel=1");
  else if ( hymode_ == "kHydroQJets")
    call_hyjgive("nhsel=2");
  else if ( hymode_ == "kJetsOnly")
    call_hyjgive("nhsel=3");
  else if ( hymode_ == "kQJetsOnly")
    call_hyjgive("nhsel=4");
  else call_hyjgive("nhsel=2");

  // minimum pT hard
  hyjpar.ptmin = ptmin_;

  return true;
}


//____________________________________________________________________
bool HydjetSource::hyjpythia_init(const ParameterSet &pset)
{
  //initialize PYTHIA

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

  LogDebug("HYDJETabeamtarget") << "abeamtarget_ =  " << abeamtarget_;
  edm::LogInfo("HYDJETcflag") << "cflag_ = " << cflag_;
  edm::LogInfo("HYDJETbmin") << "bmin_ = " << bmin_;
  edm::LogInfo("HYDJETbmax") << "bmax_ = " << bmax_;
  edm::LogInfo("HYDJETbfixed") << "bfixed_ = " << bfixed_;
  edm::LogInfo("HYDJETmultiplicity") << "nmultiplicity_ = " << nmultiplicity_;
  edm::LogInfo("HYDJETmode") << "hydjet mode_ = " << hyjpar.nhsel;
  LogDebug("HYDJETinAction") << "##### Calling HYDRO(abeamtarget_,cflag_,bmin_,bmax_,bfixed_,nmultiplicity_) ####" << endl;

  HYDRO(abeamtarget_,cflag_,bmin_,bmax_,bfixed_,nmultiplicity_);

  nsoft_    = hyfpar.nhyd;
  nhard_    = hyfpar.npyt;

  // event information
  HepMC::GenEvent *evt = new HepMC::GenEvent();
  get_hydjet_particles(evt); 

  evt->set_signal_process_id(pypars.msti[0]);      // type of the process
  evt->set_event_scale(pypars.pari[16]);           // Q^2
  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);

  add_heavy_ion_rec(evt);
 
  if (evt) {
    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());
    bare_product->addHepMCData(evt );
    e.put(bare_product);

    // print PYLIST info
    if (event() <= maxEventsToPrint_ && pythiaPylistVerbosity_)     
      call_pylist(pythiaPylistVerbosity_);      
    }
  
  return true;
}


//________________________________________________________________
