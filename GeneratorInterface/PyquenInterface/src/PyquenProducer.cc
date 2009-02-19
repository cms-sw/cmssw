/*
 *
 * Generates PYQUEN HepMC events
 *
 * Original Author: Camelia Mironov
 * $Id: PyquenProducer.cc,v 1.22 2009/02/03 22:02:00 yilmaz Exp $
*/

#include <iostream>
#include "time.h"

#include "GeneratorInterface/PyquenInterface/interface/PyquenProducer.h"
#include "GeneratorInterface/PyquenInterface/interface/PYR.h"
#include "GeneratorInterface/PyquenInterface/interface/PyquenWrapper.h"
#include "GeneratorInterface/CommonInterface/interface/PythiaCMS.h"

#include "SimDataFormats/GeneratorProducts/interface/GenInfoProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "HepMC/IO_HEPEVT.h"
#include "HepMC/PythiaWrapper.h"

#include "CLHEP/Random/RandomEngine.h"


using namespace edm;
using namespace std;

HepMC::IO_HEPEVT hepevtio2;

PyquenProducer :: PyquenProducer(const ParameterSet & pset):
EDProducer(),
abeamtarget_(pset.getParameter<double>("aBeamTarget")),
angularspecselector_(pset.getParameter<int>("angularSpectrumSelector")),
bmin_(pset.getParameter<double>("bMin")),
bmax_(pset.getParameter<double>("bMax")),
bfixed_(pset.getParameter<double>("bFixed")),
cflag_(pset.getParameter<int>("cFlag")),
comenergy(pset.getParameter<double>("comEnergy")),
doquench_(pset.getParameter<bool>("doQuench")),
doradiativeenloss_(pset.getParameter<bool>("doRadiativeEnLoss")),
docollisionalenloss_(pset.getParameter<bool>("doCollisionalEnLoss")),
doIsospin_(pset.getParameter<bool>("doIsospin")),
embedding_(pset.getParameter<bool>("embeddingMode")),
nquarkflavor_(pset.getParameter<int>("numQuarkFlavor")),
qgpt0_(pset.getParameter<double>("qgpInitialTemperature")),
qgptau0_(pset.getParameter<double>("qgpProperTimeFormation")),
maxEventsToPrint_(pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
pythiaHepMCVerbosity_(pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
pythiaPylistVerbosity_(pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0)),
eventNumber_(0)
{
  // Default constructor
  // Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
  LogDebug("PYLISTverbosity") << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_ << endl;
  
  // HepMC event verbosity Level
  pythiaHepMCVerbosity_ = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false);
  LogDebug("HepMCverbosity")  << "Pythia HepMC verbosity = " << pythiaHepMCVerbosity_ << endl; 

  //Max number of events printed on verbosity level 
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  LogDebug("Events2Print") << "Number of events to be printed = " << maxEventsToPrint_ << endl;

  //Proton to Nucleon fraction
  pfrac_ = 1./(1.98+0.015*pow(abeamtarget_,2./3));

  //initialize pythia
  pyqpythia_init(pset);

  //initilize pyquen
  pyquen_init(pset);

  // Call PYTHIA
  call_pyinit("CMS", "p", "p", comenergy);  
  
  produces<HepMCProduct>();
  produces<GenInfoProduct, edm::InRun>();
}


//_____________________________________________________________________
PyquenProducer::~PyquenProducer()
{
  // distructor

  call_pystat(1);

  clear();
}


//_____________________________________________________________________
void PyquenProducer::add_heavy_ion_rec(HepMC::GenEvent *evt)
{
  HepMC::HeavyIon *hi = new HepMC::HeavyIon(
    -1,                                 // Ncoll_hard
    -1,                                 // Npart_proj
    -1,                                 // Npart_targ
    -1,                                 // Ncoll
    -1,                                 // spectator_neutrons
    -1,                                 // spectator_protons
    -1,                                 // N_Nwounded_collisions
    -1,                                 // Nwounded_N_collisions
    -1,                                 // Nwounded_Nwounded_collisions
    plfpar.bgen,                        // impact_parameter in [fm]
    0,                                  // event_plane_angle
    0,                                  // eccentricity
    -1                                  // sigma_inel_NN
  );

  evt->set_heavy_ion(*hi);

  delete hi;
}


//______________________________________________________________________
bool PyquenProducer::call_pygive(const std::string& iParm ) 
{
  // Set Pythia parameters

  int numWarn = pydat1.mstu[26];//# warnings
  int numErr = pydat1.mstu[22]; //# errors
  // call the fortran routine pygive with a fortran string
  PYGIVE( iParm.c_str(), iParm.length() );  

  // if an error or warning happens it is problem
  return pydat1.mstu[26] == numWarn && pydat1.mstu[22] == numErr;   
}


//____________________________________________________________________
void PyquenProducer::clear()
{
}


//_____________________________________________________________________
void PyquenProducer::produce(Event & e, const EventSetup& es)
{

   //Get Parameters from the background Pb+Pb event if necessary
   HepMC::FourVector* vtx_;
   double evtPlane = 0;
   if(embedding_){
      Handle<HepMCProduct> input;
      e.getByLabel("source",input);
      const HepMC::GenEvent * inev = input->GetEvent();
      HepMC::HeavyIon* hi = inev->heavy_ion();
      if(hi){
	 bfixed_ = hi->impact_parameter();
	 evtPlane = hi->event_plane_angle();
      }else{
	 LogWarning("EventEmbedding")<<"Background event does not have heavy ion record!";
      }

      HepMC::GenVertex* genvtx = inev->signal_process_vertex();
      if(!genvtx){
	 cout<<"No Signal Process Vertex!"<<endl;
	 HepMC::GenEvent::particle_const_iterator pt=inev->particles_begin();
	 HepMC::GenEvent::particle_const_iterator ptend=inev->particles_end();
         while(!genvtx || ( genvtx->particles_in_size() == 1 && pt != ptend ) ){
	    if(!genvtx) cout<<"No Gen Vertex!"<<endl;
            ++pt;
	    if(pt == ptend) cout<<"End reached!"<<endl;
	    genvtx = (*pt)->production_vertex();
	 }
      }
      vtx_ = &(genvtx->position());
      cout<<"Vertex is at : "<<vtx_->z()<<" cm"<<endl;
   }
   
   // Generate PYQUEN event
  // generate single partonic PYTHIA jet event

  // Take into account whether it's a nn or pp or pn interaction
  if(doIsospin_) call_pyinit("CMS", nucleon(), nucleon(), comenergy);
  call_pyevnt();

  // call PYQUEN to apply parton rescattering and energy loss 
  // if doQuench=FALSE, it is pure PYTHIA
  if( doquench_ ){
    PYQUEN(abeamtarget_,cflag_,bfixed_,bmin_,bmax_);
    edm::LogInfo("PYQUENinAction") << "##### Calling PYQUEN("<<abeamtarget_<<","<<cflag_<<","<<bfixed_<<") ####";
  } else {
    edm::LogInfo("PYQUENinAction") << "##### Calling PYQUEN: QUENCHING OFF!! This is just PYTHIA !!!! ####";
  }

  // call PYTHIA to finish the hadronization
  PYEXEC();

  // fill the HEPEVT with the PYJETS event record
  call_pyhepc(1);

  // event information
  HepMC::GenEvent* evt = hepevtio2.read_next_event();
  evt->set_signal_process_id(pypars.msti[0]);      // type of the process

  evt->set_event_scale(pypars.pari[16]);           // Q^2

  ++eventNumber_;
  evt->set_event_number(eventNumber_);
  if(embedding_) rotateEvtPlane(evt,evtPlane);
  add_heavy_ion_rec(evt);

  auto_ptr<HepMCProduct> bare_product(new HepMCProduct());
  bare_product->addHepMCData(evt );

  if(embedding_) bare_product->applyVtxGen(vtx_);
  e.put(bare_product); 

  // verbosity
  if( e.id().event() <= maxEventsToPrint_ && ( pythiaPylistVerbosity_ || pythiaHepMCVerbosity_ )) { 
    // Prints PYLIST info
     if( pythiaPylistVerbosity_ ){
       call_pylist(pythiaPylistVerbosity_);
     }
      
     // Prints HepMC event
     if( pythiaHepMCVerbosity_ ){
        cout << "Event process = " << pypars.msti[0] << endl; 
	evt->print(); 
     }
  }
    
  return;
}


//_____________________________________________________________________
bool PyquenProducer::pyqpythia_init(const ParameterSet & pset)
{
  //initialize PYTHIA

  //random number seed
  edm::Service<RandomNumberGenerator> rng;
  randomEngine = fRandomEngine = &(rng->getEngine());
  uint32_t seed = rng->mySeed();
  ostringstream sRandomSet;
  sRandomSet << "MRPY(1)=" << seed;
  call_pygive(sRandomSet.str());

  //Turn Hadronization Off if there is quenching
  if(doquench_){
     string sHadOff("MSTP(111)=0");
     call_pygive(sHadOff);
  }

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
           << " Attempted to set random number using 'MRPY(1)'. NOT ALLOWED!\n"
              " Use RandomNumberGeneratorService to set the random number seed.";
      }
      if( !call_pygive(*itPar) ) {
        throw edm::Exception(edm::errors::Configuration,"PythiaError") 
           << "PYTHIA did not accept \""<<*itPar<<"\"";
      }
    }
  }

  return true;
}


//_________________________________________________________________
bool PyquenProducer::pyquen_init(const ParameterSet &pset)
{
  // PYQUEN initialization

  // angular emitted gluon  spectrum selection 
  pyqpar.ianglu = angularspecselector_;

  // type of medium induced partonic energy loss
  if( doradiativeenloss_ && docollisionalenloss_ ){
    edm::LogInfo("PYQUENinEnLoss") << "##### PYQUEN: Radiative AND Collisional partonic energy loss ON ####";
    pyqpar.ienglu = 0; 
  } else if ( doradiativeenloss_ ) {
    edm::LogInfo("PYQUENinRad") << "##### PYQUEN: Only RADIATIVE partonic energy loss ON ####";
    pyqpar.ienglu = 1; 
  } else if ( docollisionalenloss_ ) {
    edm::LogInfo("PYQUENinColl") << "##### PYQUEN: Only COLLISIONAL partonic energy loss ON ####";
    pyqpar.ienglu = 2; 
  } else {
    edm::LogInfo("PYQUENinEnLoss") << "##### PYQUEN: Radiative AND Collisional partonic energy loss ON ####";
    pyqpar.ienglu = 0; 
  }

  // number of active quark flavors in qgp
  pyqpar.nfu    = nquarkflavor_;

  // initial temperature of QGP
  pyqpar.T0u    = qgpt0_;

  // proper time of QGP formation
  pyqpar.tau0u  = qgptau0_;

  return true;
}

char* PyquenProducer::nucleon(){
  int* dummy;
  double random = pyr_(dummy);
  char* nuc;
  if(random > pfrac_) nuc = "n";
  else nuc = "p";
  
  return nuc;
}

void PyquenProducer::rotateEvtPlane(HepMC::GenEvent* evt, double angle){

   double sinphi0 = sin(angle);
   double cosphi0 = cos(angle);

   for ( HepMC::GenEvent::vertex_iterator vt=evt->vertices_begin();
	 vt!=evt->vertices_end(); ++vt )
      {
	 
	 double x0 = (*vt)->position().x();
	 double y0 = (*vt)->position().y();
	 double z = (*vt)->position().z();
	 double t = (*vt)->position().t();

	 double x = x0*cosphi0-y0*sinphi0;
	 double y = y0*cosphi0+x0*sinphi0;

	 (*vt)->set_position( HepMC::FourVector(x,y,z,t) ) ;      
      }

   for ( HepMC::GenEvent::particle_iterator vt=evt->particles_begin();
         vt!=evt->particles_end(); ++vt )
      {

         double x0 = (*vt)->momentum().x();
         double y0 = (*vt)->momentum().y();
         double z = (*vt)->momentum().z();
         double t = (*vt)->momentum().t();

         double x = x0*cosphi0-y0*sinphi0;
         double y = y0*cosphi0+x0*sinphi0;

         (*vt)->set_momentum( HepMC::FourVector(x,y,z,t) ) ;
      }




}

//____________________________________________________________________
