/*
 *
 * Generates PYQUEN HepMC events
 *
 * Original Author: Camelia Mironov
 * $Id: PyquenSource.cc,v 1.11 2008/03/03 15:52:10 yilmaz Exp $
*/

#include <iostream>
#include "time.h"

#include "GeneratorInterface/PyquenInterface/interface/PyquenSource.h"
#include "GeneratorInterface/PyquenInterface/interface/PyquenWrapper.h"
#include "GeneratorInterface/CommonInterface/interface/PythiaCMS.h"

#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "HepMC/IO_HEPEVT.h"
#include "HepMC/PythiaWrapper.h"

using namespace edm;
using namespace std;

HepMC::IO_HEPEVT hepevtio;

PyquenSource :: PyquenSource(const ParameterSet & pset, InputSourceDescription const& desc):
GeneratedInputSource(pset, desc), evt(0), 
abeamtarget_(pset.getParameter<double>("aBeamTarget")),
angularspecselector_(pset.getParameter<int>("angularSpectrumSelector")),
bfixed_(pset.getParameter<double>("bFixed")),
cflag_(pset.getParameter<int>("cFlag")),
comenergy(pset.getParameter<double>("comEnergy")),
doquench_(pset.getParameter<bool>("doQuench")),
doradiativeenloss_(pset.getParameter<bool>("doRadiativeEnLoss")),
docollisionalenloss_(pset.getParameter<bool>("doCollisionalEnLoss")),
nquarkflavor_(pset.getParameter<int>("numQuarkFlavor")),
qgpt0_(pset.getParameter<double>("qgpInitialTemperature")),
qgptau0_(pset.getParameter<double>("qgpProperTimeFormation")),
maxEventsToPrint_(pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
pythiaHepMCVerbosity_(pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
pythiaPylistVerbosity_(pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0))
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

  //initialize pythia
  pyqpythia_init(pset);

  //initilize pyquen
  pyquen_init(pset);

  // Call PYTHIA
  call_pyinit("CMS", "p", "p", comenergy);  
  
  cout<<endl;

  produces<HepMCProduct>();
  produces<GenInfoProduct, edm::InRun>();
}


//_____________________________________________________________________
PyquenSource::~PyquenSource()
{
  // distructor

  call_pystat(1);

  clear();
}


//_____________________________________________________________________
void PyquenSource::add_heavy_ion_rec(HepMC::GenEvent *evt)
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
bool PyquenSource::call_pygive(const std::string& iParm ) 
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
void PyquenSource::clear()
{
}


//_____________________________________________________________________
bool PyquenSource::produce(Event & e)
{
  edm::LogInfo("PYQUENabeamtarget") << "##### PYQUEN: beam/target A = "                     << abeamtarget_;
  edm::LogInfo("PYQUENcflag")       << "##### PYQUEN: centrality flag cflag_ = "            << cflag_;
  edm::LogInfo("PYQUENbfixed")      << "##### PYQUEN: fixed impact parameter bFixed = "     << bfixed_;
  edm::LogInfo("PYQUENinNFlav")     << "##### PYQUEN: No active quark flavor nf = "         << pyqpar.nfu;
  edm::LogInfo("PYQUENinTemp")      << "##### PYQUEN: Initial temperature of QGP, T0 = "    << pyqpar.T0u;
  edm::LogInfo("PYQUENinTau")       << "##### PYQUEN: Proper formation time of QGP, tau0 =" << pyqpar.tau0u;

  // Generate PYQUEN event
  // generate single partonic PYTHIA jet event
  call_pyevnt();

  // call PYQUEN to apply parton rescattering and energy loss 
  // if doQuench=FALSE, it is pure PYTHIA
  if( doquench_ ){
    PYQUEN(abeamtarget_,cflag_,bfixed_);
    edm::LogInfo("PYQUENinAction") << "##### Calling PYQUEN("<<abeamtarget_<<","<<cflag_<<","<<bfixed_<<") ####";
  } else {
    edm::LogInfo("PYQUENinAction") << "##### Calling PYQUEN: QUENCHING OFF!! This is just PYTHIA !!!! ####";
  }

  // call PYTHIA to finish the hadronization
  PYEXEC();

  // fill the HEPEVT with the PYJETS event record
  call_pyhepc(1);

  // event information
  HepMC::GenEvent* evt = hepevtio.read_next_event();
  evt->set_signal_process_id(pypars.msti[0]);      // type of the process
  evt->set_event_scale(pypars.pari[16]);           // Q^2
  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);

  add_heavy_ion_rec(evt);

  auto_ptr<HepMCProduct> bare_product(new HepMCProduct());
  bare_product->addHepMCData(evt );
  e.put(bare_product); 

  // verbosity
  if( event() <= maxEventsToPrint_ && ( pythiaPylistVerbosity_ || pythiaHepMCVerbosity_ )) { 
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
    
  return true;
}


//_____________________________________________________________________
bool PyquenSource::pyqpythia_init(const ParameterSet & pset)
{
  //initialize PYTHIA

  //random number seed
  edm::Service<RandomNumberGenerator> rng;
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
bool PyquenSource::pyquen_init(const ParameterSet &pset)
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


//____________________________________________________________________
