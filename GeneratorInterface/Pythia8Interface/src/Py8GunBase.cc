#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Concurrency/interface/SharedResourceNames.h"

// EvtGen plugin
//
#include "Pythia8Plugins/EvtGen.h"

using namespace Pythia8;

const std::vector<std::string> gen::Py8GunBase::p8SharedResources = { edm::SharedResourceNames::kPythia8 };

namespace gen {

Py8GunBase::Py8GunBase( edm::ParameterSet const& ps )
   : Py8InterfaceBase(ps)
{

  runInfo().setFilterEfficiency(
    ps.getUntrackedParameter<double>("filterEfficiency", -1.) );
  runInfo().setExternalXSecLO(
    GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("crossSection", -1.)) );
  runInfo().setExternalXSecNLO(
    GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("crossSectionNLO", -1.)) );

  // PGun specs
  //
  edm::ParameterSet pgun_params = 
    ps.getParameter<edm::ParameterSet>("PGunParameters");
      
  // although there's the method ParameterSet::empty(),  
  // it looks like it's NOT even necessary to check if it is,
  // before trying to extract parameters - if it is empty,
  // the default values seem to be taken
  //
  fPartIDs    = pgun_params.getParameter< std::vector<int> >("ParticleID");
  fMinPhi     = pgun_params.getParameter<double>("MinPhi"); // ,-3.14159265358979323846);
  fMaxPhi     = pgun_params.getParameter<double>("MaxPhi"); // , 3.14159265358979323846);
   
}

// specific to Py8GunHad !!!
// 
bool Py8GunBase::initializeForInternalPartons()
{

  // NO MATTER what was this setting below, override it before init 
  // - this is essencial for the PGun mode 
   
  // Key requirement: switch off ProcessLevel, and thereby also PartonLevel.
  fMasterGen->readString("ProcessLevel:all = off");
  fMasterGen->readString("ProcessLevel::resonanceDecays=on");
  fMasterGen->init();
   
  // init decayer
  fDecayer->readString("ProcessLevel:all = off"); // Same trick!
  fDecayer->readString("ProcessLevel::resonanceDecays=on");
  fDecayer->init();
  
  if (useEvtGen) {
    edm::LogInfo("Pythia8Interface") << "Creating and initializing pythia8 EvtGen plugin";

    evtgenDecays = new EvtGenDecays(fMasterGen.get(), evtgenDecFile.c_str(), evtgenPdlFile.c_str());
    evtgenDecays->readDecayFile("evtgen_userfile.dec");
  }

  return true;

}

bool Py8GunBase::residualDecay()
{

  Event* pythiaEvent = &(fMasterGen->event);
  
  int NPartsBeforeDecays = pythiaEvent->size()-1; // do NOT count the very 1st "system" particle 
                                                  // in Pythia8::Event record; it does NOT even
                                                  // get translated by the HepMCInterface to the
                                                  // HepMC::GenEvent record !!!
  int NPartsAfterDecays = event().get()->particles_size();

  if(NPartsAfterDecays == NPartsBeforeDecays) return true;
  
  bool result = true;

  for ( int ipart=NPartsAfterDecays; ipart>NPartsBeforeDecays; ipart-- )
  {

    HepMC::GenParticle* part = event().get()->barcode_to_particle( ipart );

    if ( part->status() == 1 && (fDecayer->particleData).canDecay(part->pdg_id()) )
    {
      fDecayer->event.reset();
      Particle py8part(  part->pdg_id(), 93, 0, 0, 0, 0, 0, 0,
                         part->momentum().x(),
                         part->momentum().y(),
                         part->momentum().z(),
                         part->momentum().t(),
                         part->generated_mass() );
      HepMC::GenVertex* ProdVtx = part->production_vertex();
      py8part.vProd( ProdVtx->position().x(), ProdVtx->position().y(), 
                     ProdVtx->position().z(), ProdVtx->position().t() );
      py8part.tau( (fDecayer->particleData).tau0( part->pdg_id() ) );
      fDecayer->event.append( py8part );
      int nentries = fDecayer->event.size();
      if ( !fDecayer->event[nentries-1].mayDecay() ) continue;
      fDecayer->next();
      int nentries1 = fDecayer->event.size();
      if ( nentries1 <= nentries ) continue; //same number of particles, no decays...
	    
      part->set_status(2);
	    
      result = toHepMC.fill_next_event( *(fDecayer.get()), event().get(), -1, true, part);

    }
  } 

  return result;

}
 
void Py8GunBase::finalizeEvent()
{
   
  //******** Verbosity ********

  if (maxEventsToPrint > 0 &&
      (pythiaPylistVerbosity || pythiaHepMCVerbosity ||
                                pythiaHepMCVerbosityParticles) ) 
  {
    maxEventsToPrint--;
    if (pythiaPylistVerbosity) 
    {
      fMasterGen->info.list(std::cout); 
      fMasterGen->event.list(std::cout);
    } 

    if (pythiaHepMCVerbosity) 
    {
      std::cout << "Event process = "
                << fMasterGen->info.code() << "\n"
                << "----------------------" << std::endl;
      event()->print();
    }
    if (pythiaHepMCVerbosityParticles) {
      std::cout << "Event process = "
                << fMasterGen->info.code() << "\n"
                << "----------------------" << std::endl;
      ascii_io->write_event(event().get());
    }
  }
      
  return;
}

void Py8GunBase::statistics()
{

  fMasterGen->stat();

  double xsec = fMasterGen->info.sigmaGen(); // cross section in mb
  xsec *= 1.0e9; // translate to pb (CMS/Gen "convention" as of May 2009)
  runInfo().setInternalXSec(xsec);
  return;

}

}
