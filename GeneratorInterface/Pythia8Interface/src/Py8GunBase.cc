#include "GeneratorInterface/Pythia8Interface/interface/Py8GunBase.h"

using namespace Pythia8;

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
   fMasterGen->readString("Standalone:allowResDec=on");
   // pythia->readString("ProcessLevel::resonanceDecays=on");
   fMasterGen->init();
   
   // init decayer
   fDecayer->readString("ProcessLevel:all = off"); // Same trick!
   fDecayer->readString("Standalone:allowResDec=on");
   // pythia->readString("ProcessLevel::resonanceDecays=on");
   fDecayer->init();
  
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
   int NewBarcode = NPartsAfterDecays;
   
   for ( int ipart=NPartsAfterDecays; ipart>NPartsBeforeDecays; ipart-- )
   {

      HepMC::GenParticle* part = event().get()->barcode_to_particle( ipart );

      if ( part->status() == 1 )
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
	    
         Particle& py8daughter = fDecayer->event[nentries]; // the 1st daughter
         HepMC::GenVertex* DecVtx = new HepMC::GenVertex( HepMC::FourVector(py8daughter.xProd(),
                                                          py8daughter.yProd(),
                                                          py8daughter.zProd(),
                                                          py8daughter.tProd()) );

         DecVtx->add_particle_in( part ); // this will cleanup end_vertex if exists, replace with the new one
                                          // I presume (vtx) barcode will be given automatically
	    
         HepMC::FourVector pmom( py8daughter.px(), py8daughter.py(), py8daughter.pz(), py8daughter.e() );
	    
         HepMC::GenParticle* daughter =
                             new HepMC::GenParticle( pmom, py8daughter.id(), 1 );
	    
         NewBarcode++;
         daughter->suggest_barcode( NewBarcode );
         DecVtx->add_particle_out( daughter );
	    	    
         for ( int ipart1=nentries+1; ipart1<nentries1; ipart1++ )
         {
            py8daughter = fDecayer->event[ipart1];
            HepMC::FourVector pmomN( py8daughter.px(), py8daughter.py(), py8daughter.pz(), py8daughter.e() );	    
            HepMC::GenParticle* daughterN =
                                new HepMC::GenParticle( pmomN, py8daughter.id(), 1 );
            NewBarcode++;
            daughterN->suggest_barcode( NewBarcode );
            DecVtx->add_particle_out( daughterN );
         }
	    
         event().get()->add_vertex( DecVtx );

      }
   } 

   return true;

}
 
void Py8GunBase::finalizeEvent()
{
   

// FIXME !!!

  //******** Verbosity ********

   if (maxEventsToPrint > 0 &&
      (pythiaPylistVerbosity || pythiaHepMCVerbosity)) 
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
   }
      
   return;
}

void Py8GunBase::statistics()
{

   fMasterGen->statistics();

   double xsec = fMasterGen->info.sigmaGen(); // cross section in mb
   xsec *= 1.0e9; // translate to pb (CMS/Gen "convention" as of May 2009)
   runInfo().setInternalXSec(xsec);
   return;

}

}
