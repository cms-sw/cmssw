//#include <iostream>
//#include <sstream>
//#include <string>
//#include <memory>
//#include <stdint.h>

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"
#include "GeneratorInterface/Pythia8Interface/interface/RandomP8.h"
#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace Pythia8;

namespace gen {

Py8InterfaceBase::Py8InterfaceBase( edm::ParameterSet const& ps )
{

  randomEngine = &getEngineReference();

  fMasterGen.reset(new Pythia);
  fDecayer.reset(new Pythia);

  fMasterGen->readString("Next:numberShowEvent = 0");
  fDecayer->readString("Next:numberShowEvent = 0");

  // RandomP8* RP8 = new RandomP8();
  fMasterGen->setRndmEnginePtr( new RandomP8() );
  fDecayer->setRndmEnginePtr( new RandomP8() );
  
  fParameters = ps.getParameter<edm::ParameterSet>("PythiaParameters");
  
  pythiaPylistVerbosity = ps.getUntrackedParameter<int>("pythiaPylistVerbosity", 0);
  pythiaHepMCVerbosity  = ps.getUntrackedParameter<bool>("pythiaHepMCVerbosity", false);
  maxEventsToPrint      = ps.getUntrackedParameter<int>("maxEventsToPrint", 0);

}

bool Py8InterfaceBase::readSettings( int ) 
{

   for ( ParameterCollector::const_iterator line = fParameters.begin();
         line != fParameters.end(); ++line ) 
   {
      if (line->find("Random:") != std::string::npos)
         throw cms::Exception("PythiaError") << "Attempted to set random number "
         "using Pythia commands. Please use " "the RandomNumberGeneratorService."
         << std::endl;

      if (!fMasterGen->readString(*line)) throw cms::Exception("PythiaError")
			              << "Pythia 8 did not accept \""
				      << *line << "\"." << std::endl;
   }

   if ( pythiaPylistVerbosity > 10 ) 
   {
      if ( pythiaPylistVerbosity == 11 || pythiaPylistVerbosity == 13 )
         fMasterGen->settings.listAll();
      if ( pythiaPylistVerbosity == 12 || pythiaPylistVerbosity == 13 )
         fMasterGen->particleData.listAll();
   }

   return true;

}

bool Py8InterfaceBase::declareStableParticles( const std::vector<int>& pdgIds )
{
  
  for ( size_t i=0; i<pdgIds.size(); i++ )
  {
    // FIXME: need to double check if PID's are the same in Py6 & Py8,
    //        because the HepPDT translation tool is actually for **Py6** 
    // 
    // well, actually it looks like Py8 operates in PDT id's rather than Py6's
    //
    // int PyID = HepPID::translatePDTtoPythia( pdgIds[i] ); 
    int PyID = pdgIds[i]; 
    std::ostringstream pyCard ;
    pyCard << PyID <<":mayDecay=false";
    fMasterGen->readString( pyCard.str() );
    // alternative:
    // set the 2nd input argument warn=false 
    // - this way Py8 will NOT print warnings about unknown particle code(s)
    // fMasterPtr->readString( pyCard.str(), false )
  }
        
   return true;

}

bool Py8InterfaceBase:: declareSpecialSettings( const std::vector<std::string> settings )
{

   for ( unsigned int iss=0; iss<settings.size(); iss++ )
   {
      if ( settings[iss].find("QED-brem-off") == std::string::npos ) continue;
      fMasterGen->readString( "TimeShower:QEDshowerByL=off" );
   }

   return true;
}

bool Py8InterfaceBase::residualDecay() 
{
  
/*
  Event* pythiaEvent = &(fMasterPtr->event);
  
  assert(fCurrentEventState);
  
  int NPartsBeforeDecays = pythiaEvent->size();
  // int NPartsAfterDecays = event().get()->particles_size();
  int NPartsAfterDecays = fCurrentEventState->particles_size();
  int NewBarcode = NPartsAfterDecays;
   
  for ( int ipart=NPartsAfterDecays; ipart>NPartsBeforeDecays; ipart-- )
  {

    // HepMC::GenParticle* part = event().get()->barcode_to_particle( ipart );
    HepMC::GenParticle* part = fCurrentEventState->barcode_to_particle( ipart );

    if ( part->status() == 1 )
    {
      fDecayerPtr->event.reset();
      Particle py8part(  part->pdg_id(), 93, 0, 0, 0, 0, 0, 0,
                         part->momentum().x(),
                         part->momentum().y(),
                         part->momentum().z(),
                         part->momentum().t(),
                         part->generated_mass() );
      HepMC::GenVertex* ProdVtx = part->production_vertex();
      py8part.vProd( ProdVtx->position().x(), ProdVtx->position().y(), 
                     ProdVtx->position().z(), ProdVtx->position().t() );
      py8part.tau( (fDecayerPtr->particleData).tau0( part->pdg_id() ) );
      fDecayerPtr->event.append( py8part );
      int nentries = fDecayerPtr->event.size();
      if ( !fDecayerPtr->event[nentries-1].mayDecay() ) continue;
      fDecayerPtr->next();
      int nentries1 = fDecayerPtr->event.size();
      // --> fDecayerPtr->event.list(std::cout);
      if ( nentries1 <= nentries ) continue; //same number of particles, no decays...
	    
      part->set_status(2);
	    
      Particle& py8daughter = fDecayerPtr->event[nentries]; // the 1st daughter
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
	    	    
      for ( ipart=nentries+1; ipart<nentries1; ipart++ )
      {
        py8daughter = fDecayerPtr->event[ipart];
        HepMC::FourVector pmomN( py8daughter.px(), py8daughter.py(), py8daughter.pz(), py8daughter.e() );	    
        HepMC::GenParticle* daughterN =
                        new HepMC::GenParticle( pmomN, py8daughter.id(), 1 );
        NewBarcode++;
        daughterN->suggest_barcode( NewBarcode );
        DecVtx->add_particle_out( daughterN );
      }
	    
      // event().get()->add_vertex( DecVtx );
      fCurrentEventState->add_vertex( DecVtx );

    }
 } 
*/   
 return true;
 
}


void Py8InterfaceBase::statistics()
{
  
   fMasterGen->statistics();
   return;
   
}

}
