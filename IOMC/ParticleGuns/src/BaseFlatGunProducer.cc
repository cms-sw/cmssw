/*
 *  $Date: 2008/04/10 13:04:13 $
 *  $Revision: 1.1 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace CLHEP;

namespace {
  HepRandomEngine& getEngineReference()
  {

   Service<RandomNumberGenerator> rng;
   if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
       << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
          "which appears to be absent.  Please add that service to your configuration\n"
          "or remove the modules that require it.";
   }

// The Service has already instantiated an engine.  Make contact with it.
   return (rng->getEngine());
  }
}

BaseFlatGunProducer::BaseFlatGunProducer( const ParameterSet& pset ) :
   fEvt(0),
   fRandomEngine(getEngineReference()),
   fRandomGenerator(0)
   // fPDGTable( new DefaultConfig::ParticleDataTable("PDG Table") )
{

   ParameterSet defpset ;
   //ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
   ParameterSet pgun_params = 
      pset.getUntrackedParameter<ParameterSet>("PGunParameters", defpset ) ;
  
   // although there's the method ParameterSet::empty(),  
   // it looks like it's NOT even necessary to check if it is,
   // before trying to extract parameters - if it is empty,
   // the default values seem to be taken
   vector<int> defids ;
   defids.push_back(13) ;
   fPartIDs    = pgun_params.getUntrackedParameter< vector<int> >("PartID",defids);  
   fMinEta     = pgun_params.getUntrackedParameter<double>("MinEta",-5.5);
   fMaxEta     = pgun_params.getUntrackedParameter<double>("MaxEta",5.5);
   fMinPhi     = pgun_params.getUntrackedParameter<double>("MinPhi",-3.14159265358979323846);
   fMaxPhi     = pgun_params.getUntrackedParameter<double>("MaxPhi", 3.14159265358979323846);

  //
  //fPDGTablePath = "/afs/cern.ch/sw/lcg/external/clhep/1.9.2.1/slc3_ia32_gcc323/data/HepPDT/" ;
/*
  string HepPDTBase( getenv("HEPPDT_PARAM_PATH") ) ; 
  fPDGTablePath = HepPDTBase + "/data/" ;
  fPDGTableName = "PDG_mass_width_2004.mc"; // should it be 2004 table ?

  string TableFullName = fPDGTablePath + fPDGTableName ;
  ifstream PDFile( TableFullName.c_str() ) ;
  if( !PDFile ) 
  {
      throw cms::Exception("FileNotFound", "BaseFlatGunProducer::BaseFlatGunProducer()")
	<< "File " << TableFullName << " cannot be opened.\n";
  }

  HepPDT::TableBuilder tb(*fPDGTable) ;
  if ( !addPDGParticles( PDFile, tb ) ) { cout << " Error reading PDG !" << endl; }
  // the tb dtor fills fPDGTable
*/

  fVerbosity = pset.getUntrackedParameter<int>( "Verbosity",0 ) ;

// The Service has already instantiated an engine.  Use it.
   fRandomGenerator = new RandFlat(fRandomEngine) ;
   fAddAntiParticle = pset.getUntrackedParameter("AddAntiParticle", false) ;
}

BaseFlatGunProducer::~BaseFlatGunProducer()
{
  
//if ( fRandomGenerator != NULL ) delete fRandomGenerator;
  // do I need to delete the Engine, too ?
  
  // no need to cleanup GenEvent memory - done in HepMCProduct
  // if (fEvt != NULL) delete fEvt ; // double check
  // delete fPDGTable;
  
}


void BaseFlatGunProducer::beginJob( const EventSetup& es )
{
   es.getData( fPDGTable ) ;
   return ;

}
