/*
 *  $Date: 2013/05/17 20:09:07 $
 *  $Revision: 1.9 $
 *  \author Julia Yarba
 */

#include <ostream>
#include <memory>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"


#include <iostream>

using namespace edm;
using namespace std;
using namespace CLHEP;

namespace {
  CLHEP::HepRandomEngine& getEngineReference()
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

   ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  
   // although there's the method ParameterSet::empty(),  
   // it looks like it's NOT even necessary to check if it is,
   // before trying to extract parameters - if it is empty,
   // the default values seem to be taken
   fPartIDs    = pgun_params.getParameter< vector<int> >("PartID");  
   fMinEta     = pgun_params.getParameter<double>("MinEta");
   fMaxEta     = pgun_params.getParameter<double>("MaxEta");
   fMinPhi     = pgun_params.getParameter<double>("MinPhi");
   fMaxPhi     = pgun_params.getParameter<double>("MaxPhi");

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
   fRandomGenerator = new CLHEP::RandFlat(fRandomEngine) ;
   fAddAntiParticle = pset.getParameter<bool>("AddAntiParticle") ;

   produces<GenRunInfoProduct, InRun>();
}

BaseFlatGunProducer::~BaseFlatGunProducer()
{
  
//if ( fRandomGenerator != NULL ) delete fRandomGenerator;
  // do I need to delete the Engine, too ?
  
  // no need to cleanup GenEvent memory - done in HepMCProduct
  // if (fEvt != NULL) delete fEvt ; // double check
  // delete fPDGTable;
  
}


void BaseFlatGunProducer::beginRun(const edm::Run & r, const EventSetup& es )
{
   es.getData( fPDGTable ) ;
   return ;

}
void BaseFlatGunProducer::endRun(const Run &run, const EventSetup& es ) {
}


void BaseFlatGunProducer::endRunProduce(Run &run, const EventSetup& es )
{
   // just create an empty product
   // to keep the EventContent definitions happy
   // later on we might put the info into the run info that this is a PGun
   auto_ptr<GenRunInfoProduct> genRunInfo( new GenRunInfoProduct() );
   run.put( genRunInfo );
}
