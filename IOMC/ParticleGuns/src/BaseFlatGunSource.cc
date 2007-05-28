/*
 *  $Date: 2006/07/19 02:08:50 $
 *  $Revision: 1.14 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

// #include "FWCore/Framework/intercface/ESHandle.h"
// #include "FWCore/Framework/interface/EventSetup.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace CLHEP;

BaseFlatGunSource::BaseFlatGunSource( const ParameterSet& pset,
                                      const InputSourceDescription& desc ) : 
   GeneratedInputSource (pset, desc),
   fEvt(0)
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
      throw cms::Exception("FileNotFound", "BaseFlatGunSource::BaseFlatGunSource()")
	<< "File " << TableFullName << " cannot be opened.\n";
  }

  HepPDT::TableBuilder tb(*fPDGTable) ;
  if ( !addPDGParticles( PDFile, tb ) ) { cout << " Error reading PDG !" << endl; }
  // the tb dtor fills fPDGTable
*/

  fVerbosity = pset.getUntrackedParameter<int>( "Verbosity",0 ) ;

   Service<RandomNumberGenerator> rng;
   long seed = (long)(rng->mySeed()) ;
   fRandomEngine = new HepJamesRandom(seed) ;
   fRandomGenerator = new RandFlat(fRandomEngine) ;
   
   fAddAntiParticle = pset.getUntrackedParameter("AddAntiParticle", false) ;
   
}

BaseFlatGunSource::~BaseFlatGunSource()
{
  
  if ( fRandomGenerator != NULL ) delete fRandomGenerator;
  // do I need to delete the Engine, too ?
  
  // no need to cleanup GenEvent memory - done in HepMCProduct
  // if (fEvt != NULL) delete fEvt ; // double check
  // delete fPDGTable;
  
}


void BaseFlatGunSource::beginJob( const EventSetup& es )
{

   es.getData( fPDGTable ) ;

   return ;

}
