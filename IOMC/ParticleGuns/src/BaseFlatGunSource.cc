/*
 *  $Date: 2006/03/24 00:09:45 $
 *  $Revision: 1.10 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace CLHEP;

BaseFlatGunSource::BaseFlatGunSource( const ParameterSet& pset,
                                      const InputSourceDescription& desc ) : 
  GeneratedInputSource (pset, desc),
  fEvt(0),
  fPDGTable( new DefaultConfig::ParticleDataTable("PDG Table") )
{

  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  
  fPartIDs    = pgun_params.getParameter< vector<int> >("PartID");
  
  fMinEta     = pgun_params.getParameter<double>("MinEta");
  fMaxEta     = pgun_params.getParameter<double>("MaxEta");
  fMinPhi     = pgun_params.getParameter<double>("MinPhi");
  fMaxPhi     = pgun_params.getParameter<double>("MaxPhi");

  //
  //fPDGTablePath = "/afs/cern.ch/sw/lcg/external/clhep/1.9.2.1/slc3_ia32_gcc323/data/HepPDT/" ;
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

  fVerbosity = pset.getUntrackedParameter<int>( "Verbosity",0 ) ;

   Service<RandomNumberGenerator> rng;
   long seed = (long)(rng->mySeed()) ;
//   cout << " seed= " << seed << endl ;
   fRandomEngine = new HepJamesRandom(seed) ;
   fRandomGenerator = new RandFlat(fRandomEngine) ;
 
  cout << "Internal BaseFlatGunSource is initialzed" << endl ;
   
}

BaseFlatGunSource::~BaseFlatGunSource()
{
  
  if ( fRandomGenerator != NULL ) delete fRandomGenerator;
  // do I need to delete the Engine, too ?
  
  // no need to cleanup GenEvent memory - done in HepMCProduct
  // if (fEvt != NULL) delete fEvt ; // double check
  delete fPDGTable;
  
}
