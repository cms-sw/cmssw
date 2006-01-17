/*
 *  $Date: 2005/12/17 00:12:20 $
 *  $Revision: 1.1 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"

#include "FWCore/Framework/src/TypeID.h" 

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

//#include "PluginManager/ModuleDef.h"
//#include "FWCore/Framework/interface/InputSourceMacros.h"
//#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

//used for defaults

BaseFlatGunSource::BaseFlatGunSource( const ParameterSet& pset,
                                      const InputSourceDescription& desc ) : 
  GeneratedInputSource (pset, desc),
  fPDGTable( new DefaultConfig::ParticleDataTable("PDG Table") )
{

  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  fPartIDs    = pgun_params.getParameter< vector<int> >("PartID");
  fMinEta     = pgun_params.getParameter<double>("MinEta");
  fMaxEta     = pgun_params.getParameter<double>("MaxEta");
  fMinPhi     = pgun_params.getParameter<double>("MinPhi");
  fMaxPhi     = pgun_params.getParameter<double>("MaxPhi");

  // hardcoded for now
  fPDGTablePath = "/afs/cern.ch/sw/lcg/external/clhep/1.9.2.1/slc3_ia32_gcc323/data/HepPDT/" ;
  fPDGTableName = "PDG_mass_width_2002.mc"; // should it be 2004 table ?

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

  cout << "Internal BaseFlatGunSource is initialzed" << endl ;
   
}

BaseFlatGunSource::~BaseFlatGunSource()
{
  if (fEvt != NULL) delete fEvt ; // double check
  delete fPDGTable;
}
