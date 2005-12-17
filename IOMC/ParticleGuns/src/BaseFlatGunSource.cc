/*
 *  $Date: 2005/12/11 15:30:20 $
 *  $Revision: 1.2 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/BaseFlatGunSource.h"

#include "FWCore/Framework/src/TypeID.h" 

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/EDProduct/interface/EDProduct.h"
// #include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

BaseFlatGunSource::BaseFlatGunSource( const ParameterSet& pset,
                                      const InputSourceDescription& desc ) : 
  InputSource ( desc ),
  fNEventsToProcess(pset.getUntrackedParameter<int>("maxEvents", -1)),
  fCurrentEvent(0), 
  fCurrentRun( pset.getUntrackedParameter<unsigned int>("firstRun",1)  ),
  fNextTime(pset.getUntrackedParameter<unsigned int>("firstTime",1)),  //time in ns
  fTimeBetweenEvents(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents",kNanoSecPerSec/kAveEventPerSec) ),
  fNextID( fCurrentRun, 1 ), 
  fEvt(0),
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
  if ( fEvt != NULL ) delete fEvt ; // double check
  delete fPDGTable;
}

auto_ptr<EventPrincipal> BaseFlatGunSource::insertHepMCEvent( const BranchDescription& bds )
{ 
   
   auto_ptr<EventPrincipal> epr(0) ;
   epr = auto_ptr<EventPrincipal>(new EventPrincipal(fNextID ,
                                                     Timestamp(fNextTime),  
					             *preg_ ) ) ;
						      
   if(fEvt)  
   {
       auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
       BProduct->addHepMCData( fEvt );
       edm::Wrapper<HepMCProduct>* WProduct = 
            new edm::Wrapper<HepMCProduct>(BProduct); 
       auto_ptr<EDProduct>  FinalProduct(WProduct);
       auto_ptr<Provenance> Prov(new Provenance(bds)) ;
       epr->put(FinalProduct, Prov);
   }
    
   return epr ;
   
}
      

