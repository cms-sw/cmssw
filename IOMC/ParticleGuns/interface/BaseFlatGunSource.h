#ifndef BaseFlatGunSource_H
#define BaseFlatGunSource_H

/** \class FlatRandomEGunSource
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 10/2005 
 ***************************************/
#include <string>

#include "CLHEP/HepPDT/defs.h"
#include "CLHEP/HepPDT/DefaultConfig.hh"
#include "CLHEP/HepPDT/TableBuilder.hh"
#include "CLHEP/HepPDT/ParticleDataTableT.hh"

#include "CLHEP/HepMC/GenEvent.h"

#include "FWCore/Framework/interface/InputSource.h"

#include "FWCore/Framework/interface/EventPrincipal.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/InputSourceDescription.h"

#include "FWCore/Framework/interface/ProductRegistry.h"

#include "FWCore/Framework/interface/BranchDescription.h"

#include "FWCore/EDProduct/interface/EventID.h"

namespace edm
{
  
  class BaseFlatGunSource : public InputSource
  {
  
  public:
    BaseFlatGunSource(const ParameterSet &, const InputSourceDescription&  );
    // BaseFlatGunSource( const ParameterSet& ) ;
    virtual ~BaseFlatGunSource();

  private:
   
    virtual std::auto_ptr<EventPrincipal> read() = 0 ;
    
  protected :
  
    // non-virtuals ! this and only way !
    //
    void registerBranch( const BranchDescription& bds ) { preg_->addProduct( bds ) ; } 
    std::auto_ptr<EventPrincipal> insertHepMCEvent( const BranchDescription&  ) ; 
  
    // data members
    
    unsigned long    fNEventsToProcess ;
    unsigned long    fCurrentEvent ;
    unsigned long    fCurrentRun;
    unsigned long    fNextTime;
    unsigned long    fTimeBetweenEvents;
    EventID          fNextID;    

    // gun particle(s) characteristics
    std::vector<int> fPartIDs ;
    double           fMinEta ;
    double           fMaxEta ;
    double           fMinPhi ;
    double           fMaxPhi ;

    // the event format itself
    HepMC::GenEvent* fEvt;

    // HepMC/HepPDT related things 
    // (for particle/event construction)
    std::string      fPDGTablePath ;
    std::string      fPDGTableName ; 
    DefaultConfig::ParticleDataTable* fPDGTable;
        	    	
  };
} 

#endif
