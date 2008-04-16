#ifndef BaseFlatGunSource_H
#define BaseFlatGunSource_H

/** \class FlatRandomEGunSource
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 10/2005 
 ***************************************/
#include <string>

#include "HepPDT/defs.h"
// #include "HepPDT/DefaultConfig.hh"
#include "HepPDT/TableBuilder.hh"
//#include "HepPDT/ParticleDataTableT.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "HepMC/GenEvent.h"

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandFlat.h"

#include <memory>
#include "boost/shared_ptr.hpp"

namespace edm
{
  
  class BaseFlatGunSource : public GeneratedInputSource
  {
  
  public:
    BaseFlatGunSource(const ParameterSet &, const InputSourceDescription&  );
    // BaseFlatGunSource( const ParameterSet& ) ;
    virtual ~BaseFlatGunSource();
    void beginJob( const edm::EventSetup& ) ;
  private:
   
  protected :
  
    // non-virtuals ! this and only way !
    //
    // data members
    
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
    //std::string      fPDGTablePath ;
    //std::string      fPDGTableName ; 
    // DefaultConfig::ParticleDataTable* fPDGTable;
    // DefaultConfig::ParticleDataTable* fTestTable ;
    // ESHandle<DefaultConfig::ParticleDataTable> fPDGTable ;
    ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
            	    	
    int              fVerbosity ;

    HepRandomEngine& fRandomEngine ;
    RandFlat*        fRandomGenerator; 
    
    bool             fAddAntiParticle;
    
  };
} 

#endif
