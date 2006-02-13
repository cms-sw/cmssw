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

#include "FWCore/Framework/interface/GeneratedInputSource.h"

#include <memory>
#include "boost/shared_ptr.hpp"

// fwd declaration(s)
class BaseEventVertexGenerator;

namespace edm
{
  
  class BaseFlatGunSource : public GeneratedInputSource
  {
  
  public:
    BaseFlatGunSource(const ParameterSet &, const InputSourceDescription&  );
    // BaseFlatGunSource( const ParameterSet& ) ;
    virtual ~BaseFlatGunSource();
    
    HepMC::GenVertex* generateEvtVertex() const ;

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
    std::string      fPDGTablePath ;
    std::string      fPDGTableName ; 
    DefaultConfig::ParticleDataTable* fPDGTable;
        	    	
    std::auto_ptr<BaseEventVertexGenerator> fEventVertexGenerator;
    
    int              fVerbosity ;
    
  };
} 

#endif
