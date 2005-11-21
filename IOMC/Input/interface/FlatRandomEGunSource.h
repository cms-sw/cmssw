#ifndef FlatRandomEGunSource_H
#define FlatRandomEGunSource_H

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

// #include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/InputSource.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "IOMC/InputService/src/FakeRetriever.h"

//#include "FWCore/Framework/interface/InputServiceDescription.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"

#include "FWCore/Framework/interface/ProductRegistry.h"
//#include "FWCore/Framework/interface/ProductDescription.h"
#include "FWCore/Framework/interface/BranchDescription.h"

#include "FWCore/EDProduct/interface/EventID.h"

namespace edm
{
  // class Retriever;
  
  //class FlatRandomEGunSource : public InputService 
  class FlatRandomEGunSource : public InputSource
  {
  
  public:
    FlatRandomEGunSource(const ParameterSet &, const InputSourceDescription&  );
   virtual ~FlatRandomEGunSource();

  private:
   
    virtual std::auto_ptr<EventPrincipal> read();
    
    // data members
    
    unsigned long fNEventsToProcess ;
    unsigned long fCurrentEvent ;
    unsigned long fCurrentRun;
    unsigned long fNextTime;
    unsigned long fTimeBetweenEvents;

    // gun particle(s) characteristics
    std::vector<int>        fPartIDs ;
    double                  fMinEta ;
    double                  fMaxEta ;
    double                  fMinPhi ;
    double                  fMaxPhi ;
    double                  fMinE   ;
    double                  fMaxE   ;
    // CLHEP::HepLorentzVector fVtx ;

    // the event format itself
    HepMC::GenEvent*  fEvt;

    // HepMC/HepPDT related things 
    // (for particle/event construction)
    std::string fPDGTablePath ;
    std::string fPDGTableName ; 
    DefaultConfig::ParticleDataTable* fPDGTable;
    
    // service stuff needed to put "product" into edm::Event
    EventID            fNextID;
    
    //Retriever*         fRetriever;        
    
    //ProductDescription fProdDesc;
    BranchDescription fBranchDesc ;
    	    	
  };
} 

#endif
