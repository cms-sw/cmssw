// ----------------------------------------------------------------------
// FlatEGunASCIIWriter.h
// Author: Julia Yarba
//
// This code has been molded after examples in HepMC and HepPDT, and
// after single particle gun example (private contacts with Lynn Garren)
//
// Plus, it uses the ParameterSet funtionalities for "user interface"  
//
// ----------------------------------------------------------------------

#ifndef FlatEGunASCIIWriter_h
#define FlatEGunASCIIWriter_h

// base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

// for local things (data members)
#include <string>

#include "CLHEP/HepPDT/defs.h"
#include "CLHEP/HepPDT/DefaultConfig.hh"
#include "CLHEP/HepPDT/TableBuilder.hh"
#include "CLHEP/HepPDT/ParticleDataTableT.hh"

#include "CLHEP/HepMC/GenEvent.h"

#include "CLHEP/HepMC/WriteHepMC.h"

namespace edm 
{

   class FlatEGunASCIIWriter : public edm::EDAnalyzer 
   {

      public:

         // The following is not yet used, but will be the primary
         // constructor when the parameter set system is available.
         //
         explicit FlatEGunASCIIWriter( const edm::ParameterSet& ) ;

         virtual ~FlatEGunASCIIWriter() ;

         virtual void analyze(  const edm::Event&, const edm::EventSetup& );
	 virtual void beginJob( const EventSetup& ) ;

      private:
      

         // gun particle(s) characteristics
         std::vector<int>        fPartIDs ;
         double                  fMinEta ;
         double                  fMaxEta ;
         double                  fMinPhi ;
         double                  fMaxPhi ;
         double                  fMinE   ;
         double                  fMaxE   ;

         // the event format itself
         HepMC::GenEvent*  fEvt;


         // HepMC/HepPDT related things 
         // (for particle/event construction)
         std::string    fPDGTablePath ;
         std::string    fPDGTableName ; 
         DefaultConfig::ParticleDataTable* fPDGTable;
	 
         std::string    fOutFileName;
	 std::ofstream* fOutStream ;
	 
	 int            fCurrentEvent ; // event counter - untill I learn how to get it
	                                // from edm::Event ) EventID ??? or what ?)

   };

}

#endif
