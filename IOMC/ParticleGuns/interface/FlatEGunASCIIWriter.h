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

#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

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
     
     virtual void analyze(  const edm::Event&, const edm::EventSetup&) override;
	 virtual void beginJob() ;
	 virtual void beginRun(const edm::Run&, const EventSetup&) override;
     
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
     ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
	 
     std::string    fOutFileName;
     HepMC::IO_GenEvent * fOutStream ;
     
     int            fCurrentEvent ; // event counter - untill I learn how to get it
	                                // from edm::Event ) EventID ??? or what ?)
     
   };
   
}

#endif
