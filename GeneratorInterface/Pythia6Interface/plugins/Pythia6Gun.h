#ifndef gen_Pythia6Gun_h
#define gen_Pythia6Gun_h

/** \class Pythia6Gun
 *
 * Generates single particle gun in HepMC format
 * Julia Yarba 02/2009 
 ***************************************/

#include <string>
#include <vector>

//#include "HepPDT/defs.h"
//#include "HepPDT/TableBuilder.hh"
//#include "HepPDT/ParticleDataTable.hh"

#include "HepMC/GenEvent.h"

// #include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h"

#include "HepPID/ParticleIDTranslations.hh"

namespace gen
{
  
// class Pythia6Service;
  
  class Pythia6Gun : public edm::EDProducer
  {
  
  public:

    Pythia6Gun( const edm::ParameterSet& );
    virtual ~Pythia6Gun();
    void beginJob() override;
    void beginRun( edm::Run const&, edm::EventSetup const& ) override;
    void endRun( edm::Run const&, edm::EventSetup const& ) override;
    void produce( edm::Event&, const edm::EventSetup& ) override;

  protected:
   
    void attachPy6DecaysToGenEvent();
    void loadEvent( edm::Event& );
    virtual void generateEvent() = 0;
    HepMC::GenParticle* addAntiParticle( int&, int&, double&, double&, double& ); 
    
    Pythia6Service*  fPy6Service;
    
    // gun particle(s) characteristics
    //
    std::vector<int> fPartIDs ;
    double           fMinPhi ;
    double           fMaxPhi ;
        
    // the event format itself
    //
    HepMC::GenEvent* fEvt;

    // HepMC/HepPDT related things 
    // (for particle/event construction)
    //
    // edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
            	    	
    bool             fHepMCVerbosity ;
    int              fPylistVerbosity;
    int              fMaxEventsToPrint ;
        
  };

}  // end namespace

#endif
