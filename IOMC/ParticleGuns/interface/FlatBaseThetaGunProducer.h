#ifndef FlatBaseThetaGunProducer_H
#define FlatBaseThetaGunProducer_H

#include <string>

#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "HepMC/GenEvent.h"

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "GeneratorInterface/Core/interface/EventVertexHelper.h"

#include <memory>
#include "boost/shared_ptr.hpp"

namespace edm {
  class Event;
  class HepMCProduct;
  
class FlatBaseThetaGunProducer : public one::EDProducer<one::WatchRuns,
                                                        one::WatchLuminosityBlocks,
                                                        EndRunProducer> {
  
  public:
    
    FlatBaseThetaGunProducer(const ParameterSet &);
    virtual ~FlatBaseThetaGunProducer();

  private:
    virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) override;
    virtual void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) override;
    virtual void beginRun(edm::Run const& run, edm::EventSetup const&) override;
    virtual void endRun(edm::Run const& run, edm::EventSetup const&) override;
    virtual void endRunProduce(edm::Run& run, edm::EventSetup const&) override;

    EventVertexHelper eventVertexHelper_;
    
  protected :
    void smearVertex(edm::Event const&, edm::HepMCProduct&);
  
    // non-virtuals ! this and only way !
    //
    // data members
    
    // gun particle(s) characteristics
    std::vector<int> fPartIDs ;
    double           fMinTheta ;
    double           fMaxTheta ;
    double           fMinPhi ;
    double           fMaxPhi ;

    // the event format itself
    HepMC::GenEvent* fEvt;

    // HepMC/HepPDT related things 
    // (for particle/event construction)
    ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
            	    	
    int                     fVerbosity ;

    bool                    fAddAntiParticle;
    
  };
} 

#endif
