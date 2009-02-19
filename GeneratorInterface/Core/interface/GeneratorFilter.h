// -*- C++ -*-
//
//

// class template GeneratorFilter<HAD> provides an EDFilter which uses
// the hadronizer type HAD to generate partons, hadronize them, and
// decay the resulting particles, in the CMS framework.

#ifndef gen_GeneratorFilter_h
#define gen_GeneratorFilter_h

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

//#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

namespace edm
{
  template <class HAD> class GeneratorFilter : public EDFilter
  {
  public:
    typedef HAD Hadronizer;

    // The given ParameterSet will be passed to the contained
    // Hadronizer object.
    explicit GeneratorFilter(ParameterSet const& ps);

    virtual ~GeneratorFilter();

    virtual bool filter(Event& e, EventSetup const& es);
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual bool beginRun(Run &, EventSetup const&);
    virtual bool endRun(Run &, EventSetup const&);
    virtual bool beginLuminosityBlock(LuminosityBlock &, EventSetup const&);
    virtual bool endLuminosityBlock(LuminosityBlock &, EventSetup const&);
    virtual void respondToOpenInputFile(FileBlock const& fb);
    virtual void respondToCloseInputFile(FileBlock const& fb);
    virtual void respondToOpenOutputFiles(FileBlock const& fb);
    virtual void respondToCloseOutputFiles(FileBlock const& fb);

  private:
    Hadronizer hadronizer_;
    gen::ExternalDecayDriver* decayer_;
    
  };

  //------------------------------------------------------------------------
  //
  // Implementation

  template <class HAD>
  GeneratorFilter<HAD>::GeneratorFilter(ParameterSet const& ps) :
    EDFilter(),
    hadronizer_(ps),
    decayer_(0)
  {
    // TODO:
    // Put the list of types produced by the filters here.
    // The current design calls for:
    //   * GenRunInfoProduct
    //   * HepMCProduct
    //
    // other maybe added as needs be
    //
    
    if ( ps.exists("ExternalDecays") )
    {
       decayer_ = new gen::ExternalDecayDriver(ps.getParameter<ParameterSet>("ExternalDecays"));
    }

    produces<edm::HepMCProduct>();
    produces<GenEventInfoProduct>();
    produces<GenRunInfoProduct, edm::InRun>();
  }

  template <class HAD>
  GeneratorFilter<HAD>::~GeneratorFilter()
  { if ( decayer_ ) delete decayer_;}

  template <class HAD>
  bool
  GeneratorFilter<HAD>::filter(Event& ev, EventSetup const& /* es */)
  {
    
    if ( !hadronizer_.generatePartonsAndHadronize() ) return false;

    //  this is a "fake" stuff
    // in principle, decays are done as part of full event generation,
    // except for particles that are marked as to be kept stable
    // but we currently keep in it the design, because we might want
    // to use such feature for other applications
    //
    if ( !hadronizer_.decay() ) return false;
    
    HepMC::GenEvent* event = hadronizer_.getGenEvent();
    if( !event ) return false; 

    // The external decay driver is being added to the system,
    // it should be called here
    //
    if ( decayer_ ) 
    {
      event = decayer_->decay( event );
    }
    if ( !event ) return false;

    // check and perform if there're any unstable particles after 
    // running external decay packages
    //
    // fisrt of all, put back modified event tree (after external decay)
    //
    hadronizer_.resetEvent( event );
    //
    // now run residual decays
    //
    if ( !hadronizer_.residualDecay() ) return false;

    hadronizer_.finalizeEvent();

    event = hadronizer_.getGenEvent() ;
    if ( !event ) return false;

    //
    // tutto bene - finally, form up EDM products !
    //
    std::auto_ptr<HepMCProduct> bare_product(new HepMCProduct());
    bare_product->addHepMCData( event );
    ev.put(bare_product);

    std::auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(event));
    ev.put(genEventInfo);
    
    return true;
  }

  template <class HAD>
  void
  GeneratorFilter<HAD>::beginJob(EventSetup const&)
  { 
  
    if ( decayer_ ) decayer_->init() ;
    return;
  
  }
  
  template <class HAD>
  void
  GeneratorFilter<HAD>::endJob()
  { }

  template <class HAD>
  bool
  GeneratorFilter<HAD>::beginRun(Run &, EventSetup const&)
  {
    // Create the LHEGeneratorInfo product describing the run
    // conditions here, and insert it into the Run object.

    if ( !hadronizer_.initializeForInternalPartons() )
       throw edm::Exception(errors::Configuration) 
	<< "Failed to initialize hadronizer "
	<< hadronizer_.classname()
	<< " for internal parton generation\n";
    
    if ( decayer_ )
    {
       if ( !hadronizer_.declareStableParticles( decayer_->operatesOnParticles() ) )
          throw edm::Exception(errors::Configuration)
	  << "Failed to declare stable particles in hadronizer "
	  << hadronizer_.classname()
	  << "\n";
    }

    return true;
  }

  template <class HAD>
  bool
  GeneratorFilter<HAD>::endRun(Run& r, EventSetup const&)
  {
    // If relevant, record the integrated luminosity for this run
    // here.  To do so, we would need a standard function to invoke on
    // the contained hadronizer that would report the integrated
    // luminosity.

    hadronizer_.statistics();
    
    if ( decayer_ ) decayer_->statistics();
    
    std::auto_ptr<GenRunInfoProduct> griproduct(new GenRunInfoProduct(hadronizer_.getGenRunInfo()));
    r.put(griproduct);

    return true;
  }

  template <class HAD>
  bool
  GeneratorFilter<HAD>::beginLuminosityBlock(LuminosityBlock &, EventSetup const&)
  {
    return true;
  }

  template <class HAD>
  bool
  GeneratorFilter<HAD>::endLuminosityBlock(LuminosityBlock &, EventSetup const&)
  {
    // If relevant, record the integration luminosity of this
    // luminosity block here.  To do so, we would need a standard
    // function to invoke on the contained hadronizer that would
    // report the integrated luminosity.
    return true;
  }

  template <class HAD>
  void
  GeneratorFilter<HAD>::respondToOpenInputFile(FileBlock const& fb)
  { }

  template <class HAD>
  void
  GeneratorFilter<HAD>::respondToCloseInputFile(FileBlock const& fb)
  { }

  template <class HAD>
  void
  GeneratorFilter<HAD>::respondToOpenOutputFiles(FileBlock const& fb)
  { }

  template <class HAD>
  void
  GeneratorFilter<HAD>::respondToCloseOutputFiles(FileBlock const& fb)
  { }

}

#endif // gen_GeneratorFilter_h
