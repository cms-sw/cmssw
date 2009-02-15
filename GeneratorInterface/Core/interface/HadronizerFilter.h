// -*- C++ -*-
//
//

// class template HadronizerFilter<HAD> provides an EDFilter which uses
// the hadronizer type HAD to read in external partons and hadronize them, 
// and decay the resulting particles, in the CMS framework.

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#ifndef gen_HadronizerFilter_h
#define gen_HadronizerFilter_h

// LHE Run
// #include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h" // this comes through LHERunInfo
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"


// LHE Event
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

namespace edm
{
  template <class HAD> class HadronizerFilter : public EDFilter
  {
  public:
    typedef HAD Hadronizer;

    // The given ParameterSet will be passed to the contained
    // Hadronizer object.
    explicit HadronizerFilter(ParameterSet const& ps);

    virtual ~HadronizerFilter();

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


  };

  //------------------------------------------------------------------------
  //
  // Implementation

  template <class HAD>
  HadronizerFilter<HAD>::HadronizerFilter(ParameterSet const& ps) :
    EDFilter(),
    hadronizer_(ps)
  {
    // TODO:
    // Put the list of types produced by the filters here.
    // The current design calls for:
    //   * LHEGeneratorInfo
    //   * LHEEvent
    //   * HepMCProduct
    // But I can not find the LHEGeneratorInfo class; it might need to
    // be invented.

    // Commented out because of compilation failures; inability to
    // find headers.
    //
    produces<edm::HepMCProduct>();
    produces<GenRunInfoProduct, edm::InRun>();
  }

  template <class HAD>
  HadronizerFilter<HAD>::~HadronizerFilter()
  { }

  template <class HAD>
  bool
  HadronizerFilter<HAD>::filter(Event& ev, EventSetup const& /* es */)
  {
    
    std::auto_ptr<HepMCProduct> bare_product(new HepMCProduct());
        
    
    // get LHE stuff and pass to hadronizer !
    //
    edm::Handle<LHEEventProduct> product;
    ev.getByLabel("source", product);
    
    hadronizer_.setLHEEventProd( (LHEEventProduct*)(product.product()) ) ;
    
    // hadronizer_.generatePartons();
    if ( !hadronizer_.hadronize() ) return false ;

    // When the external decay driver is added to the system, it
    // should be called here.

    if ( !hadronizer_.decay() ) return false;
    
    if( hadronizer_.getGenEvent() ) 
    {
       bare_product->addHepMCData( hadronizer_.getGenEvent() );
       ev.put(bare_product);
    }
    else 
    { 
       return false ; 
    }
       
    return true;
  }

  template <class HAD>
  void
  HadronizerFilter<HAD>::beginJob(EventSetup const&)
  { 
    
    // do things that's common through the job, such as
    // attach external decay packages, etc.
    
/*
    if (! hadronizer_.declareStableParticles())
      throw edm::Exception(errors::Configuration)
	<< "Failed to declare stable particles in hadronizer "
	<< hadronizer_.classname()
	<< "\n";
*/
  }
  
  template <class HAD>
  void
  HadronizerFilter<HAD>::endJob()
  { }

  template <class HAD>
  bool
  HadronizerFilter<HAD>::beginRun(Run& run, EventSetup const&)
  {
    
    // this is run-specific
    
    // get LHE stuff and pass to hadronizer !

    edm::Handle<LHERunInfoProduct> product;
    run.getByLabel("source", product);
            
    hadronizer_.setLHERunInfo( new lhef::LHERunInfo(*product) ) ;
   
    if (! hadronizer_.initializeForExternalPartons())
      throw edm::Exception(errors::Configuration) 
	<< "Failed to initialize hadronizer "
	<< hadronizer_.classname()
	<< " for internal parton generation\n";


    // Create the LHEGeneratorInfo product describing the run
    // conditions here, and insert it into the Run object.
    
    return true;
  
  }

  template <class HAD>
  bool
  HadronizerFilter<HAD>::endRun(Run& r, EventSetup const&)
  {
    // If relevant, record the integrated luminosity for this run
    // here.  To do so, we would need a standard function to invoke on
    // the contained hadronizer that would report the integrated
    // luminosity.
    
    hadronizer_.statistics();
    
    std::auto_ptr<GenRunInfoProduct> griproduct(new GenRunInfoProduct(hadronizer_.getGenRunInfo()));
    r.put(griproduct);
    
    return true;
  }

  template <class HAD>
  bool
  HadronizerFilter<HAD>::beginLuminosityBlock(LuminosityBlock &, EventSetup const&)
  {
    return true;
  }

  template <class HAD>
  bool
  HadronizerFilter<HAD>::endLuminosityBlock(LuminosityBlock &, EventSetup const&)
  {
    // If relevant, record the integration luminosity of this
    // luminosity block here.  To do so, we would need a standard
    // function to invoke on the contained hadronizer that would
    // report the integrated luminosity.
    return true;
  }

  template <class HAD>
  void
  HadronizerFilter<HAD>::respondToOpenInputFile(FileBlock const& fb)
  { }

  template <class HAD>
  void
  HadronizerFilter<HAD>::respondToCloseInputFile(FileBlock const& fb)
  { }

  template <class HAD>
  void
  HadronizerFilter<HAD>::respondToOpenOutputFiles(FileBlock const& fb)
  { }

  template <class HAD>
  void
  HadronizerFilter<HAD>::respondToCloseOutputFiles(FileBlock const& fb)
  { }

}

#endif // gen_HadronizerFilter_h
