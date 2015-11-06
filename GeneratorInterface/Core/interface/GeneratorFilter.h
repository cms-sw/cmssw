// -*- C++ -*-
//
//

// class template GeneratorFilter<HAD> provides an EDFilter which uses
// the hadronizer type HAD to generate partons, hadronize them, and
// decay the resulting particles, in the CMS framework.

#ifndef gen_GeneratorFilter_h
#define gen_GeneratorFilter_h

#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

// #include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

//#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

namespace edm
{
  template <class HAD, class DEC> class GeneratorFilter : public EDFilter
  {
  public:
    typedef HAD Hadronizer;
    typedef DEC Decayer;

    // The given ParameterSet will be passed to the contained
    // Hadronizer object.
    explicit GeneratorFilter(ParameterSet const& ps);

    virtual ~GeneratorFilter();

    virtual bool filter(Event& e, EventSetup const& es);
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
    Hadronizer            hadronizer_;
    //gen::ExternalDecayDriver* decayer_;
    Decayer*              decayer_;
    unsigned int          nEventsInLumiBlock_;
  };

  //------------------------------------------------------------------------
  //
  // Implementation

  template <class HAD, class DEC>
  GeneratorFilter<HAD,DEC>::GeneratorFilter(ParameterSet const& ps) :
    EDFilter(),
    hadronizer_(ps),
    decayer_(0),
    nEventsInLumiBlock_(0)
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
       //decayer_ = new gen::ExternalDecayDriver(ps.getParameter<ParameterSet>("ExternalDecays"));
       ParameterSet ps1 = ps.getParameter<ParameterSet>("ExternalDecays");
       decayer_ = new Decayer(ps1);
    }
    
    produces<edm::HepMCProduct>();
    produces<GenEventInfoProduct>();
    produces<GenLumiInfoProduct, edm::InLumi>();
    produces<GenRunInfoProduct, edm::InRun>();
 
  }

  template <class HAD, class DEC>
  GeneratorFilter<HAD, DEC>::~GeneratorFilter()
  { if ( decayer_ ) delete decayer_;}

  template <class HAD, class DEC>
  bool
  GeneratorFilter<HAD, DEC>::filter(Event& ev, EventSetup const& /* es */)
  {
    //added for selecting/filtering gen events, in the case of hadronizer+externalDecayer
      
    bool passEvtGenSelector = false;
    std::auto_ptr<HepMC::GenEvent> event(0);
   
    while(!passEvtGenSelector)
      {
	event.reset();
	hadronizer_.setEDMEvent(ev);
	
	if ( !hadronizer_.generatePartonsAndHadronize() ) return false;
	
	//  this is "fake" stuff
	// in principle, decays are done as part of full event generation,
	// except for particles that are marked as to be kept stable
	// but we currently keep in it the design, because we might want
	// to use such feature for other applications
	//
	if ( !hadronizer_.decay() ) return false;
	
	event = std::auto_ptr<HepMC::GenEvent>(hadronizer_.getGenEvent());
	if ( !event.get() ) return false; 
	
	// The external decay driver is being added to the system,
	// it should be called here
	//
	if ( decayer_ ) 
	  {
	    event.reset( decayer_->decay( event.get() ) );
	  }
	if ( !event.get() ) return false;
	
	passEvtGenSelector = hadronizer_.select( event.get() );
	
      }
    // check and perform if there're any unstable particles after 
    // running external decay packages
    //
    // fisrt of all, put back modified event tree (after external decay)
    //
    hadronizer_.resetEvent( event.release() );
	
    //
    // now run residual decays
    //
    if ( !hadronizer_.residualDecay() ) return false;
    	
    hadronizer_.finalizeEvent();
    
    event.reset( hadronizer_.getGenEvent() );
    if ( !event.get() ) return false;
    
    event->set_event_number( ev.id().event() );
    
    //
    // tutto bene - finally, form up EDM products !
    //
    std::auto_ptr<GenEventInfoProduct> genEventInfo(hadronizer_.getGenEventInfo());
    if (!genEventInfo.get())
      { 
	// create GenEventInfoProduct from HepMC event in case hadronizer didn't provide one
	genEventInfo.reset(new GenEventInfoProduct(event.get()));
      }
    ev.put(genEventInfo);
   
    std::auto_ptr<HepMCProduct> bare_product(new HepMCProduct());
    bare_product->addHepMCData( event.release() );
    ev.put(bare_product);
    nEventsInLumiBlock_ ++;
    return true;
  }

  template <class HAD, class DEC>
  void
  GeneratorFilter<HAD, DEC>::endJob()
  { }

  template <class HAD, class DEC>
  bool
  GeneratorFilter<HAD, DEC>::beginRun( Run &, EventSetup const& es )
  {
    
    return true;

  }

  template <class HAD, class DEC>
  bool
  GeneratorFilter<HAD, DEC>::endRun( Run& r, EventSetup const& )
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

  template <class HAD, class DEC>
  bool
  GeneratorFilter<HAD, DEC>::beginLuminosityBlock( LuminosityBlock &, EventSetup const& es )
  {
    nEventsInLumiBlock_ = 0;
    
    if ( !hadronizer_.readSettings(0) )
       throw edm::Exception(errors::Configuration) 
	 << "Failed to read settings for the hadronizer "
	 << hadronizer_.classname() << " \n";
        
    if ( decayer_ )
    {
       decayer_->init(es);
       if ( !hadronizer_.declareStableParticles( decayer_->operatesOnParticles() ) )
          throw edm::Exception(errors::Configuration)
            << "Failed to declare stable particles in hadronizer "
            << hadronizer_.classname()
            << "\n";
       if ( !hadronizer_.declareSpecialSettings( decayer_->specialSettings() ) )
          throw edm::Exception(errors::Configuration)
            << "Failed to declare special settings in hadronizer "
            << hadronizer_.classname()
            << "\n";
    }

    if ( !hadronizer_.initializeForInternalPartons() )
       throw edm::Exception(errors::Configuration) 
	 << "Failed to initialize hadronizer "
	 << hadronizer_.classname()
	 << " for internal parton generation\n";

    return true;

  }

  template <class HAD, class DEC>
  bool
  GeneratorFilter<HAD, DEC>::endLuminosityBlock(LuminosityBlock &lumi, EventSetup const&)
  {
    
    hadronizer_.statistics();    
    if ( decayer_ ) decayer_->statistics();

    GenRunInfoProduct genRunInfo = GenRunInfoProduct(hadronizer_.getGenRunInfo());
    std::vector<GenLumiInfoProduct::ProcessInfo> GenLumiProcess;
    GenRunInfoProduct::XSec xsec = genRunInfo.internalXSec();
    GenLumiInfoProduct::ProcessInfo temp;      
    temp.setProcess(0);
    temp.setLheXSec(xsec.value(), xsec.error()); // Pythia gives error of -1
    temp.setNPassPos(nEventsInLumiBlock_);
    temp.setNPassNeg(0);
    temp.setNTotalPos(nEventsInLumiBlock_);
    temp.setNTotalNeg(0);
    temp.setTried(nEventsInLumiBlock_, nEventsInLumiBlock_, nEventsInLumiBlock_);
    temp.setSelected(nEventsInLumiBlock_, nEventsInLumiBlock_, nEventsInLumiBlock_);
    temp.setKilled(nEventsInLumiBlock_, nEventsInLumiBlock_, nEventsInLumiBlock_);
    temp.setAccepted(0,-1,-1);
    temp.setAcceptedBr(0,-1,-1);
    GenLumiProcess.push_back(temp);

    std::auto_ptr<GenLumiInfoProduct> genLumiInfo(new GenLumiInfoProduct());
    genLumiInfo->setHEPIDWTUP(-1);
    genLumiInfo->setProcessInfo( GenLumiProcess );
    lumi.put(genLumiInfo);

    nEventsInLumiBlock_ = 0;
    
    return true;
  }

  template <class HAD, class DEC>
  void
  GeneratorFilter<HAD, DEC>::respondToOpenInputFile(FileBlock const& fb)
  { }

  template <class HAD, class DEC>
  void
  GeneratorFilter<HAD, DEC>::respondToCloseInputFile(FileBlock const& fb)
  { }

  template <class HAD, class DEC>
  void
  GeneratorFilter<HAD, DEC>::respondToOpenOutputFiles(FileBlock const& fb)
  { }

  template <class HAD, class DEC>
  void
  GeneratorFilter<HAD, DEC>::respondToCloseOutputFiles(FileBlock const& fb)
  { }

}

#endif // gen_GeneratorFilter_h
