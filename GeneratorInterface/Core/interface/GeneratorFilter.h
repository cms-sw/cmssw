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
#include <vector>

#include "HepMC3/GenEvent.h"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CLHEP/Random/RandomEngine.h"

// #include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

//#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct3.h"

namespace edm {
  template <class HAD, class DEC>
  class GeneratorFilter : public one::EDFilter<EndRunProducer,
                                               BeginLuminosityBlockProducer,
                                               EndLuminosityBlockProducer,
                                               one::WatchLuminosityBlocks,
                                               one::SharedResources> {
  public:
    typedef HAD Hadronizer;
    typedef DEC Decayer;

    // The given ParameterSet will be passed to the contained
    // Hadronizer object.
    explicit GeneratorFilter(ParameterSet const& ps);

    ~GeneratorFilter() override;

    bool filter(Event& e, EventSetup const& es) override;
    void endRunProduce(Run&, EventSetup const&) override;
    void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;
    void beginLuminosityBlockProduce(LuminosityBlock&, EventSetup const&) override;
    void endLuminosityBlock(LuminosityBlock const&, EventSetup const&) override;
    void endLuminosityBlockProduce(LuminosityBlock&, EventSetup const&) override;
    void preallocThreads(unsigned int iThreads) override;

  private:
    // two-phase construction to allow specialization of the constructor
    void init(ParameterSet const& ps);

    Hadronizer hadronizer_;
    //gen::ExternalDecayDriver* decayer_;
    Decayer* decayer_ = nullptr;
    unsigned int nEventsInLumiBlock_ = 0;
    unsigned int nThreads_{1};
    bool initialized_ = false;
    unsigned int ivhepmc = 2;
  };

  //------------------------------------------------------------------------
  //
  // Implementation

  template <class HAD, class DEC>
  GeneratorFilter<HAD, DEC>::GeneratorFilter(ParameterSet const& ps) : hadronizer_(ps) {
    init(ps);
  }

  template <class HAD, class DEC>
  void GeneratorFilter<HAD, DEC>::init(ParameterSet const& ps) {
    // TODO:
    // Put the list of types produced by the filters here.
    // The current design calls for:
    //   * GenRunInfoProduct
    //   * HepMCProduct
    //
    // other maybe added as needs be
    //

    ivhepmc = hadronizer_.getVHepMC();

    std::vector<std::string> const& sharedResources = hadronizer_.sharedResources();
    for (auto const& resource : sharedResources) {
      usesResource(resource);
    }

    if (ps.exists("ExternalDecays")) {
      //decayer_ = new gen::ExternalDecayDriver(ps.getParameter<ParameterSet>("ExternalDecays"));
      ParameterSet ps1 = ps.getParameter<ParameterSet>("ExternalDecays");
      decayer_ = new Decayer(ps1, consumesCollector());

      std::vector<std::string> const& sharedResourcesDec = decayer_->sharedResources();
      for (auto const& resource : sharedResourcesDec) {
        usesResource(resource);
      }
    }

    // This handles the case where there are no shared resources, because you
    // have to declare something when the SharedResources template parameter was used.
    if (sharedResources.empty() && (!decayer_ || decayer_->sharedResources().empty())) {
      usesResource(edm::uniqueSharedResourceName());
    }

    if (ivhepmc == 2) {
      produces<edm::HepMCProduct>("unsmeared");
      produces<GenEventInfoProduct>();
    } else if (ivhepmc == 3) {
      //produces<edm::HepMC3Product>("unsmeared");
      //produces<GenEventInfoProduct3>();
    }
    produces<GenLumiInfoHeader, edm::Transition::BeginLuminosityBlock>();
    produces<GenLumiInfoProduct, edm::Transition::EndLuminosityBlock>();
    produces<GenRunInfoProduct, edm::Transition::EndRun>();
  }

  template <class HAD, class DEC>
  GeneratorFilter<HAD, DEC>::~GeneratorFilter() {
    if (decayer_)
      delete decayer_;
  }

  template <class HAD, class DEC>
  void GeneratorFilter<HAD, DEC>::preallocThreads(unsigned int iThreads) {
    nThreads_ = iThreads;
  }

  template <class HAD, class DEC>
  bool GeneratorFilter<HAD, DEC>::filter(Event& ev, EventSetup const& /* es */) {
    RandomEngineSentry<HAD> randomEngineSentry(&hadronizer_, ev.streamID());
    RandomEngineSentry<DEC> randomEngineSentryDecay(decayer_, ev.streamID());

    //added for selecting/filtering gen events, in the case of hadronizer+externalDecayer

    bool passEvtGenSelector = false;
    std::unique_ptr<HepMC::GenEvent> event(nullptr);
    std::unique_ptr<HepMC3::GenEvent> event3(nullptr);

    while (!passEvtGenSelector) {
      event.reset();
      event3.reset();
      hadronizer_.setEDMEvent(ev);

      if (!hadronizer_.generatePartonsAndHadronize())
        return false;

      //  this is "fake" stuff
      // in principle, decays are done as part of full event generation,
      // except for particles that are marked as to be kept stable
      // but we currently keep in it the design, because we might want
      // to use such feature for other applications
      //
      if (!hadronizer_.decay())
        return false;

      event = hadronizer_.getGenEvent();
      event3 = hadronizer_.getGenEvent3();
      if (ivhepmc == 2 && !event.get())
        return false;
      if (ivhepmc == 3 && !event3.get())
        return false;

      // The external decay driver is being added to the system,
      // it should be called here
      //
      if (decayer_) {  // handle only HepMC2 for the moment
        auto t = decayer_->decay(event.get());
        if (t != event.get()) {
          event.reset(t);
        }
      }
      if (ivhepmc == 2 && !event.get())
        return false;
      if (ivhepmc == 3 && !event3.get())
        return false;

      passEvtGenSelector = hadronizer_.select(event.get());
    }
    // check and perform if there're any unstable particles after
    // running external decay packages
    //
    // fisrt of all, put back modified event tree (after external decay)
    //
    if (ivhepmc == 2)
      hadronizer_.resetEvent(std::move(event));
    else if (ivhepmc == 3)
      hadronizer_.resetEvent3(std::move(event3));

    //
    // now run residual decays
    //
    if (!hadronizer_.residualDecay())
      return false;

    hadronizer_.finalizeEvent();

    event = hadronizer_.getGenEvent();
    event3 = hadronizer_.getGenEvent3();
    if (ivhepmc == 2) {  // HepMC
      if (!event.get())
        return false;
      event->set_event_number(ev.id().event());

    } else if (ivhepmc == 3) {  // HepMC3
      if (!event3.get())
        return false;
      event3->set_event_number(ev.id().event());
    }

    //
    // tutto bene - finally, form up EDM products !
    //
    if (ivhepmc == 2) {  // HepMC
      auto genEventInfo = hadronizer_.getGenEventInfo();
      if (!genEventInfo.get()) {
        // create GenEventInfoProduct from HepMC event in case hadronizer didn't provide one
        genEventInfo.reset(new GenEventInfoProduct(event.get()));
      }

      ev.put(std::move(genEventInfo));

      std::unique_ptr<HepMCProduct> bare_product(new HepMCProduct());
      bare_product->addHepMCData(event.release());
      ev.put(std::move(bare_product), "unsmeared");

    } else if (ivhepmc == 3) {  // HepMC3
      auto genEventInfo3 = hadronizer_.getGenEventInfo3();
      if (!genEventInfo3.get()) {
        // create GenEventInfoProduct3 from HepMC3 event in case hadronizer didn't provide one
        genEventInfo3.reset(new GenEventInfoProduct3(event3.get()));
      }

      //ev.put(std::move(genEventInfo3));

      //std::unique_ptr<HepMCProduct3> bare_product(new HepMCProduct3());
      //bare_product->addHepMCData(event3.release());
      //ev.put(std::move(bare_product), "unsmeared");
    }

    nEventsInLumiBlock_++;
    return true;
  }

  template <class HAD, class DEC>
  void GeneratorFilter<HAD, DEC>::endRunProduce(Run& r, EventSetup const&) {
    // If relevant, record the integrated luminosity for this run
    // here.  To do so, we would need a standard function to invoke on
    // the contained hadronizer that would report the integrated
    // luminosity.

    if (initialized_) {
      hadronizer_.statistics();

      if (decayer_)
        decayer_->statistics();
    }

    std::unique_ptr<GenRunInfoProduct> griproduct(new GenRunInfoProduct(hadronizer_.getGenRunInfo()));
    r.put(std::move(griproduct));
  }

  template <class HAD, class DEC>
  void GeneratorFilter<HAD, DEC>::beginLuminosityBlock(LuminosityBlock const& lumi, EventSetup const& es) {}

  template <class HAD, class DEC>
  void GeneratorFilter<HAD, DEC>::beginLuminosityBlockProduce(LuminosityBlock& lumi, EventSetup const& es) {
    nEventsInLumiBlock_ = 0;
    RandomEngineSentry<HAD> randomEngineSentry(&hadronizer_, lumi.index());
    RandomEngineSentry<DEC> randomEngineSentryDecay(decayer_, lumi.index());

    hadronizer_.randomizeIndex(lumi, randomEngineSentry.randomEngine());
    hadronizer_.generateLHE(lumi, randomEngineSentry.randomEngine(), nThreads_);

    if (!hadronizer_.readSettings(0))
      throw edm::Exception(errors::Configuration)
          << "Failed to read settings for the hadronizer " << hadronizer_.classname() << " \n";

    if (decayer_) {
      decayer_->init(es);
      if (!hadronizer_.declareStableParticles(decayer_->operatesOnParticles()))
        throw edm::Exception(errors::Configuration)
            << "Failed to declare stable particles in hadronizer " << hadronizer_.classname() << "\n";
      if (!hadronizer_.declareSpecialSettings(decayer_->specialSettings()))
        throw edm::Exception(errors::Configuration)
            << "Failed to declare special settings in hadronizer " << hadronizer_.classname() << "\n";
    }

    if (!hadronizer_.initializeForInternalPartons())
      throw edm::Exception(errors::Configuration)
          << "Failed to initialize hadronizer " << hadronizer_.classname() << " for internal parton generation\n";

    std::unique_ptr<GenLumiInfoHeader> genLumiInfoHeader(hadronizer_.getGenLumiInfoHeader());
    lumi.put(std::move(genLumiInfoHeader));
    initialized_ = true;
  }

  template <class HAD, class DEC>
  void GeneratorFilter<HAD, DEC>::endLuminosityBlock(LuminosityBlock const&, EventSetup const&) {
    hadronizer_.cleanLHE();
  }

  template <class HAD, class DEC>
  void GeneratorFilter<HAD, DEC>::endLuminosityBlockProduce(LuminosityBlock& lumi, EventSetup const&) {
    hadronizer_.statistics();
    if (decayer_)
      decayer_->statistics();

    GenRunInfoProduct genRunInfo = GenRunInfoProduct(hadronizer_.getGenRunInfo());
    std::vector<GenLumiInfoProduct::ProcessInfo> GenLumiProcess;
    const GenRunInfoProduct::XSec& xsec = genRunInfo.internalXSec();
    GenLumiInfoProduct::ProcessInfo temp;
    temp.setProcess(0);
    temp.setLheXSec(xsec.value(), xsec.error());  // Pythia gives error of -1
    temp.setNPassPos(nEventsInLumiBlock_);
    temp.setNPassNeg(0);
    temp.setNTotalPos(nEventsInLumiBlock_);
    temp.setNTotalNeg(0);
    temp.setTried(nEventsInLumiBlock_, nEventsInLumiBlock_, nEventsInLumiBlock_);
    temp.setSelected(nEventsInLumiBlock_, nEventsInLumiBlock_, nEventsInLumiBlock_);
    temp.setKilled(nEventsInLumiBlock_, nEventsInLumiBlock_, nEventsInLumiBlock_);
    temp.setAccepted(0, -1, -1);
    temp.setAcceptedBr(0, -1, -1);
    GenLumiProcess.push_back(temp);

    std::unique_ptr<GenLumiInfoProduct> genLumiInfo(new GenLumiInfoProduct());
    genLumiInfo->setHEPIDWTUP(-1);
    genLumiInfo->setProcessInfo(GenLumiProcess);

    lumi.put(std::move(genLumiInfo));

    nEventsInLumiBlock_ = 0;
  }
}  // namespace edm

#endif  // gen_GeneratorFilter_h
