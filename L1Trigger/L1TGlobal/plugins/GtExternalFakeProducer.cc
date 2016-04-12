///
/// \class l1t::GtExternalFakeProducer
///
/// Description: Fill uGT external condition to allow testing stage 2 algos, e.g. Bptx
///
/// 
/// \author: D. Puigh OSU
///


// system include files
#include <boost/shared_ptr.hpp>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

//#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

using namespace std;
using namespace edm;


namespace l1t {

  //
  // class declaration
  //

  class GtExternalFakeProducer : public global::EDProducer<> {
  public:
    explicit GtExternalFakeProducer(const ParameterSet&);
    ~GtExternalFakeProducer();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

    // ----------member data ---------------------------
    // unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
    //boost::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
    //boost::shared_ptr<const FirmwareVersion> m_fwv;
    //boost::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

    // BX parameters
    int bxFirst_;
    int bxLast_;

    bool setBptxAND_;
    bool setBptxPlus_;
    bool setBptxMinus_;
    bool setBptxOR_;

  };

  //
  // constructors and destructor
  //
  GtExternalFakeProducer::GtExternalFakeProducer(const ParameterSet& iConfig) :
    bxFirst_ (iConfig.getParameter<int>("bxFirst")),
    bxLast_ (iConfig.getParameter<int>("bxLast")),
    setBptxAND_ (iConfig.getParameter<bool>("setBptxAND")),
    setBptxPlus_ (iConfig.getParameter<bool>("setBptxPlus")),
    setBptxMinus_ (iConfig.getParameter<bool>("setBptxMinus")),
    setBptxOR_ (iConfig.getParameter<bool>("setBptxOR")) 
  {
    // register what you produce
    produces<GlobalExtBlkBxCollection>();

    // Setup parameters

  }


  GtExternalFakeProducer::~GtExternalFakeProducer()
  {
  }



  //
  // member functions
  //

  // ------------ method called to produce the data ------------
  void
  GtExternalFakeProducer::produce(edm::StreamID, Event& iEvent, const EventSetup& iSetup) const
  {

    LogDebug("GtExternalFakeProducer") << "GtExternalFakeProducer::produce function called...\n";

    // Setup vectors
    GlobalExtBlk extCond_bx;

    //outputs
    std::auto_ptr<GlobalExtBlkBxCollection> extCond( new GlobalExtBlkBxCollection(0,bxFirst_,bxLast_));

    // Fill in some external conditions for testing
    if( setBptxAND_ ) extCond_bx.setExternalDecision(8,true);  //EXT_BPTX_plus_AND_minus.v0
    if( setBptxPlus_ ) extCond_bx.setExternalDecision(9,true);  //EXT_BPTX_plus.v0
    if( setBptxMinus_ ) extCond_bx.setExternalDecision(10,true); //EXT_BPTX_minus.v0
    if( setBptxOR_ ) extCond_bx.setExternalDecision(11,true); //EXT_BPTX_plus_OR_minus.v0

    // Fill Externals
    for( int iBx=bxFirst_; iBx<=bxLast_; iBx++ ){
      extCond->push_back(iBx, extCond_bx);
    }
   

    iEvent.put(extCond);

  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module ------------
  void
  GtExternalFakeProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    // simGtExtFakeProd
    edm::ParameterSetDescription desc;
    desc.add<bool>("setBptxMinus", true);
    desc.add<bool>("setBptxAND", true);
    desc.add<int>("bxFirst", -2);
    desc.add<bool>("setBptxOR", true);
    desc.add<int>("bxLast", 2);
    desc.add<bool>("setBptxPlus", true);
    descriptions.add("simGtExtFakeProd", desc);
  }

} // namespace

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::GtExternalFakeProducer);
