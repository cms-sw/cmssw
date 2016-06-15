///
/// \class l1t::L1TExtCondProducer
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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "L1Trigger/L1TGlobal/plugins/TriggerMenuParser.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

//#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

using namespace std;
using namespace edm;
using namespace l1t;

  //
  // class declaration
  //

  class L1TExtCondProducer : public stream::EDProducer<> {
  public:
    explicit L1TExtCondProducer(const ParameterSet&);
    ~L1TExtCondProducer();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&) override;

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

    unsigned long long m_l1GtMenuCacheID;
    std::map<std::string, unsigned int> m_extBitMap;
  };

  //
  // constructors and destructor
  //
  L1TExtCondProducer::L1TExtCondProducer(const ParameterSet& iConfig) :
    bxFirst_ (iConfig.getParameter<int>("bxFirst")),
    bxLast_ (iConfig.getParameter<int>("bxLast")),
    setBptxAND_ (iConfig.getParameter<bool>("setBptxAND")),
    setBptxPlus_ (iConfig.getParameter<bool>("setBptxPlus")),
    setBptxMinus_ (iConfig.getParameter<bool>("setBptxMinus")),
    setBptxOR_ (iConfig.getParameter<bool>("setBptxOR")) 
  {
    // register what you produce
    produces<GlobalExtBlkBxCollection>();

    // Initialize parameters
    m_l1GtMenuCacheID = 0ULL;
  }


  L1TExtCondProducer::~L1TExtCondProducer()
  {
  }



  //
  // member functions
  //

  // ------------ method called to produce the data ------------
  void
  L1TExtCondProducer::produce(Event& iEvent, const EventSetup& iSetup)
  {

    LogDebug("L1TExtCondProducer") << "L1TExtCondProducer::produce function called...\n";

    // get / update the trigger menu from the EventSetup
    // local cache & check on cacheIdentifier
    unsigned long long l1GtMenuCacheID = iSetup.get<L1TUtmTriggerMenuRcd>().cacheIdentifier();
    
    if (m_l1GtMenuCacheID != l1GtMenuCacheID) {

        edm::ESHandle<L1TUtmTriggerMenu> l1GtMenu;
        iSetup.get< L1TUtmTriggerMenuRcd>().get(l1GtMenu) ;
        const L1TUtmTriggerMenu* utml1GtMenu =  l1GtMenu.product();
        
	// Instantiate Parser
        TriggerMenuParser gtParser = TriggerMenuParser();   

	std::map<std::string, unsigned int> extBitMap = gtParser.getExternalSignals(utml1GtMenu);
	
	m_l1GtMenuCacheID = l1GtMenuCacheID;
	m_extBitMap = extBitMap;
    }

    // Setup vectors
    GlobalExtBlk extCond_bx;

    //outputs
    std::auto_ptr<GlobalExtBlkBxCollection> extCond( new GlobalExtBlkBxCollection(0,bxFirst_,bxLast_));

    bool foundBptxAND = ( m_extBitMap.find("BPTX_plus_AND_minus.v0")!=m_extBitMap.end() );
    bool foundBptxPlus = ( m_extBitMap.find("BPTX_plus.v0")!=m_extBitMap.end() );
    bool foundBptxMinus = ( m_extBitMap.find("BPTX_minus.v0")!=m_extBitMap.end() );
    bool foundBptxOR = ( m_extBitMap.find("BPTX_plus_OR_minus.v0")!=m_extBitMap.end() );

    // Fill in some external conditions for testing
    if( setBptxAND_ && foundBptxAND ) extCond_bx.setExternalDecision(m_extBitMap["BPTX_plus_AND_minus.v0"],true);
    if( setBptxPlus_ && foundBptxPlus ) extCond_bx.setExternalDecision(m_extBitMap["BPTX_plus.v0"],true);
    if( setBptxMinus_ && foundBptxMinus ) extCond_bx.setExternalDecision(m_extBitMap["BPTX_minus.v0"],true);
    if( setBptxOR_ && foundBptxOR ) extCond_bx.setExternalDecision(m_extBitMap["BPTX_plus_OR_minus.v0"],true);

    // Fill Externals
    for( int iBx=bxFirst_; iBx<=bxLast_; iBx++ ){
      extCond->push_back(iBx, extCond_bx);
    }
   

    iEvent.put(extCond);

  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module ------------
  void
  L1TExtCondProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
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



//define this as a plug-in
DEFINE_FWK_MODULE(L1TExtCondProducer);
