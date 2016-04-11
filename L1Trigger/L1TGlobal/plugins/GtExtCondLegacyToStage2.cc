///
/// \class l1t::GtExtCondLegacyToStage2
///
/// Description: Fill uGT external condition (stage2) with legacy information from data
///
/// 
/// \author: D. Puigh OSU
/// \revised: V. Rekovic


// system include files
#include <boost/shared_ptr.hpp>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
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

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

using namespace std;
using namespace edm;


namespace l1t {

  //
  // class declaration
  //

  class GtExtCondLegacyToStage2 : public global::EDProducer<> {
  public:
    explicit GtExtCondLegacyToStage2(const ParameterSet&);
    ~GtExtCondLegacyToStage2();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

    // ----------member data ---------------------------
    //unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
    //boost::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
    //boost::shared_ptr<const FirmwareVersion> m_fwv;
    //boost::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

    // BX parameters
    int bxFirst_;
    int bxLast_;

    // Readout Record token
    edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> gtReadoutRecordToken;

  };

  //
  // constructors and destructor
  //
  GtExtCondLegacyToStage2::GtExtCondLegacyToStage2(const ParameterSet& iConfig) :
    bxFirst_ (iConfig.getParameter<int>("bxFirst")),
    bxLast_ (iConfig.getParameter<int>("bxLast")),
    gtReadoutRecordToken (consumes <L1GlobalTriggerReadoutRecord> (iConfig.getParameter<edm::InputTag>("LegacyGtReadoutRecord")))
  {
    // register what you produce
    produces<GlobalExtBlkBxCollection>();

  }


  GtExtCondLegacyToStage2::~GtExtCondLegacyToStage2()
  {
  }



  //
  // member functions
  //

  // ------------ method called to produce the data ------------
  void
  GtExtCondLegacyToStage2::produce(edm::StreamID, Event& iEvent, const EventSetup& iSetup) const
  {

    LogDebug("GtExtCondLegacyToStage2") << "GtExtCondLegacyToStage2::produce function called...\n";

    edm::Handle<L1GlobalTriggerReadoutRecord> gtReadoutRecord;
    iEvent.getByToken(gtReadoutRecordToken, gtReadoutRecord);


    // Setup vectors
    GlobalExtBlk extCond_bx_m2;
    GlobalExtBlk extCond_bx_m1;
    GlobalExtBlk extCond_bx_0;
    GlobalExtBlk extCond_bx_p1;
    GlobalExtBlk extCond_bx_p2;

    if( gtReadoutRecord.isValid() ){
      // L1GlobalTriggerReadoutRecord const & l1tResults = * gtReadoutRecord;

      // // select PSB#9 and bunch crossing 0
      // const L1GtPsbWord & psb = l1tResults.gtPsbWord(0xbb09, 0);

      // // the four 16-bit words psb.bData(1), psb.aData(1), psb.bData(0) and psb.aData(0) yield
      // // (in this sequence) the 64 technical trigger bits from most significant to least significant bit
      // uint64_t psbTriggerWord = ( (uint64_t) psb.bData(1) << 48) |
      // 	((uint64_t) psb.aData(1) << 32) |
      // 	((uint64_t) psb.bData(0) << 16) |
      // 	((uint64_t) psb.aData(0));

      // std::cout << "psbTriggerWord = " << psbTriggerWord << std::endl;
      // //

      for( int ibx = 0; ibx < 5; ibx++ ){

	int useBx = ibx - 2;
	if( useBx<bxFirst_ || useBx>bxLast_ ) continue;

	//std::cout << "  BX = " << ibx - 2 << std::endl;
	
	// L1 technical
	const TechnicalTriggerWord& gtTTWord = gtReadoutRecord->technicalTriggerWord(useBx);
	int tbitNumber = 0;
	TechnicalTriggerWord::const_iterator GTtbitItr;

        std::vector<bool> pass_externs(4, false); //BptxAND, BptxPlus, BptxMinus, BptxOR

	for(GTtbitItr = gtTTWord.begin(); GTtbitItr != gtTTWord.end(); GTtbitItr++) {

	  int pass_l1t_tech = 0;

	  if (*GTtbitItr) pass_l1t_tech = 1;

	  if( pass_l1t_tech==1 ){

           pass_externs[tbitNumber] = true;

	  }

	  tbitNumber++;

          if(tbitNumber>3) break;
	}

	if( useBx==-2 ){

         for (unsigned int i=0;i<4;i++) extCond_bx_m2.setExternalDecision(8+i,pass_externs[tbitNumber]);

	}
	else if( useBx==-1 ){

         for (unsigned int i=0;i<4;i++) extCond_bx_m1.setExternalDecision(8+i,pass_externs[tbitNumber]);

	}
	else if( useBx==0 ){

         for (unsigned int i=0;i<4;i++) extCond_bx_0.setExternalDecision(8+i,pass_externs[tbitNumber]);

	}
	else if( useBx==1 ){

         for (unsigned int i=0;i<4;i++) extCond_bx_p1.setExternalDecision(8+i,pass_externs[tbitNumber]);

	}
	else if( useBx==2 ){

         for (unsigned int i=0;i<4;i++) extCond_bx_p2.setExternalDecision(8+i,pass_externs[tbitNumber]);

	}
      }
    }
    else {

      LogWarning("MissingProduct") << "Input L1GlobalTriggerReadoutRecord collection not found\n";

    }

    //outputs
    std::auto_ptr<GlobalExtBlkBxCollection> extCond( new GlobalExtBlkBxCollection(0,bxFirst_,bxLast_));

    // Fill Externals
    if( -2>=bxFirst_ && -2<=bxLast_ ) extCond->push_back(-2, extCond_bx_m2);
    if( -1>=bxFirst_ && -1<=bxLast_ ) extCond->push_back(-1, extCond_bx_m1);
    if(  0>=bxFirst_ &&  0<=bxLast_ ) extCond->push_back(0,  extCond_bx_0);
    if(  1>=bxFirst_ &&  1<=bxLast_ ) extCond->push_back(1,  extCond_bx_p1);
    if(  2>=bxFirst_ &&  2<=bxLast_ ) extCond->push_back(2,  extCond_bx_p2); 
   

    iEvent.put(extCond);

  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module ------------
  void
  GtExtCondLegacyToStage2::fillDescriptions(ConfigurationDescriptions& descriptions) {
    // l1GtExtCondLegacyToStage2
    edm::ParameterSetDescription desc;
    desc.add<int>("bxFirst", -2);
    desc.add<int>("bxLast", 2);
    desc.add<edm::InputTag>("LegacyGtReadoutRecord", edm::InputTag("unpackLegacyGtDigis"));
    descriptions.add("l1GtExtCondLegacyToStage2", desc);
  }

} // namespace

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::GtExtCondLegacyToStage2);
