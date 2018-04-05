// Original Author:  Anne-Marie Magnan
//         Created:  2010/01/21
//

#include <sstream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>
#include <vector>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

//for the zero suppression algorithm(s)
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"

#include "DQM/SiStripMonitorHardware/interface/SiStripFEDEmulator.h"
#include "DQM/SiStripMonitorHardware/interface/SiStripSpyUtilities.h"

using edm::LogError;
using edm::LogInfo;
using edm::LogWarning;

namespace sistrip
{

  //
  // Class declaration
  //

  class FEDEmulatorModule : public edm::EDProducer
  {
  public:

    explicit FEDEmulatorModule(const edm::ParameterSet&);
    ~FEDEmulatorModule() override;

  private:

    void produce(edm::Event&, const edm::EventSetup&) override;
    //virtual void endJob();

    //tag of spydata collection
    edm::InputTag spyReorderedDigisTag_;
    edm::InputTag spyVirginRawDigisTag_;
    edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > spyReorderedDigisToken_;
    edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > spyVirginRawDigisToken_;


    //by fedIndex or module detid
    bool byModule_;

    sistrip::FEDEmulator fedEmulator_;

    static const char* messageLabel_;
  
    std::unique_ptr<SiStripRawProcessingAlgorithms> algorithms_; //!< object for zero-suppression

    //utilities for cabling etc...
    SpyUtilities utility_;

  };

}//~sistrip

namespace sistrip {

  //
  // Constructors and destructor
  //
  const char* FEDEmulatorModule::messageLabel_ = "SiStripFEDEmulatorModule";


  FEDEmulatorModule::FEDEmulatorModule(const edm::ParameterSet& iConfig): 
    spyReorderedDigisTag_(iConfig.getParameter<edm::InputTag>("SpyReorderedDigisTag")),
    spyVirginRawDigisTag_(iConfig.getParameter<edm::InputTag>("SpyVirginRawDigisTag")),
    byModule_(iConfig.getParameter<bool>("ByModule")),
    algorithms_(SiStripRawProcessingFactory::create(iConfig.getParameter<edm::ParameterSet>("Algorithms"))) 
  {

    spyReorderedDigisToken_ = consumes<edm::DetSetVector<SiStripRawDigi> >(spyReorderedDigisTag_);
    spyVirginRawDigisToken_ = consumes<edm::DetSetVector<SiStripRawDigi> >(spyVirginRawDigisTag_);

    fedEmulator_.initialise(byModule_);

    if (!byModule_) { //if not by module
      //the medians will be produced by fed id/channel
      produces<std::map<uint32_t,std::vector<uint32_t> > >("Medians");
      produces<edm::DetSetVector<SiStripRawDigi> >("PedestalsOrdered");
      produces<edm::DetSetVector<SiStripProcessedRawDigi> >("NoisesOrdered");
      produces<edm::DetSetVector<SiStripRawDigi> >("PedSubtrDigisOrdered");
      produces<edm::DetSetVector<SiStripRawDigi> >("CMSubtrDigisOrdered");
    }
    else { //by module
      produces<edm::DetSetVector<SiStripRawDigi> >("ModulePedestals");
      produces<edm::DetSetVector<SiStripProcessedRawDigi> >("ModuleNoises");
      produces<edm::DetSetVector<SiStripRawDigi> >("PedSubtrModuleDigis");
      produces<std::map<uint32_t,std::vector<uint32_t> > >("ModuleMedians");
      produces<edm::DetSetVector<SiStripRawDigi> >("CMSubtrModuleDigis");
      produces<edm::DetSetVector<SiStripDigi> >("ZSModuleDigis");
    }//end of by module check

  }//end of FEDEmulatorModule constructor

  FEDEmulatorModule::~FEDEmulatorModule()
  {
  }

  // ------------ method called to for each event  ------------
  void
  FEDEmulatorModule::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    //update cabling and pedestals
    const SiStripFedCabling* lCabling = utility_.getCabling( iSetup );
    edm::ESHandle<SiStripPedestals> lPedsHandle = utility_.getPedestalHandle(iSetup);
    edm::ESHandle<SiStripNoises> lNoiseHandle = utility_.getNoiseHandle(iSetup);
  
    //initialise the algorithms object for the zero suppression
    algorithms_->initialize(iSetup);

    //retrieve the digis
    edm::Handle<edm::DetSetVector<SiStripRawDigi> > lDigisHandle;
    try { //to get the digis from the event
      if (!byModule_) {
	iEvent.getByToken(spyReorderedDigisToken_, lDigisHandle);
      }
      else { //digis supplied by module
	iEvent.getByToken(spyVirginRawDigisToken_, lDigisHandle);
      }//end of by module check
    } catch (const cms::Exception& e) {
      std::cout << e.what() ;
      return;
    } //end of get digis try

    const edm::DetSetVector<SiStripRawDigi> * lInputDigis = lDigisHandle.product();

    unsigned int lNDigis = lInputDigis->size();

    //define output containers
    //reserve space, will push_back elements
    std::vector<edm::DetSetVector<SiStripRawDigi>::detset > pedsData;
    pedsData.reserve(lNDigis);
    std::vector<edm::DetSetVector<SiStripProcessedRawDigi>::detset > noiseData;
    noiseData.reserve(lNDigis);
    std::vector<edm::DetSetVector<SiStripRawDigi>::detset > pedSubtrData;
    pedSubtrData.reserve(lNDigis);
    std::vector<edm::DetSetVector<SiStripRawDigi>::detset > cmSubtrData;
    cmSubtrData.reserve(lNDigis);
    //zero suppressed contained - no fixed size (could be empty) so no need to reserve elements.
    std::vector< edm::DetSet<SiStripDigi> > zsData;
    
    //this is a map: no reserve/resize
    std::map<uint32_t,std::vector<uint32_t> > medsData;

    edm::DetSetVector<SiStripRawDigi>::const_iterator inputChannel = lInputDigis->begin();

    for ( ; inputChannel!=lInputDigis->end(); ++inputChannel){//loop on input channels
      uint32_t lDetId = inputChannel->detId(); //either fedIndex or detId

      pedsData.push_back(    edm::DetSetVector<SiStripRawDigi>::detset(lDetId) );
      noiseData.push_back(   edm::DetSetVector<SiStripProcessedRawDigi>::detset(lDetId) );
      pedSubtrData.push_back(edm::DetSetVector<SiStripRawDigi>::detset(lDetId) );
      cmSubtrData.push_back( edm::DetSetVector<SiStripRawDigi>::detset(lDetId) );

      unsigned int lNStrips = inputChannel->size();

      //define output digi containers
      std::vector<SiStripRawDigi>& pedsDetSetData = pedsData.back().data;
      pedsDetSetData.reserve(lNStrips);
      std::vector<SiStripProcessedRawDigi>& noiseDetSetData = noiseData.back().data;
      noiseDetSetData.reserve(lNStrips);
      std::vector<SiStripRawDigi>& pedSubtrDetSetData = pedSubtrData.back().data;
      pedSubtrDetSetData.reserve(lNStrips);
      std::vector<SiStripRawDigi>& cmSubtrDetSetData = cmSubtrData.back().data;
      cmSubtrDetSetData.reserve(lNStrips);
      //zero suppressed - slightly different procedure as not fixed size
      edm::DetSet<SiStripDigi> zsDetSetData(lDetId);
      
      //determine the number of APV pairs in the channel
      uint32_t lNPairs = static_cast<uint32_t>(lNStrips*1./sistrip::STRIPS_PER_FEDCH);
      uint32_t lPair = 0;
      
      std::vector<uint32_t> medsDetSetData;
      medsDetSetData.reserve(lNPairs*2); //2*number of pairs per module. If not by module, lNPairs = 1...
      
      if (!byModule_) { //the input is not stored by module
	//need to retrieve the proper detId from cabling
	uint16_t lFedId = 0;
	uint16_t lFedChannel = 0;
	sistrip::SpyUtilities::fedIndex(lDetId, lFedId, lFedChannel);
    		
	const FedChannelConnection & lConnection = lCabling->fedConnection(lFedId,lFedChannel);
	lDetId = lConnection.detId();
	lNPairs = lConnection.nApvPairs();
	lPair = lConnection.apvPairNumber();
      }//end of by module check

      fedEmulator_.initialiseModule(lDetId,lNPairs,lPair);

      //get the pedestal values
      //stored by module in the database
      fedEmulator_.retrievePedestals(lPedsHandle);
      fedEmulator_.retrieveNoises(lNoiseHandle);
      
      //last option: fill medians from these ped subtr data
      //if want something else, need to call a method to fill
      //the data member medians_ of the class fedEmulator.
      fedEmulator_.subtractPedestals(inputChannel,
				     pedsDetSetData,
				     noiseDetSetData,
				     pedSubtrDetSetData,
				     medsDetSetData,
				     true);
      
      fedEmulator_.subtractCM(pedSubtrDetSetData,cmSubtrDetSetData);

      //fill the median map
      medsData[inputChannel->detId()] = medsDetSetData;

      //zero suppress the digis
      fedEmulator_.zeroSuppress(cmSubtrDetSetData, zsDetSetData, algorithms_);
      if (!zsDetSetData.empty()) zsData.push_back( zsDetSetData );
      
    }//loop on input channels


    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > lPeds(new edm::DetSetVector<SiStripRawDigi>(pedsData,true));
    std::unique_ptr<edm::DetSetVector<SiStripProcessedRawDigi> > lNoises(new edm::DetSetVector<SiStripProcessedRawDigi>(noiseData,true));

    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > lOutputPedSubtr(new edm::DetSetVector<SiStripRawDigi>(pedSubtrData,true));

    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > lOutputCMSubtr(new edm::DetSetVector<SiStripRawDigi>(cmSubtrData,true));

    std::unique_ptr<std::map<uint32_t,std::vector<uint32_t> > > lMedians(new std::map<uint32_t,std::vector<uint32_t> >(medsData));
  
    //zero suppressed digis
    std::unique_ptr< edm::DetSetVector<SiStripDigi> > lOutputZS(new edm::DetSetVector<SiStripDigi>(zsData));
    
    if (!byModule_) {
      iEvent.put(std::move(lMedians),"Medians");
      iEvent.put(std::move(lPeds),"PedestalsOrdered");
      iEvent.put(std::move(lNoises),"NoisesOrdered");
      iEvent.put(std::move(lOutputPedSubtr),"PedSubtrDigisOrdered");
      iEvent.put(std::move(lOutputCMSubtr),"CMSubtrDigisOrdered");
    }
    else {
      iEvent.put(std::move(lPeds),"ModulePedestals");
      iEvent.put(std::move(lNoises),"ModuleNoises");
      iEvent.put(std::move(lOutputPedSubtr),"PedSubtrModuleDigis");
      iEvent.put(std::move(lMedians),"ModuleMedians");
      iEvent.put(std::move(lOutputCMSubtr),"CMSubtrModuleDigis");
      iEvent.put(std::move(lOutputZS),"ZSModuleDigis");
    }

  }//produce method
}//namespace sistrip

//
// Define as a plug-in
//

#include "FWCore/Framework/interface/MakerMacros.h"
typedef sistrip::FEDEmulatorModule SiStripFEDEmulatorModule;
DEFINE_FWK_MODULE(SiStripFEDEmulatorModule);
