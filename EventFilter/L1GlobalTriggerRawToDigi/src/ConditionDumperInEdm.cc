#include "EventFilter/L1GlobalTriggerRawToDigi/interface/ConditionDumperInEdm.h"
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"


//
// constructors and destructor
//
ConditionDumperInEdm::ConditionDumperInEdm(const edm::ParameterSet& iConfig)
{
  //per LUMI products
  produces<ConditionsInLumiBlock,edm::InLumi>();
  //per RUN products
  produces<ConditionsInRunBlock,edm::InRun>();
  //per EVENT products
  produces<ConditionsInEventBlock>();

}


ConditionDumperInEdm::~ConditionDumperInEdm()
{
}


//
// member functions
//
void ConditionDumperInEdm::beginLuminosityBlock(edm::LuminosityBlock&lumi, edm::EventSetup const&setup){
}
void ConditionDumperInEdm::endLuminosityBlock(edm::LuminosityBlock&lumi, edm::EventSetup const&setup){
  std::auto_ptr<ConditionsInLumiBlock> lumiOut( new ConditionsInLumiBlock(lumiBlock_));
  lumi.put( lumiOut );
}

void ConditionDumperInEdm::beginRun(edm::Run& run , const edm::EventSetup& setup){
}

void ConditionDumperInEdm::endRun(edm::Run& run , const edm::EventSetup& setup){
  //dump of RunInfo
  {
    edm::ESHandle<RunInfo> sum;
    setup.get<RunInfoRcd>().get(sum);
    runBlock_.BStartCurrent=sum->m_start_current;
    runBlock_.BStopCurrent=sum->m_stop_current;
    runBlock_.BAvgCurrent=sum->m_avg_current;
  }

  std::auto_ptr<ConditionsInRunBlock> outBlock(new ConditionsInRunBlock(runBlock_));
  run.put(outBlock);
}

// ------------ method called to produce the data  ------------
void
ConditionDumperInEdm::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //get the L1 object 
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtReadoutRecordData;
  iEvent.getByLabel("gtEvmDigis", gtReadoutRecordData);

  const L1GtfeExtWord& gtfeBlockData = gtReadoutRecordData->gtfeWord();

  //lumi info
  lumiBlock_.beamMomentum=gtfeBlockData.beamMomentum();
  lumiBlock_.totalIntensityBeam1=gtfeBlockData.totalIntensityBeam1();
  lumiBlock_.totalIntensityBeam2=gtfeBlockData.totalIntensityBeam2();

  //run info
  runBlock_.beamMode=gtfeBlockData.beamMode();
  //  runBlock_.particleTypeBeam1=gtfeBlockData.particleTypeBeam1();
  //  runBlock_.particleTypeBeam2=gtfeBlockData.particleTypeBeam2();
  runBlock_.lhcFillNumber=gtfeBlockData.lhcFillNumber();

  //event info
  eventBlock_. bstMasterStatus= gtfeBlockData.bstMasterStatus() ;
  eventBlock_.turnCountNumber = gtfeBlockData.turnCountNumber();

  std::auto_ptr<ConditionsInEventBlock> eventOut( new ConditionsInEventBlock(eventBlock_));
  iEvent.put( eventOut );
}


