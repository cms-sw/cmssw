#include "EventFilter/L1GlobalTriggerRawToDigi/interface/ConditionDumperInEdm.h"
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtfeExtWord.h"

#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"


//
// constructors and destructor
//
ConditionDumperInEdm::ConditionDumperInEdm(const edm::ParameterSet& iConfig)
{
  
  gtEvmDigisLabel_ = iConfig.getParameter<edm::InputTag>("gtEvmDigisLabel");


  //per LUMI products
  produces<edm::ConditionsInLumiBlock,edm::InLumi>();
  //per RUN products
  produces<edm::ConditionsInRunBlock,edm::InRun>();
  //per EVENT products
  produces<edm::ConditionsInEventBlock>();

}


ConditionDumperInEdm::~ConditionDumperInEdm()
{
}


//
// member functions
//
void ConditionDumperInEdm::endLuminosityBlockProduce(edm::LuminosityBlock&lumi, edm::EventSetup const&setup){
  std::auto_ptr<edm::ConditionsInLumiBlock> lumiOut( new edm::ConditionsInLumiBlock(lumiBlock_));
  lumi.put( lumiOut );
}

void ConditionDumperInEdm::endRunProduce(edm::Run& run , const edm::EventSetup& setup){
  //dump of RunInfo
  {
    edm::ESHandle<RunInfo> sum;
    setup.get<RunInfoRcd>().get(sum);
    runBlock_.BStartCurrent=sum->m_start_current;
    runBlock_.BStopCurrent=sum->m_stop_current;
    runBlock_.BAvgCurrent=sum->m_avg_current;
  }

  std::auto_ptr<edm::ConditionsInRunBlock> outBlock(new edm::ConditionsInRunBlock(runBlock_));
  run.put(outBlock);
}

// ------------ method called to produce the data  ------------
void
ConditionDumperInEdm::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //get the L1 object 
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtReadoutRecordData;
  iEvent.getByLabel(gtEvmDigisLabel_, gtReadoutRecordData);

  if (!gtReadoutRecordData.isValid()) {
      LogDebug("ConditionDumperInEdm")
              << "\nWarning: L1GlobalTriggerEvmReadoutRecord with input tag " << gtEvmDigisLabel_
              << "\nrequested in configuration, but not found in the event."
              << "\nNo BST quantities retrieved." << std::endl;

      std::auto_ptr<edm::ConditionsInEventBlock> eventOut( new edm::ConditionsInEventBlock(eventBlock_));
      iEvent.put( eventOut );

      return;
  }

  const L1GtfeExtWord& gtfeBlockData = gtReadoutRecordData->gtfeWord();

  //lumi info
  lumiBlock_.totalIntensityBeam1=gtfeBlockData.totalIntensityBeam1();
  lumiBlock_.totalIntensityBeam2=gtfeBlockData.totalIntensityBeam2();

  //run info
  runBlock_.beamMomentum=gtfeBlockData.beamMomentum();
  runBlock_.beamMode=gtfeBlockData.beamMode();
  //  runBlock_.particleTypeBeam1=gtfeBlockData.particleTypeBeam1();
  //  runBlock_.particleTypeBeam2=gtfeBlockData.particleTypeBeam2();
  runBlock_.lhcFillNumber=gtfeBlockData.lhcFillNumber();

  //event info
  eventBlock_. bstMasterStatus= gtfeBlockData.bstMasterStatus() ;
  eventBlock_.turnCountNumber = gtfeBlockData.turnCountNumber();

  std::auto_ptr<edm::ConditionsInEventBlock> eventOut( new edm::ConditionsInEventBlock(eventBlock_));
  iEvent.put( eventOut );
}


