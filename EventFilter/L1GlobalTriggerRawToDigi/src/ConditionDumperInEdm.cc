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
ConditionDumperInEdm::ConditionDumperInEdm(const edm::ParameterSet& iConfig):
  gtEvmDigisLabel_{iConfig.getParameter<edm::InputTag>("gtEvmDigisLabel")},
  gtEvmDigisLabelToken_{consumes<L1GlobalTriggerEvmReadoutRecord>(gtEvmDigisLabel_)},
  //per LUMI products
  lumiToken_{produces<edm::ConditionsInLumiBlock,edm::Transition::EndLuminosityBlock>()},
  //per RUN products
  runToken_{produces<edm::ConditionsInRunBlock,edm::Transition::EndRun>()},
  //per EVENT products
  eventToken_{produces<edm::ConditionsInEventBlock>()}
{
}


ConditionDumperInEdm::~ConditionDumperInEdm()
{
}


//
// member functions
//
std::shared_ptr<edm::ConditionsInLumiBlock> 
ConditionDumperInEdm::globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const {
  return std::make_shared<edm::ConditionsInLumiBlock>();
}

void ConditionDumperInEdm::endLuminosityBlockProduce(edm::LuminosityBlock&lumi, edm::EventSetup const&setup){
  lumi.emplace(lumiToken_,std::move(*luminosityBlockCache(lumi.index())));
}

std::shared_ptr<edm::ConditionsInRunBlock> 
ConditionDumperInEdm::globalBeginRun(edm::Run const& , const edm::EventSetup&) const {
  return std::make_shared<edm::ConditionsInRunBlock>();
}

void ConditionDumperInEdm::endRunProduce(edm::Run& run , const edm::EventSetup& setup){
  //dump of RunInfo
  auto& runBlock = *(runCache(run.index()));
  {
    edm::ESHandle<RunInfo> sum;
    setup.get<RunInfoRcd>().get(sum);
    runBlock.BStartCurrent=sum->m_start_current;
    runBlock.BStopCurrent=sum->m_stop_current;
    runBlock.BAvgCurrent=sum->m_avg_current;
  }

  run.emplace(runToken_,std::move(runBlock));
}

// ------------ method called to produce the data  ------------
void
ConditionDumperInEdm::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //get the L1 object 
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtReadoutRecordData;
  iEvent.getByToken(gtEvmDigisLabelToken_, gtReadoutRecordData);

  if (!gtReadoutRecordData.isValid()) {
      LogDebug("ConditionDumperInEdm")
              << "\nWarning: L1GlobalTriggerEvmReadoutRecord with input tag " << gtEvmDigisLabel_
              << "\nrequested in configuration, but not found in the event."
              << "\nNo BST quantities retrieved." << std::endl;

      iEvent.emplace(eventToken_,eventBlock_);

      return;
  }

  const L1GtfeExtWord& gtfeBlockData = gtReadoutRecordData->gtfeWord();

  //lumi info
  auto& lumiBlock = *luminosityBlockCache(iEvent.getLuminosityBlock().index());
  lumiBlock.totalIntensityBeam1=gtfeBlockData.totalIntensityBeam1();
  lumiBlock.totalIntensityBeam2=gtfeBlockData.totalIntensityBeam2();

  //run info
  auto& runBlock = *runCache(iEvent.getRun().index());
  runBlock.beamMomentum=gtfeBlockData.beamMomentum();
  runBlock.beamMode=gtfeBlockData.beamMode();
  runBlock.lhcFillNumber=gtfeBlockData.lhcFillNumber();

  //event info
  eventBlock_. bstMasterStatus= gtfeBlockData.bstMasterStatus() ;
  eventBlock_.turnCountNumber = gtfeBlockData.turnCountNumber();

  iEvent.emplace(eventToken_,eventBlock_);
}


