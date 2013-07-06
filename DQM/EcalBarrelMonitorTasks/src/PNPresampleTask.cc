#include "../interface/PNPresampleTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  PNPresampleTask::PNPresampleTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "PNPresampleTask")
  {
    collectionMask_ =
      (0x1 << kPnDiodeDigi);
  }

  PNPresampleTask::~PNPresampleTask()
  {
  }

  void
  PNPresampleTask::beginRun(const edm::Run&, const edm::EventSetup&)
  {
    for(int idcc(0); idcc < 54; idcc++)
      enable_[idcc] = false;
  }

  void
  PNPresampleTask::endEvent(const edm::Event&, const edm::EventSetup&)
  {
    for(int idcc(0); idcc < 54; idcc++)
      enable_[idcc] = false;
  }

  bool
  PNPresampleTask::filterRunType(const std::vector<short>& _runType)
  {
    bool enable(false);

    for(int iDCC(0); iDCC < 54; iDCC++){
      if(_runType[iDCC] == EcalDCCHeaderBlock::LASER_STD ||
	 _runType[iDCC] == EcalDCCHeaderBlock::LASER_GAP ||
	 _runType[iDCC] == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
	 _runType[iDCC] == EcalDCCHeaderBlock::TESTPULSE_GAP){
	enable = true;
	enable_[iDCC] = true;
      }
    }

    return enable;
  }

  void
  PNPresampleTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      const EcalPnDiodeDetId& id(digiItr->id());

      float mean(0.);
      bool gainSwitch(false);

      for(int iSample(0); iSample < 4; iSample++){
	if(digiItr->sample(iSample).gainId() != 1){
	  gainSwitch = true;
	  break;
	}
	mean += digiItr->sample(iSample).adc();
      }
      if(gainSwitch) continue;

      mean /= 4.;

      MEs_[kPedestal]->fill(id, mean);
    }
  }

  /*static*/
  void
  PNPresampleTask::setMEData(std::vector<MEData>& _data)
  {
    _data[kPedestal] = MEData("Pedestal", BinService::kSMMEM, BinService::kCrystal, MonitorElement::DQM_KIND_TPROFILE2D);
  }

  DEFINE_ECALDQM_WORKER(PNPresampleTask);
}
