#include "../interface/PresampleTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

namespace ecaldqm {

  PresampleTask::PresampleTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "PresampleTask")
  {
    collectionMask_ =
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi);
  }

  PresampleTask::~PresampleTask()
  {
  }

  bool
  PresampleTask::filterRunType(const std::vector<short>& _runType)
  {
    for(int iFED(0); iFED < 54; iFED++){
      if ( _runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
           _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
           _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL ) return true;
    }

    return false;
  }

  void
  PresampleTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      float mean(0.);
      bool gainSwitch(false);

      for(int iSample(0); iSample < 3; iSample++){
	if(dataFrame.sample(iSample).gainId() != 1){
	  gainSwitch = true;
	  break;
	}

	mean += dataFrame.sample(iSample).adc();
      }
      if(gainSwitch) continue;

      mean /= 3.;

      MEs_[kPedestal]->fill(id, mean);
    }
  }

  /*static*/
  void
  PresampleTask::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs axis;
    axis.low = 160.;
    axis.high = 240.;
    _data[kPedestal] = MEData("Pedestal", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &axis);
  }

  DEFINE_ECALDQM_WORKER(PresampleTask);
}

