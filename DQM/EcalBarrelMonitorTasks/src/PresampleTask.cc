#include "../interface/PresampleTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

namespace ecaldqm {

  PresampleTask::PresampleTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "PresampleTask")
  {
    collectionMask_[kEBDigi] = true;
    collectionMask_[kEEDigi] = true;
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
    MESet* mePedestal(MEs_["Pedestal"]);

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      bool gainSwitch(false);
      int iMax(-1);
      int maxADC(0);
      for(int iSample(0); iSample < 10; ++iSample){
        int adc(dataFrame.sample(iSample).adc());
        if(adc > maxADC){
          iMax = iSample;
          maxADC = adc;
        }
        if(iSample < 3 && dataFrame.sample(iSample).gainId() != 1){
          gainSwitch = true;
          break;
        }
      }
      if(iMax != 5 || gainSwitch) continue;

      for(int iSample(0); iSample < 3; ++iSample)
	mePedestal->fill(id, double(dataFrame.sample(iSample).adc()));
    }
  }

  DEFINE_ECALDQM_WORKER(PresampleTask);
}

