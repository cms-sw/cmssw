#include "../interface/PedestalTask.h"

#include <iomanip>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  PedestalTask::PedestalTask() :
    DQWorkerTask(),
    gainToME_(),
    pnGainToME_()
  {
    std::fill_n(enable_, nDCC, false);
  }

  void
  PedestalTask::setParams(edm::ParameterSet const& _params)
  {
    std::vector<int> MGPAGains(_params.getUntrackedParameter<std::vector<int> >("MGPAGains"));
    std::vector<int> MGPAGainsPN(_params.getUntrackedParameter<std::vector<int> >("MGPAGainsPN"));

    MESet::PathReplacements repl;

    MESetMulti& pedestal(static_cast<MESetMulti&>(MEs_.at("Pedestal")));
    unsigned nG(MGPAGains.size());
    for(unsigned iG(0); iG != nG; ++iG){
      int gain(MGPAGains[iG]);
      if(gain != 1 && gain != 6 && gain != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain";
      repl["gain"] = std::to_string(gain);
      gainToME_[gain] = pedestal.getIndex(repl);
    }

    repl.clear();

    MESetMulti& pnPedestal(static_cast<MESetMulti&>(MEs_.at("PNPedestal")));
    unsigned nGPN(MGPAGainsPN.size());
    for(unsigned iG(0); iG != nGPN; ++iG){
      int gain(MGPAGainsPN[iG]);
      if(gain != 1 && gain != 16) throw cms::Exception("InvalidConfiguration") << "PN MGPA gain";
      repl["pngain"] = std::to_string(gain);
      pnGainToME_[gain] = pnPedestal.getIndex(repl);
    }
  }

  bool
  PedestalTask::filterRunType(short const* _runType)
  {
    bool enable(false);

    for(int iFED(0); iFED < nDCC; iFED++){
      if(_runType[iFED] == EcalDCCHeaderBlock::PEDESTAL_STD ||
	 _runType[iFED] == EcalDCCHeaderBlock::PEDESTAL_GAP){
	enable = true;
	enable_[iFED] = true;
      }
      else
        enable_[iFED] = false;
    }

    return enable;
  }

  template<typename DigiCollection>
  void
  PedestalTask::runOnDigis(DigiCollection const& _digis)
  {
    MESet& mePedestal(MEs_.at("Pedestal"));
    MESet& meOccupancy(MEs_.at("Occupancy"));

    unsigned iME(-1);

    for(typename DigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      int gain(0);
      switch(dataFrame.sample(0).gainId()){
      case 1: gain = 12; break;
      case 2: gain = 6; break;
      case 3: gain = 1; break;
      default: continue;
      }

      if(gainToME_.find(gain) == gainToME_.end()) continue;

      if(iME != gainToME_[gain]){
        iME = gainToME_[gain];
        static_cast<MESetMulti&>(mePedestal).use(iME);
      }

      meOccupancy.fill(id);

      for(int iSample(0); iSample < EcalDataFrame::MAXSAMPLES; iSample++)
	mePedestal.fill(id, double(dataFrame.sample(iSample).adc()));
    }
  }

  void
  PedestalTask::runOnPnDigis(EcalPnDiodeDigiCollection const& _digis)
  {
    MESet& mePNPedestal(MEs_.at("PNPedestal"));

    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      EcalPnDiodeDetId id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      int gain(0);
      switch(digiItr->sample(0).gainId()){
      case 0: gain = 1; break;
      case 1: gain = 16; break;
      default: continue;
      }

      if(pnGainToME_.find(gain) == pnGainToME_.end()) continue;

      if(iME != pnGainToME_[gain]){
        iME = pnGainToME_[gain];
        static_cast<MESetMulti&>(mePNPedestal).use(iME);
      }

      for(int iSample(0); iSample < 50; iSample++)
        mePNPedestal.fill(id, double(digiItr->sample(iSample).adc()));
    }
  }

  DEFINE_ECALDQM_WORKER(PedestalTask);
}
