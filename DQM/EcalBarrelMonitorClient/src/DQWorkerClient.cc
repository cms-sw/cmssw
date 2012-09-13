#include "../interface/DQWorkerClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetChannel.h"
#include "DQM/EcalCommon/interface/MESetUtils.h"

#include "../interface/EcalDQMClientUtils.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace ecaldqm {

  DQWorkerClient::DQWorkerClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams, std::string const& _name) :
    DQWorker(_workerParams, _commonParams, _name),
    sources_(0),
    usedSources_(0)
  {
    using namespace std;

    string topDir;
    if(_workerParams.existsAs<string>("topDirectory", false))
      topDir = _workerParams.getUntrackedParameter<string>("topDirectory");
    else
      topDir = _commonParams.getUntrackedParameter<string>("topDirectory");

    if(_workerParams.existsAs<edm::ParameterSet>("sources", false)){
      BinService const* binService(&(*(edm::Service<EcalDQMBinningService>())));
      if(!binService)
        throw cms::Exception("Service") << "EcalDQMBinningService not found" << std::endl;

      edm::ParameterSet const& sourceParams(_workerParams.getUntrackedParameterSet("sources"));
      vector<string> const& sourceNames(sourceParams.getParameterNames());

      sources_.resize(sourceNames.size());

      // existence already checked in DQWorker Ctor
      map<string, unsigned> const& nameToIndex(meOrderingMaps[name_]);

      for(unsigned iS(0); iS < sourceNames.size(); iS++){
        string const& sourceName(sourceNames[iS]);

        map<string, unsigned>::const_iterator nItr(nameToIndex.find(sourceName));
        if(nItr == nameToIndex.end())
          throw cms::Exception("InvalidConfiguration") << "Cannot find ME index for " << sourceName;

        MESet const* meSet(createMESet(sourceParams.getUntrackedParameterSet(sourceName), binService, topDir));
        if(meSet){
          sources_[nItr->second] = meSet;
          if(meSet->getBinType() != BinService::kTrend || online_)
            usedSources_ |= (0x1 << nItr->second);
        }
      }
    }
  }

  void
  DQWorkerClient::endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    for(std::vector<MESet const*>::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr){
      MESetChannel const* meset(dynamic_cast<MESetChannel const*>(*sItr));
      if(meset) meset->checkDirectory();
    }
  }

  void
  DQWorkerClient::reset()
  {
    DQWorker::reset();
    for(std::vector<MESet const*>::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr)
      (*sItr)->clear();
  }

  void
  DQWorkerClient::initialize()
  {
    initialized_ = true;
    for(unsigned iS(0); iS < sources_.size(); ++iS){
      if(!using_(iS)) continue;
      initialized_ &= sources_[iS]->retrieve();
    }
  }

  bool
  DQWorkerClient::applyMask_(unsigned _iME, DetId const& _id, uint32_t _mask)
  {
    return applyMask(MEs_[_iME]->getBinType(), _id, _mask);
  }

  void
  DQWorkerClient::towerAverage_(unsigned _target, unsigned _source, float _threshold)
  {
    MESet::iterator tEnd(MEs_[_target]->end());
    for(MESet::iterator tItr(MEs_[_target]->beginChannel()); tItr != tEnd; tItr.toNextChannel()){
      DetId towerId(tItr->getId());

      std::vector<DetId> cryIds;
      if(towerId.subdetId() == EcalTriggerTower)
        cryIds = getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(towerId));
      else{
        cryIds = scConstituents(EcalScDetId(towerId));
      }

      if(cryIds.size() == 0) return;

      float mean(0.);
      float nValid(0.);
      bool masked(false);
      for(unsigned iId(0); iId < cryIds.size(); ++iId){
        float content(MEs_[_source]->getBinContent(cryIds[iId]));
        if(content < 0. || content == 2.) continue;
        if(content == 5.) masked = true;
        else{
          nValid += 1;
          if(content > 2.){
            masked = true;
            mean += content - 3.;
          }
          else
            mean += content;
        }
      }
      
      if(nValid < 1.) tItr->setBinContent(masked ? 5. : 2.);
      else{
        mean /= nValid;
        if(mean < _threshold) tItr->setBinContent(masked ? 3. : 0.);
        else tItr->setBinContent(masked ? 4. : 1.);
      }
    }
  }
}
