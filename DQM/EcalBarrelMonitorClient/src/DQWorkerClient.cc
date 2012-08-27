#include "../interface/DQWorkerClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetChannel.h"

#include "../interface/EcalDQMClientUtils.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  DQWorkerClient::DQWorkerClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams, std::string const& _name) :
    DQWorker(_workerParams, _commonParams, _name),
    sources_(0)
  {
    using namespace std;

    string topDir;
    if(_workerParams.existsAs<string>("topDirectory", false))
      topDir = _workerParams.getUntrackedParameter<string>("topDirectory");
    else
      topDir = _commonParams.getUntrackedParameter<string>("topDirectory");

    if(_workerParams.existsAs<edm::ParameterSet>("sources", false)){
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

        MESet const* meSet(createMESet_(topDir, sourceParams.getUntrackedParameterSet(sourceName)));
        if(meSet) sources_[nItr->second] = meSet;
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
    for(std::vector<MESet const*>::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr)
      initialized_ &= (*sItr)->retrieve();
  }

  float
  DQWorkerClient::maskQuality_(unsigned _iME, DetId const& _id, uint32_t _mask, int _quality)
  {
    return maskQuality(MEs_[_iME]->getBinType(), _id, _mask, _quality);
  }

  float
  DQWorkerClient::maskQuality_(MESet::iterator const& _itr, uint32_t _mask, int _quality)
  {
    uint32_t id(_itr->getId());
    if(id == 0) return _quality;
    else return maskQuality(_itr->getMESet()->getBinType(), DetId(id), _mask, _quality);
  }

  float
  DQWorkerClient::maskPNQuality_(unsigned, EcalPnDiodeDetId const&, int _quality)
  {
    return _quality;
  }

  float
  DQWorkerClient::maskPNQuality_(MESet::iterator const& _itr, int _quality)
  {
    return _quality;
  }

  void
  DQWorkerClient::towerAverage_(unsigned _target, unsigned _source, float _threshold)
  {
    MESet::iterator meEnd(MEs_[_target]->end());
    for(MESet::iterator meItr(MEs_[_target]->beginChannel()); meItr != meEnd; meItr.toNextChannel()){
      DetId towerId(meItr->getId());

      std::vector<DetId> cryIds;
      if(towerId.subdetId() == EcalTriggerTower)
        cryIds = getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(towerId));
      else{
        std::pair<int, int> dccsc(getElectronicsMap()->getDCCandSC(EcalScDetId(towerId)));
        cryIds = getElectronicsMap()->dccTowerConstituents(dccsc.first, dccsc.second);
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
      
      if(nValid < 1.) meItr->setBinContent(masked ? 5. : 2.);
      else{
        mean /= nValid;
        if(mean < _threshold) meItr->setBinContent(masked ? 3. : 0.);
        else meItr->setBinContent(masked ? 4. : 1.);
      }
    }
  }
}
