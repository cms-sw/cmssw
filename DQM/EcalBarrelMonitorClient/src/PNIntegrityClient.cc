#include "../interface/PNIntegrityClient.h"

namespace ecaldqm
{
  PNIntegrityClient::PNIntegrityClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "PNIntegrityClient"),
    errFractionThreshold_(_workerParams.getUntrackedParameter<double>("errFractionThreshold"))
  {
  }

  void
  PNIntegrityClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQualitySummary]->resetAll(-1.);
  }

  void
  PNIntegrityClient::producePlots()
  {
    MESet::const_iterator oEnd(sources_[kOccupancy]->end());
    MESet::iterator qsItr(MEs_[kQualitySummary]);
    for(MESet::const_iterator oItr(sources_[kOccupancy]->beginChannel()); oItr != oEnd; oItr.toNextChannel()){

      qsItr = oItr;

      EcalPnDiodeDetId id(oItr->getId());

      float entries(oItr->getBinContent());

      float chid(sources_[kMEMChId]->getBinContent(id));
      float gain(sources_[kMEMGain]->getBinContent(id));

      float blocksize(sources_[kMEMBlockSize]->getBinContent(id));
      float towerid(sources_[kMEMTowerId]->getBinContent(id));

      if(entries + gain + chid + blocksize + towerid < 1.){
        qsItr->setBinContent(maskPNQuality_(qsItr, 2));
        continue;
      }

      float chErr((gain + chid + blocksize + towerid) / (entries + gain + chid + blocksize + towerid));

      if(chErr > errFractionThreshold_)
        qsItr->setBinContent(maskPNQuality_(qsItr, 0));
      else
        qsItr->setBinContent(maskPNQuality_(qsItr, 1));
    }
  }

  /*static*/
  void
  PNIntegrityClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["QualitySummary"] = kQualitySummary;

    _nameToIndex["kOccupancy"] = kOccupancy;
    _nameToIndex["MEMChId"] = kMEMChId;
    _nameToIndex["MEMGain"] = kMEMGain;
    _nameToIndex["MEMBlockSize"] = kMEMBlockSize;
    _nameToIndex["MEMTowerId"] = kMEMTowerId;
  }

  DEFINE_ECALDQM_WORKER(PNIntegrityClient);
}
