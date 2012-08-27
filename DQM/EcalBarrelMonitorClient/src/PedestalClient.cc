#include "../interface/PedestalClient.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

namespace ecaldqm
{

  PedestalClient::PedestalClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "PedestalTask"),
    expectedMean_(0),
    toleranceMean_(0),
    toleranceRMS_(0),
    expectedPNMean_(0),
    tolerancePNMean_(0),
    tolerancePNRMS_(0)
  {
    using namespace std;

    vector<int> MGPAGains(_commonParams.getUntrackedParameter<vector<int> >("MGPAGains"));
    vector<int> MGPAGainsPN(_commonParams.getUntrackedParameter<vector<int> >("MGPAGainsPN"));

    unsigned iMEGain(0);
    for(vector<int>::iterator gainItr(MGPAGains.begin()); gainItr != MGPAGains.end(); ++gainItr){
      if(*gainItr != 1 && *gainItr != 6 && *gainItr != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain" << endl;
      gainToME_[*gainItr] = iMEGain++;
    }

    unsigned iMEPNGain(0);
    for(vector<int>::iterator gainItr(MGPAGainsPN.begin()); gainItr != MGPAGainsPN.end(); ++gainItr){
      if(*gainItr != 1 && *gainItr != 16) throw cms::Exception("InvalidConfiguration") << "PN diode gain" << endl;	
      pnGainToME_[*gainItr] = iMEPNGain++;
    }

    stringstream ss;

    expectedMean_.resize(iMEGain);
    toleranceMean_.resize(iMEGain);
    toleranceRMS_.resize(iMEGain);

    for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      ss.str("");
      ss << "G" << gainItr->first;

      expectedMean_[gainItr->second] = _workerParams.getUntrackedParameter<double>("expectedMean" + ss.str());
      toleranceMean_[gainItr->second] = _workerParams.getUntrackedParameter<double>("toleranceMean" + ss.str());
      toleranceRMS_[gainItr->second] = _workerParams.getUntrackedParameter<double>("toleranceRMS" + ss.str());
    }

    expectedPNMean_.resize(iMEPNGain);
    tolerancePNMean_.resize(iMEPNGain);
    tolerancePNRMS_.resize(iMEPNGain);

    for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      ss.str("");
      ss << "G" << gainItr->first;

      expectedPNMean_[gainItr->second] = _workerParams.getUntrackedParameter<double>("expectedPNMean" + ss.str());
      tolerancePNMean_[gainItr->second] = _workerParams.getUntrackedParameter<double>("tolerancePNMean" + ss.str());
      tolerancePNRMS_[gainItr->second] = _workerParams.getUntrackedParameter<double>("tolerancePNRMS" + ss.str());
    }

    map<string, string> replacements;

    unsigned apdPlots[] = {kQuality, kMean, kRMS, kQualitySummary};
    for(unsigned iS(0); iS < sizeof(apdPlots) / sizeof(unsigned); ++iS){
      unsigned plot(apdPlots[iS]);
      MESet* temp(MEs_[plot]);
      MESetMulti* meSet(new MESetMulti(*temp, iMEGain));

      for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
        meSet->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["gain"] = ss.str();

        meSet->formPath(replacements);
      }

      MEs_[plot] = meSet;
      delete temp;
    }

    unsigned pnPlots[] = {kPNRMS, kPNQualitySummary};
    for(unsigned iS(0); iS < sizeof(pnPlots) / sizeof(unsigned); ++iS){
      unsigned plot(pnPlots[iS]);
      MESet* temp(MEs_[plot]);
      MESetMulti* meSet(new MESetMulti(*temp, iMEPNGain));

      for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
        meSet->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["pngain"] = ss.str();

        meSet->formPath(replacements);
      }

      MEs_[plot] = meSet;
      delete temp;
    }

    unsigned apdSources[] = {kPedestal};
    for(unsigned iS(0); iS < sizeof(apdSources) / sizeof(unsigned); ++iS){
      unsigned plot(apdSources[iS]);
      MESet const* temp(sources_[plot]);
      MESetMulti const* meSet(new MESetMulti(*temp, iMEGain));

      for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
        meSet->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["gain"] = ss.str();

        meSet->formPath(replacements);
      }

      sources_[plot] = meSet;
      delete temp;
    }

    unsigned pnSources[] = {kPNPedestal};
    for(unsigned iS(0); iS < sizeof(pnSources) / sizeof(unsigned); ++iS){
      unsigned plot(pnSources[iS]);
      MESet const* temp(sources_[plot]);
      MESetMulti const* meSet(new MESetMulti(*temp, iMEPNGain));

      for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
        meSet->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["pngain"] = ss.str();

        meSet->formPath(replacements);
      }

      sources_[plot] = meSet;
      delete temp;
    }
  }

  void
  PedestalClient::beginRun(edm::Run const&, edm::EventSetup const&)
  {
    for(unsigned iME(0); iME < gainToME_.size(); ++iME){
      static_cast<MESetMulti*>(MEs_[kQuality])->use(iME);
      static_cast<MESetMulti*>(MEs_[kQualitySummary])->use(iME);

      MEs_[kQuality]->resetAll(-1.);
      MEs_[kQualitySummary]->resetAll(-1.);
    }

    for(unsigned iME(0); iME < pnGainToME_.size(); ++iME){
      static_cast<MESetMulti*>(MEs_[kPNQualitySummary])->use(iME);

      MEs_[kPNQualitySummary]->resetAll(-1.);
    }
  }

  void
  PedestalClient::producePlots()
  {
    using namespace std;

    for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      static_cast<MESetMulti*>(MEs_[kQuality])->use(gainItr->second);
      static_cast<MESetMulti*>(MEs_[kQualitySummary])->use(gainItr->second);
      static_cast<MESetMulti*>(MEs_[kMean])->use(gainItr->second);
      static_cast<MESetMulti*>(MEs_[kRMS])->use(gainItr->second);

      static_cast<MESetMulti const*>(sources_[kPedestal])->use(gainItr->second);

      MEs_[kMean]->reset();
      MEs_[kRMS]->reset();

      uint32_t mask(0);
      switch(gainItr->first){
      case 1:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR);
        break;
      case 6:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR);
        break;
      case 12:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR);
        break;
      default:
        break;
      }

      MESet::const_iterator meEnd(sources_[kPedestal]->end());
      for(MESet::const_iterator meItr(sources_[kPedestal]->beginChannel()); meItr != meEnd; meItr.toNextChannel()){

        DetId id(meItr->getId());

        float mean(meItr->getBinContent());
        float entries(meItr->getBinEntries());
        float rms(meItr->getBinError() * sqrt(entries));

        int quality(2);

        if(entries < 1.){
          MEs_[kQuality]->setBinContent(id, maskQuality_(kQuality, id, mask, quality));
          continue;
        }

        MEs_[kMean]->fill(id, mean);
        MEs_[kRMS]->fill(id, mean);

        if(abs(mean - expectedMean_[gainItr->second]) > toleranceMean_[gainItr->second] || rms > toleranceRMS_[gainItr->second])
          quality = 0;
        else
          quality = 1;

        MEs_[kQuality]->setBinContent(id, maskQuality_(kQuality, id, mask, quality));
      }
        
      towerAverage_(kQualitySummary, kQuality, 0.5);
    }

    for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      static_cast<MESetMulti*>(MEs_[kPNQualitySummary])->use(gainItr->second);
      static_cast<MESetMulti*>(MEs_[kPNRMS])->use(gainItr->second);

      static_cast<MESetMulti const*>(sources_[kPNPedestal])->use(gainItr->second);

      MEs_[kPNRMS]->reset();

      uint32_t mask(0);
      switch(gainItr->first){
      case 1:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR);
        break;
      case 16:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR);
        break;
      default:
        break;
      }

      MESet::const_iterator meEnd(sources_[kPNPedestal]->end());
      for(MESet::const_iterator meItr(sources_[kPNPedestal]->beginChannel()); meItr != meEnd; meItr.toNextChannel()){

        EcalPnDiodeDetId id(meItr->getId());

        float mean(meItr->getBinContent());
        float entries(meItr->getBinEntries());
        float rms(meItr->getBinError() * sqrt(entries));

        int quality(2);

        if(entries < 1.){
          MEs_[kPNQualitySummary]->setBinContent(id, maskQuality_(kPNQualitySummary, id, mask, quality));
          continue;
        }

        MEs_[kPNRMS]->fill(id, rms);

        if(abs(mean - expectedPNMean_[gainItr->second]) > tolerancePNMean_[gainItr->second] || rms > tolerancePNRMS_[gainItr->second])
          quality = 0;
        else
          quality = 1;

        MEs_[kPNQualitySummary]->setBinContent(id, maskQuality_(kPNQualitySummary, id, mask, quality));
      }
    }
  }

  void
  PedestalClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Quality"] = kQuality;
    _nameToIndex["Mean"] = kMean;
    _nameToIndex["RMS"] = kRMS;
    _nameToIndex["PNRMS"] = kPNRMS;
    _nameToIndex["QualitySummary"] = kQualitySummary;
    _nameToIndex["PNQualitySummary"] = kPNQualitySummary;

    _nameToIndex["Pedestal"] = kPedestal;
    _nameToIndex["PNPedestal"] = kPNPedestal;
  }

  DEFINE_ECALDQM_WORKER(PedestalClient);
}
