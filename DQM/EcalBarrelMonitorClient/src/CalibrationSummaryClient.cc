#include "../interface/CalibrationSummaryClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

namespace ecaldqm {

  CalibrationSummaryClient::CalibrationSummaryClient(edm::ParameterSet const&  _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "CalibrationSummaryClient"),
    laserWlToME_(),
    ledWlToME_(),
    tpGainToME_(),
    tpPNGainToME_(),
    pedGainToME_(),
    pedPNGainToME_()
  {
    using namespace std;

    usedSources_ = 
      (0x1 << kPNIntegrity);

    vector<std::string> sourceList(_workerParams.getUntrackedParameter<std::vector<std::string> >("activeSources"));
    for(unsigned iS(0); iS < sourceList.size(); ++iS){
      std::string& sourceName(sourceList[iS]);
      if(sourceName == "Laser") usedSources_ |= (0x1 << kLaser) | (0x1 << kLaserPN);
      else if(sourceName == "Led") usedSources_ |= (0x1 << kLed) | (0x1 << kLedPN);
      else if(sourceName == "TestPulse") usedSources_ |= (0x1 << kTestPulse) | (0x1 << kTestPulsePN);
      else if(sourceName == "Pedestal") usedSources_ |= (0x1 << kPedestal) | (0x1 << kPedestalPN);
    }

    stringstream ss;
    map<string, string> replacements;

    if(using_(kLaser)){
      vector<int> laserWavelengths(_workerParams.getUntrackedParameter<vector<int> >("laserWavelengths", vector<int>()));
      unsigned iMELaserWL(0);
      for(vector<int>::iterator wlItr(laserWavelengths.begin()); wlItr != laserWavelengths.end(); ++wlItr){
        if(*wlItr <= 0 || *wlItr >= 5) throw cms::Exception("InvalidConfiguration") << "Laser Wavelength" << endl;
        laserWlToME_[*wlItr] = iMELaserWL++;
      }
      unsigned laserPlots[] = {kLaser, kLaserPN};
      for(unsigned iP(0); iP < sizeof(laserPlots) / sizeof(int); ++iP){
        unsigned plot(laserPlots[iP]);
        MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[plot]));
        for(map<int, unsigned>::iterator wlItr(laserWlToME_.begin()); wlItr != laserWlToME_.end(); ++wlItr){
          multi->use(wlItr->second);

          ss.str("");
          ss << wlItr->first;
          replacements["wl"] = ss.str();
        
          multi->formPath(replacements);
        }
      }
    }

    if(using_(kLed)){
      vector<int> ledWavelengths(_workerParams.getUntrackedParameter<vector<int> >("ledWavelengths", vector<int>()));
      unsigned iMELedWL(0);
      for(vector<int>::iterator wlItr(ledWavelengths.begin()); wlItr != ledWavelengths.end(); ++wlItr){
        if(*wlItr <= 0 || *wlItr >= 3) throw cms::Exception("InvalidConfiguration") << "Led Wavelength" << endl;
        ledWlToME_[*wlItr] = iMELedWL++;
      }
      unsigned ledPlots[] = {kLed, kLedPN};
      for(unsigned iP(0); iP < sizeof(ledPlots) / sizeof(int); ++iP){
        unsigned plot(ledPlots[iP]);
        MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[plot]));
        for(map<int, unsigned>::iterator wlItr(ledWlToME_.begin()); wlItr != ledWlToME_.end(); ++wlItr){
          multi->use(wlItr->second);

          ss.str("");
          ss << wlItr->first;
          replacements["wl"] = ss.str();

          multi->formPath(replacements);
        }
      }
    }

    if(using_(kTestPulse)){
      vector<int> tpMGPAGains(_workerParams.getUntrackedParameter<vector<int> >("testPulseMGPAGains", vector<int>()));
      vector<int> tpMGPAGainsPN(_workerParams.getUntrackedParameter<vector<int> >("testPulseMGPAGainsPN", vector<int>()));
      unsigned iMETPGain(0);
      for(vector<int>::iterator gainItr(tpMGPAGains.begin()); gainItr != tpMGPAGains.end(); ++gainItr){
        if(*gainItr != 1 && *gainItr != 6 && *gainItr != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain" << endl;
        tpGainToME_[*gainItr] = iMETPGain++;
      }
      unsigned iMETPPNGain(0);
      for(vector<int>::iterator gainItr(tpMGPAGainsPN.begin()); gainItr != tpMGPAGainsPN.end(); ++gainItr){
        if(*gainItr != 1 && *gainItr != 16) throw cms::Exception("InvalidConfiguration") << "PN diode gain" << endl;	
        tpPNGainToME_[*gainItr] = iMETPPNGain++;
      }

      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[kTestPulse]));
      for(map<int, unsigned>::iterator gainItr(tpGainToME_.begin()); gainItr != tpGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }

      multi = static_cast<MESetMulti const*>(sources_[kTestPulsePN]);
      for(map<int, unsigned>::iterator gainItr(tpPNGainToME_.begin()); gainItr != tpPNGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    if(using_(kPedestal)){
      vector<int> pedMGPAGains(_workerParams.getUntrackedParameter<vector<int> >("pedestalMGPAGains", vector<int>()));
      vector<int> pedMGPAGainsPN(_workerParams.getUntrackedParameter<vector<int> >("pedestalMGPAGainsPN", vector<int>()));
      unsigned iMEPedGain(0);
      for(vector<int>::iterator gainItr(pedMGPAGains.begin()); gainItr != pedMGPAGains.end(); ++gainItr){
        if(*gainItr != 1 && *gainItr != 6 && *gainItr != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain" << endl;
        pedGainToME_[*gainItr] = iMEPedGain++;
      }

      unsigned iMEPedPNGain(0);
      for(vector<int>::iterator gainItr(pedMGPAGainsPN.begin()); gainItr != pedMGPAGainsPN.end(); ++gainItr){
        if(*gainItr != 1 && *gainItr != 16) throw cms::Exception("InvalidConfiguration") << "PN diode gain" << endl;	
        pedPNGainToME_[*gainItr] = iMEPedPNGain++;
      }

      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_[kPedestal]));
      for(map<int, unsigned>::iterator gainItr(pedGainToME_.begin()); gainItr != pedGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }


      multi = static_cast<MESetMulti const*>(sources_[kPedestalPN]);
      for(map<int, unsigned>::iterator gainItr(pedPNGainToME_.begin()); gainItr != pedPNGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }
  }

  void
  CalibrationSummaryClient::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    MEs_[kQualitySummary]->resetAll(-1.);
    MEs_[kQualitySummary]->reset(kGood);
    MEs_[kPNQualitySummary]->resetAll(-1.);
    MEs_[kPNQualitySummary]->reset(kGood);
  }

  void
  CalibrationSummaryClient::producePlots()
  {
    using namespace std;

    MESet::iterator qEnd(MEs_[kQualitySummary]->end());
    MESet::const_iterator lItr(sources_[kLaser], using_(kLaser) ? 0 : -1, 0);
    MESet::const_iterator tItr(sources_[kTestPulse], using_(kTestPulse) ? 0 : -1, 0);
    MESet::const_iterator pItr(sources_[kPedestal], using_(kPedestal) ? 0 : -1, 0);
    for(MESet::iterator qItr(MEs_[kQualitySummary]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      int status(kGood);

      if(status == kGood && using_(kLaser)){
        lItr = qItr;
        for(map<int, unsigned>::iterator wlItr(laserWlToME_.begin()); wlItr != laserWlToME_.end(); ++wlItr){
          static_cast<MESetMulti const*>(sources_[kLaser])->use(wlItr->second);
          if(lItr->getBinContent() == kBad){
            status = kBad;
            break;
          }
        }
      }

      if(status == kGood && using_(kLed)){
        DetId id(qItr->getId());
        if(id.subdetId() == EcalEndcap){
          for(map<int, unsigned>::iterator wlItr(ledWlToME_.begin()); wlItr != ledWlToME_.end(); ++wlItr){
            static_cast<MESetMulti const*>(sources_[kLed])->use(wlItr->second);
            if(sources_[kLed]->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }
      }

      if(status == kGood && using_(kTestPulse)){
        tItr = qItr;
        for(map<int, unsigned>::iterator gainItr(tpGainToME_.begin()); gainItr != tpGainToME_.end(); ++gainItr){
          static_cast<MESetMulti const*>(sources_[kTestPulse])->use(gainItr->second);
          if(tItr->getBinContent() == kBad){
            status = kBad;
            break;
          }
        }
      }

      if(status == kGood && using_(kPedestal)){
        pItr = qItr;
        for(map<int, unsigned>::iterator gainItr(pedGainToME_.begin()); gainItr != pedGainToME_.end(); ++gainItr){
          static_cast<MESetMulti const*>(sources_[kPedestal])->use(gainItr->second);
          if(pItr->getBinContent() == kBad){
            status = kBad;
            break;
          }
        }
      }

      qItr->setBinContent(status);
    }

    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;
      for(unsigned iPN(0); iPN < 10; ++iPN){
        int subdet(0);
        if(iDCC >= kEBmLow && iDCC <= kEBpHigh) subdet = EcalBarrel;
        else subdet = EcalEndcap;

        EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

        int status(kGood);

        if(sources_[kPNIntegrity]->getBinContent(id) == kBad) status = kBad;

        if(status == kGood && using_(kLaserPN)){
          for(map<int, unsigned>::iterator wlItr(laserWlToME_.begin()); wlItr != laserWlToME_.end(); ++wlItr){
            static_cast<MESetMulti const*>(sources_[kLaserPN])->use(wlItr->second);
            if(sources_[kLaserPN]->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && using_(kLedPN)){
          for(map<int, unsigned>::iterator wlItr(ledWlToME_.begin()); wlItr != ledWlToME_.end(); ++wlItr){
            static_cast<MESetMulti const*>(sources_[kLedPN])->use(wlItr->second);
            if(sources_[kLedPN]->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && using_(kTestPulsePN)){
          for(map<int, unsigned>::iterator gainItr(tpPNGainToME_.begin()); gainItr != tpPNGainToME_.end(); ++gainItr){
            static_cast<MESetMulti const*>(sources_[kTestPulsePN])->use(gainItr->second);
            if(sources_[kTestPulsePN]->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && using_(kPedestalPN)){
          for(map<int, unsigned>::iterator gainItr(pedPNGainToME_.begin()); gainItr != pedPNGainToME_.end(); ++gainItr){
            static_cast<MESetMulti const*>(sources_[kPedestalPN])->use(gainItr->second);
            if(sources_[kPedestalPN]->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        MEs_[kPNQualitySummary]->setBinContent(id, status);
      }
    }
  }

  /*static*/
  void
  CalibrationSummaryClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["QualitySummary"] = kQualitySummary;
    _nameToIndex["PNQualitySummary"] = kPNQualitySummary;

    _nameToIndex["PNIntegrity"] = kPNIntegrity;
    _nameToIndex["Laser"] = kLaser;
    _nameToIndex["LaserPN"] = kLaserPN;
    _nameToIndex["Led"] = kLed;
    _nameToIndex["LedPN"] = kLedPN;
    _nameToIndex["TestPulse"] = kTestPulse;
    _nameToIndex["TestPulsePN"] = kTestPulsePN;
    _nameToIndex["Pedestal"] = kPedestal;
    _nameToIndex["PedestalPN"] = kPedestalPN;
  }

  DEFINE_ECALDQM_WORKER(CalibrationSummaryClient);
}

