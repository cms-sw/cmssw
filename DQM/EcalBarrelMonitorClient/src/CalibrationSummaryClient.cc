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

    usedSources_.clear();
    usedSources_.insert("PNIntegrity");

    vector<std::string> sourceList(_workerParams.getUntrackedParameter<std::vector<std::string> >("activeSources"));
    for(unsigned iS(0); iS < sourceList.size(); ++iS){
      std::string& sourceName(sourceList[iS]);
      if(sourceName == "Laser"){
        usedSources_.insert("Laser");
        usedSources_.insert("LaserPN");
      }
      else if(sourceName == "Led"){
        usedSources_.insert("Led");
        usedSources_.insert("LedPN");
      }
      else if(sourceName == "TestPulse"){
        usedSources_.insert("TestPulse");
        usedSources_.insert("TestPulsePN");
      }
      else if(sourceName == "Pedestal"){
        usedSources_.insert("Pedestal");
        usedSources_.insert("PedestalPN");
      }
    }

    stringstream ss;
    map<string, string> replacements;

    if(using_("Laser")){
      vector<int> laserWavelengths(_workerParams.getUntrackedParameter<vector<int> >("laserWavelengths"));
      unsigned iMELaserWL(0);
      for(vector<int>::iterator wlItr(laserWavelengths.begin()); wlItr != laserWavelengths.end(); ++wlItr){
        if(*wlItr <= 0 || *wlItr >= 5) throw cms::Exception("InvalidConfiguration") << "Laser Wavelength" << endl;
        laserWlToME_[*wlItr] = iMELaserWL++;
      }
      std::string laserPlots[] = {"Laser", "LaserPN"};
      for(unsigned iP(0); iP < sizeof(laserPlots) / sizeof(std::string); ++iP){
        std::string& plot(laserPlots[iP]);
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

    if(using_("Led")){
      vector<int> ledWavelengths(_workerParams.getUntrackedParameter<vector<int> >("ledWavelengths"));
      unsigned iMELedWL(0);
      for(vector<int>::iterator wlItr(ledWavelengths.begin()); wlItr != ledWavelengths.end(); ++wlItr){
        if(*wlItr <= 0 || *wlItr >= 3) throw cms::Exception("InvalidConfiguration") << "Led Wavelength" << endl;
        ledWlToME_[*wlItr] = iMELedWL++;
      }
      std::string ledPlots[] = {"Led", "LedPN"};
      for(unsigned iP(0); iP < sizeof(ledPlots) / sizeof(std::string); ++iP){
        std::string& plot(ledPlots[iP]);
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

    if(using_("TestPulse")){
      vector<int> tpMGPAGains(_workerParams.getUntrackedParameter<vector<int> >("testPulseMGPAGains"));
      vector<int> tpMGPAGainsPN(_workerParams.getUntrackedParameter<vector<int> >("testPulseMGPAGainsPN"));
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

      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_["TestPulse"]));
      for(map<int, unsigned>::iterator gainItr(tpGainToME_.begin()); gainItr != tpGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }

      multi = static_cast<MESetMulti const*>(sources_["TestPulsePN"]);
      for(map<int, unsigned>::iterator gainItr(tpPNGainToME_.begin()); gainItr != tpPNGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    if(using_("Pedestal")){
      vector<int> pedMGPAGains(_workerParams.getUntrackedParameter<vector<int> >("pedestalMGPAGains"));
      vector<int> pedMGPAGainsPN(_workerParams.getUntrackedParameter<vector<int> >("pedestalMGPAGainsPN"));
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

      MESetMulti const* multi(static_cast<MESetMulti const*>(sources_["Pedestal"]));
      for(map<int, unsigned>::iterator gainItr(pedGainToME_.begin()); gainItr != pedGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }


      multi = static_cast<MESetMulti const*>(sources_["PedestalPN"]);
      for(map<int, unsigned>::iterator gainItr(pedPNGainToME_.begin()); gainItr != pedPNGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("PNQualitySummary");
  }

  void
  CalibrationSummaryClient::producePlots()
  {
    using namespace std;

    MESet* meQualitySummary(MEs_["QualitySummary"]);
    MESet* mePNQualitySummary(MEs_["PNQualitySummary"]);

    MESetMulti const* sLaser(static_cast<MESetMulti const*>(sources_["Laser"]));
    MESetMulti const* sLaserPN(static_cast<MESetMulti const*>(sources_["LaserPN"]));
    MESetMulti const* sLed(static_cast<MESetMulti const*>(sources_["Led"]));
    MESetMulti const* sLedPN(static_cast<MESetMulti const*>(sources_["LedPN"]));
    MESetMulti const* sTestPulse(static_cast<MESetMulti const*>(sources_["TestPulse"]));
    MESetMulti const* sTestPulsePN(static_cast<MESetMulti const*>(sources_["TestPulsePN"]));
    MESetMulti const* sPedestal(static_cast<MESetMulti const*>(sources_["Pedestal"]));
    MESetMulti const* sPedestalPN(static_cast<MESetMulti const*>(sources_["PedestalPN"]));
    MESet const* sPNIntegrity(sources_["PNIntegrity"]);

    bool useLaser(using_("Laser"));
    bool useLaserPN(using_("LaserPN"));
    bool useLed(using_("Led"));
    bool useLedPN(using_("LedPN"));
    bool useTestPulse(using_("TestPulse"));
    bool useTestPulsePN(using_("TestPulsePN"));
    bool usePedestal(using_("Pedestal"));
    bool usePedestalPN(using_("PedestalPN"));

    MESet::iterator qEnd(meQualitySummary->end());
    MESet::const_iterator lItr(sLaser, useLaser ? 0 : -1, 0);
    MESet::const_iterator tItr(sTestPulse, useTestPulse ? 0 : -1, 0);
    MESet::const_iterator pItr(sPedestal, usePedestal ? 0 : -1, 0);
    for(MESet::iterator qItr(meQualitySummary->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      int status(kGood);

      if(status == kGood && useLaser){
        lItr = qItr;
        for(map<int, unsigned>::iterator wlItr(laserWlToME_.begin()); wlItr != laserWlToME_.end(); ++wlItr){
          sLaser->use(wlItr->second);
          if(lItr->getBinContent() == kBad){
            status = kBad;
            break;
          }
        }
      }

      if(status == kGood && useLed){
        DetId id(qItr->getId());
        if(id.subdetId() == EcalEndcap){
          for(map<int, unsigned>::iterator wlItr(ledWlToME_.begin()); wlItr != ledWlToME_.end(); ++wlItr){
            sLed->use(wlItr->second);
            if(sLed->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }
      }

      if(status == kGood && useTestPulse){
        tItr = qItr;
        for(map<int, unsigned>::iterator gainItr(tpGainToME_.begin()); gainItr != tpGainToME_.end(); ++gainItr){
          sTestPulse->use(gainItr->second);
          if(tItr->getBinContent() == kBad){
            status = kBad;
            break;
          }
        }
      }

      if(status == kGood && usePedestal){
        pItr = qItr;
        for(map<int, unsigned>::iterator gainItr(pedGainToME_.begin()); gainItr != pedGainToME_.end(); ++gainItr){
          sPedestal->use(gainItr->second);
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

        if(sPNIntegrity->getBinContent(id) == kBad) status = kBad;

        if(status == kGood && useLaserPN){
          for(map<int, unsigned>::iterator wlItr(laserWlToME_.begin()); wlItr != laserWlToME_.end(); ++wlItr){
            sLaserPN->use(wlItr->second);
            if(sLaserPN->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && useLedPN){
          for(map<int, unsigned>::iterator wlItr(ledWlToME_.begin()); wlItr != ledWlToME_.end(); ++wlItr){
            sLedPN->use(wlItr->second);
            if(sLedPN->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && useTestPulsePN){
          for(map<int, unsigned>::iterator gainItr(tpPNGainToME_.begin()); gainItr != tpPNGainToME_.end(); ++gainItr){
            sTestPulsePN->use(gainItr->second);
            if(sTestPulsePN->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && usePedestalPN){
          for(map<int, unsigned>::iterator gainItr(pedPNGainToME_.begin()); gainItr != pedPNGainToME_.end(); ++gainItr){
            sPedestalPN->use(gainItr->second);
            if(sPedestalPN->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        mePNQualitySummary->setBinContent(id, status);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(CalibrationSummaryClient);
}

