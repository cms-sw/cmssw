#include "../interface/CalibrationSummaryClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>

namespace ecaldqm
{
  CalibrationSummaryClient::CalibrationSummaryClient() :
    DQWorkerClient(),
    laserWlToME_(),
    ledWlToME_(),
    tpGainToME_(),
    tpPNGainToME_(),
    pedGainToME_(),
    pedPNGainToME_()
  {
  }

  void
  CalibrationSummaryClient::setParams(edm::ParameterSet const& _params)
  {
    std::vector<std::string> sourceList(_params.getUntrackedParameter<std::vector<std::string> >("activeSources"));
    if(std::find(sourceList.begin(), sourceList.end(), "Laser") == sourceList.end()){
      sources_.erase(std::string("Laser"));
      sources_.erase(std::string("LaserPN"));
    }
    if(std::find(sourceList.begin(), sourceList.end(), "Led") == sourceList.end()){
      sources_.erase(std::string("Led"));
      sources_.erase(std::string("LedPN"));
    }
    if(std::find(sourceList.begin(), sourceList.end(), "TestPulse") == sourceList.end()){
      sources_.erase(std::string("TestPulse"));
      sources_.erase(std::string("TestPulsePN"));
    }
    if(std::find(sourceList.begin(), sourceList.end(), "Pedestal") == sourceList.end()){
      sources_.erase(std::string("Pedestal"));
      sources_.erase(std::string("PedestalPN"));
    }

    MESet::PathReplacements repl;

    if(using_("Laser")){
      repl.clear();
      std::vector<int> laserWavelengths(_params.getUntrackedParameter<std::vector<int> >("laserWavelengths"));
      MESetMulti const& laser(static_cast<MESetMulti const&>(sources_.at("Laser")));
      unsigned nWL(laserWavelengths.size());
      for(unsigned iWL(0); iWL != nWL; ++iWL){
        int wl(laserWavelengths[iWL]);
        if(wl <= 0 || wl >= 5) throw cms::Exception("InvalidConfiguration") << "Laser Wavelength";
        repl["wl"] = std::to_string(wl);
        laserWlToME_[wl] = laser.getIndex(repl);
      }
    }

    if(using_("Led")){
      repl.clear();
      std::vector<int> ledWavelengths(_params.getUntrackedParameter<std::vector<int> >("ledWavelengths"));
      MESetMulti const& led(static_cast<MESetMulti const&>(sources_.at("Led")));
      unsigned nWL(ledWavelengths.size());
      for(unsigned iWL(0); iWL != nWL; ++iWL){
        int wl(ledWavelengths[iWL]);
        if(wl <= 0 || wl >= 5) throw cms::Exception("InvalidConfiguration") << "Led Wavelength";
        repl["wl"] = std::to_string(wl);
        ledWlToME_[wl] = led.getIndex(repl);
      }
    }

    if(using_("TestPulse")){
      repl.clear();
      std::vector<int> tpMGPAGains(_params.getUntrackedParameter<std::vector<int> >("testPulseMGPAGains"));
      MESetMulti const& tp(static_cast<MESetMulti const&>(sources_.at("TestPulse")));
      unsigned nG(tpMGPAGains.size());
      for(unsigned iG(0); iG != nG; ++iG){
        int gain(tpMGPAGains[iG]);
        if(gain != 1 && gain != 6 && gain != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain";
        repl["gain"] = std::to_string(gain);
        tpGainToME_[gain] = tp.getIndex(repl);
      }

      repl.clear();
      std::vector<int> tpMGPAGainsPN(_params.getUntrackedParameter<std::vector<int> >("testPulseMGPAGainsPN"));
      MESetMulti const& tppn(static_cast<MESetMulti const&>(sources_.at("TestPulsePN")));
      unsigned nGPN(tpMGPAGainsPN.size());
      for(unsigned iG(0); iG != nGPN; ++iG){
        int gain(tpMGPAGainsPN[iG]);
        if(gain != 1 && gain != 16) throw cms::Exception("InvalidConfiguration") << "PN MGPA gain";
        repl["pngain"] = std::to_string(gain);
        tpPNGainToME_[gain] = tppn.getIndex(repl);
      }
    }

    if(using_("Pedestal")){
      repl.clear();
      std::vector<int> pedMGPAGains(_params.getUntrackedParameter<std::vector<int> >("pedestalMGPAGains"));
      MESetMulti const& ped(static_cast<MESetMulti const&>(sources_.at("Pedestal")));
      unsigned nG(pedMGPAGains.size());
      for(unsigned iG(0); iG != nG; ++iG){
        int gain(pedMGPAGains[iG]);
        if(gain != 1 && gain != 6 && gain != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain";
        repl["gain"] = std::to_string(gain);
        pedGainToME_[gain] = ped.getIndex(repl);
      }

      repl.clear();
      std::vector<int> pedMGPAGainsPN(_params.getUntrackedParameter<std::vector<int> >("pedestalMGPAGainsPN"));
      MESetMulti const& pedpn(static_cast<MESetMulti const&>(sources_.at("PedestalPN")));
      unsigned nGPN(pedMGPAGainsPN.size());
      for(unsigned iG(0); iG != nGPN; ++iG){
        int gain(pedMGPAGainsPN[iG]);
        if(gain != 1 && gain != 16) throw cms::Exception("InvalidConfiguration") << "PN MGPA gain";
        repl["pngain"] = std::to_string(gain);
        pedPNGainToME_[gain] = pedpn.getIndex(repl);
      }
    }

    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("PNQualitySummary");
  }

  void
  CalibrationSummaryClient::producePlots(ProcessType)
  {
    using namespace std;

    MESet& meQualitySummary(MEs_.at("QualitySummary"));
    MESet& mePNQualitySummary(MEs_.at("PNQualitySummary"));

    MESetMulti const* sLaser(using_("Laser") ? static_cast<MESetMulti const*>(&sources_.at("Laser")) : 0);
    MESetMulti const* sLaserPN(using_("LaserPN") ? static_cast<MESetMulti const*>(&sources_.at("LaserPN")) : 0);
    MESetMulti const* sLed(using_("Led") ? static_cast<MESetMulti const*>(&sources_.at("Led")) : 0);
    MESetMulti const* sLedPN(using_("LedPN") ? static_cast<MESetMulti const*>(&sources_.at("LedPN")) : 0);
    MESetMulti const* sTestPulse(using_("TestPulse") ? static_cast<MESetMulti const*>(&sources_.at("TestPulse")) : 0);
    MESetMulti const* sTestPulsePN(using_("TestPulsePN") ? static_cast<MESetMulti const*>(&sources_.at("TestPulsePN")) : 0);
    MESetMulti const* sPedestal(using_("Pedestal") ? static_cast<MESetMulti const*>(&sources_.at("Pedestal")) : 0);
    MESetMulti const* sPedestalPN(using_("PedestalPN") ? static_cast<MESetMulti const*>(&sources_.at("PedestalPN")) : 0);
    MESet const& sPNIntegrity(sources_.at("PNIntegrity"));

    MESet::iterator qEnd(meQualitySummary.end());
    for(MESet::iterator qItr(meQualitySummary.beginChannel()); qItr != qEnd; qItr.toNextChannel()){
      DetId id(qItr->getId());

      int status(kGood);

      if(status == kGood && sLaser){
        for(map<int, unsigned>::iterator wlItr(laserWlToME_.begin()); wlItr != laserWlToME_.end(); ++wlItr){
          sLaser->use(wlItr->second);
          if(sLaser->getBinContent(id) == kBad){
            status = kBad;
            break;
          }
        }
      }

      if(status == kGood && sLed){
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

      if(status == kGood && sTestPulse){
        for(map<int, unsigned>::iterator gainItr(tpGainToME_.begin()); gainItr != tpGainToME_.end(); ++gainItr){
          sTestPulse->use(gainItr->second);
          if(sTestPulse->getBinContent(id) == kBad){
            status = kBad;
            break;
          }
        }
      }

      if(status == kGood && sPedestal){
        for(map<int, unsigned>::iterator gainItr(pedGainToME_.begin()); gainItr != pedGainToME_.end(); ++gainItr){
          sPedestal->use(gainItr->second);
          if(sPedestal->getBinContent(id) == kBad){
            status = kBad;
            break;
          }
        }
      }

      qItr->setBinContent(status);
    }

    for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){
      if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;
      for(unsigned iPN(0); iPN < 10; ++iPN){
        int subdet(0);
        if(iDCC >= kEBmLow && iDCC <= kEBpHigh) subdet = EcalBarrel;
        else subdet = EcalEndcap;

        EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

        int status(kGood);

        if(sPNIntegrity.getBinContent(id) == kBad) status = kBad;

        if(status == kGood && sLaserPN){
          for(map<int, unsigned>::iterator wlItr(laserWlToME_.begin()); wlItr != laserWlToME_.end(); ++wlItr){
            sLaserPN->use(wlItr->second);
            if(sLaserPN->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && sLedPN){
          for(map<int, unsigned>::iterator wlItr(ledWlToME_.begin()); wlItr != ledWlToME_.end(); ++wlItr){
            sLedPN->use(wlItr->second);
            if(sLedPN->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && sTestPulsePN){
          for(map<int, unsigned>::iterator gainItr(tpPNGainToME_.begin()); gainItr != tpPNGainToME_.end(); ++gainItr){
            sTestPulsePN->use(gainItr->second);
            if(sTestPulsePN->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        if(status == kGood && sPedestalPN){
          for(map<int, unsigned>::iterator gainItr(pedPNGainToME_.begin()); gainItr != pedPNGainToME_.end(); ++gainItr){
            sPedestalPN->use(gainItr->second);
            if(sPedestalPN->getBinContent(id) == kBad){
              status = kBad;
              break;
            }
          }
        }

        mePNQualitySummary.setBinContent(id, status);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(CalibrationSummaryClient);
}
