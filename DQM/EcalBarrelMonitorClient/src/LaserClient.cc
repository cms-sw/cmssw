#include "../interface/LaserClient.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <cmath>

namespace ecaldqm {

  LaserClient::LaserClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "LaserClient"),
    laserWavelengths_(),
    MGPAGainsPN_(),
    minChannelEntries_(0),
    expectedAmplitude_(),
    amplitudeThreshold_(),
    amplitudeRMSThreshold_(),
    expectedTiming_(),
    timingThreshold_(),
    timingRMSThreshold_(),
    expectedPNAmplitude_(),
    pnAmplitudeThreshold_(),
    pnAmplitudeRMSThreshold_(),
    towerThreshold_()
  {
    using namespace std;

    edm::ParameterSet const& commonParams(_params.getUntrackedParameterSet("Common"));
    MGPAGainsPN_ = commonParams.getUntrackedParameter<vector<int> >("MGPAGainsPN");

    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));
    laserWavelengths_ = taskParams.getUntrackedParameter<vector<int> >("laserWavelengths");
    minChannelEntries_ = taskParams.getUntrackedParameter<int>("minChannelEntries");
    expectedAmplitude_ = taskParams.getUntrackedParameter<vector<double> >("expectedAmplitude");
    amplitudeThreshold_ = taskParams.getUntrackedParameter<vector<double> >("amplitudeThreshold");
    amplitudeRMSThreshold_ = taskParams.getUntrackedParameter<vector<double> >("amplitudeRMSThreshold");
    expectedTiming_ = taskParams.getUntrackedParameter<vector<double> >("expectedTiming");
    timingThreshold_ = taskParams.getUntrackedParameter<vector<double> >("timingThreshold");
    timingRMSThreshold_ = taskParams.getUntrackedParameter<vector<double> >("timingRMSThreshold");
    expectedPNAmplitude_ = taskParams.getUntrackedParameter<vector<double> >("expectedPNAmplitude");
    pnAmplitudeThreshold_ = taskParams.getUntrackedParameter<vector<double> >("pnAmplitudeThreshold");
    pnAmplitudeRMSThreshold_ = taskParams.getUntrackedParameter<vector<double> >("pnAmplitudeRMSThreshold");
    towerThreshold_ = taskParams.getUntrackedParameter<double>("towerThreshold");

    for(vector<int>::iterator wlItr(laserWavelengths_.begin()); wlItr != laserWavelengths_.end(); ++wlItr)
      if(*wlItr <= 0 || *wlItr >= 5) throw cms::Exception("InvalidConfiguration") << "Laser Wavelength" << endl;

    for(vector<int>::iterator gainItr(MGPAGainsPN_.begin()); gainItr != MGPAGainsPN_.end(); ++gainItr)
      if(*gainItr != 1 && *gainItr != 16) throw cms::Exception("InvalidConfiguration") << "PN diode gain" << endl;

    if(expectedAmplitude_.size() != nWL ||
       amplitudeThreshold_.size() != nWL ||
       amplitudeRMSThreshold_.size() != nWL ||
       expectedTiming_.size() != nWL ||
       timingThreshold_.size() != nWL ||
       timingRMSThreshold_.size() != nWL ||
       expectedPNAmplitude_.size() != nWL * nPNGain ||
       pnAmplitudeThreshold_.size() != nWL * nPNGain ||
       pnAmplitudeRMSThreshold_.size() != nWL * nPNGain)
      throw cms::Exception("InvalidConfiguration") << "Size of quality cut parameter vectors" << endl;

    map<string, string> replacements;
    stringstream ss;

    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    for(vector<int>::iterator wlItr(laserWavelengths_.begin()); wlItr != laserWavelengths_.end(); ++wlItr){
      ss.str("");
      ss << *wlItr;
      replacements["wl"] = ss.str();

      unsigned offset(*wlItr - 1);
      source_(sAmplitude + offset, "LaserTask", LaserTask::kAmplitude + offset, sources);
      source_(sTiming + offset, "LaserTask", LaserTask::kTiming + offset, sources);

      sources_[sAmplitude + offset]->name(replacements);
      sources_[sTiming + offset]->name(replacements);

      for(vector<int>::iterator gainItr(MGPAGainsPN_.begin()); gainItr != MGPAGainsPN_.end(); ++gainItr){
	ss.str("");
	ss << *gainItr;
	replacements["pngain"] = ss.str();

	offset = (*wlItr - 1) * nPNGain + (*gainItr == 1 ? 0 : 1);
	source_(sPNAmplitude + offset, "LaserTask", LaserTask::kPNAmplitude, sources);

	sources_[sPNAmplitude + offset]->name(replacements);
      }
    }

    BinService::AxisSpecs axis;

    for(vector<int>::iterator wlItr(laserWavelengths_.begin()); wlItr != laserWavelengths_.end(); ++wlItr){
      ss.str("");
      ss << *wlItr;
      replacements["wl"] = ss.str();

      unsigned offset(*wlItr - 1);

      MEs_[kQuality + offset]->name(replacements);
      MEs_[kQualitySummary + offset]->name(replacements);
      MEs_[kAmplitudeMean + offset]->name(replacements);
      MEs_[kAmplitudeRMS + offset]->name(replacements);
      MEs_[kTimingMean + offset]->name(replacements);
      MEs_[kTimingRMS + offset]->name(replacements);
      MEs_[kPNQualitySummary + offset]->name(replacements);

      for(vector<int>::iterator gainItr(MGPAGainsPN_.begin()); gainItr != MGPAGainsPN_.end(); ++gainItr){
	ss.str("");
	ss << *gainItr;
	replacements["pngain"] = ss.str();

	offset = (*wlItr - 1) * nPNGain + (*gainItr == 1 ? 0 : 1);

	MEs_[kPNAmplitudeMean + offset]->name(replacements);
	MEs_[kPNAmplitudeRMS + offset]->name(replacements);
      }
    }
  }

  void
  LaserClient::initialize()
  {
    initialized_ = true;

    for(std::vector<int>::iterator wlItr(laserWavelengths_.begin()); wlItr != laserWavelengths_.end(); ++wlItr){
      unsigned offset(*wlItr - 1);

      initialized_ &= sources_[sAmplitude + offset]->retrieve();
      initialized_ &= sources_[sTiming + offset]->retrieve();

      for(std::vector<int>::iterator gainItr(MGPAGainsPN_.begin()); gainItr != MGPAGainsPN_.end(); ++gainItr){
	offset = (*wlItr - 1) * nPNGain + (*gainItr == 1 ? 0 : 1);

	initialized_ &= sources_[sPNAmplitude + offset]->retrieve();
      }
    }
  }

  void
  LaserClient::bookMEs()
  {
    for(std::vector<int>::iterator wlItr(laserWavelengths_.begin()); wlItr != laserWavelengths_.end(); ++wlItr){
      unsigned offset(*wlItr - 1);

      MEs_[kQuality + offset]->book();
      MEs_[kQualitySummary + offset]->book();
      MEs_[kAmplitudeMean + offset]->book();
      MEs_[kAmplitudeRMS + offset]->book();
      MEs_[kTimingMean + offset]->book();
      MEs_[kTimingRMS + offset]->book();
      MEs_[kPNQualitySummary + offset]->book();

      for(std::vector<int>::iterator gainItr(MGPAGainsPN_.begin()); gainItr != MGPAGainsPN_.end(); ++gainItr){
	offset = (*wlItr - 1) * nPNGain + (*gainItr == 1 ? 0 : 1);

	MEs_[kPNAmplitudeMean + offset]->book();
	MEs_[kPNAmplitudeRMS + offset]->book();
      }
    }
  }

  void
  LaserClient::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    for(std::vector<int>::iterator wlItr(laserWavelengths_.begin()); wlItr != laserWavelengths_.end(); ++wlItr){
      unsigned offset(*wlItr - 1);
      MEs_[kQuality + offset]->resetAll(-1.);
      MEs_[kQualitySummary + offset]->resetAll(-1.);

      for(std::vector<int>::iterator gainItr(MGPAGainsPN_.begin()); gainItr != MGPAGainsPN_.end(); ++gainItr){
	offset = (*wlItr - 1) * nPNGain + (*gainItr == 1 ? 0 : 1);
	MEs_[kPNQualitySummary + offset]->resetAll(-1.);
      }
    }
  }

  void
  LaserClient::producePlots()
  {
    using namespace std;

    for(vector<int>::iterator wlItr(laserWavelengths_.begin()); wlItr != laserWavelengths_.end(); ++wlItr){

      unsigned offset(*wlItr - 1);

      MEs_[kQuality + offset]->reset(2.);
      MEs_[kQualitySummary + offset]->reset(2.);
      MEs_[kPNQualitySummary + offset]->reset(2.);

      MEs_[kAmplitudeMean + offset]->reset();
      MEs_[kAmplitudeRMS + offset]->reset();
      MEs_[kTimingMean + offset]->reset();
      MEs_[kTimingRMS + offset]->reset();

      for(vector<int>::iterator gainItr(MGPAGainsPN_.begin()); gainItr != MGPAGainsPN_.end(); ++gainItr){
	unsigned suboffset((*wlItr - 1) * nPNGain + (*gainItr == 1 ? 0 : 1));

	MEs_[kPNAmplitudeMean + suboffset]->reset();
	MEs_[kPNAmplitudeRMS + suboffset]->reset();
      }

      for(unsigned dccid(1); dccid <= 54; dccid++){

	for(unsigned tower(1); tower <= getNSuperCrystals(dccid); tower++){
	  std::vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	  if(ids.size() == 0) continue;

	  float nBad(0.);

	  for(std::vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	    float aEntries(sources_[sAmplitude + offset]->getBinEntries(*idItr));

	    if(aEntries < minChannelEntries_) continue;

	    float aMean(sources_[sAmplitude + offset]->getBinContent(*idItr));
	    float aRms(sources_[sAmplitude + offset]->getBinError(*idItr) * std::sqrt(aEntries));

	    MEs_[kAmplitudeMean + offset]->fill(*idItr, aMean);
	    MEs_[kAmplitudeRMS + offset]->fill(*idItr, aRms);

	    float tEntries(sources_[sTiming + offset]->getBinEntries(*idItr));

	    if(tEntries < minChannelEntries_) continue;

	    float tMean(sources_[sTiming + offset]->getBinContent(*idItr));
	    float tRms(sources_[sTiming + offset]->getBinError(*idItr) * std::sqrt(tEntries));

	    MEs_[kTimingMean + offset]->fill(*idItr, tMean);
	    MEs_[kTimingRMS + offset]->fill(*idItr, tRms);

	    if(std::abs(aMean - expectedAmplitude_[offset]) > amplitudeThreshold_[offset] || aRms > amplitudeRMSThreshold_[offset] ||
	       std::abs(tMean - expectedTiming_[offset]) > timingThreshold_[offset] || tRms > timingRMSThreshold_[offset]){
	      MEs_[kQuality + offset]->setBinContent(*idItr, 0.);
	      nBad += 1.;
	    }
	    else
	      MEs_[kQuality + offset]->setBinContent(*idItr, 1.);
	  }

	  if(nBad / ids.size() > towerThreshold_)
	    MEs_[kQualitySummary + offset]->setBinContent(ids[0], 0.);
	  else
	    MEs_[kQualitySummary + offset]->setBinContent(ids[0], 1.);
	}

	unsigned subdet;
	if(dccid <= 9 || dccid >= 46) subdet = EcalEndcap;
	else subdet = EcalBarrel;

	for(unsigned pn(1); pn <= 10; pn++){
	  EcalPnDiodeDetId pnid(subdet, dccid, pn);

	  bool bad(false);
	  for(vector<int>::iterator gainItr(MGPAGainsPN_.begin()); gainItr != MGPAGainsPN_.end(); ++gainItr){
	    unsigned suboffset((*wlItr - 1) * nPNGain + (*gainItr == 1 ? 0 : 1));

	    float pEntries(sources_[sPNAmplitude + suboffset]->getBinEntries(pnid));

	    if(pEntries < minChannelEntries_) continue;

	    float pMean(sources_[sPNAmplitude + suboffset]->getBinContent(pnid));
	    float pRms(sources_[sPNAmplitude + suboffset]->getBinError(pnid) * std::sqrt(pEntries));

	    MEs_[kPNAmplitudeMean + suboffset]->fill(pnid, pMean);
	    MEs_[kPNAmplitudeRMS + suboffset]->fill(pnid, pRms);

	    if(std::abs(pMean - expectedPNAmplitude_[suboffset]) > pnAmplitudeThreshold_[suboffset] || pRms > pnAmplitudeRMSThreshold_[suboffset])
	      bad = true;
	  }

	  if(bad)
	    MEs_[kPNQualitySummary + offset]->setBinContent(pnid, 0.);
	  else
	    MEs_[kPNQualitySummary + offset]->setBinContent(pnid, 1.);
	}

      }
    }
  }

  /*static*/
  void
  LaserClient::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs axis;

    for(unsigned iWL(0); iWL < nWL; iWL++){
      _data[kQuality + iWL] = MEData("Quality", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);

      axis.nbins = 100;
      axis.low = 0.;
      axis.high = 4096.;
      _data[kAmplitudeMean + iWL] = MEData("AmplitudeMean", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

      axis.low = 0.;
      axis.high = 400.;
      _data[kAmplitudeRMS + iWL] = MEData("AmplitudeRMS", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

      axis.low = 3.5;
      axis.high = 5.5;
      _data[kTimingMean + iWL] = MEData("TimingMean", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

      axis.low = 0.;
      axis.high = 0.5;
      _data[kTimingRMS + iWL] = MEData("TimingRMS", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

      _data[kQualitySummary + iWL] = MEData("QualitySummary", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
      _data[kPNQualitySummary + iWL] = MEData("PNQualitySummary", BinService::kEcalMEM2P, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);

      for(unsigned iPNGain(0); iPNGain < nPNGain; iPNGain++){
	unsigned offset(iWL * nPNGain + iPNGain);

	axis.low = 0.;
	axis.high = 4096.;
	_data[kPNAmplitudeMean + offset] = MEData("PNAmplitudeMean", BinService::kSMMEM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

	axis.low = 0.;
	axis.high = 200.;
	_data[kPNAmplitudeRMS + offset] = MEData("PNAmplitudeRMS", BinService::kSMMEM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(LaserClient);
}
