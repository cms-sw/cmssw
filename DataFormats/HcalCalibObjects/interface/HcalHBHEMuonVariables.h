#ifndef DataFormatsHcalCalibObjectsHcalHBHEMuonVariables_h
#define DataFormatsHcalCalibObjectsHcalHBHEMuonVariables_h
#include <string>
#include <vector>

class HcalHBHEMuonVariables {
public:
  HcalHBHEMuonVariables() { clear(); }

  void clear() {
    runNumber_ = eventNumber_ = lumiNumber_ = bxNumber_ = goodVertex_ = 0;
    muonGood_ = muonGlobal_ = muonTracker_ = muonTight_ = muonMedium_ = false;
    ptGlob_ = etaGlob_ = phiGlob_ = energyMuon_ = pMuon_ = 0;
    muonTrkKink_ = muonChi2LocalPosition_ = muonSegComp_ = 0;
    trackerLayer_ = numPixelLayers_ = tightPixelHits_ = 0;
    innerTrack_ = outerTrack_ = globalTrack_ = false;
    chiTracker_ = dxyTracker_ = dzTracker_ = 0;
    innerTrackPt_ = innerTrackEta_ = innerTrackPhi_ = 0;
    outerTrackHits_ = outerTrackRHits_ = 0;
    outerTrackPt_ = outerTrackEta_ = outerTrackPhi_ = outerTrackChi_ = 0;
    globalMuonHits_ = matchedStat_ = 0;
    globalTrackPt_ = globalTrackEta_ = globalTrackPhi_ = chiGlobal_ = 0;
    tightValidFraction_ = tightLongPara_ = tightTransImpara_ = 0;
    isolationR04_ = isolationR03_ = 0;
    ecalDetId_ = 0;
    ecalEnergy_ = ecal3x3Energy_ = 0;
    hcalDetId_ = ehcalDetId_ = 0;
    matchedId_ = hcalHot_ = false;
    hcalIeta_ = hcalIphi_ = 0;
    hcalEnergy_ = hoEnergy_ = 0;
    hcal1x1Energy_ = hcal1x1EnergyAux_ = hcal1x1EnergyRaw_ = 0;
    hcalDepthActiveLength_.clear();
    hcalDepthActiveLengthHot_.clear();
    hcalDepthEnergy_.clear();
    hcalDepthEnergyHot_.clear();
    hcalDepthEnergyCorr_.clear();
    hcalDepthEnergyHotCorr_.clear();
    hcalDepthChargeHot_.clear();
    hcalDepthChargeHotBG_.clear();
    hcalDepthEnergyAux_.clear();
    hcalDepthEnergyHotAux_.clear();
    hcalDepthEnergyCorrAux_.clear();
    hcalDepthEnergyHotCorrAux_.clear();
    hcalDepthChargeHotAux_.clear();
    hcalDepthChargeHotBGAux_.clear();
    hcalDepthEnergyRaw_.clear();
    hcalDepthEnergyHotRaw_.clear();
    hcalDepthEnergyCorrRaw_.clear();
    hcalDepthEnergyHotCorrRaw_.clear();
    hcalDepthChargeHotRaw_.clear();
    hcalDepthChargeHotBGRaw_.clear();
    hcalDepthMatch_.clear();
    hcalDepthMatchHot_.clear();
    hcalActiveLength_ = hcalActiveLengthHot_ = 0;
    allTriggers_.clear();
    hltResults_.clear();
  }

  unsigned int runNumber_, eventNumber_, lumiNumber_, bxNumber_, goodVertex_;
  bool muonGood_, muonGlobal_, muonTracker_, muonTight_, muonMedium_;
  float ptGlob_, etaGlob_, phiGlob_, energyMuon_, pMuon_;
  float muonTrkKink_, muonChi2LocalPosition_, muonSegComp_;
  int trackerLayer_, numPixelLayers_, tightPixelHits_;
  bool innerTrack_, outerTrack_, globalTrack_;
  float chiTracker_, dxyTracker_, dzTracker_;
  float innerTrackPt_, innerTrackEta_, innerTrackPhi_;
  int outerTrackHits_, outerTrackRHits_;
  float outerTrackPt_, outerTrackEta_, outerTrackPhi_, outerTrackChi_;
  int globalMuonHits_, matchedStat_;
  float globalTrackPt_, globalTrackEta_, globalTrackPhi_, chiGlobal_;
  float tightValidFraction_, tightLongPara_, tightTransImpara_;
  float isolationR04_, isolationR03_;
  unsigned int ecalDetId_;
  float ecalEnergy_, ecal3x3Energy_;
  unsigned int hcalDetId_, ehcalDetId_;
  bool matchedId_, hcalHot_;
  int hcalIeta_, hcalIphi_;
  float hcalEnergy_, hoEnergy_;
  float hcal1x1Energy_, hcal1x1EnergyAux_, hcal1x1EnergyRaw_;
  std::vector<float> hcalDepthActiveLength_, hcalDepthActiveLengthHot_;
  std::vector<float> hcalDepthEnergy_, hcalDepthEnergyHot_;
  std::vector<float> hcalDepthEnergyCorr_, hcalDepthEnergyHotCorr_;
  std::vector<float> hcalDepthChargeHot_, hcalDepthChargeHotBG_;
  std::vector<float> hcalDepthEnergyAux_, hcalDepthEnergyHotAux_;
  std::vector<float> hcalDepthEnergyCorrAux_, hcalDepthEnergyHotCorrAux_;
  std::vector<float> hcalDepthChargeHotAux_, hcalDepthChargeHotBGAux_;
  std::vector<float> hcalDepthEnergyRaw_, hcalDepthEnergyHotRaw_;
  std::vector<float> hcalDepthEnergyCorrRaw_, hcalDepthEnergyHotCorrRaw_;
  std::vector<float> hcalDepthChargeHotRaw_, hcalDepthChargeHotBGRaw_;
  std::vector<bool> hcalDepthMatch_, hcalDepthMatchHot_;
  float hcalActiveLength_, hcalActiveLengthHot_;
  std::vector<std::string> allTriggers_;
  std::vector<int> hltResults_;
};

typedef std::vector<HcalHBHEMuonVariables> HcalHBHEMuonVariablesCollection;
#endif
