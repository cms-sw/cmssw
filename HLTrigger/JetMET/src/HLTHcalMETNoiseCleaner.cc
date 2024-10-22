// -*- C++ -*-
//
// Class:      HLTHcalMETNoiseCleaner
//
/**\class HLTHcalMETNoiseCleaner

 Description: HLT filter module for cleaning HCal Noise from MET or MHT

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alexander Mott
//         Created:  Mon Nov 21 11:32:00 CEST 2011
//
//
//

#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseCleaner.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/METReco/interface/SpecificCaloMETData.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>
#include <fstream>
#include <TVector3.h>
#include <TLorentzVector.h>
//#include <Point.h>

HLTHcalMETNoiseCleaner::HLTHcalMETNoiseCleaner(const edm::ParameterSet& iConfig)
    : HcalNoiseRBXCollectionTag_(iConfig.getParameter<edm::InputTag>("HcalNoiseRBXCollection")),
      CaloMetCollectionTag_(iConfig.getParameter<edm::InputTag>("CaloMetCollection")),
      CaloMetCut_(iConfig.getParameter<double>("CaloMetCut")),
      severity_(iConfig.getParameter<int>("severity")),
      maxNumRBXs_(iConfig.getParameter<int>("maxNumRBXs")),
      numRBXsToConsider_(iConfig.getParameter<int>("numRBXsToConsider")),
      accept2NoiseRBXEvents_(iConfig.getParameter<bool>("accept2NoiseRBXEvents")),
      needEMFCoincidence_(iConfig.getParameter<bool>("needEMFCoincidence")),
      minRBXEnergy_(iConfig.getParameter<double>("minRBXEnergy")),
      minRatio_(iConfig.getParameter<double>("minRatio")),
      maxRatio_(iConfig.getParameter<double>("maxRatio")),
      minHPDHits_(iConfig.getParameter<int>("minHPDHits")),
      minRBXHits_(iConfig.getParameter<int>("minRBXHits")),
      minHPDNoOtherHits_(iConfig.getParameter<int>("minHPDNoOtherHits")),
      minZeros_(iConfig.getParameter<int>("minZeros")),
      minHighEHitTime_(iConfig.getParameter<double>("minHighEHitTime")),
      maxHighEHitTime_(iConfig.getParameter<double>("maxHighEHitTime")),
      maxRBXEMF_(iConfig.getParameter<double>("maxRBXEMF")),
      minRecHitE_(iConfig.getParameter<double>("minRecHitE")),
      minLowHitE_(iConfig.getParameter<double>("minLowHitE")),
      minHighHitE_(iConfig.getParameter<double>("minHighHitE")),
      minR45HitE_(iConfig.getParameter<double>("minR45HitE")),
      TS4TS5EnergyThreshold_(iConfig.getParameter<double>("TS4TS5EnergyThreshold")) {
  std::vector<double> TS4TS5UpperThresholdTemp = iConfig.getParameter<std::vector<double> >("TS4TS5UpperThreshold");
  std::vector<double> TS4TS5UpperCutTemp = iConfig.getParameter<std::vector<double> >("TS4TS5UpperCut");
  std::vector<double> TS4TS5LowerThresholdTemp = iConfig.getParameter<std::vector<double> >("TS4TS5LowerThreshold");
  std::vector<double> TS4TS5LowerCutTemp = iConfig.getParameter<std::vector<double> >("TS4TS5LowerCut");

  for (int i = 0; i < (int)TS4TS5UpperThresholdTemp.size() && i < (int)TS4TS5UpperCutTemp.size(); i++)
    TS4TS5UpperCut_.push_back(std::pair<double, double>(TS4TS5UpperThresholdTemp[i], TS4TS5UpperCutTemp[i]));
  sort(TS4TS5UpperCut_.begin(), TS4TS5UpperCut_.end());

  for (int i = 0; i < (int)TS4TS5LowerThresholdTemp.size() && i < (int)TS4TS5LowerCutTemp.size(); i++)
    TS4TS5LowerCut_.push_back(std::pair<double, double>(TS4TS5LowerThresholdTemp[i], TS4TS5LowerCutTemp[i]));
  sort(TS4TS5LowerCut_.begin(), TS4TS5LowerCut_.end());

  m_theCaloMetToken = consumes<reco::CaloMETCollection>(CaloMetCollectionTag_);
  m_theHcalNoiseToken = consumes<reco::HcalNoiseRBXCollection>(HcalNoiseRBXCollectionTag_);

  produces<reco::CaloMETCollection>();
}

HLTHcalMETNoiseCleaner::~HLTHcalMETNoiseCleaner() = default;

void HLTHcalMETNoiseCleaner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HcalNoiseRBXCollection", edm::InputTag("hltHcalNoiseInfoProducer"));
  desc.add<edm::InputTag>("CaloMetCollection", edm::InputTag("hltMet"));
  desc.add<double>("CaloMetCut", 0.0);
  desc.add<int>("severity", 1);
  desc.add<int>("maxNumRBXs", 2);
  desc.add<int>("numRBXsToConsider", 2);
  desc.add<bool>("accept2NoiseRBXEvents", true);
  desc.add<bool>("needEMFCoincidence", true);
  desc.add<double>("minRBXEnergy", 50.0);
  desc.add<double>("minRatio", -999.);
  desc.add<double>("maxRatio", 999.);
  desc.add<int>("minHPDHits", 17);
  desc.add<int>("minRBXHits", 999);
  desc.add<int>("minHPDNoOtherHits", 10);
  desc.add<int>("minZeros", 10);
  desc.add<double>("minHighEHitTime", -9999.0);
  desc.add<double>("maxHighEHitTime", 9999.0);
  desc.add<double>("maxRBXEMF", 0.02);
  desc.add<double>("minRecHitE", 1.5);
  desc.add<double>("minLowHitE", 10.0);
  desc.add<double>("minHighHitE", 25.0);
  desc.add<double>("minR45HitE", 5.0);
  desc.add<double>("TS4TS5EnergyThreshold", 50.0);

  double TS4TS5UpperThresholdArray[5] = {70, 90, 100, 400, 4000};
  double TS4TS5UpperCutArray[5] = {1, 0.8, 0.75, 0.72, 0.72};
  double TS4TS5LowerThresholdArray[7] = {100, 120, 150, 200, 300, 400, 500};
  double TS4TS5LowerCutArray[7] = {-1, -0.7, -0.4, -0.2, -0.08, 0, 0.1};
  std::vector<double> TS4TS5UpperThreshold(TS4TS5UpperThresholdArray, TS4TS5UpperThresholdArray + 5);
  std::vector<double> TS4TS5UpperCut(TS4TS5UpperCutArray, TS4TS5UpperCutArray + 5);
  std::vector<double> TS4TS5LowerThreshold(TS4TS5LowerThresholdArray, TS4TS5LowerThresholdArray + 7);
  std::vector<double> TS4TS5LowerCut(TS4TS5LowerCutArray, TS4TS5LowerCutArray + 7);

  desc.add<std::vector<double> >("TS4TS5UpperThreshold", TS4TS5UpperThreshold);
  desc.add<std::vector<double> >("TS4TS5UpperCut", TS4TS5UpperCut);
  desc.add<std::vector<double> >("TS4TS5LowerThreshold", TS4TS5LowerThreshold);
  desc.add<std::vector<double> >("TS4TS5LowerCut", TS4TS5LowerCut);
  descriptions.add("hltHcalMETNoiseCleaner", desc);
}

//
// member functions
//

bool HLTHcalMETNoiseCleaner::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace reco;

  //output collection
  std::unique_ptr<CaloMETCollection> CleanedMET(new CaloMETCollection);

  //get the calo MET / MHT
  edm::Handle<CaloMETCollection> met_h;
  iEvent.getByToken(m_theCaloMetToken, met_h);

  if (not met_h.isValid() or met_h->empty() or
      met_h->front().pt() < 0) {  //No Valid MET, don't do anything and accept the event
    return true;                  // we shouldn't get here, but lets not crash
  }

  reco::CaloMET inCaloMet = met_h->front();

  // in this case, do not filter anything
  if (severity_ == 0) {
    CleanedMET->push_back(inCaloMet);
    iEvent.put(std::move(CleanedMET));
    return true;
  }

  // get the RBXs produced by RecoMET/METProducers/HcalNoiseInfoProducer
  edm::Handle<HcalNoiseRBXCollection> rbxs_h;
  iEvent.getByToken(m_theHcalNoiseToken, rbxs_h);
  if (!rbxs_h.isValid()) {
    edm::LogError("DataNotFound") << "HLTHcalMETNoiseCleaner: Could not find HcalNoiseRBXCollection product named "
                                  << HcalNoiseRBXCollectionTag_ << "." << std::endl;
    CleanedMET->push_back(inCaloMet);
    iEvent.put(std::move(CleanedMET));
    return true;  // no valid RBXs
  }

  // create a sorted set of the RBXs, ordered by energy
  noisedataset_t data;
  for (auto const& rbx : *rbxs_h) {
    CommonHcalNoiseRBXData d(rbx,
                             minRecHitE_,
                             minLowHitE_,
                             minHighHitE_,
                             TS4TS5EnergyThreshold_,
                             TS4TS5UpperCut_,
                             TS4TS5LowerCut_,
                             minR45HitE_);
    data.insert(d);
  }
  //if 0 RBXs are in the list, just accept
  if (data.empty()) {
    CleanedMET->push_back(inCaloMet);
    iEvent.put(std::move(CleanedMET));
    return true;
  }
  // data is now sorted by RBX energy
  // only consider top N=numRBXsToConsider_ energy RBXs
  int cntr = 0;
  int nNoise = 0;

  TVector3 metVec;
  metVec.SetPtEtaPhi(met_h->front().pt(), 0, met_h->front().phi());

  TVector3 noiseHPDVector(0, 0, 0);
  TVector3 secondHPDVector(0, 0, 0);
  for (auto it = data.begin(); it != data.end() && cntr < numRBXsToConsider_; it++, cntr++) {
    bool isNoise = false;
    bool passFilter = true;
    bool passEMF = true;
    if (it->energy() > minRBXEnergy_) {
      if (it->validRatio() && it->ratio() < minRatio_)
        passFilter = false;
      else if (it->validRatio() && it->ratio() > maxRatio_)
        passFilter = false;
      else if (it->numHPDHits() >= minHPDHits_)
        passFilter = false;
      else if (it->numRBXHits() >= minRBXHits_)
        passFilter = false;
      else if (it->numHPDNoOtherHits() >= minHPDNoOtherHits_)
        passFilter = false;
      else if (it->numZeros() >= minZeros_)
        passFilter = false;
      else if (it->minHighEHitTime() < minHighEHitTime_)
        passFilter = false;
      else if (it->maxHighEHitTime() > maxHighEHitTime_)
        passFilter = false;
      else if (!it->PassTS4TS5())
        passFilter = false;

      if (it->RBXEMF() < maxRBXEMF_) {
        passEMF = false;
      }
    }

    if ((needEMFCoincidence_ && !passEMF && !passFilter) || (!needEMFCoincidence_ && !passFilter)) {  // check for noise
      LogDebug("") << "HLTHcalMETNoiseCleaner debug: Found a noisy RBX: "
                   << "energy=" << it->energy() << "; "
                   << "ratio=" << it->ratio() << "; "
                   << "# RBX hits=" << it->numRBXHits() << "; "
                   << "# HPD hits=" << it->numHPDHits() << "; "
                   << "# Zeros=" << it->numZeros() << "; "
                   << "min time=" << it->minHighEHitTime() << "; "
                   << "max time=" << it->maxHighEHitTime() << "; "
                   << "passTS4TS5=" << it->PassTS4TS5() << "; "
                   << "RBX EMF=" << it->RBXEMF() << std::endl;
      nNoise++;
      isNoise = true;
    }  // OK, checked for noise

    //------------First Noisy RBX-----------------------
    if (isNoise && nNoise == 1) {
      edm::RefVector<CaloTowerCollection> noiseTowers = it->rbxTowers();
      edm::RefVector<CaloTowerCollection>::const_iterator noiseTowersIt;
      // get the energy vector for this RBX from the calotowers
      for (noiseTowersIt = noiseTowers.begin(); noiseTowersIt != noiseTowers.end(); noiseTowersIt++) {
        TVector3 towerVec;
        towerVec.SetPtEtaPhi((*noiseTowersIt)->pt(), (*noiseTowersIt)->eta(), (*noiseTowersIt)->phi());
        noiseHPDVector += towerVec;  // add this tower to the vector for the RBX
      }
      if (noiseHPDVector.Mag() > 0)
        noiseHPDVector.SetPtEtaPhi(noiseHPDVector.Pt(), 0, noiseHPDVector.Phi());  // make the noise transverse
      else
        noiseHPDVector.SetPtEtaPhi(0, 0, 0);
    }
    //-----------FOUND a SECOND NOISY RBX-------------------
    if (isNoise && cntr > 0) {
      CleanedMET->push_back(inCaloMet);
      iEvent.put(std::move(CleanedMET));
      return accept2NoiseRBXEvents_;  // don't try to clean these for the moment, just keep or throw away
    }
    //----------LEADING RBX is NOT NOISY--------------------
    if (!isNoise && cntr == 0) {
      CleanedMET->push_back(inCaloMet);
      iEvent.put(std::move(CleanedMET));
      return true;  // don't reject the event if the leading RBX isn't noise
    }
    //-----------SUBLEADING RBX is NOT NOISY: STORE INFO----
    if (!isNoise && nNoise > 0) {  //second RBX isn't noisy (and first one was), so clean
      edm::RefVector<CaloTowerCollection> noiseTowers = it->rbxTowers();
      edm::RefVector<CaloTowerCollection>::const_iterator noiseTowersIt;
      for (noiseTowersIt = noiseTowers.begin(); noiseTowersIt != noiseTowers.end(); noiseTowersIt++) {
        TVector3 towerVec;
        towerVec.SetPtEtaPhi((*noiseTowersIt)->pt(), (*noiseTowersIt)->eta(), (*noiseTowersIt)->phi());
        secondHPDVector += towerVec;  // add this tower to the vector for the RBX
      }
      if (secondHPDVector.Mag() > 0)
        secondHPDVector.SetPtEtaPhi(secondHPDVector.Pt(), 0, secondHPDVector.Phi());  // make the second transverse
      else
        secondHPDVector.SetPtEtaPhi(0, 0, 0);
      break;
    }
  }  // end RBX loop

  if (noiseHPDVector.Mag() == 0) {
    CleanedMET->push_back(inCaloMet);
    iEvent.put(std::move(CleanedMET));
    return true;  // don't reject the event if the leading RBX isn't noise
  }

  //********************************************************************************
  //The Event gets here only if it had exactly 1 noisy RBX in the lead position
  //********************************************************************************

  float METsumet = met_h->front().energy();

  metVec += noiseHPDVector;

  float ZMETsumet = METsumet - noiseHPDVector.Mag();
  float ZMETpt = metVec.Pt();
  float ZMETphi = metVec.Phi();

  //put the second RBX vector in the eta phi position of the leading RBX vector

  float SMETsumet = 0;
  float SMETpt = 0;
  float SMETphi = 0;
  if (secondHPDVector.Mag() > 0.) {
    secondHPDVector.SetPtEtaPhi(secondHPDVector.Pt(), noiseHPDVector.Eta(), noiseHPDVector.Phi());
    metVec -= secondHPDVector;
    SMETsumet = METsumet - noiseHPDVector.Mag();
    SMETpt = metVec.Pt();
    SMETphi = metVec.Phi();
  }
  //Get the maximum MET:
  float CorMetSumEt, CorMetPt, CorMetPhi;
  if (ZMETpt > SMETpt) {
    CorMetSumEt = ZMETsumet;
    CorMetPt = ZMETpt;
    CorMetPhi = ZMETphi;
  } else {
    CorMetSumEt = SMETsumet;
    CorMetPt = SMETpt;
    CorMetPhi = SMETphi;
  }

  reco::CaloMET corMet = BuildCaloMet(CorMetSumEt, CorMetPt, CorMetPhi);
  CleanedMET->push_back(corMet);
  iEvent.put(std::move(CleanedMET));

  return (corMet.pt() > CaloMetCut_);
}

reco::CaloMET HLTHcalMETNoiseCleaner::BuildCaloMet(float sumet, float pt, float phi) const {
  // Instantiate the container to hold the calorimeter specific information

  typedef math::XYZPoint Point;
  typedef math::XYZTLorentzVector LorentzVector;

  SpecificCaloMETData specific;
  // Initialise the container
  specific.MaxEtInEmTowers = 0.0;     // Maximum energy in EM towers
  specific.MaxEtInHadTowers = 0.0;    // Maximum energy in HCAL towers
  specific.HadEtInHO = 0.0;           // Hadronic energy fraction in HO
  specific.HadEtInHB = 0.0;           // Hadronic energy in HB
  specific.HadEtInHF = 0.0;           // Hadronic energy in HF
  specific.HadEtInHE = 0.0;           // Hadronic energy in HE
  specific.EmEtInEB = 0.0;            // Em energy in EB
  specific.EmEtInEE = 0.0;            // Em energy in EE
  specific.EmEtInHF = 0.0;            // Em energy in HF
  specific.EtFractionHadronic = 0.0;  // Hadronic energy fraction
  specific.EtFractionEm = 0.0;        // Em energy fraction
  specific.CaloSETInpHF = 0.0;        // CaloSET in HF+
  specific.CaloSETInmHF = 0.0;        // CaloSET in HF-
  specific.CaloMETInpHF = 0.0;        // CaloMET in HF+
  specific.CaloMETInmHF = 0.0;        // CaloMET in HF-
  specific.CaloMETPhiInpHF = 0.0;     // CaloMET-phi in HF+
  specific.CaloMETPhiInmHF = 0.0;     // CaloMET-phi in HF-
  specific.METSignificance = 0.0;

  TLorentzVector p4TL;
  p4TL.SetPtEtaPhiM(pt, 0., phi, 0.);
  const LorentzVector p4(p4TL.X(), p4TL.Y(), 0, p4TL.T());
  const Point vtx(0.0, 0.0, 0.0);
  reco::CaloMET specificmet(specific, sumet, p4, vtx);
  return specificmet;
}
