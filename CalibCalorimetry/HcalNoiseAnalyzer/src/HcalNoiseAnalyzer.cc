//---------------------------------------------------------------------------
// -*- C++ -*-
//
// Package:    HcalNoiseAnalyzer
// Class:      HcalNoiseAnalyzer
// 
/**\class HcalNoiseAnalyzer HcalNoiseAnalyzer.cc HcalNoise/HcalNoiseAnalyzer/src/HcalNoiseAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yi Chen,40 3-B12,+41227675736,
//         Created:  Wed Oct 27 11:08:10 CEST 2010
// $Id: HcalNoiseAnalyzer.cc,v 1.3 2013/03/04 14:23:21 chenyi Exp $
//
//
//---------------------------------------------------------------------------
#include <memory>
#include <string>
#include <map>
#include <iostream>
using namespace std;
//---------------------------------------------------------------------------
#include "TTree.h"
#include "TFile.h"
//---------------------------------------------------------------------------
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "DataFormats/METReco/interface/HcalNoiseSummary.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"
#include "RecoMET/METAlgorithms/interface/HcalHPDRBXMap.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "L1Trigger/GlobalTrigger/plugins/L1GlobalTrigger.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
//---------------------------------------------------------------------------
static double MaximumFractionalError = 0.002;
class HcalNoiseAnalyzer;
double ECorr(int ieta, int iphi, double energy);
//---------------------------------------------------------------------------
class HcalNoiseAnalyzer : public edm::EDAnalyzer
{
public:
   explicit HcalNoiseAnalyzer(const edm::ParameterSet&);
   ~HcalNoiseAnalyzer();

private:
   virtual void beginJob();
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endJob();
   void beginRun(const edm::Run&, const edm::EventSetup&);
   void endRun(const edm::Run&, const edm::EventSetup&);

private:
   bool FillHBHE;                  // Whether to store HBHE digi-level information or not
   bool FillHF;                    // Whether to store HF digi-level information or not
   bool FillHO;                    // Whether to store HO digi-level information or not
   double TotalChargeThreshold;    // To avoid trees from overweight, only store digis above some threshold
   string sHBHERecHitCollection;   // Name of the HBHE rechit collection
   edm::Service<TFileService> FileService;

   // Basic event coordinates
   long long RunNumber;
   long long EventNumber;
   long long LumiSection;
   long long Bunch;
   long long Orbit;
   long long Time;

   // Trigger bits
   bool TTrigger[64];
   bool L1Trigger[128];
   bool HLTrigger[256];

   // Sum \vec{ET}
   double EBET[2];
   double EEET[2];
   double HBET[2];
   double HEET[2];
   double HFET[2];
   double NominalMET[2];

   // Sum |E|
   double EBSumE;
   double EESumE;
   double HBSumE;
   double HESumE;
   double HFSumE;

   // Sum |ET|
   double EBSumET;
   double EESumET;
   double HBSumET;
   double HESumET;
   double HFSumET;

   // Summary variables for tracks and muons (and PV)
   int NumberOfGoodTracks;
   int NumberOfGoodTracks15;
   int NumberOfGoodTracks30;
   double TotalPTTracks[2];
   double SumPTTracks;
   double SumPTracks;
   int NumberOfGoodPrimaryVertices;
   int NumberOfMuonCandidates;
   int NumberOfCosmicMuonCandidates;

   // HBHE rechits and digis
   int PulseCount;
   double Charge[5184][10];
   double Pedestal[5184][10];
   double Energy[5184];
   int IEta[5184];
   int IPhi[5184];
   int Depth[5184];
   double RecHitTime[5184];
   uint32_t FlagWord[5184];
   uint32_t AuxWord[5184];
   double RespCorrGain[5184];
   double FCorr[5184];
   double SamplesToAdd[5184];

   // HF rechits and digis
   int HFPulseCount;
   int HFADC[5184][10];
   double HFCharge[5184][10];
   double HFPedestal[5184][10];
   int HFIEta[5184];
   int HFIPhi[5184];
   int HFDepth[5184];
   double HFEnergy[5184];

   // HO rechits and digis
   int HOPulseCount;
   int HOADC[2160][10];
   double HOCharge[2160][10];
   double HOPedestal[2160][10];
   int HOIEta[2160];
   int HOIPhi[2160];
   double HOEnergy[2160];

   // HBHE RBX energy and mega-pulse shape
   double RBXCharge[72][10];
   double RBXEnergy[72];
   double RBXCharge15[72][10];
   double RBXEnergy15[72];

   // Summary variables for baseline Hcal noise filter
   int HPDHits;
   int HPDNoOtherHits;
   int MaxZeros;
   double MinE2E10;
   double MaxE2E10;
   bool HasBadRBXR45;
   bool HasBadRBXRechitR45Loose;
   bool HasBadRBXRechitR45Tight;

   // Official decision from the baseline hcal noise filter
   bool OfficialDecision;

   // Summary variables for (PF) jets
   double LeadingJetEta;
   double LeadingJetPhi;
   double LeadingJetPt;
   double LeadingJetHad;
   double LeadingJetEM;
   double FollowingJetEta;
   double FollowingJetPhi;
   double FollowingJetPt;
   double FollowingJetHad;
   double FollowingJetEM;
   int JetCount20;
   int JetCount30;
   int JetCount50;
   int JetCount100;

   // Summary variables for HO
   double HOMaxEnergyRing0, HOSecondMaxEnergyRing0;
   int HOMaxEnergyIDRing0, HOSecondMaxEnergyIDRing0;
   int HOHitCount100Ring0, HOHitCount150Ring0;
   double HOMaxEnergyRing12, HOSecondMaxEnergyRing12;
   int HOMaxEnergyIDRing12, HOSecondMaxEnergyIDRing12;
   int HOHitCount100Ring12, HOHitCount150Ring12;

private:
   TTree *OutputTree;

   const CaloGeometry *Geometry;
   std::auto_ptr<HcalPulseContainmentManager> pulseCorr_;
   bool correctForTimeslew;
   bool correctForPhaseContainment;
   double correctionPhaseNS;
   HcalRecoParams *paramTS;  // firstSample & samplesToAdd from DB  
   HcalTopology *theTopology;

   void ClearVariables();
   void CalculateTotalEnergiesHBHE(const HBHERecHitCollection &RecHits);
   void CalculateTotalEnergiesHF(const HFRecHitCollection &RecHits);
   void CalculateTotalEnergiesEB(const EcalRecHitCollection &RecHits);
   void CalculateTotalEnergiesEE(const EcalRecHitCollection &RecHits);

   void Initialize();
};
//---------------------------------------------------------------------------
HcalNoiseAnalyzer::HcalNoiseAnalyzer(const edm::ParameterSet& iConfig) :
   correctForTimeslew(true), correctForPhaseContainment(true),
   correctionPhaseNS(6.0)
{
   // Get stuff and initialize here
   FillHBHE = iConfig.getUntrackedParameter<bool>("FillHBHE", true);
   FillHF = iConfig.getUntrackedParameter<bool>("FillHF", false);
   FillHO = iConfig.getUntrackedParameter<bool>("FillHO", false);
   TotalChargeThreshold = iConfig.getUntrackedParameter<double>("TotalChargeThreshold", 10);

   sHBHERecHitCollection = iConfig.getUntrackedParameter<string>("HBHERecHits", "hbhereco");

   Initialize();
}
//---------------------------------------------------------------------------
HcalNoiseAnalyzer::~HcalNoiseAnalyzer()
{
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   ClearVariables();

   // get stuff
   Handle<HBHERecHitCollection> hRecHits;
   iEvent.getByLabel(InputTag(sHBHERecHitCollection), hRecHits);

   Handle<HFRecHitCollection> hHFRecHits;
   iEvent.getByLabel(InputTag("hfreco"), hHFRecHits);

   Handle<HORecHitCollection> hHORecHits;
   iEvent.getByLabel(InputTag("horeco"), hHORecHits);

   Handle<HBHEDigiCollection> hHBHEDigis;
   iEvent.getByLabel(InputTag("hcalDigis"), hHBHEDigis);

   Handle<HFDigiCollection> hHFDigis;
   iEvent.getByLabel(InputTag("hcalDigis"), hHFDigis);

   Handle<HODigiCollection> hHODigis;
   iEvent.getByLabel(InputTag("hcalDigis"), hHODigis);

   Handle<EcalRecHitCollection> hEBRecHits;
   iEvent.getByLabel(InputTag("ecalRecHit", "EcalRecHitsEB"), hEBRecHits);

   Handle<EcalRecHitCollection> hEERecHits;
   iEvent.getByLabel(InputTag("ecalRecHit", "EcalRecHitsEE"), hEERecHits);

   Handle<CaloMETCollection> hCaloMET;
   iEvent.getByLabel(InputTag("caloMet"), hCaloMET);

   ESHandle<HcalDbService> hConditions;
   iSetup.get<HcalDbRecord>().get(hConditions);

   ESHandle<CaloGeometry> hGeometry;
   iSetup.get<CaloGeometryRecord>().get(hGeometry);
   Geometry = hGeometry.product();

   Handle<TriggerResults> hTrigger;
   iEvent.getByLabel(InputTag("TriggerResults::HLT"), hTrigger);

   Handle<L1GlobalTriggerReadoutRecord> hL1GlobalTrigger;
   iEvent.getByLabel(InputTag("gtDigis"), hL1GlobalTrigger);

   Handle<VertexCollection> hVertices;
   iEvent.getByLabel(InputTag("offlinePrimaryVertices"), hVertices);

   Handle<TrackCollection> hTracks;
   iEvent.getByLabel(InputTag("generalTracks"), hTracks);

   Handle<MuonCollection> hStandardMuon;
   iEvent.getByLabel("muons", hStandardMuon);
   
   Handle<MuonCollection> hCosmicMuon;
   iEvent.getByLabel("muonsFromCosmics", hCosmicMuon);

   Handle<HcalNoiseSummary> hSummary;
   iEvent.getByLabel("hcalnoise", hSummary);

   Handle<CaloJetCollection> hCaloJets;
   iEvent.getByLabel("ak5CaloJets", hCaloJets);

   Handle<bool> hNoiseResult;
   iEvent.getByLabel(InputTag("HBHENoiseFilterResultProducer", "HBHENoiseFilterResult"), hNoiseResult);
   OfficialDecision = *hNoiseResult;

   // basic event coordinates
   RunNumber = iEvent.id().run();
   EventNumber = iEvent.id().event();
   LumiSection = iEvent.luminosityBlock();
   Bunch = iEvent.bunchCrossing();
   Orbit = iEvent.orbitNumber();
   Time = iEvent.time().value();

   // event cleaning related - vertices
   NumberOfGoodPrimaryVertices = 0;
   for(int i = 0; i < (int)hVertices->size(); i++)
   {
      if((*hVertices)[i].ndof() <= 4)
         continue;
      if((*hVertices)[i].z() > 15)
         continue;
      if((*hVertices)[i].position().rho() > 2)
         continue;
      NumberOfGoodPrimaryVertices = NumberOfGoodPrimaryVertices + 1;
   }

   // event cleaning related - tracks
   for(int i = 0; i < (int)hTracks->size(); i++)
   {
      if((*hTracks)[i].numberOfValidHits() < 6)
         continue;
      if((*hTracks)[i].pt() < 0.5 || (*hTracks)[i].pt() > 500)
         continue;
      if((*hTracks)[i].eta() < -2.4 || (*hTracks)[i].eta() > 2.4)
         continue;
      if((*hTracks)[i].chi2() / (*hTracks)[i].ndof() > 20)
         continue;

      NumberOfGoodTracks = NumberOfGoodTracks + 1;
      TotalPTTracks[0] = TotalPTTracks[0] + (*hTracks)[i].px();
      TotalPTTracks[1] = TotalPTTracks[1] + (*hTracks)[i].py();
      SumPTTracks = SumPTTracks + (*hTracks)[i].pt();
      SumPTracks = SumPTracks + (*hTracks)[i].p();
      
      if((*hTracks)[i].pt() > 1.5)
         NumberOfGoodTracks15 = NumberOfGoodTracks15 + 1;
      if((*hTracks)[i].pt() > 3.0)
         NumberOfGoodTracks30 = NumberOfGoodTracks30 + 1;
   }

   // HBHE rechit maps - we want to link rechits and digis together
   map<HcalDetId, int> RecHitIndex;
   for(int i = 0; i < (int)hRecHits->size(); i++)
   {
      HcalDetId id = (*hRecHits)[i].id();
      RecHitIndex.insert(pair<HcalDetId, int>(id, i));
   }

   // HF rechit maps
   map<HcalDetId, int> HFRecHitIndex;
   for(int i = 0; i < (int)hHFRecHits->size(); i++)
   {
      HcalDetId id = (*hHFRecHits)[i].id();
      HFRecHitIndex.insert(pair<HcalDetId, int>(id, i));
   }

   // HO rechit maps
   map<HcalDetId, int> HORecHitIndex;
   for(int i = 0; i < (int)hHORecHits->size(); i++)
   {
      HcalDetId id = (*hHORecHits)[i].id();
      HORecHitIndex.insert(pair<HcalDetId, int>(id, i));
   }

   // triggers
   int TTriggerSize = hL1GlobalTrigger->technicalTriggerWord().size();
   for(int i = 0; i < 64 && i < TTriggerSize; i++)
      TTrigger[i] = hL1GlobalTrigger->technicalTriggerWord()[i];
   
   int L1TriggerSize = hL1GlobalTrigger->decisionWord().size();
   for(int i = 0; i < 128 && i < L1TriggerSize; i++)
      L1Trigger[i] = hL1GlobalTrigger->decisionWord()[i];

   const TriggerNames &Names = iEvent.triggerNames(*hTrigger);
   int HLTriggerSize = Names.triggerNames().size();
   for(int i = 0; i < 256 && i < HLTriggerSize; i++)
      HLTrigger[i] = hTrigger->accept(i);
   
   // total calorimeter energies - from rechit
   CalculateTotalEnergiesHBHE(*hRecHits);
   CalculateTotalEnergiesHF(*hHFRecHits);
   CalculateTotalEnergiesEB(*hEBRecHits);
   CalculateTotalEnergiesEE(*hEERecHits);

   if(hCaloMET->size() > 0)
   {
      NominalMET[0] = (*hCaloMET)[0].px();
      NominalMET[1] = (*hCaloMET)[0].py();
   }

   // loop over digis
   for(HBHEDigiCollection::const_iterator iter = hHBHEDigis->begin(); iter != hHBHEDigis->end(); iter++)
   {
      HcalDetId id = iter->id();
     
      int RBXIndex = HcalHPDRBXMap::indexRBX(id);

      // First let's convert ADC to deposited charge
      const HcalCalibrations &Calibrations = hConditions->getHcalCalibrations(id);
      const HcalQIECoder *ChannelCoder = hConditions->getHcalCoder(id);
      const HcalQIEShape *Shape = hConditions->getHcalShape(ChannelCoder);
      HcalCoderDb Coder(*ChannelCoder, *Shape);
      CaloSamples Tool;
      Coder.adc2fC(*iter, Tool);

      // Total charge of the digi
      double TotalCharge = 0;
      for(int i = 0; i < (int)iter->size(); i++)
         TotalCharge = TotalCharge + Tool[i] - Calibrations.pedestal(iter->sample(i).capid());

      // Add this rechit/digi into RBX total charge and total energy
      for(int i = 0; i < (int)iter->size(); i++)
      {
         const HcalQIESample &QIE = iter->sample(i);
         RBXCharge[RBXIndex][i] = RBXCharge[RBXIndex][i] + Tool[i] - Calibrations.pedestal(QIE.capid());

         if((*hRecHits)[RecHitIndex[id]].energy() > 1.5)
            RBXCharge15[RBXIndex][i] = RBXCharge15[RBXIndex][i] + Tool[i] - Calibrations.pedestal(QIE.capid());
      }
      RBXEnergy[RBXIndex] = RBXEnergy[RBXIndex] + (*hRecHits)[RecHitIndex[id]].energy();
  
      if((*hRecHits)[RecHitIndex[id]].energy() > 1.5)
         RBXEnergy15[RBXIndex] = RBXEnergy15[RBXIndex] + (*hRecHits)[RecHitIndex[id]].energy();
      
      // If total charge is smaller than threshold, don't store this rechit/digi into the tree
      if(TotalCharge < TotalChargeThreshold)
         continue;

      // Safety check - there are only 5184 channels in HBHE, but just in case...
      if(PulseCount >= 5184)
      {
         PulseCount = PulseCount + 1;
         continue;
      }

      // Fill things into the tree
      for(int i = 0; i < (int)iter->size(); i++)
      {
         const HcalQIESample &QIE = iter->sample(i);

         Pedestal[PulseCount][i] = Calibrations.pedestal(QIE.capid());
         Charge[PulseCount][i] = Tool[i] - Pedestal[PulseCount][i];
      }

      Energy[PulseCount] = (*hRecHits)[RecHitIndex[id]].energy();
      RecHitTime[PulseCount] = (*hRecHits)[RecHitIndex[id]].time();

      FlagWord[PulseCount] = (*hRecHits)[RecHitIndex[id]].flags();
      AuxWord[PulseCount] = (*hRecHits)[RecHitIndex[id]].aux();

      IEta[PulseCount] = id.ieta();
      IPhi[PulseCount] = id.iphi();
      Depth[PulseCount] = id.depth();

      const HcalRecoParam *RecoParameters = paramTS->getValues(DetId(id).rawId());
      int FirstSample = RecoParameters->firstSample();
      int WindowSize = RecoParameters->samplesToAdd();
      float FixedPhaseNs = RecoParameters->correctionPhaseNS();

      double GainCorrection[4];
      for(int iCap = 0; iCap < 4; iCap++)
         GainCorrection[iCap] = Calibrations.respcorrgain(iCap);

      double ChargeInWindow = 0;
      for(int i = FirstSample; i < FirstSample + WindowSize; i++)
         ChargeInWindow = ChargeInWindow + Charge[PulseCount][i];

      const HcalPulseContainmentCorrection *Correction = pulseCorr_->get(id, WindowSize, FixedPhaseNs);
      double fCorr = Correction->getCorrection(ChargeInWindow);

      // These will give us the eCorr factor
      // double EBeforeECorr = ChargeInWindow * RespCorrGain[0] * fCorr;
      // double Ecorr = ECorr(id.ieta(), id.iphi(), EBeforeECorr);
      // double FinalEnergy = EBeforeECorr * Ecorr;

      RespCorrGain[PulseCount] = GainCorrection[0];
      FCorr[PulseCount] = fCorr;
      SamplesToAdd[PulseCount] = WindowSize;

      PulseCount = PulseCount + 1;
   }

   // Loop over HF digis
   HFPulseCount = 0;
   for(HFDigiCollection::const_iterator iter = hHFDigis->begin(); iter != hHFDigis->end(); iter++)
   {
      HcalDetId id = iter->id();

      HFIEta[HFPulseCount] = id.ieta();
      HFIPhi[HFPulseCount] = id.iphi();
      HFDepth[HFPulseCount] = id.depth();

      // ADC -> fC
      const HcalCalibrations &Calibrations = hConditions->getHcalCalibrations(id);
      const HcalQIECoder *ChannelCoder = hConditions->getHcalCoder(id);
      const HcalQIEShape *Shape = hConditions->getHcalShape(ChannelCoder);
      HcalCoderDb Coder(*ChannelCoder, *Shape);
      CaloSamples Tool;
      Coder.adc2fC(*iter, Tool);

      // Fill!
      for(int i = 0; i < iter->size(); i++)
      {
         const HcalQIESample &QIE = iter->sample(i);

         HFADC[HFPulseCount][i] = iter->sample(i).adc();

         HFPedestal[HFPulseCount][i] = Calibrations.pedestal(QIE.capid());
         HFCharge[HFPulseCount][i] = Tool[i] - HFPedestal[HFPulseCount][i];
      }

      HFEnergy[HFPulseCount] = (*hHFRecHits)[HFRecHitIndex[id]].energy();

      HFPulseCount = HFPulseCount + 1;
   }

   // Loop over HO digis
   HOPulseCount = 0;
   for(HODigiCollection::const_iterator iter = hHODigis->begin(); iter != hHODigis->end(); iter++)
   {
      HcalDetId id = iter->id();

      HOIEta[HOPulseCount] = id.ieta();
      HOIPhi[HOPulseCount] = id.iphi();

      // ADC -> fC
      const HcalCalibrations &Calibrations = hConditions->getHcalCalibrations(id);
      const HcalQIECoder *ChannelCoder = hConditions->getHcalCoder(id);
      const HcalQIEShape *Shape = hConditions->getHcalShape(ChannelCoder);
      HcalCoderDb Coder(*ChannelCoder, *Shape);
      CaloSamples Tool;
      Coder.adc2fC(*iter, Tool);

      // Fill!
      for(int i = 0; i < iter->size(); i++)
      {
         const HcalQIESample &QIE = iter->sample(i);

         HOADC[HOPulseCount][i] = iter->sample(i).adc();

         HOPedestal[HOPulseCount][i] = Calibrations.pedestal(QIE.capid());
         HOCharge[HOPulseCount][i] = Tool[i] - HOPedestal[HOPulseCount][i];
      }

      HOEnergy[HOPulseCount] = (*hHORecHits)[HORecHitIndex[id]].energy();

      HOPulseCount = HOPulseCount + 1;
   }

   // muons
   NumberOfMuonCandidates = hStandardMuon->size();
   NumberOfCosmicMuonCandidates = hCosmicMuon->size();

   // hcal sumamry objects
   HPDHits = hSummary->maxHPDHits();
   HPDNoOtherHits = hSummary->maxHPDNoOtherHits();
   MaxZeros = hSummary->maxZeros();
   MinE2E10 = hSummary->minE2Over10TS();
   MaxE2E10 = hSummary->maxE2Over10TS();
   HasBadRBXR45 = hSummary->HasBadRBXTS4TS5();
   HasBadRBXRechitR45Loose = hSummary->HasBadRBXRechitR45Loose();
   HasBadRBXRechitR45Tight = hSummary->HasBadRBXRechitR45Tight();

   // jets
   int JetCollectionCount = hCaloJets->size();
   map<double, int, greater<double> > JetPTMap;
   for(int i = 0; i < JetCollectionCount; i++)
   {
      JetPTMap.insert(pair<double, int>((*hCaloJets)[i].pt(), i));

      if((*hCaloJets)[i].pt() > 20)
         JetCount20 = JetCount20 + 1;
      if((*hCaloJets)[i].pt() > 30)
         JetCount30 = JetCount30 + 1;
      if((*hCaloJets)[i].pt() > 50)
         JetCount50 = JetCount50 + 1;
      if((*hCaloJets)[i].pt() > 100)
         JetCount100 = JetCount100 + 1;
   }

   map<double, int, greater<double> >::iterator iter = JetPTMap.begin();
   if(JetPTMap.size() > 0)
   {
      if(iter->second < (int)hCaloJets->size())
      {
         LeadingJetEta = (*hCaloJets)[iter->second].eta();
         LeadingJetPhi = (*hCaloJets)[iter->second].phi();
         LeadingJetPt = (*hCaloJets)[iter->second].pt();
         LeadingJetHad = (*hCaloJets)[iter->second].hadEnergyInHB()
            + (*hCaloJets)[iter->second].hadEnergyInHE() + (*hCaloJets)[iter->second].hadEnergyInHF();
         LeadingJetEM = (*hCaloJets)[iter->second].emEnergyInEB()
            + (*hCaloJets)[iter->second].emEnergyInEE() + (*hCaloJets)[iter->second].emEnergyInHF();
      }
   }

   if(JetPTMap.size() > 1)
   {
      iter++;
      if(iter->second < (int)hCaloJets->size())
      {
         FollowingJetEta = (*hCaloJets)[iter->second].eta();
         FollowingJetPhi = (*hCaloJets)[iter->second].phi();
         FollowingJetPt = (*hCaloJets)[iter->second].pt();
         FollowingJetHad = (*hCaloJets)[iter->second].hadEnergyInHB()
            + (*hCaloJets)[iter->second].hadEnergyInHE() + (*hCaloJets)[iter->second].hadEnergyInHF();
         FollowingJetEM = (*hCaloJets)[iter->second].emEnergyInEB()
            + (*hCaloJets)[iter->second].emEnergyInEE() + (*hCaloJets)[iter->second].emEnergyInHF();
      }
   }

   // HO Summary variables
   for(int i = 0; i < (int)hHORecHits->size(); i++)
   {
      HcalDetId id = (*hHORecHits)[i].id();

      double energy = (*hHORecHits)[i].energy();
      int ieta = id.ieta();
      int iphi = id.iphi();
      int InternalHOID = 100 * (ieta + 50) + iphi;

      bool IsRing0 = false;
      if(ieta >= -4 && ieta <= 4)   // otherwise it's ring 1/2
         IsRing0 = true;

      if(IsRing0 == true && energy > 100)
         HOHitCount100Ring0 = HOHitCount100Ring0 + 1;
      if(IsRing0 == true && energy > 150)
         HOHitCount150Ring0 = HOHitCount150Ring0 + 1;
      if(IsRing0 == true && energy > HOMaxEnergyRing0)
      {
         HOSecondMaxEnergyRing0 = HOMaxEnergyRing0;
         HOSecondMaxEnergyIDRing0 = HOMaxEnergyIDRing0;
         HOMaxEnergyRing0 = energy;
         HOMaxEnergyIDRing0 = InternalHOID;
      }
      else if(IsRing0 == true && energy > HOSecondMaxEnergyRing0)
      {
         HOSecondMaxEnergyRing0 = energy;
         HOSecondMaxEnergyIDRing0 = InternalHOID;
      }
      
      if(IsRing0 == false && energy > 100)
         HOHitCount100Ring12 = HOHitCount100Ring12 + 1;
      if(IsRing0 == false && energy > 150)
         HOHitCount150Ring12 = HOHitCount150Ring12 + 1;
      if(IsRing0 == false && energy > HOMaxEnergyRing12)
      {
         HOSecondMaxEnergyRing12 = HOMaxEnergyRing12;
         HOSecondMaxEnergyIDRing12 = HOMaxEnergyIDRing12;
         HOMaxEnergyRing12 = energy;
         HOMaxEnergyIDRing12 = InternalHOID;
      }
      else if(IsRing0 == false && energy > HOSecondMaxEnergyRing12)
      {
         HOSecondMaxEnergyRing12 = energy;
         HOSecondMaxEnergyIDRing12 = InternalHOID;
      }
   }

   // finally actually fill the tree
   OutputTree->Fill();
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::beginJob()
{
   // Make branches in the output trees
   OutputTree = FileService->make<TTree>("HcalNoiseTree", "Hcal noise tree version 1,2134");

   OutputTree->Branch("RunNumber", &RunNumber, "RunNumber/LL");
   OutputTree->Branch("EventNumber", &EventNumber, "EventNumber/LL");
   OutputTree->Branch("LumiSection", &LumiSection, "LumiSection/LL");
   OutputTree->Branch("Bunch", &Bunch, "Bunch/LL");
   OutputTree->Branch("Orbit", &Orbit, "Orbit/LL");
   OutputTree->Branch("Time", &Time, "Time/LL");

   OutputTree->Branch("TTrigger", &TTrigger, "TTrigger[64]/O");
   OutputTree->Branch("L1Trigger", &L1Trigger, "L1Trigger[128]/O");
   OutputTree->Branch("HLTrigger", &HLTrigger, "HLTrigger[256]/O");

   OutputTree->Branch("EBET", &EBET, "EBET[2]/D");
   OutputTree->Branch("EEET", &EEET, "EEET[2]/D");
   OutputTree->Branch("HBET", &HBET, "HBET[2]/D");
   OutputTree->Branch("HEET", &HEET, "HEET[2]/D");
   OutputTree->Branch("HFET", &HFET, "HFET[2]/D");
   OutputTree->Branch("NominalMET", &NominalMET, "NominalMET[2]/D");
   
   OutputTree->Branch("EBSumE", &EBSumE, "EBSumE/D");
   OutputTree->Branch("EESumE", &EESumE, "EESumE/D");
   OutputTree->Branch("HBSumE", &HBSumE, "HBSumE/D");
   OutputTree->Branch("HESumE", &HESumE, "HESumE/D");
   OutputTree->Branch("HFSumE", &HFSumE, "HFSumE/D");

   OutputTree->Branch("EBSumET", &EBSumET, "EBSumET/D");
   OutputTree->Branch("EESumET", &EESumET, "EESumET/D");
   OutputTree->Branch("HBSumET", &HBSumET, "HBSumET/D");
   OutputTree->Branch("HESumET", &HESumET, "HESumET/D");
   OutputTree->Branch("HFSumET", &HFSumET, "HFSumET/D");

   OutputTree->Branch("NumberOfGoodTracks", &NumberOfGoodTracks, "NumberOfGoodTracks/I");
   OutputTree->Branch("NumberOfGoodTracks15", &NumberOfGoodTracks15, "NumberOfGoodTracks15/I");
   OutputTree->Branch("NumberOfGoodTracks30", &NumberOfGoodTracks30, "NumberOfGoodTracks30/I");
   OutputTree->Branch("TotalPTTracks", &TotalPTTracks, "TotalPTTracks[2]/D");
   OutputTree->Branch("SumPTTracks", &SumPTTracks, "SumPTTracks/D");
   OutputTree->Branch("SumPTracks", &SumPTracks, "SumPTracks/D");
   OutputTree->Branch("NumberOfGoodPrimaryVertices", &NumberOfGoodPrimaryVertices, "NumberOfGoodPrimaryVertices/I");
   OutputTree->Branch("NumberOfMuonCandidates", &NumberOfMuonCandidates, "NumberOfMuonCandidates/I");
   OutputTree->Branch("NumberOfCosmicMuonCandidates", &NumberOfCosmicMuonCandidates,
      "NumberOfCosmicMuonCandidates/I");

   if(FillHBHE == true)
   {
      OutputTree->Branch("PulseCount", &PulseCount, "PulseCount/I");
      OutputTree->Branch("Charge", &Charge, "Charge[5184][10]/D");
      OutputTree->Branch("Pedestal", &Pedestal, "Pedestal[5184][10]/D");
      OutputTree->Branch("Energy", &Energy, "Energy[5184]/D");
      OutputTree->Branch("IEta", &IEta, "IEta[5184]/I");
      OutputTree->Branch("IPhi", &IPhi, "IPhi[5184]/I");
      OutputTree->Branch("Depth", &Depth, "Depth[5184]/I");
      OutputTree->Branch("RecHitTime", &RecHitTime, "RecHitTime[5184]/D");
      OutputTree->Branch("FlagWord", &FlagWord, "FlagWord[5184]/i");
      OutputTree->Branch("AuxWord", &AuxWord, "AuxWord[5184]/i");
      OutputTree->Branch("RespCorrGain", &RespCorrGain, "RespCorrGain[5184]/D");
      OutputTree->Branch("fCorr", &FCorr, "fCorr[5184]/D");
      OutputTree->Branch("SamplesToAdd", &SamplesToAdd, "SamplesToAdd[5184]/D");
   }

   if(FillHF == true)
   {
      OutputTree->Branch("HFPulseCount", &HFPulseCount, "HFPulseCount/I");
      OutputTree->Branch("HFADC", &HFADC, "HFADC[5184][10]/I");
      OutputTree->Branch("HFCharge", &HFCharge, "HFCharge[5184][10]/D");
      OutputTree->Branch("HFPedestal", &HFPedestal, "HFPedestal[5184][10]/D");
      OutputTree->Branch("HFIPhi", &HFIPhi, "HFIPhi[5184]/I");
      OutputTree->Branch("HFIEta", &HFIEta, "HFIEta[5184]/I");
      OutputTree->Branch("HFDepth", &HFDepth, "HFDepth[5184]/I");
      OutputTree->Branch("HFEnergy", &HFEnergy, "HFEnergy[5184]/D");
   }

   if(FillHO == true)
   {
      OutputTree->Branch("HOPulseCount", &HOPulseCount, "HOPulseCount/I");
      OutputTree->Branch("HOADC", &HOADC, "HOADC[2160][10]/I");
      OutputTree->Branch("HOCharge", &HOCharge, "HOCharge[2160][10]/D");
      OutputTree->Branch("HOPedestal", &HOPedestal, "HOPedestal[2160][10]/D");
      OutputTree->Branch("HOIPhi", &HOIPhi, "HOIPhi[2160]/I");
      OutputTree->Branch("HOIEta", &HOIEta, "HOIEta[2160]/I");
      OutputTree->Branch("HOEnergy", &HOEnergy, "HOEnergy[2160]/D");
   }
   
   OutputTree->Branch("RBXCharge", &RBXCharge, "RBXCharge[72][10]/D");
   OutputTree->Branch("RBXEnergy", &RBXEnergy, "RBXEnergy[72]/D");
   OutputTree->Branch("RBXCharge15", &RBXCharge15, "RBXCharge15[72][10]/D");
   OutputTree->Branch("RBXEnergy15", &RBXEnergy15, "RBXEnergy15[72]/D");

   OutputTree->Branch("HPDHits", &HPDHits, "HPDHits/I");
   OutputTree->Branch("HPDNoOtherHits", &HPDNoOtherHits, "HPDNoOtherHits/I");
   OutputTree->Branch("MaxZeros", &MaxZeros, "MaxZeros/I");
   OutputTree->Branch("MinE2E10", &MinE2E10, "MinE2E10/D");
   OutputTree->Branch("MaxE2E10", &MaxE2E10, "MaxE2E10/D");
   OutputTree->Branch("HasBadRBXR45", &HasBadRBXR45, "HasBadRBXR45/O");
   OutputTree->Branch("HasBadRBXRechitR45Loose", &HasBadRBXRechitR45Loose, "HasBadRBXRechitR45Loose/O");
   OutputTree->Branch("HasBadRBXRechitR45Tight", &HasBadRBXRechitR45Tight, "HasBadRBXRechitR45Tight/O");

   OutputTree->Branch("LeadingJetEta", &LeadingJetEta, "LeadingJetEta/D");
   OutputTree->Branch("LeadingJetPhi", &LeadingJetPhi, "LeadingJetPhi/D");
   OutputTree->Branch("LeadingJetPt", &LeadingJetPt, "LeadingJetPt/D");
   OutputTree->Branch("LeadingJetHad", &LeadingJetHad, "LeadingJetHad/D");
   OutputTree->Branch("LeadingJetEM", &LeadingJetEM, "LeadingJetEM/D");
   OutputTree->Branch("FollowingJetEta", &FollowingJetEta, "FollowingJetEta/D");
   OutputTree->Branch("FollowingJetPhi", &FollowingJetPhi, "FollowingJetPhi/D");
   OutputTree->Branch("FollowingJetPt", &FollowingJetPt, "FollowingJetPt/D");
   OutputTree->Branch("FollowingJetHad", &FollowingJetHad, "FollowingJetHad/D");
   OutputTree->Branch("FollowingJetEM", &FollowingJetEM, "FollowingJetEM/D");
   OutputTree->Branch("JetCount20", &JetCount20, "JetCount20/I");
   OutputTree->Branch("JetCount30", &JetCount30, "JetCount30/I");
   OutputTree->Branch("JetCount50", &JetCount50, "JetCount50/I");
   OutputTree->Branch("JetCount100", &JetCount100, "JetCount100/I");
   
   OutputTree->Branch("HOMaxEnergyRing0", &HOMaxEnergyRing0, "HOMaxEnergyRing0/D");
   OutputTree->Branch("HOSecondMaxEnergyRing0", &HOSecondMaxEnergyRing0, "HOSecondMaxEnergyRing0/D");
   OutputTree->Branch("HOMaxEnergyIDRing0", &HOMaxEnergyIDRing0, "HOMaxEnergyIDRing0/I");
   OutputTree->Branch("HOSecondMaxEnergyIDRing0", &HOSecondMaxEnergyIDRing0, "HOSecondMaxEnergyIDRing0/I");
   OutputTree->Branch("HOHitCount100Ring0", &HOHitCount100Ring0, "HOHitCount100Ring0/I");
   OutputTree->Branch("HOHitCount150Ring0", &HOHitCount150Ring0, "HOHitCount150Ring0/I");
   OutputTree->Branch("HOMaxEnergyRing12", &HOMaxEnergyRing12, "HOMaxEnergyRing12/D");
   OutputTree->Branch("HOSecondMaxEnergyRing12", &HOSecondMaxEnergyRing12, "HOSecondMaxEnergyRing12/D");
   OutputTree->Branch("HOMaxEnergyIDRing12", &HOMaxEnergyIDRing12, "HOMaxEnergyIDRing12/I");
   OutputTree->Branch("HOSecondMaxEnergyIDRing12", &HOSecondMaxEnergyIDRing12, "HOSecondMaxEnergyIDRing12/I");
   OutputTree->Branch("HOHitCount100Ring12", &HOHitCount100Ring12, "HOHitCount100Ring12/I");
   OutputTree->Branch("HOHitCount150Ring12", &HOHitCount150Ring12, "HOHitCount150Ring12/I");
   
   OutputTree->Branch("OfficialDecision", &OfficialDecision, "OfficialDecision/O");
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::endJob()
{
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::beginRun(const edm::Run & r, const edm::EventSetup & es)
{
   edm::ESHandle<HcalRecoParams> p;
   es.get<HcalRecoParamsRcd>().get(p);
   paramTS = new HcalRecoParams(*p.product());

   edm::ESHandle<HcalTopology> htopo;
   es.get<IdealGeometryRecord>().get(htopo);
   theTopology=new HcalTopology(*htopo);
   paramTS->setTopo(theTopology);
   
   // --------------- dump of ResoParams DB ---------------------------
   //std::cout << " skdump in HcalHitReconstructor::beginRun   dupm RecoParams " << std::endl;
   //std::ofstream skfile("skdumpRecoParamsNewFormat.txt");
   //HcalDbASCIIIO::dumpObject(skfile, (*paramTS) );
   // -----------------------------------------------------------------

   pulseCorr_->beginRun(es); // to initialize HcalPulseShapes
}

//------------------------------------------------------------------------------
void HcalNoiseAnalyzer::endRun(const edm::Run &r, const edm::EventSetup &es)
{
   if (paramTS) delete paramTS;
}
//------------------------------------------------------------------------------
void HcalNoiseAnalyzer::ClearVariables()
{
   RunNumber = 0;
   EventNumber = 0;
   LumiSection = 0;
   Bunch = 0;
   Orbit = 0;
   Time = 0;

   for(int i = 0; i < 64; i++)
      TTrigger[i] = false;
   for(int i = 0; i < 128; i++)
      L1Trigger[i] = false;
   for(int i = 0; i < 256; i++)
      HLTrigger[i] = false;

   EBET[0] = 0;   EBET[1] = 0;
   EEET[0] = 0;   EEET[1] = 0;
   HBET[0] = 0;   HBET[1] = 0;
   HEET[0] = 0;   HEET[1] = 0;
   HFET[0] = 0;   HFET[1] = 0;
   NominalMET[0] = 0;   NominalMET[1] = 0;

   EBSumE = 0;
   EESumE = 0;
   HBSumE = 0;
   HESumE = 0;
   HFSumE = 0;

   EBSumET = 0;
   EESumET = 0;
   HBSumET = 0;
   HESumET = 0;
   HFSumET = 0;

   NumberOfGoodTracks = 0;
   NumberOfGoodTracks15 = 0;
   NumberOfGoodTracks30 = 0;
   TotalPTTracks[0] = 0;   TotalPTTracks[1] = 0;
   SumPTTracks = 0;
   SumPTracks = 0;
   NumberOfGoodPrimaryVertices = 0;
   NumberOfMuonCandidates = 0;
   NumberOfCosmicMuonCandidates = 0;

   PulseCount = 0;
   for(int i = 0; i < 5184; i++)
   {
      for(int j = 0; j < 10; j++)
      {
         Charge[i][j] = 0;
         Pedestal[i][j] = 0;
      }

      Energy[i] = 0;
      IEta[i] = 0;
      IPhi[i] = 0;
      Depth[i] = 0;
      RecHitTime[i] = 0;
      FlagWord[i] = 0;
      AuxWord[i] = 0;
      RespCorrGain[i] = 0;
      FCorr[i] = 0;
      SamplesToAdd[i] = 0;
   }

   HFPulseCount = 0;
   for(int i = 0; i < 5184; i++)
   {
      for(int j = 0; j < 10; j++)
      {
         HFADC[i][j] = 0;
         HFCharge[i][j] = 0;
         HFPedestal[i][j] = 0;
      }
      HFIPhi[i] = 0;
      HFIEta[i] = 0;
      HFDepth[i] = 0;
      HFEnergy[i] = 0;
   }

   HOPulseCount = 0;
   for(int i = 0; i < 2160; i++)
   {
      for(int j = 0; j < 10; j++)
      {
         HOADC[i][j] = 0;
         HOCharge[i][j] = 0;
         HOPedestal[i][j] = 0;
      }
      HOIPhi[i] = 0;
      HOIEta[i] = 0;
      HOEnergy[i] = 0;
   }

   for(int i = 0; i < 72; i++)
   {
      for(int j = 0; j < 10; j++)
      {
         RBXCharge[i][j] = 0;
         RBXCharge15[i][j] = 0;
      }

      RBXEnergy[i] = 0;
      RBXEnergy15[i] = 0;
   }

   HPDHits = 0;
   HPDNoOtherHits = 0;
   MaxZeros = 0;
   MinE2E10 = 0;
   MaxE2E10 = 0;
   HasBadRBXR45 = false;
   HasBadRBXRechitR45Loose = false;
   HasBadRBXRechitR45Tight = false;

   LeadingJetEta = 0;
   LeadingJetPhi = 0;
   LeadingJetPt = 0;
   FollowingJetEta = 0;
   FollowingJetPhi = 0;
   FollowingJetPt = 0;
   JetCount20 = 0;
   JetCount30 = 0;
   JetCount50 = 0;
   JetCount100 = 0;

   HOMaxEnergyRing0 = 0;
   HOSecondMaxEnergyRing0 = 0;
   HOMaxEnergyIDRing0 = 0;
   HOSecondMaxEnergyIDRing0 = 0;
   HOHitCount100Ring0 = 0;
   HOHitCount150Ring0 = 0;
   HOMaxEnergyRing12 = 0;
   HOSecondMaxEnergyRing12 = 0;
   HOMaxEnergyIDRing12 = 0; 
   HOSecondMaxEnergyIDRing12 = 0;
   HOHitCount100Ring12 = 0;
   HOHitCount150Ring12 = 0;
   
   OfficialDecision = false;
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::CalculateTotalEnergiesHBHE(const HBHERecHitCollection &RecHits)
{
   for(int i = 0; i < (int)RecHits.size(); i++)
   {
      bool IsHB = true;
      if(RecHits[i].id().subdet() == HcalEndcap)
         IsHB = false;

      double eta = Geometry->getPosition(RecHits[i].id()).eta();
      double phi = Geometry->getPosition(RecHits[i].id()).phi();
      double energy = RecHits[i].energy();
      double et = energy / cosh(eta);

      if(IsHB == true)
      {
         HBET[0] = HBET[0] + et * cos(phi);
         HBET[1] = HBET[1] + et * sin(phi);
         HBSumE = HBSumE + energy;
         HBSumET = HBSumET + et;
      }
      else   // is HE
      {
         HEET[0] = HEET[0] + et * cos(phi);
         HEET[1] = HEET[1] + et * sin(phi);
         HESumE = HESumE + energy;
         HESumET = HESumET + et;
      }
   }
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::CalculateTotalEnergiesHF(const HFRecHitCollection &RecHits)
{
   for(int i = 0; i < (int)RecHits.size(); i++)
   {
      double eta = Geometry->getPosition(RecHits[i].id()).eta();
      double phi = Geometry->getPosition(RecHits[i].id()).phi();
      double energy = RecHits[i].energy();
      double et = energy / cosh(eta);
         
      HFET[0] = HFET[0] + et * cos(phi);
      HFET[1] = HFET[1] + et * sin(phi);
      HFSumE = HFSumE + energy;
      HFSumET = HFSumET + et;
   }
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::CalculateTotalEnergiesEB(const EcalRecHitCollection &RecHits)
{
   for(int i = 0; i < (int)RecHits.size(); i++)
   {
      double eta = Geometry->getPosition(RecHits[i].id()).eta();
      double phi = Geometry->getPosition(RecHits[i].id()).phi();
      double energy = RecHits[i].energy();
      double et = energy / cosh(eta);
         
      EBET[0] = EBET[0] + et * cos(phi);
      EBET[1] = EBET[1] + et * sin(phi);
      EBSumE = EBSumE + energy;
      EBSumET = EBSumET + et;
   }
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::CalculateTotalEnergiesEE(const EcalRecHitCollection &RecHits)
{
   for(int i = 0; i < (int)RecHits.size(); i++)
   {
      double eta = Geometry->getPosition(RecHits[i].id()).eta();
      double phi = Geometry->getPosition(RecHits[i].id()).phi();
      double energy = RecHits[i].energy();
      double et = energy / cosh(eta);
         
      EEET[0] = EEET[0] + et * cos(phi);
      EEET[1] = EEET[1] + et * sin(phi);
      EESumE = EESumE + energy;
      EESumET = EESumET + et;
   }
}
//---------------------------------------------------------------------------
void HcalNoiseAnalyzer::Initialize()
{
   pulseCorr_ =
      std::auto_ptr<HcalPulseContainmentManager>(new HcalPulseContainmentManager(MaximumFractionalError));

   /*
   // Reads in ideal pulse shape - for fitting purposes
   vector<double> PulseShape;

   HcalPulseShapes Shapes;
   HcalPulseShapes::Shape HPDShape = Shapes.hbShape();

   for(int i = 0; i < 200; i++)
      PulseShape.push_back(HPDShape.at(i));
   PulseShape.insert(PulseShape.begin(), 150, 0);

   CumulativeIdealPulse.clear();
   CumulativeIdealPulse.push_back(0);
   for(unsigned int i = 1; i < PulseShape.size(); i++)
      CumulativeIdealPulse.push_back(CumulativeIdealPulse[i-1] + PulseShape[i]);
   */
}
//---------------------------------------------------------------------------
double ECorr(int ieta, int iphi, double energy)
{
   // return energy correction factor for HBM channels 
   // iphi=6 ieta=(-1,-15) and iphi=32 ieta=(-1,-7)
   // I.Vodopianov 28 Feb. 2011
   static const float low32[7]  = {0.741,0.721,0.730,0.698,0.708,0.751,0.861};
   static const float high32[7] = {0.973,0.925,0.900,0.897,0.950,0.935,1};
   static const float low6[15]  = {0.635,0.623,0.670,0.633,0.644,0.648,0.600,
      0.570,0.595,0.554,0.505,0.513,0.515,0.561,0.579};
   static const float high6[15] = {0.875,0.937,0.942,0.900,0.922,0.925,0.901,
      0.850,0.852,0.818,0.731,0.717,0.782,0.853,0.778};

   double slope, mid, en;
   double corr = 1.0;

   if (!(iphi==6 && ieta<0 && ieta>-16) && !(iphi==32 && ieta<0 && ieta>-8)) 
      return corr;

   int jeta = -ieta-1;
   double xeta = (double) ieta;
   if (energy > 0.) en=energy;
   else en = 0.;

   if (iphi == 32) {
      slope = 0.2272;
      mid = 17.14 + 0.7147*xeta;
      if (en > 100.) corr = high32[jeta];
      else corr = low32[jeta]+(high32[jeta]-low32[jeta])/(1.0+exp(-(en-mid)*slope));
   }
   else if (iphi == 6) {
      slope = 0.1956;
      mid = 15.96 + 0.3075*xeta;
      if (en > 100.0) corr = high6[jeta];
      else corr = low6[jeta]+(high6[jeta]-low6[jeta])/(1.0+exp(-(en-mid)*slope));
   }

   return corr;
}
//---------------------------------------------------------------------------
DEFINE_FWK_MODULE(HcalNoiseAnalyzer);
