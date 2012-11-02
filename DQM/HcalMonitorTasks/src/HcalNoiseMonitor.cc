#include <cmath>
#include <fstream>
#include <algorithm>

#include "DQM/HcalMonitorTasks/interface/HcalNoiseMonitor.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"

#include "FWCore/Common/interface/TriggerNames.h"

HcalNoiseMonitor::HcalNoiseMonitor(const edm::ParameterSet& ps)
{
   Online_                = ps.getUntrackedParameter<bool>("online",false);
   mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
   enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
   debug_                 = ps.getUntrackedParameter<int>("debug",0);
   prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
   if(prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
      prefixME_.append("/");
   subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","NoiseMonitor_Hcal");
   if(subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
      subdir_.append("/");
   subdir_=prefixME_+subdir_;
   AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
   skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
   NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
   makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

   triggers_=ps.getUntrackedParameter<std::vector<std::string> >("nzsHLTnames");
      //["HLT_HcalPhiSym","HLT_HcalNoise_8E29]
   period_=ps.getUntrackedParameter<int>("NoiseeventPeriod",4096); //4096
   rawdataLabel_          = ps.getUntrackedParameter<edm::InputTag>("RawDataLabel");
   hltresultsLabel_       = ps.getUntrackedParameter<edm::InputTag>("HLTResultsLabel");
   hbheDigiLabel_         = ps.getUntrackedParameter<edm::InputTag>("hbheDigiLabel");
   hbheRechitLabel_       = ps.getUntrackedParameter<edm::InputTag>("hbheRechitLabel");
   noiseLabel_            = ps.getUntrackedParameter<edm::InputTag>("noiseLabel");

   mTrianglePeakTS        = 4;   // for now...

   mE2E10MinEnergy        = ps.getUntrackedParameter<double>("E2E10MinEnergy");
   mMinE2E10              = ps.getUntrackedParameter<double>("MinE2E10");
   mMaxE2E10              = ps.getUntrackedParameter<double>("MaxE2E10");
   mMaxHPDHitCount        = ps.getUntrackedParameter<int>("MaxHPDHitCount");
   mMaxHPDNoOtherHitCount = ps.getUntrackedParameter<int>("MaxHPDNoOtherHitCount");
   mMaxADCZeros           = ps.getUntrackedParameter<int>("MaxADCZeros");
   mTotalZeroMinEnergy    = ps.getUntrackedParameter<double>("TotalZeroMinEnergy");
   setupDone_ = false;
}

HcalNoiseMonitor::~HcalNoiseMonitor() {}

void HcalNoiseMonitor::reset()
{
}

void HcalNoiseMonitor::cleanup()
{
   if(dbe_)
   {
      dbe_->setCurrentFolder(subdir_);
      dbe_->removeContents();
   }
}

void HcalNoiseMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
   if(debug_ > 1)
      std::cout <<"HcalNoiseMonitor::beginRun"<< std::endl;

   HcalBaseDQMonitor::beginRun(run,c);

   if(tevt_ == 0)
      setup();

   if(mergeRuns_ == false)
      reset();

   return;
}


void HcalNoiseMonitor::setup()
{
   if (setupDone_)
     return;
   setupDone_ = true;
   HcalBaseDQMonitor::setup();

   if(debug_ > 1)
      std::cout << "<HcalNoiseMonitor::setup> Creating histograms" << std::endl;

   if(dbe_)
   {
      dbe_->setCurrentFolder(subdir_);

      // Fit-based
      dbe_->setCurrentFolder(subdir_ + "DoubleChi2/");

      hNominalChi2 = dbe_->book1D("Nominal_fit_chi2", "Nominal fit chi2, total charge > 20 fC", 100, 0, 200);
      hNominalChi2->setAxisTitle("Nominal fit #chi^{2}", 1);

      hLinearChi2 = dbe_->book1D("Linear_fit_chi2", "Linear fit chi2, total charge > 20 fC", 100, 0, 200);
      hLinearChi2->setAxisTitle("Linear fit #chi^{2}", 1);
      
      hLinearTestStatistics = dbe_->book1D("Lambda_linear", "#Lambda_{linear}, total charge > 20 fC", 100, -10, 10);
      hLinearTestStatistics->setAxisTitle("#Lambda_{linear}", 1);

      hRMS8OverMax = dbe_->book1D("RMS8_over_Max", "RMS8/max, total charge > 20 fC", 100, 0, 2);
      hRMS8OverMax->setAxisTitle("RMS8/max", 1);

      hRMS8OverMaxTestStatistics = dbe_->book1D("Lambda_RMS8_over_max", "#Lambda_{RMS8/Max}, total charge > 20 fC",
         100, -30, 10);
      hRMS8OverMaxTestStatistics->setAxisTitle("#Lambda_{RMS8/Max}", 1);

      hLambdaLinearVsTotalCharge = dbe_->book2D("Lambda_linear_vs_total_charge", "#Lambda_{Linear}",
         50, -5, 5, 25, 0, 500);
      hLambdaLinearVsTotalCharge->setAxisTitle("#Lambda_{linear}", 1);
      hLambdaLinearVsTotalCharge->setAxisTitle("Total charge", 2);

      hLambdaRMS8MaxVsTotalCharge = dbe_->book2D("Lambda_RMS8Max_vs_total_charge", "#Lambda_{RMS8/Max}",
         50, -15, 5, 25, 0, 500);
      hLambdaRMS8MaxVsTotalCharge->setAxisTitle("#Lambda_{RMS8/Max}", 1);
      hLambdaRMS8MaxVsTotalCharge->setAxisTitle("Total charge", 2);

      hTriangleLeftSlopeVsTS4 = dbe_->book2D("Triangle_fit_left_slope",
         "Triangle fit left distance vs. TS4", 50, 0, 10, 25, 0, 500);
      hTriangleLeftSlopeVsTS4->setAxisTitle("Left slope", 1);
      hTriangleLeftSlopeVsTS4->setAxisTitle("Peak time slice", 2);

      hTriangleRightSlopeVsTS4 = dbe_->book2D("Triangle_fit_right_slope",
         "Triangle fit right distance vs. peak time slice", 50, 0, 10, 25, 0, 500);
      hTriangleRightSlopeVsTS4->setAxisTitle("Left slope", 1);
      hTriangleRightSlopeVsTS4->setAxisTitle("Peak time slice", 2);

      SetupEtaPhiHists(hFailLinearEtaPhi, "Fail_linear_Eta_Phi_Map", "");
      SetupEtaPhiHists(hFailRMSMaxEtaPhi, "Fail_RMS8Max_Eta_Phi_Map", "");
      SetupEtaPhiHists(hFailTriangleEtaPhi, "Fail_triangle_Eta_Phi_Map", "");

      // High-level isolation filter
      dbe_->setCurrentFolder(subdir_ + "IsolationVariable/");

      SetupEtaPhiHists(hFailIsolationEtaPhi, "Fail_isolation_Eta_Phi_Map", "");
      
      // TS4 vs. TS5 variable
      dbe_->setCurrentFolder(subdir_ + "TS4TS5Variable/");
      
      hTS4TS5RelativeDifference = dbe_->book1D("TS4_TS5_relative_difference",
         "(TS4-TS5)/(TS4+TS5), total charge > 20 fC", 100, -1, 1);
      hTS4TS5RelativeDifference->setAxisTitle("(TS4 - TS5) / (TS4 + TS5)", 1);

      hTS4TS5RelativeDifferenceVsCharge = dbe_->book2D("TS4_TS5_relative_difference_charge",
         "(TS4-TS5)/(TS4+TS5) vs. Charge", 25, 0, 400, 75, -1, 1);
      hTS4TS5RelativeDifferenceVsCharge->setAxisTitle("Charge", 1);
      hTS4TS5RelativeDifferenceVsCharge->setAxisTitle("(TS4 - TS5) / (TS4 + TS5)", 2);

      // Noise summary object
      dbe_->setCurrentFolder(subdir_ + "NoiseMonitoring/");
   
      hMaxZeros = dbe_->book1D("Max_Zeros", "Max zeros", 15, -0.5, 14.5);
      
      hTotalZeros = dbe_->book1D("Total_Zeros", "Total zeros", 15, -0.5, 14.5);
      
      hE2OverE10Digi = dbe_->book1D("E2OverE10Digi", "E2/E10 of the highest digi in an HPD", 100, 0, 2);
      
      hE2OverE10Digi5 = dbe_->book1D("E2OverE10Digi5", "E2/E10 of the highest 5 digi in an HPD", 100, 0, 2);
      
      hE2OverE10RBX = dbe_->book1D("E2OverE10RBX", "E2/E10 of RBX", 100, 0, 2);
      
      hHPDHitCount = dbe_->book1D("HPDHitCount", "HPD hit count (1.5 GeV)", 19, -0.5, 18.5);
      
      hRBXHitCount = dbe_->book1D("RBXHitCount", "Number of hits in RBX", 74, -0.5, 73.5);

      hHcalNoiseCategory = dbe_->book1D("Hcal_noise_category", "Hcal noise category", 10, 0.5, 10.5);
      hHcalNoiseCategory->setBinLabel(1, "RBX noise", 1);
      hHcalNoiseCategory->setBinLabel(2, "RBX pedestal flatter", 1);
      hHcalNoiseCategory->setBinLabel(3, "RBX pedestal sharper", 1);
      hHcalNoiseCategory->setBinLabel(4, "RBX flash large hit count", 1);
      hHcalNoiseCategory->setBinLabel(5, "RBX flash small hit count", 1);
      hHcalNoiseCategory->setBinLabel(7, "HPD discharge", 1);
      hHcalNoiseCategory->setBinLabel(8, "HPD ion feedback", 1);

      hBadZeroRBX = dbe_->book1D("BadZeroRBX", "RBX with bad ADC zero counts", 72, 0.5, 72.5);
      hBadCountHPD = dbe_->book1D("BadCountHPD", "HPD with bad hit counts", 72 * 4, 0.5, 72 * 4 + 0.5);
      hBadNoOtherCountHPD = dbe_->book1D("BadNoOtherCountHPD", "HPD with bad \"no other\" hit counts", 72 * 4, 0.5, 72 * 4 + 0.5);
      hBadE2E10RBX = dbe_->book1D("BadE2E10RBX", "RBX with bad E2/E10 value", 72, 0.5, 72.5);
   }

   ReadHcalPulse();

   return;
}


void HcalNoiseMonitor::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup)
{
   edm::Handle<HBHEDigiCollection> hHBHEDigis;
   iEvent.getByLabel(edm::InputTag(hbheDigiLabel_),hHBHEDigis);

   edm::ESHandle<HcalDbService> hConditions;
   iSetup.get<HcalDbRecord>().get(hConditions);

   edm::Handle<HBHERecHitCollection> hRecHits;
   iEvent.getByLabel(edm::InputTag(hbheRechitLabel_), hRecHits);

   edm::Handle<reco::HcalNoiseRBXCollection> hRBXCollection;
   iEvent.getByLabel(edm::InputTag(noiseLabel_),hRBXCollection);

   HcalBaseDQMonitor::analyze(iEvent, iSetup);

   if(dbe_ == NULL)
   {
      if(debug_ > 0)
         std::cout << "HcalNoiseMonitor::processEvent DQMStore not instantiated!!!"<< std::endl;
      return;
   }

   // loop over digis
   for(HBHEDigiCollection::const_iterator iter = hHBHEDigis->begin(); iter != hHBHEDigis->end(); iter++)
   {
      HcalDetId id = iter->id();
      const HcalCalibrations &Calibrations = hConditions->getHcalCalibrations(id);
      const HcalQIECoder *ChannelCoder = hConditions->getHcalCoder(id);
      const HcalQIEShape *Shape = hConditions->getHcalShape(ChannelCoder);
      HcalCoderDb Coder(*ChannelCoder, *Shape);
      CaloSamples Tool;
      Coder.adc2fC(*iter, Tool);

      // int ieta = id.ieta();
      // int iphi = id.iphi();
      // int depth = id.depth();

      double Charge[10] = {0};
      for(int i = 0; i < iter->size(); i++)
         Charge[i] = Tool[i] - Calibrations.pedestal(iter->sample(i).capid());

      double TotalCharge = 0;
      for(int i = 0; i < 10; i++)
         TotalCharge = TotalCharge + Charge[i];

      if(TotalCharge > 20)
      {
         double NominalChi2 = 10000000;
         NominalChi2 = PerformNominalFit(Charge);
         
         double LinearChi2 = PerformLinearFit(Charge);
         double RMS8Max = CalculateRMS8Max(Charge);
         TriangleFitResult TriangleResult = PerformTriangleFit(Charge);
      
         double TS4LeftSlope = 100000;
         double TS4RightSlope = 100000;
         
         if(TriangleResult.LeftSlope > 1e-5)
            TS4LeftSlope = Charge[4] / fabs(TriangleResult.LeftSlope);
         if(TriangleResult.RightSlope < -1e-5)
            TS4RightSlope = Charge[4] / fabs(TriangleResult.RightSlope);

         if(TS4LeftSlope < -1000 || TS4LeftSlope > 1000)
            TS4LeftSlope = 1000;
         if(TS4RightSlope < -1000 || TS4RightSlope > 1000)
            TS4RightSlope = 1000;

         hNominalChi2->Fill(NominalChi2);
         hLinearChi2->Fill(LinearChi2);
         hLinearTestStatistics->Fill(log(LinearChi2) - log(NominalChi2));
         hRMS8OverMax->Fill(RMS8Max);
         hRMS8OverMaxTestStatistics->Fill(log(RMS8Max) - log(NominalChi2));

         hLambdaLinearVsTotalCharge->Fill(log(LinearChi2) - log(NominalChi2), TotalCharge);
         hLambdaRMS8MaxVsTotalCharge->Fill(log(RMS8Max) - log(NominalChi2), TotalCharge);
         hTriangleLeftSlopeVsTS4->Fill(TS4LeftSlope, Charge[4]);
         hTriangleRightSlopeVsTS4->Fill(TS4RightSlope, Charge[4]);
      }

      if(Charge[4] + Charge[5] > 1e-5)
      {
         hTS4TS5RelativeDifference->Fill((Charge[4] - Charge[5]) / (Charge[4] + Charge[5]));
         hTS4TS5RelativeDifferenceVsCharge->Fill(TotalCharge, (Charge[4] - Charge[5]) / (Charge[4] + Charge[5]));
      }
   }

   // loop over rechits - noise bits (fit-based, isolation)
   for(HBHERecHitCollection::const_iterator iter = hRecHits->begin(); iter != hRecHits->end(); iter++)
   {
      HcalDetId id = iter->id();

      int ieta = id.ieta();
      int iphi = id.iphi();
      int depth = id.depth();

      if(iter->flagField(HcalCaloFlagLabels::HBHEFlatNoise) == 1)
         hFailLinearEtaPhi.depth[depth-1]->Fill(ieta, iphi);

      if(iter->flagField(HcalCaloFlagLabels::HBHESpikeNoise) == 1)
         hFailRMSMaxEtaPhi.depth[depth-1]->Fill(ieta, iphi);
      
      if(iter->flagField(HcalCaloFlagLabels::HBHETriangleNoise) == 1)
         hFailTriangleEtaPhi.depth[depth-1]->Fill(ieta, iphi);

      if(iter->flagField(HcalCaloFlagLabels::HBHEIsolatedNoise) == 1)
         hFailIsolationEtaPhi.depth[depth-1]->Fill(ieta, iphi);
   }

   // Code analagous to Yifei's
   for(reco::HcalNoiseRBXCollection::const_iterator rbx = hRBXCollection->begin();
      rbx != hRBXCollection->end(); rbx++)
   {
      const reco::HcalNoiseRBX RBX = *rbx;

      int NumberRBXHits = RBX.numRecHits(1.5);
      double RBXEnergy = RBX.recHitEnergy(1.5);
      double RBXE2 = RBX.allChargeHighest2TS();
      double RBXE10 = RBX.allChargeTotal();

      std::vector<reco::HcalNoiseHPD> HPDs = RBX.HPDs();
      
      int RBXID = RBX.idnumber();

      if(RBXEnergy > mTotalZeroMinEnergy && RBX.totalZeros() >= mMaxADCZeros)
         hBadZeroRBX->Fill(RBXID);
      if(RBXEnergy > mE2E10MinEnergy && RBXE10 > 1e-5 && (RBXE2 / RBXE10 > mMaxE2E10 || RBXE2 / RBXE10 < mMinE2E10))
         hBadE2E10RBX->Fill(RBXID);
      for(std::vector<reco::HcalNoiseHPD>::const_iterator hpd = HPDs.begin(); hpd != HPDs.end(); hpd++)
      {
         reco::HcalNoiseHPD HPD = *hpd;
         int HPDHitCount = HPD.numRecHits(1.5);
         if(HPDHitCount >= mMaxHPDHitCount)
            hBadCountHPD->Fill(HPD.idnumber());
         if(HPDHitCount == NumberRBXHits && HPDHitCount >= mMaxHPDNoOtherHitCount)
            hBadNoOtherCountHPD->Fill(HPD.idnumber());
      }
      
      if(NumberRBXHits == 0 || RBXEnergy <= 10)
         continue;

      hRBXHitCount->Fill(NumberRBXHits);

      hMaxZeros->Fill(RBX.maxZeros());
      hTotalZeros->Fill(RBX.totalZeros());
   
      double HighestHPDEnergy = 0;
      int HighestHPDHits = 0;

      for(std::vector<reco::HcalNoiseHPD>::const_iterator hpd = HPDs.begin(); hpd != HPDs.end(); hpd++)
      {
         reco::HcalNoiseHPD HPD = *hpd;

         if(HPD.recHitEnergy(1.5) > HighestHPDEnergy)
         {
            HighestHPDEnergy = HPD.recHitEnergy(1.5);
            HighestHPDHits = HPD.numRecHits(1.5);
         }

         if(HPD.numRecHits(5) < 1)
            continue;

         if(HPD.bigChargeTotal() > 1e-5)
            hE2OverE10Digi->Fill(HPD.bigChargeHighest2TS() / HPD.bigChargeTotal());
         if(HPD.big5ChargeTotal() > 1e-5)
            hE2OverE10Digi5->Fill(HPD.big5ChargeHighest2TS() / HPD.big5ChargeTotal());

         hHPDHitCount->Fill(HPD.numRecHits(1.5));
      }

      int NoiseCategory = 0;
      bool IsRBXNoise = false;
      //      bool IsHPDNoise = false;
      //      bool IsHPDIonFeedback = false;
      //      bool IsHPDDischarge = false;

      if(RBXEnergy > 1e-5 && HighestHPDEnergy / RBXEnergy > 0.98)
      {
	//         IsHPDNoise = true;

         if(HighestHPDHits >= 9)
         {
	   //            IsHPDDischarge = true;
            NoiseCategory = 7;
         }
         else
         {
	   //            IsHPDIonFeedback = true;
            NoiseCategory = 8;
         }
      }
      else
      {
         IsRBXNoise = true;
         NoiseCategory = 1;

         if(RBXE10 > 1e-5)
         {
            if(RBXE2 / RBXE10 < 0.33)
               NoiseCategory = 2;
            else if(RBXE2 / RBXE10 < 0.8)
               NoiseCategory = 3;
            else if(RBXE2 / RBXE10 > 0.8 && NumberRBXHits > 10)
               NoiseCategory = 4;
            else if(RBXE2 / RBXE10 > 0.8 && NumberRBXHits < 10)  // [hic]
               NoiseCategory = 5;
         }
      }

      hHcalNoiseCategory->Fill(NoiseCategory);

      if(IsRBXNoise == true && RBXE10 > 1e-5)
         hE2OverE10RBX->Fill(RBXE2 / RBXE10);
   }

   return;
}

double HcalNoiseMonitor::PerformNominalFit(double Charge[10])
{
   //
   // Performs a fit to the ideal pulse shape.  Returns best chi2
   //
   // A scan over different timing offset (for the ideal pulse) is carried out,
   //    and for each offset setting a one-parameter fit is performed
   //

   int DigiSize = 10;

   double MinimumChi2 = 100000;

   double F = 0;

   double SumF2 = 0;
   double SumTF = 0;
   double SumT2 = 0;

   for(int i = 0; i + 250 < (int)CumulativeIdealPulse.size(); i++)
   {
      if(CumulativeIdealPulse[i+250] - CumulativeIdealPulse[i] < 1e-5)
         continue;

      SumF2 = 0;
      SumTF = 0;
      SumT2 = 0;

      for(int j = 0; j < DigiSize; j++)
      {
         // get ideal pulse component for this time slice....
         F = CumulativeIdealPulse[i+j*25+25] - CumulativeIdealPulse[i+j*25];

         double Error2 = Charge[j];
         if(Error2 < 1)
            Error2 = 1;

         // ...and increment various summations
         SumF2 += F * F / Error2;
         SumTF += F * Charge[j] / Error2;
         SumT2 += Charge[j] * Charge[j] / Error2;
      }

      /* chi2= sum((Charge[j]-aF)^2/|Charge[j]|
         ( |Charge[j]| = assumed sigma^2 for Charge[j]; a bit wonky for Charge[j]<1 )
         chi2 = sum(|Charge[j]|) - 2*sum(aF*Charge[j]/|Charge[j]|) +sum( a^2*F^2/|Charge[j]|)
         chi2 minimimized when d(chi2)/da = 0:
         a = sum(F*Charge[j])/sum(F^2)
         ...
         chi2= sum(|Q[j]|) - sum(Q[j]/|Q[j]|*F)*sum(Q[j]/|Q[j]|*F)/sum(F^2/|Q[j]|), where Q = Charge
         chi2 = SumT2 - SumTF*SumTF/SumF2
      */
      
      double Chi2 = SumT2 - SumTF * SumTF / SumF2;

      if(Chi2 < MinimumChi2)
         MinimumChi2 = Chi2;
   }

   // safety protection in case of perfect fit - don't want the log(...) to explode
   if(MinimumChi2 < 1e-5)
      MinimumChi2 = 1e-5;

   return MinimumChi2;
}

double HcalNoiseMonitor::PerformDualNominalFit(double Charge[10])
{
   //
   // Perform dual nominal fit and returns the chi2
   //
   // In this function we do a scan over possible "distance" (number of time slices between two components)
   //    and overall offset for the two components; first coarse, then finer
   // All the fitting is done in the DualNominalFitSingleTry function
   //                  
   
   double OverallMinimumChi2 = 1000000;

   int AvailableDistance[] = {-100, -75, -50, 50, 75, 100};

   // loop over possible pulse distances between two components
   for(int k = 0; k < 6; k++)
   {
      double SingleMinimumChi2 = 1000000;
      int MinOffset = 0;

      // scan coarsely through different offsets and find the minimum
      for(int i = 0; i + 250 < (int)CumulativeIdealPulse.size(); i += 10)
      {
         double Chi2 = DualNominalFitSingleTry(Charge, i, AvailableDistance[k]);

         if(Chi2 < SingleMinimumChi2)
         {
            SingleMinimumChi2 = Chi2;
            MinOffset = i;
         }
      }

      // around the minimum, scan finer for better a better minimum
      for(int i = MinOffset - 15; i + 250 < (int)CumulativeIdealPulse.size() && i < MinOffset + 15; i++)
      {
         double Chi2 = DualNominalFitSingleTry(Charge, i, AvailableDistance[k]);
         if(Chi2 < SingleMinimumChi2)
            SingleMinimumChi2 = Chi2;
      }

      // update overall minimum chi2
      if(SingleMinimumChi2 < OverallMinimumChi2)
         OverallMinimumChi2 = SingleMinimumChi2;
   }

   return OverallMinimumChi2;
}

double HcalNoiseMonitor::DualNominalFitSingleTry(double Charge[10], int Offset, int Distance)
{
   //
   // Does a fit to dual signal pulse hypothesis given offset and distance of the two target pulses
   //
   // The only parameters to fit here are the two pulse heights of in-time and out-of-time components
   //    since offset is given
   // The calculation here is based from writing down the Chi2 formula and minimize against the two parameters,
   //    ie., Chi2 = Sum{((T[i] - a1 * F1[i] - a2 * F2[i]) / (Sigma[i]))^2}, where T[i] is the input pulse shape,
   //    and F1[i], F2[i] are the two ideal pulse components
   //

   int DigiSize = 10;

   if(Offset < 0 || Offset + 250 >= (int)CumulativeIdealPulse.size())
      return 1000000;
   if(CumulativeIdealPulse[Offset+250] - CumulativeIdealPulse[Offset] < 1e-5)
      return 1000000;

   static std::vector<double> F1;
   static std::vector<double> F2;

   F1.resize(DigiSize);
   F2.resize(DigiSize);

   double SumF1F1 = 0;
   double SumF1F2 = 0;
   double SumF2F2 = 0;
   double SumTF1 = 0;
   double SumTF2 = 0;

   double Error = 0;

   for(int j = 0; j < DigiSize; j++)
   {
      // this is the TS value for in-time component - no problem we can do a subtraction directly
      F1[j] = CumulativeIdealPulse[Offset+j*25+25] - CumulativeIdealPulse[Offset+j*25];

      // However for the out-of-time component the index might go out-of-bound.
      // Let's protect against this.

      int OffsetTemp = Offset + j * 25 + Distance;
      
      double C1 = 0;   // lower-indexed value in the cumulative pulse shape
      double C2 = 0;   // higher-indexed value in the cumulative pulse shape
      
      if(OffsetTemp + 25 < (int)CumulativeIdealPulse.size() && OffsetTemp + 25 >= 0)
         C1 = CumulativeIdealPulse[OffsetTemp+25];
      if(OffsetTemp + 25 >= (int)CumulativeIdealPulse.size())
         C1 = CumulativeIdealPulse[CumulativeIdealPulse.size()-1];
      if(OffsetTemp < (int)CumulativeIdealPulse.size() && OffsetTemp >= 0)
         C2 = CumulativeIdealPulse[OffsetTemp];
      if(OffsetTemp >= (int)CumulativeIdealPulse.size())
         C2 = CumulativeIdealPulse[CumulativeIdealPulse.size()-1];
      F2[j] = C1 - C2;

      Error = Charge[j];
      if(Error < 1)
         Error = 1;

      SumF1F1 += F1[j] * F1[j] / Error;
      SumF1F2 += F1[j] * F2[j] / Error; 
      SumF2F2 += F2[j] * F2[j] / Error;
      SumTF1  += F1[j] * Charge[j] / Error; 
      SumTF2  += F2[j] * Charge[j] / Error; 
   }

   double Height = (SumF1F2 * SumTF2 - SumF2F2 * SumTF1) / (SumF1F2 * SumF1F2 - SumF1F1 * SumF2F2);
   double Height2 = (SumF1F2 * SumTF1 - SumF1F1 * SumTF2) / (SumF1F2 * SumF1F2 - SumF1F1 * SumF2F2);

   double Chi2 = 0;
   for(int j = 0; j < DigiSize; j++)
   {
      double Error = Charge[j];
      if(Error < 1)
         Error = 1;

      double Residual = Height * F1[j] + Height2 * F2[j] - Charge[j];  
      Chi2 += Residual * Residual / Error;                             
   } 

   // Safety protection in case of zero
   if(Chi2 < 1e-5)
      Chi2 = 1e-5;

   return Chi2;
}

double HcalNoiseMonitor::PerformLinearFit(double Charge[10])
{
   //
   // Performs a straight-line fit over all time slices, and returns the chi2 value
   //
   // The calculation here is based from writing down the formula for chi2 and minimize
   //    with respect to the parameters in the fit, ie., slope and intercept of the straight line
   // By doing two differentiation, we will get two equations, and the best parameters are determined by these
   //

   int DigiSize = 10;

   double SumTS2OverTi = 0;
   double SumTSOverTi = 0;
   double SumOverTi = 0;
   double SumTiTS = 0;
   double SumTi = 0;

   double Error2 = 0;
   for(int i = 0; i < DigiSize; i++)
   {
      Error2 = Charge[i];
      if(Charge[i] < 1)
         Error2 = 1;

      SumTS2OverTi += 1.* i * i / Error2;
      SumTSOverTi  += 1.* i / Error2;
      SumOverTi    += 1. / Error2;
      SumTiTS      += Charge[i] * i / Error2;
      SumTi        += Charge[i] / Error2;
   }

   double CM1 = SumTS2OverTi;   // Coefficient in front of slope in equation 1
   double CM2 = SumTSOverTi;   // Coefficient in front of slope in equation 2
   double CD1 = SumTSOverTi;   // Coefficient in front of intercept in equation 1
   double CD2 = SumOverTi;   // Coefficient in front of intercept in equation 2
   double C1 = SumTiTS;   // Constant coefficient in equation 1
   double C2 = SumTi;   // Constant coefficient in equation 2

   double Slope = (C1 * CD2 - C2 * CD1) / (CM1 * CD2 - CM2 * CD1);
   double Intercept = (C1 * CM2 - C2 * CM1) / (CD1 * CM2 - CD2 * CM1);

   // now that the best parameters are found, calculate chi2 from those
   double Chi2 = 0;
   for(int i = 0; i < DigiSize; i++)
   {
      double Deviation = Slope * i + Intercept - Charge[i];
      double Error2 = Charge[i];
      if(Charge[i] < 1)
         Error2 = 1;
      Chi2 += Deviation * Deviation / Error2;  
   }

   // safety protection in case of perfect fit
   if(Chi2 < 1e-5)
      Chi2 = 1e-5;

   return Chi2;
}

double HcalNoiseMonitor::CalculateRMS8Max(double Charge[10])
{
   //
   // CalculateRMS8Max
   //
   // returns "RMS" divided by the largest charge in the time slices
   //    "RMS" is calculated using all but the two largest time slices.
   //    The "RMS" is not quite the actual RMS (see note below), but the value is only
   //    used for determining max values, and is not quoted as the actual RMS anywhere.
   //

   int DigiSize = 10;

   // Copy Charge vector again, since we are passing references around
   std::vector<double> TempCharge(Charge, Charge + 10);

   // Sort TempCharge vector from smallest to largest charge
   sort(TempCharge.begin(), TempCharge.end());

   double Total = 0;
   double Total2 = 0;
   for(int i = 0; i < DigiSize - 2; i++)
   {
      Total = Total + TempCharge[i];
      Total2 = Total2 + TempCharge[i] * TempCharge[i];
   }

   // This isn't quite the RMS (both Total2 and Total*Total need to be
   // divided by an extra (DigiSize-2) within the sqrt to get the RMS.)
   // We're only using this value for relative comparisons, though; we
   // aren't explicitly interpreting it as the RMS.  It might be nice
   // to either change the calculation or rename the variable in the future, though.

   double RMS = sqrt(Total2 - Total * Total / (DigiSize - 2));

   double RMS8Max = 99999;
   if(TempCharge[DigiSize-1] > 1e-5)
      RMS8Max = RMS / TempCharge[DigiSize-1];
   if(RMS8Max < 1e-5)   // protection against zero
      RMS8Max = 1e-5;

   return RMS8Max;
}

TriangleFitResult HcalNoiseMonitor::PerformTriangleFit(double Charge[10])
{
   //
   // Perform a "triangle fit", and extract the slopes
   //
   // Left-hand side and right-hand side are not correlated to each other - do them separately
   //

   TriangleFitResult result;
   result.Chi2 = 0;
   result.LeftSlope = 0;
   result.RightSlope = 0;

   int DigiSize = 10;

   // right side, starting from TS4
   double MinimumRightChi2 = 1000000;
   double Numerator = 0;
   double Denominator = 0;

   for(int iTS = mTrianglePeakTS + 2; iTS <= DigiSize; iTS++)   // the place where first TS center in flat line
   {
      // fit a straight line to the triangle part
      Numerator = 0;
      Denominator = 0;

      for(int i = mTrianglePeakTS + 1; i < iTS; i++)
      {
         Numerator += (i - mTrianglePeakTS) * (Charge[i] - Charge[mTrianglePeakTS]);
         Denominator += (i - mTrianglePeakTS) * (i - mTrianglePeakTS);
      }

      double BestSlope = Numerator / Denominator;
      if(BestSlope > 0)
         BestSlope = 0;

      // check if the slope is reasonable
      if(iTS != DigiSize)
      {
         if(BestSlope > -1 * Charge[mTrianglePeakTS] / (iTS - mTrianglePeakTS))
            BestSlope = -1 * Charge[mTrianglePeakTS] / (iTS - mTrianglePeakTS);
         if(BestSlope < -1 * Charge[mTrianglePeakTS] / (iTS - 1 - mTrianglePeakTS))
            BestSlope = -1 * Charge[mTrianglePeakTS] / (iTS - 1 - mTrianglePeakTS);
      }
      else
      {
         if(BestSlope < -1 * Charge[mTrianglePeakTS] / (iTS - 1 - mTrianglePeakTS)) 
            BestSlope = -1 * Charge[mTrianglePeakTS] / (iTS - 1 - mTrianglePeakTS);
      }

      // calculate partial chi2

      // The shape I'm fitting is more like a tent than a triangle.
      // After the end of triangle edge assume a flat line

      double Chi2 = 0;
      for(int i = mTrianglePeakTS + 1; i < iTS; i++)
         Chi2 += (Charge[mTrianglePeakTS] - Charge[i] + (i - mTrianglePeakTS) * BestSlope)
            * (Charge[mTrianglePeakTS] - Charge[i] + (i - mTrianglePeakTS) * BestSlope);
      for(int i = iTS; i < DigiSize; i++)    // Assumes fit line = 0 for iTS > fit
         Chi2 += Charge[i] * Charge[i];

      if(Chi2 < MinimumRightChi2)
      {
         MinimumRightChi2 = Chi2;
         result.RightSlope = BestSlope;
      }
   }   // end of right-hand side loop

   // left side
   double MinimumLeftChi2 = 1000000;

   for(int iTS = 0; iTS < (int)mTrianglePeakTS; iTS++)   // the first time after linear fit ends
   {
      // fit a straight line to the triangle part
      Numerator = 0;
      Denominator = 0;
      for(int i = iTS; i < (int)mTrianglePeakTS; i++)
      {
         Numerator = Numerator + (i - mTrianglePeakTS) * (Charge[i] - Charge[mTrianglePeakTS]);
         Denominator = Denominator + (i - mTrianglePeakTS) * (i - mTrianglePeakTS);
      }

      double BestSlope = Numerator / Denominator;
      if(BestSlope < 0)
         BestSlope = 0;

      // check slope
      if(iTS != 0)
      {
         if(BestSlope > Charge[mTrianglePeakTS] / (mTrianglePeakTS - iTS))
            BestSlope = Charge[mTrianglePeakTS] / (mTrianglePeakTS - iTS);
         if(BestSlope < Charge[mTrianglePeakTS] / (mTrianglePeakTS + 1 - iTS))
            BestSlope = Charge[mTrianglePeakTS] / (mTrianglePeakTS + 1 - iTS);
      }
      else
      {
         if(BestSlope > Charge[mTrianglePeakTS] / (mTrianglePeakTS - iTS))
            BestSlope = Charge[mTrianglePeakTS] / (mTrianglePeakTS - iTS);
      }

      // calculate minimum chi2
      double Chi2 = 0;
      for(int i = 0; i < iTS; i++)
         Chi2 += Charge[i] * Charge[i];
      for(int i = iTS; i < (int)mTrianglePeakTS; i++)
         Chi2 += (Charge[mTrianglePeakTS] - Charge[i] + (i - mTrianglePeakTS) * BestSlope)
            * (Charge[mTrianglePeakTS] - Charge[i] + (i - mTrianglePeakTS) * BestSlope);

      if(MinimumLeftChi2 > Chi2)
      {
         MinimumLeftChi2 = Chi2;
         result.LeftSlope = BestSlope;
      }
   }   // end of left-hand side loop

   result.Chi2 = MinimumLeftChi2 + MinimumRightChi2;

   return result;
}

void HcalNoiseMonitor::ReadHcalPulse()
{
   std::vector<double> PulseShape;

   HcalPulseShapes Shapes;
   HcalPulseShapes::Shape HPDShape = Shapes.hbShape();

   PulseShape.reserve(350);
   for(int i = 0; i < 200; i++)
      PulseShape.push_back(HPDShape.at(i));
   PulseShape.insert(PulseShape.begin(), 150, 0);   // Safety margin of a lot of zeros in the beginning

   CumulativeIdealPulse.reserve(350);
   CumulativeIdealPulse.clear();
   CumulativeIdealPulse.push_back(0);
   for(unsigned int i = 1; i < PulseShape.size(); i++)
      CumulativeIdealPulse.push_back(CumulativeIdealPulse[i-1] + PulseShape[i]);
}

DEFINE_FWK_MODULE(HcalNoiseMonitor);

