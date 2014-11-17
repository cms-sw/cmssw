#ifndef DQM_HCALMONITORTASKS_HCALNoiseMONITOR_H
#define DQM_HCALMONITORTASKS_HCALNoiseMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"  // need for emap
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"

class HcalNoiseMonitor;
struct TriangleFitResult;

class HcalNoiseMonitor: public HcalBaseDQMonitor
{
public:
   HcalNoiseMonitor(const edm::ParameterSet& ps);
   ~HcalNoiseMonitor();


   void setup(DQMStore::IBooker &);
   void bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c);
   void analyze(edm::Event const&e, edm::EventSetup const&s);

   void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
   void reset();

private:
   std::vector<std::string> triggers_;
   int period_;
   bool setupDone_;
  
private:
   int mTrianglePeakTS;
   double mE2E10MinEnergy;
   double mMinE2E10;
   double mMaxE2E10;
   int mMaxHPDHitCount;
   int mMaxHPDNoOtherHitCount;
   int mMaxADCZeros;
   double mTotalZeroMinEnergy;

private:
   // Monitoring elements
   edm::InputTag rawdataLabel_;  // no data access associated - 
   edm::InputTag hltresultsLabel_; // no data access associated

   edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
   edm::EDGetTokenT<HBHERecHitCollection> tok_hbherec_;
   edm::EDGetTokenT<reco::HcalNoiseRBXCollection> tok_noise_;

   // Double-chi2 related stuff
   MonitorElement *hNominalChi2;
   MonitorElement *hLinearChi2;
   MonitorElement *hLinearTestStatistics;
   MonitorElement *hRMS8OverMax;
   MonitorElement *hRMS8OverMaxTestStatistics;
  
   MonitorElement *hLambdaLinearVsTotalCharge;
   MonitorElement *hLambdaRMS8MaxVsTotalCharge;
   MonitorElement *hTriangleLeftSlopeVsTS4;
   MonitorElement *hTriangleRightSlopeVsTS4;

   EtaPhiHists hFailLinearEtaPhi;
   EtaPhiHists hFailRMSMaxEtaPhi;
   EtaPhiHists hFailTriangleEtaPhi;
   
   // Isolation Filter
   EtaPhiHists hFailIsolationEtaPhi;
   
   // Jason variable
   MonitorElement *hTS4TS5RelativeDifference;
   MonitorElement *hTS4TS5RelativeDifferenceVsCharge;

   // Hcal noise summary object variable
   MonitorElement *hMaxZeros;
   MonitorElement *hTotalZeros;
   MonitorElement *hE2OverE10Digi;
   MonitorElement *hE2OverE10Digi5;
   MonitorElement *hE2OverE10RBX;
   MonitorElement *hHPDHitCount;
   MonitorElement *hRBXHitCount;
   MonitorElement *hHcalNoiseCategory;

   MonitorElement *hBadZeroRBX;
   MonitorElement *hBadCountHPD;
   MonitorElement *hBadNoOtherCountHPD;
   MonitorElement *hBadE2E10RBX;

   std::vector<double> CumulativeIdealPulse;

private:
   double PerformNominalFit(double Charge[10]);
   double PerformDualNominalFit(double Charge[10]);
   double DualNominalFitSingleTry(double Charge[10], int Offset, int Distance);
   double PerformLinearFit(double Charge[10]);
   double CalculateRMS8Max(double Charge[10]);
   TriangleFitResult PerformTriangleFit(double Charge[10]);
   void ReadHcalPulse();
};

struct TriangleFitResult
{
   double Chi2;
   double LeftSlope;
   double RightSlope;
};



#endif

