#ifndef DQM_HCALMONITORTASKS_HCALDIGIDQMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDIGIDQMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"

#define DIGI_BQ_FRAC_NBINS 101
#define DIGI_NUM 9072
#define DIGI_SUBDET_NUM 2593 

#define ETABINS   85
#define PHIBINS   72
#define DEPTHBINS  4

/** \class HcalDigiMonitor
  *  
  * $Date: 2012/06/27 13:20:29 $
  * $Revision: 1.71 $
  * \author J. Temple - Univ. of Maryland
  */

struct DigiHists
{
  // structure of MonitorElements for each subdetector, along with their associated counters

  MonitorElement* shape;
  MonitorElement* shapeThresh;
  MonitorElement* ThreshCount;
  MonitorElement* presample;
  MonitorElement* BQ;
  MonitorElement* BQFrac;
  MonitorElement* DigiFirstCapID;
  MonitorElement* DVerr;
  MonitorElement* CapID;
  MonitorElement* ADC;
  MonitorElement* ADCsum;
  MonitorElement* fibBCNOff;
  std::vector<MonitorElement*> TS_sum_plus, TS_sum_minus;

  int count_shape[10];
  double count_shapeThresh[10];
  int count_presample[50];
  int count_bad;
  int count_good;
  int count_BQ[DIGI_SUBDET_NUM];
  int count_BQFrac[DIGI_BQ_FRAC_NBINS];

  int capIDdiff[8]; // only bins 0-7 used for expected real values of cap ID difference (since cap IDs run from 0-3); bin 8 is overflow
  int dverr[4];
  int capid[4];
  int adc[200];
  int adcsum[200];
  int fibbcnoff[15];
  int tssumplus[50][10];
  int tssumminus[50][10];
};

class HcalDigiMonitor: public HcalBaseDQMonitor {
public:
  HcalDigiMonitor(const edm::ParameterSet& ps); 
  ~HcalDigiMonitor(); 

  void setup();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void cleanup();

  void analyze(const edm::Event& e, const edm::EventSetup& c);

  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond,
		    const HcalUnpackerReport& report,
		    int orN, int bcN);		

  // Begin LumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                            const edm::EventSetup& c) ;

  // End LumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                          const edm::EventSetup& c);


  void endRun(const edm::Run& run, const edm::EventSetup& c);
  void endJob();
  void reset();

private:  ///Methods, variables accessible only within class code
  bool passedMinBiasHLT_;
  bool excludeHORing2_;
  bool excludeHO1P02_;
  bool excludeBadQPLLs_;
 
  void fill_Nevents();
  void zeroCounters();
  void setupSubdetHists(DigiHists& hist,  std::string subdet); // enable this feature at some point

  template<class T> int process_Digi(T& digi, DigiHists& hist, int& firstcap);
  void UpdateHists(DigiHists& h);

  bool doFCpeds_;
  bool setupDone_;
  
  int HO0bad, HO12bad, HFlumibad;

  int shapeThresh_;
  int shapeThreshHB_;
  int shapeThreshHE_;
  int shapeThreshHF_;
  int shapeThreshHO_;

  int alarmer_counter_;
  int knownbadQPLLs;
  bool hbhedcsON, hfdcsON;

  int mindigisizeHBHE_, maxdigisizeHBHE_;
  int mindigisizeHO_, maxdigisizeHO_;
  int mindigisizeHF_, maxdigisizeHF_;

  bool digi_checkoccupancy_;
  bool digi_checkcapid_;
  bool digi_checkdigisize_;
  bool digi_checkadcsum_;
  bool digi_checkdverr_;

  bool shutOffOrbitTest_;

  int DigiMonitor_ExpectedOrbitMessageTime_;
  int hbcount_, hecount_, hocount_, hfcount_;  // Counter # of good digis each event
  uint64_t uniqcounter[ETABINS][PHIBINS][DEPTHBINS]; // HFd1,2 at 'depths' 3,4 to avoid collision with HE
  uint64_t uniqcounter2[ETABINS][PHIBINS][DEPTHBINS]; // HFd1,2 at 'depths' 3,4 to avoid collision with HE

  // Monitoring elements

  EtaPhiHists DigiErrorsByDepth;
  EtaPhiHists DigiErrorsBadCapID;
  EtaPhiHists DigiErrorsDVErr;
  EtaPhiHists DigiErrorsBadDigiSize;
  EtaPhiHists DigiErrorsBadADCSum;
  EtaPhiHists DigiErrorsUnpacker;
  EtaPhiHists DigiErrorsBadFibBCNOff;

  MonitorElement* DigiSize;
  MonitorElement* DigiExpectedSize;
  int baddigis[85][72][4]; // sum of individual digi problems
  int badcapID[85][72][4];
  int baddigisize[85][72][4];
  int badFibBCNOff[85][72][4];
  int badunpackerreport[85][72][4];
  int digisize[20][4];
  int digierrorsdverr[85][72][4];

  // Count events with valid digis
  MonitorElement* h_valid_digis;
  MonitorElement* h_invalid_orbitnumMod103;
  MonitorElement* h_invalid_bcn;

  // Digi Occupancy Plots
  EtaPhiHists DigiOccupancyByDepth;
  MonitorElement* DigiOccupancyEta;
  MonitorElement* DigiOccupancyPhi;
  MonitorElement* DigiOccupancyVME;
  MonitorElement* DigiOccupancySpigot;
  MonitorElement *ProblemDigisInLastNLB_HBHEHF_alarm;
  
  // Counters for good and bad digis
  int occupancyEtaPhi[85][72][4];
  int occupancyEta[85];
  int occupancyPhi[72];
  int occupancyVME[40][18];
  int occupancySpigot[40][36];

  // Plots for Digis that are present, but with errors
  EtaPhiHists DigiErrorOccupancyByDepth;
 
  MonitorElement* DigiErrorVME;
  MonitorElement* DigiErrorSpigot;
  
  MonitorElement* DigiBQ;
  MonitorElement* DigiBQFrac;
  MonitorElement* DigiUnpackerErrorCount;
  MonitorElement* DigiUnpackerErrorFrac;

  int occupancyErrorEtaPhi[85][72][4];
  int occupancyErrorEta[85];
  int occupancyErrorPhi[72];
  int errorVME[40][18];
  int errorSpigot[15][36];// 15 is the value of SPIGOT_COUNT; may need to change this in the future?

  MonitorElement* HBocc_vs_LB;
  MonitorElement* HEocc_vs_LB;
  MonitorElement* HOocc_vs_LB;
  MonitorElement* HFocc_vs_LB;

  MonitorElement* DigiNum;
  int diginum[DIGI_NUM];

  //  Pawel's HF timing study plots
  MonitorElement* HFtiming_totaltime2D;
  MonitorElement* HFtiming_occupancy2D;
  MonitorElement* HFtiming_etaProfile;
  MonitorElement* HFP_shape;
  MonitorElement* HFM_shape;

  DigiHists hbHists, heHists, hfHists, hoHists;

  edm::InputTag digiLabel_;
  edm::InputTag FEDRawDataCollection_;

  edm::ESHandle<HcalDbService> conditions_;

  edm::InputTag hltresultsLabel_;
  std::vector <std::string> MinBiasHLTBits_;

  edm::InputTag hfRechitLabel_; // used for calculating HF total ET
  double HT_HFP_, HT_HFM_;

  // Should be able to make this a vector of ints, right?
  std::map<HcalDetId, std::vector<double> > PedestalsByCapId_;

  double pedSubtractedADC_[10]; // stores ped-subtracted ADCs for each digi time slice;  use this instead of a vector because of memory allocation issues
};

float bins_cellcount_new[]={-0.5, 0.5, 1.5, 2.5, 3.5, 4.5,
			5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
			11.5, 12.5, 13.5, 14.5, 15.5, 
			16.5, 17.5, 18.5, 19.5, 20.5,
			21.5, 22.5, 23.5, 24.5, 25.5,
			26.5, 27.5, 28.5, 29.5, 30.5,
			31.5, 32.5, 33.5, 34.5, 35.5,
			36.5, 37.5, 38.5, 39.5, 40.5,
			41.5, 42.5, 43.5, 44.5, 45.5,
			46.5, 47.5, 48.5, 49.5, 50.5,
			60.5, 70.5, 80.5, 90.5, 100.5,
			150.5, 200.5, 250.5, 300.5,
			400.5, 500.5, 600.5, 700.5, 
			800.5, 900.5, 1000.5, 1100.5,
			1200.5, 1300.5, 1400.5, 1500.5, 
			1600.5, 1700.5, 1800.5, 1900.5,
			2000.5, 2100.5, 2200.5, 2300.5, 
			2400.5, 2500.5, 2600.5, 2700.5,
			2800.5, 2900.5, 3000.5, 3100.5, 
			3200.5, 3300.5, 3400.5, 3500.5,
			3600.5, 3700.5, 3800.5, 3900.5,
			4000.5, 4100.5, 4200.5, 4300.5,
			4400.5, 4500.5, 4600.5, 4700.5,
			4800.5, 4900.5, 5000.5, 5100.5,
			5200.5, 5300.5, 5400.5, 5500.5,
			5600.5, 5700.5, 5800.5, 5900.5,
			6000.5, 6100.5, 6200.5, 6300.5, 
			6400.5, 6500.5, 6600.5, 6700.5,
			6800.5, 6900.5, 7000.5, 7100.5,
			7200.5, 7300.5, 7400.5, 7500.5,
			7600.5, 7700.5, 7800.5, 7900.5, 
			8000.5, 8100.5, 8200.5, 8300.5,
			8400.5, 8500.5, 8600.5, 8700.5, 
			8800.5, 8900.5, 9000.5, 9100.5};

// binning is 1 part per 10k -- so one channel takes ~ 1 bin at low values
float bins_fraccount_new[]={-0.00005,  0.00005, 0.0001,  0.0002,  
			0.0003,  
			0.0004,  0.0005,  0.0006,  0.0007,  0.0008, 
			0.0009, 
			0.001,  0.0011,  0.0012,  0.0013,  0.0014, 
			0.0015,  0.0016,  0.0017,  0.0018,  0.0019, 
			0.002, 0.0021,  0.0022,  0.0023,  0.0024, 
			0.0025,  0.0026,  0.0027,  0.0028,  0.0029,  
			0.003,  0.0031, 0.0032,  0.0033,  0.0034, 
			0.0035,  0.0036,  0.0037,  0.0038,  0.0039,  
			0.004,  0.0041,  0.0042,  0.0043,  0.0044, 
			0.0045,  0.0046,  0.0047,  0.0048,  0.0049,  
			0.005,  0.0051,  0.0052,  0.0053,  0.0054, 
			0.0055,  0.0056,  0.0057,  0.0058,  0.0059,  
			0.006,  0.0061,  0.0062,  0.0063,  0.0064, 
			0.0065,  0.0066,  0.0067,  0.0068,  0.0069,  
			0.007,  0.0071,  0.0072,  0.0073,  0.0074, 
			0.0075,  0.0076,  0.0077, 0.0078,  0.0079,  
			0.008,  0.0081,  0.0082,  0.0083,  0.0084, 
			0.0085,  0.0086,  0.0087,  0.0088,  0.0089,  
			0.009,  0.0091,  0.0092,  0.0093,  0.0094, 
			0.0095,  0.0096,  0.0097,  0.0098,  0.0099,  
			0.01,  0.0101,  0.0102,  0.0103,  0.0104,
			0.0105,  0.0106,  0.0107,  0.0108,  0.0109,  
			0.011,  0.0111,  0.0112,  0.0113,  0.0114, 
			0.0115,  0.0116,  0.0117,  0.0118,  0.0119,  
			0.012,  0.0121,  0.0122,  0.0123,  0.0124,
			0.0125,  0.0126,  0.0127,  0.0128,  0.0129,  
			0.013,  0.0131,  0.0132,  0.0133,  0.0134,
			0.0135,  0.0136,  0.0137,  0.0138,  0.0139,  
			0.014,  0.0141,  0.0142,  0.0143,  0.0144,
			0.0145,  0.0146,  0.0147,  0.0148,  0.0149,  
			0.015,  0.0151,  0.0152,  0.0153,  0.0154,
			0.0155,  0.0156,  0.0157,  0.0158,  0.0159,  
			0.016,  0.0161,  0.0162,  0.0163,  0.0164,
			0.0165,  0.0166, 0.0167,  0.0168,  0.0169,  
			0.017,  0.0171,  0.0172,  0.0173,  0.0174,
			0.0175,  0.0176,  0.0177,  0.0178,  0.0179,  
			0.018,  0.0181,  0.0182,  0.0183,  0.0184, 
			0.0185,  0.0186,  0.0187,  0.0188,  0.0189,  
			0.019,  0.0191,  0.0192,  0.0193,  0.0194,
			0.0195,  0.0196,  0.0197,  0.0198,  0.0199,  
			0.02,  0.0201,  0.0202,  0.0203,  0.0204, 
			0.0205,  0.0206,  0.0207,  0.0208,  0.0209,  
			0.021,  0.0211,  0.0212,  0.0213,  0.0214,
			0.0215,  0.0216,  0.0217,  0.0218,  0.0219,  
			0.022,  0.0221,  0.0222,  0.0223,  0.0224, 
			0.0225,  0.0226,  0.0227,  0.0228,  0.0229,  
			0.023,  0.0231,  0.0232,  0.0233,  0.0234,
			0.0235,  0.0236,  0.0237,  0.0238,  0.0239,  
			0.024,  0.0241,  0.0242,  0.0243,  0.0244,
			0.0245,  0.0246,  0.0247,  0.0248,  0.0249,  
			0.025,  0.0251,  0.0252,  0.0253,  0.0254,
			0.0255, 0.0256,  0.0257,  0.0258,  0.0259,  
			0.026,  0.0261,  0.0262,  0.0263,  0.0264,
			0.0265,  0.0266,  0.0267,  0.0268,  0.0269,  
			0.027,  0.0271,  0.0272,  0.0273,  0.0274,
			0.0275,  0.0276,  0.0277,  0.0278,  0.0279,  
			0.028,  0.0281,  0.0282,  0.0283,  0.0284,
			0.0285,  0.0286,  0.0287,  0.0288,  0.0289,  
			0.029,  0.0291,  0.0292,  0.0293,  0.0294, 
			0.0295,  0.0296,  0.0297,  0.0298,  0.0299,  
			0.03,  0.0301,  0.0302,  0.0303,  0.0304, 
			0.0305,  0.0306,  0.0307,  0.0308,  0.0309,  
			0.031,  0.0311,  0.0312,  0.0313,  0.0314, 
			0.0315,  0.0316,  0.0317,  0.0318,  0.0319,  
			0.032,  0.0321,  0.0322,  0.0323,  0.0324,
			0.0325,  0.0326,  0.0327,  0.0328,  0.0329,  
			0.033,  0.0331,  0.0332,  0.0333,  0.0334, 
			0.0335,  0.0336,  0.0337,  0.0338,  0.0339,  
			0.034,  0.0341,  0.0342,  0.0343,  0.0344,
			0.0345,  0.0346,  0.0347,  0.0348,  0.0349,  
			0.035,  0.0351,  0.0352,  0.0353,  0.0354,
			0.0355,  0.0356,  0.0357,  0.0358,  0.0359,  
			0.036,  0.0361,  0.0362,  0.0363,  0.0364, 
			0.0365,  0.0366,  0.0367,  0.0368,  0.0369,  
			0.037,  0.0371,  0.0372,  0.0373,  0.0374,
			0.0375,  0.0376,  0.0377,  0.0378,  0.0379,  
			0.038,  0.0381,  0.0382,  0.0383,  0.0384, 
			0.0385,  0.0386,  0.0387,  0.0388,  0.0389,  
			0.039,  0.0391,  0.0392,  0.0393,  0.0394, 
			0.0395,  0.0396,  0.0397,  0.0398,  0.0399,  
			0.04,  0.0401,  0.0402,  0.0403,  0.0404, 
			0.0405,  0.0406,  0.0407,  0.0408,  0.0409,  
			0.041,  0.0411,  0.0412,  0.0413,  0.0414, 
			0.0415,  0.0416,  0.0417,  0.0418,  0.0419,  
			0.042,  0.0421,  0.0422,  0.0423,  0.0424, 
			0.0425,  0.0426,  0.0427,  0.0428,  0.0429,  
			0.043,  0.0431,  0.0432,  0.0433, 0.0434, 
			0.0435,  0.0436,  0.0437,  0.0438,  0.0439,  
			0.044,  0.0441,  0.0442,  0.0443,  0.0444, 
			0.0445,  0.0446,  0.0447,  0.0448,  0.0449,  
			0.045,  0.0451,  0.0452,  0.0453,  0.0454,
			0.0455,  0.0456,  0.0457,  0.0458,  0.0459,  
			0.046,  0.0461,  0.0462,  0.0463,  0.0464,
			0.0465,  0.0466,  0.0467,  0.0468,  0.0469,  
			0.047,  0.0471,  0.0472,  0.0473,  0.0474, 
			0.0475,  0.0476,  0.0477,  0.0478,  0.0479,  
			0.048,  0.0481,  0.0482,  0.0483,  0.0484, 
			0.0485,  0.0486,  0.0487,  0.0488,  0.0489,  
			0.049,  0.0491,  0.0492,  0.0493,  0.0494,
			0.0495,  0.0496,  0.0497,  0.0498,  0.0499,
			0.05,  0.0501,  0.0502,  0.0503,  0.0504,
			0.0505,  0.0506,  0.0507,  0.0508,  0.0509,  
			0.051,  0.0511,  0.0512,  0.0513,  0.0514,
			0.0515,  0.0516,  0.0517,  0.0518,  0.0519,  
			0.052,  0.0521,  0.0522,  0.0523,  0.0524,
			0.0525,  0.0526,  0.0527,  0.0528,  0.0529,  
			0.053,  0.0531,  0.0532,  0.0533, 0.0534, 
			0.0535,  0.0536,  0.0537,  0.0538,  0.0539,  
			0.054,  0.0541,  0.0542,  0.0543,  0.0544,
			0.0545,  0.0546,  0.0547,  0.0548,  0.0549,  
			0.055,  0.0551,  0.0552,  0.0553,  0.0554, 
			0.0555,  0.0556,  0.0557,  0.0558,  0.0559,  
			0.056,  0.0561,  0.0562,  0.0563,  0.0564, 
			0.0565,  0.0566,  0.0567,  0.0568,  0.0569,  
			0.057,  0.0571,  0.0572,  0.0573,  0.0574,
			0.0575,  0.0576,  0.0577,  0.0578,  0.0579,  
			0.058,  0.0581,  0.0582,  0.0583,  0.0584, 
			0.0585,  0.0586,  0.0587,  0.0588,  0.0589,  
			0.059,  0.0591,  0.0592,  0.0593,  0.0594,
			0.0595,  0.0596,  0.0597,  0.0598,  0.0599,
			0.06,  0.0601,  0.0602,  0.0603,  0.0604,
			0.0605,  0.0606,  0.0607,  0.0608,  0.0609,  
			0.061,  0.0611,  0.0612,  0.0613,  0.0614, 
			0.0615,  0.0616,  0.0617,  0.0618,  0.0619,  
			0.062,  0.0621,  0.0622,  0.0623,  0.0624,
			0.0625,  0.0626,  0.0627,  0.0628,  0.0629,  
			0.063,  0.0631,  0.0632,  0.0633, 0.0634, 
			0.0635,  0.0636,  0.0637,  0.0638,  0.0639,  
			0.064,  0.0641,  0.0642,  0.0643,  0.0644, 
			0.0645,  0.0646,  0.0647,  0.0648,  0.0649,  
			0.065,  0.0651,  0.0652,  0.0653,  0.0654, 
			0.0655,  0.0656,  0.0657,  0.0658,  0.0659,  
			0.066,  0.0661,  0.0662,  0.0663,  0.0664,
			0.0665,  0.0666,  0.0667,  0.0668,  0.0669,  
			0.067,  0.0671,  0.0672,  0.0673,  0.0674,
			0.0675,  0.0676,  0.0677,  0.0678,  0.0679,  
			0.068,  0.0681,  0.0682,  0.0683,  0.0684, 
			0.0685,  0.0686,  0.0687,  0.0688,  0.0689,  
			0.069,  0.0691,  0.0692,  0.0693,  0.0694, 
			0.0695,  0.0696,  0.0697,  0.0698,  0.0699,
			0.07,  0.0701,  0.0702,  0.0703,  0.0704, 
			0.0705,  0.0706,  0.0707,  0.0708,  0.0709,  
			0.071,  0.0711,  0.0712,  0.0713,  0.0714,
			0.0715,  0.0716,  0.0717,  0.0718,  0.0719,  
			0.072,  0.0721,  0.0722,  0.0723,  0.0724, 
			0.0725,  0.0726,  0.0727,  0.0728,  0.0729,  
			0.073,  0.0731,  0.0732,  0.0733, 0.0734, 
			0.0735,  0.0736,  0.0737,  0.0738,  0.0739,  
			0.074,  0.0741,  0.0742,  0.0743,  0.0744, 
			0.0745,  0.0746,  0.0747,  0.0748,  0.0749,  
			0.075,  0.0751,  0.0752,  0.0753,  0.0754,
			0.0755,  0.0756,  0.0757,  0.0758,  0.0759,  
			0.076,  0.0761,  0.0762,  0.0763,  0.0764, 
			0.0765,  0.0766,  0.0767,  0.0768,  0.0769,  
			0.077,  0.0771,  0.0772,  0.0773,  0.0774, 
			0.0775,  0.0776,  0.0777,  0.0778,  0.0779,  
			0.078,  0.0781,  0.0782,  0.0783,  0.0784, 
			0.0785,  0.0786,  0.0787,  0.0788,  0.0789,  
			0.079,  0.0791,  0.0792,  0.0793,  0.0794, 
			0.0795,  0.0796,  0.0797,  0.0798,  0.0799,
			0.08,  0.0801,  0.0802,  0.0803,  0.0804, 
			0.0805,  0.0806,  0.0807,  0.0808,  0.0809,  
			0.081,  0.0811,  0.0812,  0.0813,  0.0814,
			0.0815,  0.0816,  0.0817,  0.0818,  0.0819,  
			0.082,  0.0821,  0.0822,  0.0823,  0.0824, 
			0.0825,  0.0826,  0.0827,  0.0828,  0.0829,  
			0.083,  0.0831,  0.0832,  0.0833, 0.0834,  
			0.0835,  0.0836,  0.0837,  0.0838,  0.0839,  
			0.084,  0.0841,  0.0842,  0.0843,  0.0844, 
			0.0845,  0.0846,  0.0847,  0.0848,  0.0849,  
			0.085,  0.0851,  0.0852,  0.0853,  0.0854, 
			0.0855,  0.0856,  0.0857,  0.0858,  0.0859,  
			0.086,  0.0861,  0.0862,  0.0863,  0.0864, 
			0.0865,  0.0866,  0.0867,  0.0868,  0.0869,  
			0.087,  0.0871,  0.0872,  0.0873,  0.0874,
			0.0875,  0.0876,  0.0877,  0.0878,  0.0879,  
			0.088,  0.0881,  0.0882,  0.0883,  0.0884, 
			0.0885,  0.0886,  0.0887,  0.0888,  0.0889,  
			0.089,  0.0891,  0.0892,  0.0893,  0.0894,
			0.0895,  0.0896,  0.0897,  0.0898,  0.0899,
			0.09,  0.0901,  0.0902,  0.0903,  0.0904, 
			0.0905,  0.0906,  0.0907,  0.0908,  0.0909,  
			0.091,  0.0911,  0.0912,  0.0913,  0.0914, 
			0.0915,  0.0916,  0.0917,  0.0918,  0.0919,  
			0.092,  0.0921,  0.0922,  0.0923,  0.0924,
			0.0925,  0.0926,  0.0927,  0.0928,  0.0929,  
			0.093,  0.0931,  0.0932,  0.0933, 0.0934, 
			0.0935,  0.0936,  0.0937,  0.0938,  0.0939,  
			0.094,  0.0941,  0.0942,  0.0943,  0.0944,
			0.0945,  0.0946,  0.0947,  0.0948,  0.0949,  
			0.095,  0.0951,  0.0952,  0.0953,  0.0954, 
			0.0955,  0.0956,  0.0957,  0.0958,  0.0959,  
			0.096,  0.0961,  0.0962,  0.0963,  0.0964, 
			0.0965,  0.0966,  0.0967,  0.0968,  0.0969,  
			0.097,  0.0971,  0.0972,  0.0973,  0.0974, 
			0.0975,  0.0976,  0.0977,  0.0978,  0.0979,  
			0.098,  0.0981,  0.0982,  0.0983,  0.0984, 
			0.0985,  0.0986,  0.0987,  0.0988,  0.0989,  
			0.099,  0.0991,  0.0992,  0.0993,  0.0994,  
			0.0995,  0.0996,  0.0997,  0.0998,  0.0999,
			//Above 10%, switch to 1% binning
			0.10, 0.11, 0.12, 0.13,0.14,
			0.15, 0.16, 0.17, 0.18, 0.19,
			// Above 20%, switch to 5% binning
			0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
			0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 
			// Make special bin for 100% value,
			0.9999, 1.01
};

#endif
