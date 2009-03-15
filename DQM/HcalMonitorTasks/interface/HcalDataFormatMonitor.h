#ifndef DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDATAFORMATMONITOR_H
#define DEPTHBINS      4

#define  IETAMIN     -43
#define  IETAMAX      43
#define  IPHIMIN       0
#define  IPHIMAX      71
#define  HBHE_LO_DCC 700
#define  HBHE_HI_DCC 717
#define  HF_LO_DCC   718
#define  HF_HI_DCC   724
#define  HO_LO_DCC   725
#define  HO_HI_DCC   731
#define  NUMDCCS      32
#define  HTRCHANMAX   24
//The four Data Integrity Plots & Arrays
#define  RCDIX        55
#define  RCDIY        22
#define  HHDIX        97
#define  HHDIY        61
#define  CSDIX        97
// CSDIY == HHDIY.
#define  CIX          73
#define  CIY          46

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <math.h>

/** \class Hcaldataformatmonitor
 *
 * $Date: 2008/12/08 15:19:55 $
 * $Revision: 1.39 $
 * \author W. Fisher - FNAL
 */
class HcalDataFormatMonitor: public HcalBaseMonitor {
 public:
  HcalDataFormatMonitor();
  ~HcalDataFormatMonitor();
  
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  //  void setup(const edm::ParameterSet& ps, DQMStore* dbe,const HcalElectronicsMap& emap );
  void processEvent(const FEDRawDataCollection& rawraw, const
		    HcalUnpackerReport& report, const HcalElectronicsMap& emap);
  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void clearME();
  void reset();

  void HTRPrint(const HcalHTRData& htr,int prtlvl);
  void labelHTRBits(MonitorElement* mePlot,unsigned int axisType);
  void labelthezoo (MonitorElement* zoo);

 public: //Electronics map -> geographic channel map
  void smuggleMaps(std::map<uint32_t, std::vector<HcalDetId> >& givenDCCtoCell,
  		   std::map<pair <int,int> , std::vector<HcalDetId> >& givenHTRtoCell);
  std::map<uint32_t, std::vector<HcalDetId> > DCCtoCell;
  std::map<uint32_t, std::vector<HcalDetId> > ::iterator thisDCC;
  std::map<pair <int,int> , std::vector<HcalDetId> > HTRtoCell;
  std::map<pair <int,int> , std::vector<HcalDetId> > ::iterator thisHTR;

 private: 
  //backstage accounting mechanisms for the ProblemMap
  bool problemfound[ETABINS][PHIBINS][DEPTHBINS];     // HFd1,2 at 'depths' 3,4 to avoid collision with HE
  uint64_t problemcount[ETABINS][PHIBINS][DEPTHBINS]; // HFd1,2 at 'depths' 3,4 to avoid collision with HE
  void mapHTRproblem (int dcc, int spigot) ;    // Increment problem counters for affected cells
  void mapDCCproblem(int dcc) ;                 // Increment problem counters for affected cells

  void fillzoos(int bin, int dccid);

  // Data accessors
  vector<int> fedUnpackList_;
  vector<int> dccCrate_;
  vector<HcalSubdetector> dccSubdet_;
  int firstFED_;
  int ievt_;
  int lastEvtN_;
  int lastBCN_;
  //   int dccnum_;
  //int cratenum_;

  int prtlvl_;
  int dfmon_checkNevents;

 private:  //Monitoring elements
   
  MonitorElement* meEVT_;
  MonitorElement* HWProblems_;
  std::vector<MonitorElement*> HWProblemsByDepth_;

  MonitorElement* DATAFORMAT_PROBLEM_MAP;
  MonitorElement* DATAFORMAT_PROBLEM_ZOO;
  MonitorElement* HB_DATAFORMAT_PROBLEM_MAP;
  MonitorElement* HBHE_DATAFORMAT_PROBLEM_ZOO;
  MonitorElement* HE_DATAFORMAT_PROBLEM_MAP;
  MonitorElement* HF_DATAFORMAT_PROBLEM_MAP;
  MonitorElement* HF_DATAFORMAT_PROBLEM_ZOO;
  MonitorElement* HO_DATAFORMAT_PROBLEM_MAP;
  MonitorElement* HO_DATAFORMAT_PROBLEM_ZOO;
   
  //MEs for hcalunpacker report info
  MonitorElement* meSpigotFormatErrors_;
  MonitorElement* meBadQualityDigis_;
  MonitorElement* meUnmappedDigis_;
  MonitorElement* meUnmappedTPDigis_;
  MonitorElement* meFEDerrorMap_;

  MonitorElement* meFEDRawDataSizes_;
  MonitorElement* meUSFractSpigs_;
  MonitorElement* meUSEvtSizes2D_;// implement me!
  MonitorElement* meUSEvtSizes1D_;// implement me!
  
  MonitorElement* me_HBHE_ZS_SlidingSum;
  MonitorElement* me_HF_ZS_SlidingSum;
  MonitorElement* me_HO_ZS_SlidingSum;
  MonitorElement* me_HBHE_ZS_SlidingSum_US;
  MonitorElement* me_HF_ZS_SlidingSum_US;
  MonitorElement* me_HO_ZS_SlidingSum_US;

  MonitorElement* fedEntries_;
  MonitorElement* fedFatal_;

  //Check that evt numbers are synchronized across all HTRs
  MonitorElement* meEvtNumberSynch_;
  MonitorElement* meBCNSynch_;
  MonitorElement* meBCN_;
  MonitorElement* medccBCN_;

  MonitorElement* meDCC_DataIntegrityCheck_;
  MonitorElement* meHalfHTR_DataIntegrityCheck_;
  MonitorElement* meChannSumm_DataIntegrityCheck_;
  float DCC_DataIntegrityCheck_      [RCDIX][RCDIY];	  
  float HalfHTR_DataIntegrityCheck_  [HHDIX][HHDIY];  
  float ChannSumm_DataIntegrityCheck_[CSDIX][HHDIY];
  float Chann_DataIntegrityCheck_    [NUMDCCS][CIX][CIY];
  void UpdateMEs ();  //Prescalable copy into MonitorElements

  //Histogram labelling functions
  void label_ySpigots(MonitorElement* me_ptr,int ybins);
  void label_xFEDs   (MonitorElement* me_ptr,int xbins);
  void label_xChanns (MonitorElement* me_ptr,int xbins);

  MonitorElement* meInvHTRData_;
  MonitorElement* meBCNCheck_; // htr BCN compared to dcc BCN
  MonitorElement* meEvtNCheck_; // htr Evt # compared to dcc Evt #
  MonitorElement* meFibBCN_;

  MonitorElement* meFWVersion_;
  MonitorElement* meEvFragSize_;
  MonitorElement* meEvFragSize2_;

  MonitorElement* meErrWdCrate_;  //HTR error bits by crate

  // The following MEs map specific conditons from the EventFragment headers as specified in
  //   http://cmsdoc.cern.ch/cms/HCAL/document/CountingHouse/DCC/DCC_1Jul06.pdf

  MonitorElement* meFEDId_;               //All of HCAL, as a stupidcheck.
  MonitorElement* meCDFErrorFound_;       //Summary histo of Common Data Format violations by FED ID
  MonitorElement* meDCCEventFormatError_; //Summary histo of DCC Event Format violations by FED ID 
  //Summary histo for HTR Status bits, DCC Error&Warn Counters Flagged Nonzero
  MonitorElement* meDCCErrorAndWarnConditions_;  
  MonitorElement* meDCCStatusFlags_;
  MonitorElement* meDCCSummariesOfHTRs_;  //Summary histo of HTR Summaries from DCC

  // The following MEs map specific conditons from the HTR/DCC headers as specified in
  //   http://cmsdoc.cern.ch/cms/HCAL/document/CountingHouse/HTR/design/Rev4MainFPGA.pdf

  MonitorElement* meCrate0HTRErr_;   //Map of HTR errors into Crate 0
  MonitorElement* meCrate1HTRErr_;   //Map of HTR errors into Crate 1
  MonitorElement* meCrate2HTRErr_;   //Map of HTR errors into Crate 2
  MonitorElement* meCrate3HTRErr_;   //Map of HTR errors into Crate 3
  MonitorElement* meCrate4HTRErr_;   //Map of HTR errors into Crate 4
  MonitorElement* meCrate5HTRErr_;   //Map of HTR errors into Crate 5
  MonitorElement* meCrate6HTRErr_;   //Map of HTR errors into Crate 6
  MonitorElement* meCrate7HTRErr_;   //Map of HTR errors into Crate 7
  MonitorElement* meCrate8HTRErr_;   //Map of HTR errors into Crate 8
  MonitorElement* meCrate9HTRErr_;   //Map of HTR errors into Crate 9
  MonitorElement* meCrate10HTRErr_;   //Map of HTR errors into Crate 10
  MonitorElement* meCrate11HTRErr_;   //Map of HTR errors into Crate 11
  MonitorElement* meCrate12HTRErr_;   //Map of HTR errors into Crate 12
  MonitorElement* meCrate13HTRErr_;   //Map of HTR errors into Crate 13
  MonitorElement* meCrate14HTRErr_;   //Map of HTR errors into Crate 14
  MonitorElement* meCrate15HTRErr_;   //Map of HTR errors into Crate 15
  MonitorElement* meCrate16HTRErr_;   //Map of HTR errors into Crate 16
  MonitorElement* meCrate17HTRErr_;   //Map of HTR errors into Crate 17

  MonitorElement* meCh_DataIntegrityFED00_;   //DataIntegrity for channels in FED 00
  MonitorElement* meCh_DataIntegrityFED01_;   //DataIntegrity for channels in FED 01
  MonitorElement* meCh_DataIntegrityFED02_;   //DataIntegrity for channels in FED 02
  MonitorElement* meCh_DataIntegrityFED03_;   //DataIntegrity for channels in FED 03
  MonitorElement* meCh_DataIntegrityFED04_;   //DataIntegrity for channels in FED 04
  MonitorElement* meCh_DataIntegrityFED05_;   //DataIntegrity for channels in FED 05
  MonitorElement* meCh_DataIntegrityFED06_;   //DataIntegrity for channels in FED 06
  MonitorElement* meCh_DataIntegrityFED07_;   //DataIntegrity for channels in FED 07
  MonitorElement* meCh_DataIntegrityFED08_;   //DataIntegrity for channels in FED 08
  MonitorElement* meCh_DataIntegrityFED09_;   //DataIntegrity for channels in FED 09
  MonitorElement* meCh_DataIntegrityFED10_;   //DataIntegrity for channels in FED 10
  MonitorElement* meCh_DataIntegrityFED11_;   //DataIntegrity for channels in FED 11
  MonitorElement* meCh_DataIntegrityFED12_;   //DataIntegrity for channels in FED 12
  MonitorElement* meCh_DataIntegrityFED13_;   //DataIntegrity for channels in FED 13
  MonitorElement* meCh_DataIntegrityFED14_;   //DataIntegrity for channels in FED 14
  MonitorElement* meCh_DataIntegrityFED15_;   //DataIntegrity for channels in FED 15
  MonitorElement* meCh_DataIntegrityFED16_;   //DataIntegrity for channels in FED 16
  MonitorElement* meCh_DataIntegrityFED17_;   //DataIntegrity for channels in FED 17
  MonitorElement* meCh_DataIntegrityFED18_;   //DataIntegrity for channels in FED 18
  MonitorElement* meCh_DataIntegrityFED19_;   //DataIntegrity for channels in FED 19
  MonitorElement* meCh_DataIntegrityFED20_;   //DataIntegrity for channels in FED 20
  MonitorElement* meCh_DataIntegrityFED21_;   //DataIntegrity for channels in FED 21
  MonitorElement* meCh_DataIntegrityFED22_;   //DataIntegrity for channels in FED 22
  MonitorElement* meCh_DataIntegrityFED23_;   //DataIntegrity for channels in FED 23
  MonitorElement* meCh_DataIntegrityFED24_;   //DataIntegrity for channels in FED 24
  MonitorElement* meCh_DataIntegrityFED25_;   //DataIntegrity for channels in FED 25
  MonitorElement* meCh_DataIntegrityFED26_;   //DataIntegrity for channels in FED 26
  MonitorElement* meCh_DataIntegrityFED27_;   //DataIntegrity for channels in FED 27
  MonitorElement* meCh_DataIntegrityFED28_;   //DataIntegrity for channels in FED 28
  MonitorElement* meCh_DataIntegrityFED29_;   //DataIntegrity for channels in FED 29
  MonitorElement* meCh_DataIntegrityFED30_;   //DataIntegrity for channels in FED 30
  MonitorElement* meCh_DataIntegrityFED31_;   //DataIntegrity for channels in FED 31
  // handy array of pointers to pointers...
  MonitorElement* meChann_DataIntegrityCheck_[32];

  MonitorElement* meFib1OrbMsgBCN_;  //BCN of Fiber 1 Orb Msg
  MonitorElement* meFib2OrbMsgBCN_;  //BCN of Fiber 2 Orb Msg
  MonitorElement* meFib3OrbMsgBCN_;  //BCN of Fiber 3 Orb Msg
  MonitorElement* meFib4OrbMsgBCN_;  //BCN of Fiber 4 Orb Msg
  MonitorElement* meFib5OrbMsgBCN_;  //BCN of Fiber 5 Orb Msg
  MonitorElement* meFib6OrbMsgBCN_;  //BCN of Fiber 6 Orb Msg
  MonitorElement* meFib7OrbMsgBCN_;  //BCN of Fiber 7 Orb Msg
  MonitorElement* meFib8OrbMsgBCN_;  //BCN of Fiber 8 Orb Msg

  MonitorElement* DCC_ErrWd_HBHE;
  MonitorElement* DCC_ErrWd_HF;
  MonitorElement* DCC_ErrWd_HO;

  int currFiberChan;
  void LabelChannInteg(MonitorElement* me_ptr);
  bool isUnsuppressed (HcalHTRData& payload); //Return the US bit: ExtHdr7[bit 15]
  uint64_t UScount[32][15];

  //Member variables for reference values to be used in consistency checks.
  std::map<int, short> CDFversionNumber_list;
  std::map<int, short>::iterator CDFvers_it;
  std::map<int, short> CDFEventType_list;
  std::map<int, short>::iterator CDFEvT_it;
  std::map<int, short> CDFReservedBits_list;
  std::map<int, short>::iterator CDFReservedBits_it;
  std::map<int, short> DCCEvtFormat_list;
  std::map<int, short>::iterator DCCEvtFormat_it;
  std::map<int, short> DCCRsvdBits_list;
  std::map<int, short>::iterator DCCRsvdBits_it;
  
  //static member variables 
  static float DIMbin[32];
};

// For crate numbers:
float HcalDataFormatMonitor::DIMbin[]={ 4, 4.5, // FED 700, 701
		       0, 0.5, // FED 702, 703
		       1, 1.5, // FED 704, 705
		       5, 5.5, // FED 706, 707
		       11, 11.5, // FED 708, 709
		       15, 15.5, // FED 710, 711
		       17, 17.5, // FED 712, 713
		       14, 14.5, // FED 714, 715
		       10, 10.5, // FED 716, 717
		       2, 2.5, // FED 718, 719
		       9, 9.5, // FED 720, 721
		       12, 12.5, // FED 722, 723
		       3, 3.5, // FED 724, 725
		       7, 7.5, // FED 726, 727
		       6, 6.5, // FED 728, 729
		       13, 13.5 // FED 730, 731
};

#endif
