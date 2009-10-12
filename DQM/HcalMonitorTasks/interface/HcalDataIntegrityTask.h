#ifndef DQM_HCALMONITORTASKS_HCALDATAINTEGRITYTASK_H
#define DQM_HCALMONITORTASKS_HCALDATAINTEGRITYTASK_H

#define  IETAMIN -43
#define  IETAMAX 43
#define  IPHIMIN 0
#define  IPHIMAX 71
#define  HBHE_LO_DCC 700
#define  HBHE_HI_DCC 717
#define  HF_LO_DCC   718
#define  HF_HI_DCC   724
#define  HO_LO_DCC   725
#define  HO_HI_DCC   731

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/** \class HcalDataIntegrityTask
 *
 * $Date: 2008/11/04 18:57:37 $
 * $Revision: 1.2 $
 * \author J. Temple -- University of Maryland
 * copied from W. Fisher/J. St. John's DataFormat code
 */

class HcalDataIntegrityTask: public HcalBaseMonitor 
{
 public:
  HcalDataIntegrityTask();
  ~HcalDataIntegrityTask();
  
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const FEDRawDataCollection& rawraw, const
		    HcalUnpackerReport& report, const HcalElectronicsMap& emap);
  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void clearME();
  void reset();

 public: //Electronics map -> geographic channel map
  std::map<uint32_t, std::vector<HcalDetId> > DCCtoCell;
  std::map<uint32_t, std::vector<HcalDetId> > ::iterator thisDCC;
  std::map<pair <int,int> , std::vector<HcalDetId> > HTRtoCell;
  std::map<pair <int,int> , std::vector<HcalDetId> > ::iterator thisHTR;

 private: 
  //backstage accounting mechanisms for the ProblemMap
  static size_t iphirange; // = IPHIMAX - IPHIMIN;
  static size_t ietarange; // = IETAMAX - IETAMIN;
  std::vector<std::vector<bool> > problemhere;  // Whole HCAL
  std::vector<std::vector<bool> > problemHB;    //  
  std::vector<std::vector<bool> > problemHE;    //  
  std::vector<std::vector<bool> > problemHF;    // Includes ZDC?
  std::vector<std::vector<bool> > problemHO;    //  
  void mapHTRproblem (int dcc, int spigot) ;
  void mapDCCproblem(int dcc) ;
  void fillzoos(int bin, int dccid);
  std::vector<std::vector<uint64_t> > phatmap;  // iphi/ieta projection of all hcal cells
  std::vector<std::vector<uint64_t> > HBmap;    // iphi/ieta projection of hb
  std::vector<std::vector<uint64_t> > HEmap;    // iphi/ieta projection of he
  std::vector<std::vector<uint64_t> > HFmap;    // iphi/ieta projection of hf
  std::vector<std::vector<uint64_t> > HOmap;    // iphi/ieta projection of ho
  void UpdateMap();

  // Data accessors
  vector<int> fedUnpackList_;
  vector<int> dccCrate_;
  vector<HcalSubdetector> dccSubdet_;
  int firstFED_;
  int lastEvtN_;
  int lastBCN_;
  //   int dccnum_;
  //int cratenum_;

  int prtlvl_;

 private:  //Monitoring elements

  MonitorElement* fedEntries_;
  MonitorElement* fedFatal_;
  MonitorElement* fedNonFatal_;

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
float HcalDataIntegrityTask::DIMbin[]={ 4, 4.5, // FED 700, 701
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
