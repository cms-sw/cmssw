#ifndef DQM_HCALMONITORTASKS_HCALRAWDATAMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRAWDATAMONITOR_H

#define  NUMDCCS      32
#define  NUMSPIGS     15
#define  HTRCHANMAX   24
//Dimensions of 'LED' plots, grouping bits by hardware space
//  NUMBER_HDWARE = 1 + ((NUMBER +1)*(NUM_HRDWARE_PIECES))
#define  TWO___FED   (1+((2+1)*NUMDCCS)   )
#define  THREE_FED   (1+((3+1)*NUMDCCS)   )
#define  TWO__SPGT   (1+((2+1)*NUMSPIGS)  )
#define  THREE_SPG   (1+((3+1)*NUMSPIGS)  ) 
#define  TWO_CHANN   (1+((2+1)*HTRCHANMAX))

#define ETABINS   85
#define PHIBINS   72
#define DEPTHBINS  4

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include <math.h>

/** \class HcalRawDataMonitor
 *
 * $Date: 2012/06/21 13:40:22 $
 * $Revision: 1.6 $
 * \author J. St. John - Boston University
 */
class HcalRawDataMonitor: public HcalBaseDQMonitor {
 public:
  HcalRawDataMonitor(const edm::ParameterSet& ps);
  //Constructor with no arguments
  HcalRawDataMonitor(){};
  ~HcalRawDataMonitor();
 protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                            const edm::EventSetup& c) ;
  // End LumiBlock 
  // Dump the backstage arrays into MEs, for normalization by the Client
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                          const edm::EventSetup& c);
  // Within Analyze(), processEvent
  void processEvent(const FEDRawDataCollection& rawraw, 
		    const HcalUnpackerReport& report);
  // Within processEvent, unpack(fed)
  void unpack(const FEDRawData& raw);
  void endJob(void);
  void endRun(const edm::Run& run, const edm::EventSetup& c);
  void setup(void);
  void reset(void);

  edm::InputTag FEDRawDataCollection_;
  edm::InputTag digiLabel_;
  const HcalElectronicsMap*    readoutMap_;
  //Electronics map -> geographic channel map
  inline int hashup(uint32_t d=0, uint32_t s=0, uint32_t c=1) {
    return (int) ( (d*NUMSPIGS*HTRCHANMAX)+(s*HTRCHANMAX)+(c)); }
  void stashHDI(int thehash, HcalDetId thehcaldetid);
  //Protect against indexing past array.
  inline HcalDetId HashToHDI(int thehash) {
    return ( ( (thehash<0) || (thehash>(NUMDCCS*NUMSPIGS*HTRCHANMAX)) )
	     ?(HcalDetId::Undefined)
	     :(hashedHcalDetId_[thehash]));
  };
  uint64_t uniqcounter[ETABINS][PHIBINS][DEPTHBINS]; // HFd1,2 at 'depths' 3,4 to avoid collision with HE
  uint64_t problemcount[ETABINS][PHIBINS][DEPTHBINS]; // HFd1,2 at 'depths' 3,4 to avoid collision with HE
  bool     problemfound[ETABINS][PHIBINS][DEPTHBINS]; // HFd1,2 at 'depths' 3,4 to avoid collision with HE
  void mapDCCproblem  (int dcc);                          // Set problemfound[][][] = true for the hardware's ieta/iphi/depth's
  void mapHTRproblem  (int dcc, int spigot);              // Set problemfound[][][] = true for the hardware's ieta/iphi/depth's
  void mapChannproblem(int dcc, int spigot, int htrchan); // Set problemfound[][][] = true for the hardware's ieta/iphi/depth 
  void whosebad(int subdet);        //Increment the NumBad counter for this LS, for this Hcal subdet

 private:
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
  MonitorElement* meChann_DataIntegrityCheck_[NUMDCCS];

  uint64_t UScount[NUMDCCS][NUMSPIGS];
  float HalfHTRDataCorruptionIndicators_  [THREE_FED][THREE_SPG];  
  float LRBDataCorruptionIndicators_      [THREE_FED][THREE_SPG];  
  float ChannSumm_DataIntegrityCheck_     [TWO___FED][TWO__SPGT];
  float Chann_DataIntegrityCheck_[NUMDCCS][TWO_CHANN][TWO__SPGT];
  float DataFlowInd_                      [TWO___FED][THREE_SPG];

  MonitorElement* meHalfHTRDataCorruptionIndicators_;
  MonitorElement* meLRBDataCorruptionIndicators_;
  MonitorElement* meChannSumm_DataIntegrityCheck_;
  MonitorElement* meDataFlowInd_;

  //Histogram labelling functions
  void label_ySpigots(MonitorElement* me_ptr,int ybins);
  void label_xFEDs   (MonitorElement* me_ptr,int xbins);
  void label_xChanns (MonitorElement* me_ptr,int xbins);

  HcalDetId hashedHcalDetId_[NUMDCCS * NUMSPIGS * HTRCHANMAX];

  // Transfer internal problem counts to ME's, & reset internal counters.
  void UpdateMEs (void );

  //Member variables for reference values to be used in consistency checks.
  std::map<int, short> CDFversionNumber_list;
  std::map<int, short>::iterator CDFvers_it;
  std::map<int, short> CDFReservedBits_list;
  std::map<int, short>::iterator CDFReservedBits_it;
  std::map<int, short> DCCEvtFormat_list;
  std::map<int, short>::iterator DCCEvtFormat_it;

  // The following MEs map specific conditons from the EventFragment headers as specified in
  //   http://cmsdoc.cern.ch/cms/HCAL/document/CountingHouse/DCC/DCC_1Jul06.pdf
  MonitorElement* meCDFErrorFound_;       //Summary histo of Common Data Format violations by FED ID
  MonitorElement* meDCCEventFormatError_; //Summary histo of DCC Event Format violations by FED ID 

  //Check that evt numbers are synchronized across all half-HTRs and their DCC
  MonitorElement* meBCN_;            // Bunch count number distributions
  MonitorElement* medccBCN_;         // Bunch count number distributions
  MonitorElement* meBCNCheck_;       // HTR BCN compared to DCC BCN
  MonitorElement* meBCNSynch_;       // htr-htr disagreement location

  MonitorElement* meEvtNCheck_;      // HTR Evt # compared to DCC Evt #
  MonitorElement* meEvtNumberSynch_; // htr-htr disagreement location

  MonitorElement* meOrNCheck_;       // htr OrN compared to dcc OrN
  MonitorElement* meOrNSynch_;       // htr-htr disagreement location
  MonitorElement* meBCNwhenOrNDiff_; // BCN distribution (subset)

  MonitorElement* mefedEntries_;
  MonitorElement* meFEDRawDataSizes_;
  MonitorElement* meEvFragSize_;
  MonitorElement* meEvFragSize2_;
  MonitorElement* meDCCVersion_;

  void labelHTRBits(MonitorElement* mePlot,unsigned int axisType);
  MonitorElement* HTR_StatusWd_HBHE;
  MonitorElement* HTR_StatusWd_HF;
  MonitorElement* HTR_StatusWd_HO;
  MonitorElement* meStatusWdCrate_;  //HTR status bits by crate
  MonitorElement* meInvHTRData_;
  MonitorElement* meFibBCN_;

  // The following MEs map specific conditons from the HTR/DCC headers as specified in
  //   http://cmsdoc.cern.ch/cms/HCAL/document/CountingHouse/HTR/design/Rev4MainFPGA.pdf
  MonitorElement* meCrate0HTRStatus_ ;   //Map of HTR status bits into Crate 0
  MonitorElement* meCrate1HTRStatus_ ;   //Map of HTR status bits into Crate 1
  MonitorElement* meCrate2HTRStatus_ ;   //Map of HTR status bits into Crate 2
  MonitorElement* meCrate3HTRStatus_ ;   //Map of HTR status bits into Crate 3
  MonitorElement* meCrate4HTRStatus_ ;   //Map of HTR status bits into Crate 4
  MonitorElement* meCrate5HTRStatus_ ;   //Map of HTR status bits into Crate 5
  MonitorElement* meCrate6HTRStatus_ ;   //Map of HTR status bits into Crate 6
  MonitorElement* meCrate7HTRStatus_ ;   //Map of HTR status bits into Crate 7
  MonitorElement* meCrate9HTRStatus_ ;   //Map of HTR status bits into Crate 9
  MonitorElement* meCrate10HTRStatus_;   //Map of HTR status bits into Crate 10
  MonitorElement* meCrate11HTRStatus_;   //Map of HTR status bits into Crate 11
  MonitorElement* meCrate12HTRStatus_;   //Map of HTR status bits into Crate 12
  MonitorElement* meCrate13HTRStatus_;   //Map of HTR status bits into Crate 13
  MonitorElement* meCrate14HTRStatus_;   //Map of HTR status bits into Crate 14
  MonitorElement* meCrate15HTRStatus_;   //Map of HTR status bits into Crate 15
  MonitorElement* meCrate17HTRStatus_;   //Map of HTR status bits into Crate 17

  MonitorElement* meUSFractSpigs_;
  MonitorElement* meHTRFWVersion_;
  MonitorElement* meFib1OrbMsgBCN_;  //BCN of Fiber 1 Orb Msg
  MonitorElement* meFib2OrbMsgBCN_;  //BCN of Fiber 2 Orb Msg
  MonitorElement* meFib3OrbMsgBCN_;  //BCN of Fiber 3 Orb Msg
  MonitorElement* meFib4OrbMsgBCN_;  //BCN of Fiber 4 Orb Msg
  MonitorElement* meFib5OrbMsgBCN_;  //BCN of Fiber 5 Orb Msg
  MonitorElement* meFib6OrbMsgBCN_;  //BCN of Fiber 6 Orb Msg
  MonitorElement* meFib7OrbMsgBCN_;  //BCN of Fiber 7 Orb Msg
  MonitorElement* meFib8OrbMsgBCN_;  //BCN of Fiber 8 Orb Msg

  int NumBadHB, NumBadHE, NumBadHO, NumBadHF, NumBadHFLUMI, NumBadHO0, NumBadHO12;

  void HTRPrint(const HcalHTRData& htr,int prtlvl);

  bool excludeHORing2_;
};

#endif
