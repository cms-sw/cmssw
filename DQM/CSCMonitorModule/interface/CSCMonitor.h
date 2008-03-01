#ifndef CSCMonitor_h
#define CSCMonitor_h


#include <iostream>
#include <string>
#include <signal.h>
#include <map>
#include <string>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdint.h>

/*
  #include "xdaq.h"
  #include "xdata.h"
  #include "toolbox.h"
*/


#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNodeList.hpp>

// == CMSSW Section
/*  actually calls  emuDQM/CMSSWLibs/FWCore/MessageLogger/interface/MessageLogger.h */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h"


/* Normal calls to CMSSW source tree */
#include "EventFilter/CSCRawToDigi/interface/CSCMonitorInterface.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDCCExaminer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCALCTTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCAnodeData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include "EventFilter/CSCRawToDigi/interface/CSCCLCTData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDDUTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBTrailer.h"
#include "EventFilter/CSCRawToDigi/interface/CSCEventData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBData.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBTrailer.h"
#include "EventFilter/CSCRawToDigi/src/bitset_append.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"



// ==  ROOT Section
#include <TROOT.h>
#include <TApplication.h>
#include <TSystem.h>
#include <TH1.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TString.h>
#include <TCanvas.h>

// == DDU Bin Examiner
// #include "dduBinExaminer.hh"


#define DEFAULT_IMAGE_FORMAT "png"
#define DEFAULT_CANVAS_WIDTH  800
#define DEFAULT_CANVAS_HEIGHT 600

#define UNPACK_NONE	0
#define UNPACK_DDU	1
#define UNPACK_CSC	1<<1
#define UNPACK_DMB	(uint32_t)1<<2
#define UNPACK_ALCT	(uint32_t)1<<3
#define UNPACK_CLCT	(uint32_t)1<<4
#define UNPACK_CFEB	(uint32_t)1<<5
#define UNPACK_CFEB_CLUSTERS (uint32_t) 1<<6
#define UNPACK_ALL	(uint32_t)0xFFFFFFFF

#include "CSCConstants.h"
#include "CSCMonitorObject.h"
#include "CSCLogger.h"

using namespace std;

class CSCMonitor : public CSCMonitorInterface {
 public:
  explicit CSCMonitor( const edm::ParameterSet& );
  ~CSCMonitor();

  // === Load Booking information
  void loadBooking();

  // === Book Histograms
  ME_List bookChamber(int chamberID);
  ME_List bookCommon(int nodeNumber);
  ME_List bookDDU(int dduNumber);


  void process(CSCDCCExaminer * examiner, CSCDCCEventData * dccData);
  void process(const char * data, int32_t dataSize, uint32_t errorStat, int32_t nodeNumber = 0);
  void monitorDDU(const CSCDDUEventData& dduEvent, int dduNumber);
  void monitorCSC(const CSCEventData& data,int32_t nodeID, int32_t dduID);
  void binExaminer(CSCDCCExaminer & examiner, int32_t nodeNumber = 0);

  void setHistoFile(string hfile) {RootHistoFile = hfile;};
  void setDDUCheckMask(uint32_t mask) { dduCheckMask = mask;}
  void setBinCheckMask(uint32_t mask) { binCheckMask = mask;}
  void setUnpackMask(uint32_t mask) { unpackMask = mask;}

  void saveToROOTFile(std::string filename);
  void saveImages(std::string path, 
		  std::string format=DEFAULT_IMAGE_FORMAT, 
		  int width=DEFAULT_CANVAS_WIDTH, 
		  int height=DEFAULT_CANVAS_HEIGHT);
  
  void setXMLHistosBookingCfgFile(string filename) {xmlHistosBookingCfgFile = filename;}
  std::string xigGetXMLHistosBookingCfgFile() const {return xmlHistosBookingCfgFile;}
   
  bool isBusy() { return fBusy;};
  
  bool isMEvalid(ME_List&, std::string, CSCMonitorObject* &, uint32_t mask=UNPACK_ALL);
  map<string, ME_List >  GetMEs() { return MEs;};
 
  int getUnpackedDMBCount() const {return unpackedDMBcount;}


 protected:

  void setParameters(); 
  void getCSCFromMap(int crate, int slot, int& csctype, int& cscposition);

  int loadXMLBookingInfo(string xmlFile);
  int loadXMLCanvasesInfo(string xmlFile);
  void clearMECollection(ME_List &collection);
  void printMECollection(ME_List &collection);
  CSCMonitorObject* createME(DOMNode* MEInfo);  

  void createHTMLNavigation(std::string path);
  void createTreeTemplate(std::string path);
  void createTreeEngine(std::string path);
  void createTreePage(std::string path);
  std::map<std::string, int> getCSCTypeToBinMap();
  std::string getCSCTypeLabel(int endcap, int station, int ring );
 




 private:
  // == list of Monitored Elements 
  map<string, ME_List > MEs;

  ME_List dduMEfactory;
  ME_List chamberMEfactory;
  ME_List commonMEfactory;


  map<string,int> nDMBEvents;
  int unpackedDMBcount;

  uint32_t nEvents;
  uint32_t L1ANumber;
  uint32_t BXN;

  // Logger logger_;
  std::string logger_;

  uint32_t dduCheckMask;
  uint32_t binCheckMask;
  uint32_t unpackMask;

  std::string RootHistoFile;
  bool fSaveHistos;
  bool fBusy;
  bool printout;

  int  saveRootFileEventsInterval;

  
  std::string xmlHistosBookingCfgFile;

  // back-end interface
  DQMStore * dbe;
  // CSC Mapping
  CSCReadoutMappingFromFile cscMapping;
  std::map<std::string, int> tmap;


};

#endif
