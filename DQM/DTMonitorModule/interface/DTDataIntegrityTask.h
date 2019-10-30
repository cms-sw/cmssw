#ifndef DTDataIntegrityTask_H
#define DTDataIntegrityTask_H

/** \class DTDataIntegrityTask
 *
 * Class for DT Data Integrity.
 *
 *
 * \author Marco Zanetti (INFN Padova), Gianluca Cerminara (INFN Torino)
 *
 */

#include "EventFilter/DTRawToDigi/interface/DTROChainCoding.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/DQMOneEDAnalyzer.h>

#include "DataFormats/DTDigi/interface/DTControlData.h"
#include "DataFormats/DTDigi/interface/DTuROSControlData.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <list>

class DTROS25Data;
class DTDDUData;
//to remove
class DTuROSROSData;
class DTuROSFEDData;
class DTTimeEvolutionHisto;

class DTDataIntegrityTask : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  DTDataIntegrityTask(const edm::ParameterSet& ps);

  ~DTDataIntegrityTask() override;

  void TimeHistos(DQMStore::IBooker&, std::string histoType);

  void processuROS(DTuROSROSData& data, int fed, int uRos);
  void processFED(DTuROSFEDData& data, int fed);
  void processROS25(DTROS25Data& data, int dduID, int ros);
  void processFED(DTDDUData& dduData, const std::vector<DTROS25Data>& rosData, int dduID);

  // log number of times the payload of each fed is unpacked
  void fedEntry(int dduID);
  // log number of times the payload of each fed is skipped (no ROS inside)
  void fedFatal(int dduID);
  // log number of times the payload of each fed is partially skipped (some ROS skipped)
  void fedNonFatal(int dduID);

  bool eventHasErrors() const;

  void beginLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) override;
  void endLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  void bookHistos(DQMStore::IBooker&, const int fedMin, const int fedMax);
  void bookHistos(DQMStore::IBooker&, std::string folder, DTROChainCoding code);
  void bookHistos(DQMStore::IBooker&, std::string folder, const int fed);
  void bookHistosROS25(DQMStore::IBooker&, DTROChainCoding code);
  void bookHistosuROS(DQMStore::IBooker&, const int fed, const int uRos);
  void bookHistosROS(DQMStore::IBooker&, const int wheel, const int ros);

  void channelsInCEROS(int cerosId, int chMask, std::vector<int>& channels);
  void channelsInROS(int cerosMask, std::vector<int>& channels);

  std::string topFolder(bool isFEDIntegrity) const;

  std::multimap<std::string, std::string> names;
  std::multimap<std::string, std::string>::iterator it;

  edm::ParameterSet parameters;

  //conversions
  int theDDU(int crate, int slot, int link, bool tenDDU);
  int theROS(int slot, int link);

  //If you want info VS time histos
  bool doTimeHisto;
  // Plot quantities about SC
  bool getSCInfo;
  // Check FEDs from uROS, otherwise standard ROS
  bool checkUros;

  int nevents;

  DTROChainCoding coding;

  // Monitor Elements
  MonitorElement* nEventMonitor;
  // <histoType, <index , histo> >
  std::map<std::string, std::map<int, MonitorElement*> > fedHistos;
  // <histoType, histo> >
  std::map<std::string, std::map<int, MonitorElement*> > summaryHistos;
  // <histoType, <index , histo> >
  std::map<std::string, std::map<int, MonitorElement*> > rosHistos;
  // <key , histo> >
  std::map<unsigned int, MonitorElement*> urosHistos;
  // <histoType, <tdcID, histo> >
  std::map<std::string, std::map<int, MonitorElement*> > robHistos;

  //enum histoTypes for reduced map of MEs urosHistos
  // key = stringEnum*1000 + (fed-minFED)#*100 + (uROS-minuROS)#
  enum histoTypes { uROSEventLength = 0, uROSError = 1, TDCError = 4, TTSValues = 7 };

  // standard ME for monitoring of FED integrity
  MonitorElement* hFEDEntry;
  MonitorElement* hFEDFatal;
  MonitorElement* hFEDNonFatal;
  MonitorElement* hCorruptionSummary;

  // one for all FEDS
  MonitorElement* hTTSSummary;

  //time histos for DDU/ROS
  std::map<std::string, std::map<int, DTTimeEvolutionHisto*> > fedTimeHistos;
  std::map<std::string, std::map<int, DTTimeEvolutionHisto*> > rosTimeHistos;
  // <key, histo> >
  std::map<unsigned int, DTTimeEvolutionHisto*> urosTimeHistos;
  //key =  (fed-minFED)#*100 + (uROS-minuROS)#

  int nEventsLS;

  int neventsFED;
  int neventsuROS;

  float trigger_counter;
  std::string outputFile;
  double rob_max[25];
  double link_max[72];

  int FEDIDmin;
  int FEDIDmax;

  // Number of ROS/uROS per FED
  const int NuROS = 12;

  //Event counter for the graphs VS time
  int myPrevEv;

  //Monitor TTS,ROS,FIFO VS time
  int myPrevTtsVal;
  int myPrevRosVal;
  int myPrevFifoVal[7];

  // event error flag: true when errors are detected
  // can be used for the selection of the debug stream
  bool eventErrorFlag;

  std::map<int, std::set<int> > rosBxIdsPerFED;
  std::set<int> fedBXIds;
  std::map<int, std::set<int> > rosL1AIdsPerFED;

  // flag to toggle the creation of only the summaries (for HLT running)
  int mode;
  std::string fedIntegrityFolder;

  // The label to retrieve the digis
  edm::EDGetTokenT<DTDDUCollection> dduToken;

  edm::EDGetTokenT<DTROS25Collection> ros25Token;

  edm::EDGetTokenT<DTuROSFEDDataCollection> fedToken;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
