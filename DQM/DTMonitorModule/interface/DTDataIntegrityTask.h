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
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "DataFormats/DTDigi/interface/DTControlData.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <list>

class DTROS25Data;
class DTDDUData;
class DTTimeEvolutionHisto;

class DTDataIntegrityTask: public thread_unsafe::DQMEDAnalyzer {

public:

  DTDataIntegrityTask( const edm::ParameterSet& ps);

  virtual ~DTDataIntegrityTask();

  void TimeHistos(DQMStore::IBooker &, std::string histoType);

  void processROS25(DTROS25Data & data, int dduID, int ros);
  void processFED(DTDDUData & dduData, const std::vector<DTROS25Data> & rosData, int dduID);

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

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:

  void bookHistos(DQMStore::IBooker &, const int fedMin, const int fedMax);
  void bookHistos(DQMStore::IBooker &, std::string folder, DTROChainCoding code);
  void bookHistosROS25(DQMStore::IBooker &, DTROChainCoding code);

  void channelsInCEROS(int cerosId, int chMask, std::vector<int>& channels);
  void channelsInROS(int cerosMask, std::vector<int>& channels);

  std::string topFolder(bool isFEDIntegrity) const;

  std::multimap<std::string, std::string> names;
  std::multimap<std::string, std::string>::iterator it;

  edm::ParameterSet parameters;

  //If you want info VS time histos
  bool doTimeHisto;
  // Plot quantities about SC
  bool getSCInfo;

  int nevents;

  DTROChainCoding coding;

  // Monitor Elements
  MonitorElement* nEventMonitor;
  // <histoType, <index , histo> >
  std::map<std::string, std::map<int, MonitorElement*> > dduHistos;
  // <histoType, histo> >
  std::map<std::string, std::map<int, MonitorElement*> > rosSHistos;
  // <histoType, <index , histo> >
  std::map<std::string, std::map<int, MonitorElement*> > rosHistos;
  // <histoType, <tdcID, histo> >
  std::map<std::string, std::map<int, MonitorElement*> > robHistos;

  // standard ME for monitoring of FED integrity
  MonitorElement* hFEDEntry;
  MonitorElement* hFEDFatal;
  MonitorElement* hFEDNonFatal;
  MonitorElement* hCorruptionSummary;

  // one for all FEDS
  MonitorElement* hTTSSummary;

  //time histos for DDU/ROS
  std::map<std::string, std::map<int, DTTimeEvolutionHisto*> > dduTimeHistos;
  std::map<std::string, std::map<int, DTTimeEvolutionHisto*> > rosTimeHistos;

  int nEventsLS;

  int neventsDDU;
  int neventsROS25;
  float trigger_counter;
  std::string outputFile;
  double rob_max[25];

  int FEDIDmin;
  int FEDIDMax;


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
 
};


#endif


/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
