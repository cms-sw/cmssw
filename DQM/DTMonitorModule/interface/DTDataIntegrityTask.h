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
#include <DQMServices/Core/interface/oneDQMEDAnalyzer.h>

#include "DataFormats/DTDigi/interface/DTControlData.h"
#include "DataFormats/DTDigi/interface/DTuROSControlData.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <list>

class DTuROSROSData;
class DTuROSFEDData;
class DTTimeEvolutionHisto;

class DTDataIntegrityTask : public one::DQMEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  DTDataIntegrityTask(const edm::ParameterSet& ps);

  ~DTDataIntegrityTask() override;

  void TimeHistos(DQMStore::IBooker&, std::string histoType);

  void processuROS(DTuROSROSData& data, int fed, int uRos);
  void processFED(DTuROSFEDData& data, int fed);

  void beginLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) override;
  void endLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  void bookHistos(DQMStore::IBooker&, const int fedMin, const int fedMax);
  void bookHistos(DQMStore::IBooker&, std::string folder, DTROChainCoding code);
  void bookHistos(DQMStore::IBooker&, std::string folder, const int fed);
  void bookHistosuROS(DQMStore::IBooker&, const int fed, const int uRos);
  void bookHistosROS(DQMStore::IBooker&, const int wheel, const int ros);

  std::string topFolder(bool isFEDIntegrity) const;

  edm::ParameterSet parameters;

  //conversions
  int theDDU(int crate, int slot, int link, bool tenDDU);
  int theROS(int slot, int link);

  // Check FEDs from uROS, otherwise standard ROS
  bool checkUros;

  int nevents;

  // Monitor Elements
  MonitorElement* nEventMonitor;
  // <histoType, <index , histo> >
  std::map<std::string, std::map<int, MonitorElement*> > fedHistos;
  // <histoType, histo> >
  std::map<std::string, std::map<int, MonitorElement*> > summaryHistos;
  // <key , histo> >
  std::map<unsigned int, MonitorElement*> urosHistos;

  //enum histoTypes for reduced map of MEs urosHistos
  // key = stringEnum*1000 + (fed-minFED)#*100 + (uROS-minuROS)#
  enum histoTypes { uROSEventLength = 0, uROSError = 1, TDCError = 4, TTSValues = 7 };

  // standard ME for monitoring of FED integrity
  MonitorElement* hFEDEntry;
  MonitorElement* hFEDFatal;
  MonitorElement* hFEDNonFatal;

  //time histos for ROS
  std::map<std::string, std::map<int, DTTimeEvolutionHisto*> > fedTimeHistos;
  // <key, histo> >
  std::map<unsigned int, DTTimeEvolutionHisto*> urosTimeHistos;
  //key =  (fed-minFED)#*100 + (uROS-minuROS)#

  int nEventsLS;

  int neventsFED;
  int neventsuROS;

  int FEDIDmin;
  int FEDIDmax;

  // Number of ROS/uROS per FED
  const int NuROS = 12;

  // flag to toggle the creation of only the summaries (for HLT running)
  int mode;
  std::string fedIntegrityFolder;

  // The label to retrieve the digis

  edm::EDGetTokenT<DTuROSFEDDataCollection> fedToken;
};

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
