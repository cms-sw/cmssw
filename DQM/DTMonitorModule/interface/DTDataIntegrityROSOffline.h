#ifndef DTDataIntegrityROSOffline_H
#define DTDataIntegrityROSOffline_H

/** \class DTDataIntegrityROSOffline
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

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

#include "DataFormats/DTDigi/interface/DTControlData.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <list>

class DTROS25Data;
class DTDDUData;

class DTDataIntegrityROSOffline : public DQMEDAnalyzer {
public:
  DTDataIntegrityROSOffline(const edm::ParameterSet& ps);

  ~DTDataIntegrityROSOffline() override;

  void processROS25(DTROS25Data& data, int dduID, int ros);
  void processFED(DTDDUData& dduData, const std::vector<DTROS25Data>& rosData, int dduID);

  // log number of times the payload of each fed is unpacked
  void fedEntry(int dduID);
  // log number of times the payload of each fed is skipped (no ROS inside)
  void fedFatal(int dduID);
  // log number of times the payload of each fed is partially skipped (some ROS skipped)
  void fedNonFatal(int dduID);

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  void bookHistos(DQMStore::IBooker&, const int fedMin, const int fedMax);
  void bookHistos(DQMStore::IBooker&, std::string folder, DTROChainCoding code);
  void bookHistosROS25(DQMStore::IBooker&, DTROChainCoding code);

  void channelsInCEROS(int cerosId, int chMask, std::vector<int>& channels);
  void channelsInROS(int cerosMask, std::vector<int>& channels);

  std::string topFolder(bool isFEDIntegrity) const;

  // Plot quantities about SC
  bool getSCInfo;

  int nevents;

  // Monitor Elements
  MonitorElement* nEventMonitor;
  // <histoType, <index , histo> >
  std::map<std::string, std::map<int, MonitorElement*> > fedHistos;
  // <histoType, histo> >
  std::map<std::string, std::map<int, MonitorElement*> > summaryHistos;
  // <histoType, <index , histo> >
  std::map<std::string, std::map<int, MonitorElement*> > rosHistos;

  // standard ME for monitoring of FED integrity
  MonitorElement* hFEDEntry;
  MonitorElement* hFEDFatal;
  MonitorElement* hFEDNonFatal;
  MonitorElement* hCorruptionSummary;

  // one for all FEDS
  MonitorElement* hTTSSummary;

  int neventsFED;
  int neventsROS;

  // Number of ROS per FED
  const int nROS = 12;

  int FEDIDmin;
  int FEDIDmax;

  // event error flag: true when errors are detected
  // can be used for the selection of the debug stream
  bool eventErrorFlag;

  std::map<int, std::set<int> > rosBxIdsPerFED;
  std::set<int> fedBXIds;
  std::map<int, std::set<int> > rosL1AIdsPerFED;

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
