#ifndef DQM_BeamMonitor__BeamSpotDipServer_h
#define DQM_BeamMonitor__BeamSpotDipServer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/OnlineMetaData/interface/DCSRecord.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"

#include <string>
#include <bits/stdc++.h>

class DipFactory;
class DipData;
class DipPublication;

class LuminosityBlock;

class BeamSpotDipServer : public DQMOneLumiEDAnalyzer<> {
public:
  explicit BeamSpotDipServer(const edm::ParameterSet&);

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void dqmBeginRun(const edm::Run& r, const edm::EventSetup&) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
  void dqmBeginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup&) override;
  void dqmEndLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup&) override;
  void dqmEndRun(const edm::Run&, const edm::EventSetup& iSetup) override;

private:
  long getFileSize(std::string filename);
  time_t getLastTime(std::string filename);

  std::vector<std::string> parse(std::string line, const std::string& delimiter);
  std::string tkStatus();

  bool readRcd(const BeamSpotOnlineObjects& bs);  // read from database
  bool readRcd(std::ifstream& file);              // read from NFS

  void problem();
  void CMS2LHCRF_POS(float x, float y, float z);

  void trueRcd();
  void fakeRcd();
  void publishRcd(std::string qlty, std::string err, bool pubCMS, bool fitTime);

  std::string getDateTime();
  std::string getDateTime(long epoch);

  // constants
  const char* qualities[3] = {"Uncertain", "Bad", "Good"};
  const bool publishStatErrors = true;
  const int rad2urad = 1000000;
  const int cm2um = 10000;
  const int cm2mm = 10;
  const int intLS = 1;  // for CMS scaler

  // variables
  long lastFitTime = 0;
  long lastModTime = 0;
  std::bitset<8> alive;
  int lsCount = 0;
  int currentLS = 0;

  // DIP objects
  DipFactory* dip;
  DipData* messageCMS;
  DipData* messageLHC;
  DipData* messagePV;
  DipPublication* publicationCMS;
  DipPublication* publicationLHC;
  DipPublication* publicationPV;

  // initial values of beamspot object
  int runnum;
  std::string startTime = getDateTime();
  std::string endTime = getDateTime();
  time_t startTimeStamp = 0;
  time_t endTimeStamp = 0;
  std::string lumiRange = "0 - 0";
  std::string quality = "Uncertain";
  int type = -1;
  float x = 0;
  float y = 0;
  float z = 0;
  float dxdz = 0;
  float dydz = 0;
  float err_x = 0;
  float err_y = 0;
  float err_z = 0;
  float err_dxdz = 0;
  float err_dydz = 0;
  float width_x = 0;
  float width_y = 0;
  float sigma_z = 0;
  float err_width_x = 0;
  float err_width_y = 0;
  float err_sigma_z = 0;

  // added for PV information
  int events = 0;
  float meanPV = 0;
  float err_meanPV = 0;
  float rmsPV = 0;
  float err_rmsPV = 0;
  int maxPV = 0;
  int nPV = 0;

  //
  float Size[3];
  float Centroid[3];
  float Tilt[2];

  // tracker status
  edm::InputTag dcsRecordInputTag_;
  edm::EDGetTokenT<DCSRecord> dcsRecordToken_;

  int lastlumi = -1;
  bool wholeTrackerOn = false;

  // online beamspot
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> bsLegacyToken_;

  // inputs
  bool verbose;
  bool testing;

  std::string subjectCMS;
  std::string subjectLHC;
  std::string subjectPV;

  bool readFromNFS;

  std::string sourceFile;
  std::string sourceFile1;

  std::vector<int> timeoutLS;
};

#endif
