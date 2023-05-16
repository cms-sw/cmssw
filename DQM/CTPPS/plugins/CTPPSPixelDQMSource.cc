/******************************************
 *
 * This is a part of CTPPSDQM software.
 * Authors:
 *   F.Ferro INFN Genova
 *   Vladimir Popov (vladimir.popov@cern.ch)
 *
 *******************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "CondFormats/PPSObjects/interface/CTPPSPixelIndices.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDataError.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelDAQMapping.h"

#include <string>

//-----------------------------------------------------------------------------

class CTPPSPixelDQMSource : public DQMEDAnalyzer {
public:
  CTPPSPixelDQMSource(const edm::ParameterSet &ps);
  ~CTPPSPixelDQMSource() override;

protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

private:
  unsigned int verbosity;
  long int nEvents = 0;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDigi>> tokenDigi;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDataError>> tokenError;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelCluster>> tokenCluster;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> tokenTrack;
  edm::EDGetTokenT<edm::TriggerResults> tokenTrigResults;
  edm::ESGetToken<CTPPSPixelDAQMapping, CTPPSPixelDAQMappingRcd> tokenPixelDAQMapping;
  std::string randomHLTPath;
  std::string mappingLabel;

  static constexpr int NArms = 2;
  static constexpr int NStationMAX = 3;  // in an arm
  static constexpr int NRPotsMAX = 6;    // per station
  static constexpr int NplaneMAX = 6;    // per RPot
  static constexpr int NROCsMAX = 6;     // per plane
  static constexpr int RPn_first = 3, RPn_last = 4;
  static constexpr int ADCMax = 256;
  static constexpr int StationIDMAX = 4;  // possible range of ID
  static constexpr int RPotsIDMAX = 8;    // possible range of ID
  static constexpr int NLocalTracksMAX = 20;
  static constexpr int hitMultMAX = 50;   // tuned
  static constexpr int ClusMultMAX = 10;  // tuned
  static constexpr int ClusterSizeMax = 9;
  static constexpr int errCodeSize = 15;
  static constexpr int minFedNumber = 1462;
  static constexpr int numberOfFeds = 2;
  static constexpr int mapXbins = 200;
  static constexpr int mapYbins = 240;
  static constexpr float mapYmin = -16.;
  static constexpr float mapYmax = 8.;
  const float mapXmin = 0. * TMath::Cos(18.4 / 180. * TMath::Pi());
  const float mapXmax = 30. * TMath::Cos(18.4 / 180. * TMath::Pi());

  CTPPSPixelIndices thePixIndices;

  int TrackFitDimension = 4;

  static constexpr int NRPotBinsInStation = RPn_last - RPn_first;
  static constexpr int NPlaneBins = NplaneMAX * NRPotBinsInStation;

  MonitorElement *hBX, *hBXshort, *h2AllPlanesActive, *hpixLTrack;
  MonitorElement *hpRPactive;

  MonitorElement *h2HitsMultipl[NArms][NStationMAX];
  MonitorElement *h2CluSize[NArms][NStationMAX];

  static constexpr int RPotsTotalNumber = NArms * NStationMAX * NRPotsMAX;

  int RPindexValid[RPotsTotalNumber];
  MonitorElement *h2trackXY0[RPotsTotalNumber];
  MonitorElement *h2ErrorCodeRP[RPotsTotalNumber];
  MonitorElement *h2ErrorCodeFED[2];
  MonitorElement *h2TBMMessageFED[2];
  MonitorElement *h2TBMTypeFED[2];
  MonitorElement *h2ErrorCodeUnidDet;
  MonitorElement *h2TBMMessageUnidDet;
  MonitorElement *h2TBMTypeUnidDet;

  MonitorElement *h2FullType;
  MonitorElement *h2TBMMessageRP[RPotsTotalNumber];
  MonitorElement *h2TBMTypeRP[RPotsTotalNumber];

  MonitorElement *htrackMult[RPotsTotalNumber];
  MonitorElement *htrackHits[RPotsTotalNumber];
  MonitorElement *hRPotActivPlanes[RPotsTotalNumber];
  MonitorElement *hRPotActivBX[RPotsTotalNumber];
  MonitorElement *hRPotActivBXroc[RPotsTotalNumber];
  MonitorElement *hRPotActivBXroc_3[RPotsTotalNumber];
  MonitorElement *hRPotActivBXroc_2[RPotsTotalNumber];
  MonitorElement *h2HitsMultROC[RPotsTotalNumber];
  MonitorElement *hp2HitsMultROC_LS[RPotsTotalNumber];
  MonitorElement *hHitsMult[RPotsTotalNumber][NplaneMAX];
  MonitorElement *h2xyHits[RPotsTotalNumber][NplaneMAX];
  MonitorElement *hp2xyADC[RPotsTotalNumber][NplaneMAX];
  MonitorElement *h2Efficiency[RPotsTotalNumber][NplaneMAX];
  MonitorElement *h2xyROCHits[RPotsTotalNumber * NplaneMAX][NROCsMAX];
  MonitorElement *hROCadc[RPotsTotalNumber * NplaneMAX][NROCsMAX];
  MonitorElement *hRPotActivBXall[RPotsTotalNumber];
  MonitorElement *h2HitsVsBXRandoms[RPotsTotalNumber];
  int HitsMultROC[RPotsTotalNumber * NplaneMAX][NROCsMAX];
  int HitsMultPlane[RPotsTotalNumber][NplaneMAX];

  //--------------------------------------------------------
  static constexpr int LINK_bits = 6;
  static constexpr int ROC_bits = 5;
  static constexpr int DCOL_bits = 5;
  static constexpr int PXID_bits = 8;
  static constexpr int ADC_bits = 8;
  static constexpr int DataBit_bits = 1;

  static constexpr int ADC_shift = 0;
  static constexpr int PXID_shift = ADC_shift + ADC_bits;
  static constexpr int DCOL_shift = PXID_shift + PXID_bits;
  static constexpr int ROC_shift = DCOL_shift + DCOL_bits;
  static constexpr int LINK_shift = ROC_shift + ROC_bits;
  static constexpr int DB0_shift = 0;
  static constexpr int DB1_shift = DB0_shift + DataBit_bits;
  static constexpr int DB2_shift = DB1_shift + DataBit_bits;
  static constexpr int DB3_shift = DB2_shift + DataBit_bits;
  static constexpr int DB4_shift = DB3_shift + DataBit_bits;
  static constexpr int DB5_shift = DB4_shift + DataBit_bits;
  static constexpr int DB6_shift = DB5_shift + DataBit_bits;
  static constexpr int DB7_shift = DB6_shift + DataBit_bits;

  static constexpr uint32_t LINK_mask = ~(~uint32_t(0) << LINK_bits);
  static constexpr uint32_t ROC_mask = ~(~uint32_t(0) << ROC_bits);
  static constexpr uint32_t DCOL_mask = ~(~uint32_t(0) << DCOL_bits);
  static constexpr uint32_t PXID_mask = ~(~uint32_t(0) << PXID_bits);
  static constexpr uint32_t ADC_mask = ~(~uint32_t(0) << ADC_bits);
  static constexpr uint32_t DataBit_mask = ~(~uint32_t(0) << DataBit_bits);
  // Flags for disabling set of plots
  bool offlinePlots = true;
  bool onlinePlots = true;

  // Flags for disabling plots of a plane
  bool isPlanePlotsTurnedOff[NArms][NStationMAX][NRPotsMAX][NplaneMAX] = {};

  unsigned int rpStatusWord = 0x8008;      // 220_fr_hr(stn2rp3)+ 210_fr_hr
  int RPstatus[StationIDMAX][RPotsIDMAX];  // symmetric in both arms
  int StationStatus[StationIDMAX];         // symmetric in both arms
  const int IndexNotValid = 0;

  int getRPindex(int arm, int station, int rp) {
    if (arm < 0 || station < 0 || rp < 0)
      return (IndexNotValid);
    if (arm > 1 || station >= NStationMAX || rp >= NRPotsMAX)
      return (IndexNotValid);
    int rc = (arm * NStationMAX + station) * NRPotsMAX + rp;
    return (rc);
  }

  int getPlaneIndex(int arm, int station, int rp, int plane) {
    if (plane < 0 || plane >= NplaneMAX)
      return (IndexNotValid);
    int rc = getRPindex(arm, station, rp);
    if (rc == IndexNotValid)
      return (IndexNotValid);
    return (rc * NplaneMAX + plane);
  }

  int getRPInStationBin(int rp) { return (rp - RPn_first + 1); }

  static constexpr int NRPglobalBins = 4;  // 2 arms w. 2 stations w. 1 RP

  int getRPglobalBin(int arm, int stn) {
    static constexpr int stationBinOrder[NStationMAX] = {0, 4, 1};
    return (arm * 2 + stationBinOrder[stn] + 1);
  }

  int prIndex(int rp, int plane)  // plane index in station

  {
    return ((rp - RPn_first) * NplaneMAX + plane);
  }
  int getDet(int id) { return (id >> DetId::kDetOffset) & 0xF; }
  int getPixPlane(int id) { return ((id >> 16) & 0x7); }
  //  int getSubdet(int id) { return ((id>>kSubdetOffset)&0x7); }

  float x0_MIN, x0_MAX, y0_MIN, y0_MAX;
};

//----------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//-------------------------------------------------------------------------------

CTPPSPixelDQMSource::CTPPSPixelDQMSource(const edm::ParameterSet &ps)
    : verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
      randomHLTPath(ps.getUntrackedParameter<std::string>("randomHLTPath", "")),
      rpStatusWord(ps.getUntrackedParameter<unsigned int>("RPStatusWord", 0x8008)) {
  tokenDigi = consumes<DetSetVector<CTPPSPixelDigi>>(ps.getUntrackedParameter<edm::InputTag>("tagRPixDigi"));
  tokenError = consumes<DetSetVector<CTPPSPixelDataError>>(ps.getUntrackedParameter<edm::InputTag>("tagRPixError"));
  tokenCluster = consumes<DetSetVector<CTPPSPixelCluster>>(ps.getUntrackedParameter<edm::InputTag>("tagRPixCluster"));
  tokenTrack = consumes<DetSetVector<CTPPSPixelLocalTrack>>(ps.getUntrackedParameter<edm::InputTag>("tagRPixLTrack"));
  tokenTrigResults = consumes<edm::TriggerResults>(ps.getUntrackedParameter<edm::InputTag>("tagTrigResults"));
  tokenPixelDAQMapping = esConsumes<CTPPSPixelDAQMapping, CTPPSPixelDAQMappingRcd>();
  mappingLabel = ps.getUntrackedParameter<std::string>("mappingLabel");
  offlinePlots = ps.getUntrackedParameter<bool>("offlinePlots", true);
  onlinePlots = ps.getUntrackedParameter<bool>("onlinePlots", true);

  vector<string> disabledPlanePlotsVec =
      ps.getUntrackedParameter<vector<string>>("turnOffPlanePlots", vector<string>());
  // Parse the strings in disabledPlanePlotsVec and set the flags in
  // isPlanePlotsTurnedOff
  for (auto s : disabledPlanePlotsVec) {
    // Check that the format is <arm>_<station>_<RP>_<Plane>
    if (count(s.begin(), s.end(), '_') != 3)
      throw cms::Exception("RPixPlaneCombinatoryTracking") << "Invalid string in turnOffPlanePlots: " << s;
    else {
      vector<string> armStationRpPlane;
      size_t pos = 0;
      while ((pos = s.find('_')) != string::npos) {
        armStationRpPlane.push_back(s.substr(0, pos));
        s.erase(0, pos + 1);
      }
      armStationRpPlane.push_back(s);

      int arm = stoi(armStationRpPlane.at(0));
      int station = stoi(armStationRpPlane.at(1));
      int rp = stoi(armStationRpPlane.at(2));
      int plane = stoi(armStationRpPlane.at(3));

      if (arm < NArms && station < NStationMAX && rp < NRPotsMAX && plane < NplaneMAX) {
        if (verbosity)
          LogPrint("CTPPSPixelDQMSource")
              << "Shutting off plots for: Arm " << arm << " Station " << station << " Rp " << rp << " Plane " << plane;
        isPlanePlotsTurnedOff[arm][station][rp][plane] = true;
      } else {
        throw cms::Exception("RPixPlaneCombinatoryTracking") << "Invalid string in turnOffPlanePlots: " << s;
      }
    }
  }
}

//----------------------------------------------------------------------------------

CTPPSPixelDQMSource::~CTPPSPixelDQMSource() {}

//--------------------------------------------------------------------------

void CTPPSPixelDQMSource::dqmBeginRun(edm::Run const &run, edm::EventSetup const &) {
  if (verbosity)
    LogPrint("CTPPSPixelDQMSource") << "RPstatusWord= " << rpStatusWord;
  nEvents = 0;

  CTPPSPixelLocalTrack thePixelLocalTrack;
  TrackFitDimension = thePixelLocalTrack.dimension;

  for (int stn = 0; stn < StationIDMAX; stn++) {
    StationStatus[stn] = 0;
    for (int rp = 0; rp < RPotsIDMAX; rp++)
      RPstatus[stn][rp] = 0;
  }

  unsigned int rpSts = rpStatusWord << 1;
  for (int stn = 0; stn < NStationMAX; stn++) {
    int stns = 0;
    for (int rp = 0; rp < NRPotsMAX; rp++) {
      rpSts = (rpSts >> 1);
      RPstatus[stn][rp] = rpSts & 1;
      if (RPstatus[stn][rp] > 0)
        stns = 1;
    }
    StationStatus[stn] = stns;
  }

  for (int ind = 0; ind < 2 * 3 * NRPotsMAX; ind++)
    RPindexValid[ind] = 0;

  x0_MIN = y0_MIN = 1.0e06;
  x0_MAX = y0_MAX = -1.0e06;
}

//-------------------------------------------------------------------------------------

void CTPPSPixelDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS/TrackingPixel");
  char s[50];
  string armTitleShort, stnTitleShort;

  TAxis *yah1st = nullptr;
  TAxis *xaRPact = nullptr;
  TAxis *xah1trk = nullptr;
  if (onlinePlots) {
    hBX = ibooker.book1D("events per BX", "ctpps_pixel;Event.BX", 4002, -1.5, 4000. + 0.5);
    hBXshort = ibooker.book1D("events per BX(short)", "ctpps_pixel;Event.BX", 102, -1.5, 100. + 0.5);

    string str1st = "Pixel planes activity";
    h2AllPlanesActive = ibooker.book2DD(
        str1st, str1st + "(digi task);Plane #", NplaneMAX, 0, NplaneMAX, NRPglobalBins, 0.5, NRPglobalBins + 0.5);
    TH2D *h1st = h2AllPlanesActive->getTH2D();
    h1st->SetOption("colz");
    yah1st = h1st->GetYaxis();

    string str2 = "Pixel RP active";
    hpRPactive = ibooker.bookProfile(
        str2, str2 + " per event(digi task)", NRPglobalBins, 0.5, NRPglobalBins + 0.5, -0.1, 1.1, "");
    xaRPact = hpRPactive->getTProfile()->GetXaxis();
    hpRPactive->getTProfile()->SetOption("hist");
    hpRPactive->getTProfile()->SetMinimum(0.);
    hpRPactive->getTProfile()->SetMaximum(1.1);

    str2 = "Pixel Local Tracks";
    hpixLTrack = ibooker.bookProfile(
        str2, str2 + " per event", NRPglobalBins, 0.5, NRPglobalBins + 0.5, -0.1, NLocalTracksMAX, "");

    xah1trk = hpixLTrack->getTProfile()->GetXaxis();
    hpixLTrack->getTProfile()->GetYaxis()->SetTitle("average number of tracks per event");
    hpixLTrack->getTProfile()->SetOption("hist");
  }
  const float minErrCode = 25.;
  const string errCode[errCodeSize] = {
      "Masked channel                     ",  // error 25
      "Gap word",                             // error 26
      "Dummy word",                           // error 27
      "FIFO nearly full",                     // error 28
      "Channel Timeout",                      // error 29
      "TBM Trailer",                          // error 30
      "Event number mismatch",                // error 31
      "Invalid/no FED header",                // error 32
      "Invalid/no FED trailer",               // error 33
      "Size mismatch",                        // error 34
      "Conversion: inv. channel",             // error 35
      "Conversion: inv. ROC number",          // error 36
      "Conversion: inv. pixel address",       // error 37
      "-",
      "CRC"  //error 39
  };

  const string tbmMessage[8] = {"Stack full ",
                                "Pre-cal issued ",
                                "Clear trigger counter",
                                "Synch trigger",
                                "Synch trigger error",
                                "Reset ROC",
                                "Reset TBM",
                                "No token pass"};

  const string tbmType[5] = {
      "All bits 0", "bit 8 : Overflow", "bit 9 : PKAM", "bit 10 : Auto Reset", "bit 11 : Number of ROC Error"};

  h2ErrorCodeUnidDet = ibooker.book2D("Errors in Unidentified Det",
                                      "Errors in Unidentified Det;;fed",
                                      errCodeSize,
                                      minErrCode - 0.5,
                                      minErrCode + float(errCodeSize) - 0.5,
                                      2,
                                      1461.5,
                                      1463.5);
  for (unsigned int iBin = 1; iBin <= errCodeSize; iBin++)
    h2ErrorCodeUnidDet->setBinLabel(iBin, errCode[iBin - 1]);
  h2ErrorCodeUnidDet->setBinLabel(1, "1462", 2);
  h2ErrorCodeUnidDet->setBinLabel(2, "1463", 2);
  h2ErrorCodeUnidDet->getTH2F()->SetOption("colz");
  h2TBMMessageUnidDet =
      ibooker.book2D("TBM Message Unid Det", "TBM Message Unid Det;;fed", 8, -0.5, 7.5, 2, 1461.5, 1463.5);
  for (unsigned int iBin = 1; iBin <= 8; iBin++)
    h2TBMMessageUnidDet->setBinLabel(iBin, tbmMessage[iBin - 1]);
  h2TBMMessageUnidDet->getTH2F()->SetOption("colz");

  h2TBMTypeUnidDet =
      ibooker.book2D("TBM Type in Unid Det", "TBM Type in Unid Det;;fed", 5, -0.5, 4.5, 2, 1461.5, 1463.5);
  for (unsigned int iBin = 1; iBin <= 5; iBin++)
    h2TBMTypeUnidDet->setBinLabel(iBin, tbmType[iBin - 1]);
  h2TBMTypeUnidDet->getTH2F()->SetOption("colz");

  h2FullType = ibooker.book2D("Full FIFO", "Full FIFO;;fed", 7, 0.5, 7.5, 2, 1461.5, 1463.5);
  h2FullType->setBinLabel(1, "1462", 2);
  h2FullType->setBinLabel(2, "1463", 2);
  h2FullType->getTH2F()->SetOption("colz");

  for (int iFed = 0; iFed < numberOfFeds; iFed++) {
    int fedId = minFedNumber + iFed;
    auto s_fed = std::to_string(fedId);

    h2ErrorCodeFED[iFed] = ibooker.book2D("Errors in FED" + s_fed,
                                          "Errors in FED" + s_fed + ";;channel",
                                          errCodeSize,
                                          minErrCode - 0.5,
                                          minErrCode + float(errCodeSize) - 0.5,
                                          37,
                                          -0.5,
                                          36.5);
    for (unsigned int iBin = 1; iBin <= errCodeSize; iBin++)
      h2ErrorCodeFED[iFed]->setBinLabel(iBin, errCode[iBin - 1]);
    h2ErrorCodeFED[iFed]->getTH2F()->SetOption("colz");

    h2TBMMessageFED[iFed] = ibooker.book2D(
        "TBM Message in FED" + s_fed, "TBM Message in FED" + s_fed + ";;channel", 8, -0.5, 7.5, 37, -0.5, 36.5);
    for (unsigned int iBin = 1; iBin <= 8; iBin++)
      h2TBMMessageFED[iFed]->setBinLabel(iBin, tbmMessage[iBin - 1]);
    h2TBMMessageFED[iFed]->getTH2F()->SetOption("colz");

    h2TBMTypeFED[iFed] = ibooker.book2D(
        "TBM Type in FED" + s_fed, "TBM Type in FED" + s_fed + ";;channel", 5, -0.5, 4.5, 37, -0.5, 36.5);
    for (unsigned int iBin = 1; iBin <= 5; iBin++)
      h2TBMTypeFED[iFed]->setBinLabel(iBin, tbmType[iBin - 1]);
    h2TBMTypeFED[iFed]->getTH2F()->SetOption("colz");
  }

  for (int arm = 0; arm < 2; arm++) {
    CTPPSDetId ID(CTPPSDetId::sdTrackingPixel, arm, 0);
    string sd, armTitle;
    ID.armName(sd, CTPPSDetId::nPath);
    ID.armName(armTitle, CTPPSDetId::nFull);
    ID.armName(armTitleShort, CTPPSDetId::nShort);

    ibooker.setCurrentFolder(sd);

    for (int stn = 0; stn < NStationMAX; stn++) {
      if (StationStatus[stn] == 0)
        continue;
      ID.setStation(stn);
      string stnd, stnTitle;

      CTPPSDetId(ID.stationId()).stationName(stnd, CTPPSDetId::nPath);
      CTPPSDetId(ID.stationId()).stationName(stnTitle, CTPPSDetId::nFull);
      CTPPSDetId(ID.stationId()).stationName(stnTitleShort, CTPPSDetId::nShort);

      ibooker.setCurrentFolder(stnd);
      //--------- RPots ---
      int pixBinW = 4;
      for (int rp = RPn_first; rp < RPn_last; rp++) {  // only installed pixel pots
        ID.setRP(rp);
        string rpd, rpTitle;
        CTPPSDetId(ID.rpId()).rpName(rpTitle, CTPPSDetId::nShort);
        string rpBinName = armTitleShort + "_" + stnTitleShort + "_" + rpTitle;
        if (onlinePlots) {
          yah1st->SetBinLabel(getRPglobalBin(arm, stn), rpBinName.c_str());
          xah1trk->SetBinLabel(getRPglobalBin(arm, stn), rpBinName.c_str());
          xaRPact->SetBinLabel(getRPglobalBin(arm, stn), rpBinName.c_str());
        }
        if (RPstatus[stn][rp] == 0)
          continue;
        int indexP = getRPindex(arm, stn, rp);
        RPindexValid[indexP] = 1;

        CTPPSDetId(ID.rpId()).rpName(rpTitle, CTPPSDetId::nFull);
        CTPPSDetId(ID.rpId()).rpName(rpd, CTPPSDetId::nPath);

        ibooker.setCurrentFolder(rpd);

        const float x0Maximum = 70.;
        const float y0Maximum = 15.;
        string st = "track intercept point";
        string st2 = ": " + stnTitle;
        h2trackXY0[indexP] = ibooker.book2D(
            st, st + st2 + ";x0;y0", int(x0Maximum) * 2, 0., x0Maximum, int(y0Maximum) * 4, -y0Maximum, y0Maximum);
        h2trackXY0[indexP]->getTH2F()->SetOption("colz");
        st = "Error Code";
        h2ErrorCodeRP[indexP] = ibooker.book2D(st,
                                               st + st2 + ";;plane",
                                               errCodeSize,
                                               minErrCode - 0.5,
                                               minErrCode + float(errCodeSize) - 0.5,
                                               6,
                                               -0.5,
                                               5.5);
        for (unsigned int iBin = 1; iBin <= errCodeSize; iBin++)
          h2ErrorCodeRP[indexP]->setBinLabel(iBin, errCode[iBin - 1]);
        h2ErrorCodeRP[indexP]->getTH2F()->SetOption("colz");

        h2TBMMessageRP[indexP] = ibooker.book2D("TBM Message", "TBM Message;;plane", 8, -0.5, 7.5, 6, -0.5, 5.5);
        for (unsigned int iBin = 1; iBin <= 8; iBin++)
          h2TBMMessageRP[indexP]->setBinLabel(iBin, tbmMessage[iBin - 1]);
        h2TBMMessageRP[indexP]->getTH2F()->SetOption("colz");

        h2TBMTypeRP[indexP] = ibooker.book2D("TBM Type", "TBM Type;;plane", 5, -0.5, 4.5, 6, -0.5, 5.5);
        for (unsigned int iBin = 1; iBin <= 5; iBin++)
          h2TBMTypeRP[indexP]->setBinLabel(iBin, tbmType[iBin - 1]);
        h2TBMTypeRP[indexP]->getTH2F()->SetOption("colz");

        st = "number of tracks per event";
        htrackMult[indexP] = ibooker.bookProfile(st,
                                                 rpTitle + ";number of tracks",
                                                 NLocalTracksMAX + 1,
                                                 -0.5,
                                                 NLocalTracksMAX + 0.5,
                                                 -0.5,
                                                 NLocalTracksMAX + 0.5,
                                                 "");
        htrackMult[indexP]->getTProfile()->SetOption("hist");

        hRPotActivPlanes[indexP] = ibooker.bookProfile("number of fired planes per event",
                                                       rpTitle + ";nPlanes;Probability",
                                                       NplaneMAX + 1,
                                                       -0.5,
                                                       NplaneMAX + 0.5,
                                                       -0.5,
                                                       NplaneMAX + 0.5,
                                                       "");
        hRPotActivPlanes[indexP]->getTProfile()->SetOption("hist");

        hp2HitsMultROC_LS[indexP] = ibooker.bookProfile2D("ROCs hits multiplicity per event vs LS",
                                                          rpTitle + ";LumiSection;Plane#___ROC#",
                                                          1000,
                                                          0.,
                                                          1000.,
                                                          NplaneMAX * NROCsMAX,
                                                          0.,
                                                          double(NplaneMAX * NROCsMAX),
                                                          0.,
                                                          rpixValues::ROCSizeInX *rpixValues::ROCSizeInY,
                                                          "");
        hp2HitsMultROC_LS[indexP]->getTProfile2D()->SetOption("colz");
        hp2HitsMultROC_LS[indexP]->getTProfile2D()->SetMinimum(1.0e-10);
        hp2HitsMultROC_LS[indexP]->getTProfile2D()->SetCanExtend(TProfile2D::kXaxis);
        TAxis *yahp2 = hp2HitsMultROC_LS[indexP]->getTProfile2D()->GetYaxis();
        for (int p = 0; p < NplaneMAX; p++) {
          sprintf(s, "plane%d_0", p);
          yahp2->SetBinLabel(p * NplaneMAX + 1, s);
          for (int r = 1; r < NROCsMAX; r++) {
            sprintf(s, "   %d_%d", p, r);
            yahp2->SetBinLabel(p * NplaneMAX + r + 1, s);
          }
        }

        // Hits per plane per bx
        h2HitsVsBXRandoms[indexP] = ibooker.book2D(
            "Hits per plane per BX - random triggers", rpTitle + ";Event.BX;Plane", 4002, -1.5, 4000. + 0.5, 6, 0, 6);

        if (onlinePlots) {
          string st3 = ";PlaneIndex(=pixelPot*PlaneMAX + plane)";

          st = "hit multiplicity in planes";
          h2HitsMultipl[arm][stn] = ibooker.book2DD(
              st, st + st2 + st3 + ";multiplicity", NPlaneBins, 0, NPlaneBins, hitMultMAX, 0, hitMultMAX);
          h2HitsMultipl[arm][stn]->getTH2D()->SetOption("colz");

          st = "cluster size in planes";
          h2CluSize[arm][stn] = ibooker.book2D(st,
                                               st + st2 + st3 + ";Cluster size",
                                               NPlaneBins,
                                               0,
                                               NPlaneBins,
                                               ClusterSizeMax + 1,
                                               0,
                                               ClusterSizeMax + 1);
          h2CluSize[arm][stn]->getTH2F()->SetOption("colz");

          st = "number of hits per track";
          htrackHits[indexP] = ibooker.bookProfile(st, rpTitle + ";number of hits", 5, 1.5, 6.5, -0.1, 1.1, "");
          htrackHits[indexP]->getTProfile()->SetOption("hist");

          h2HitsMultROC[indexP] = ibooker.bookProfile2D("ROCs hits multiplicity per event",
                                                        rpTitle + ";plane # ;ROC #",
                                                        NplaneMAX,
                                                        -0.5,
                                                        NplaneMAX - 0.5,
                                                        NROCsMAX,
                                                        -0.5,
                                                        NROCsMAX - 0.5,
                                                        0.,
                                                        rpixValues::ROCSizeInX * rpixValues::ROCSizeInY,
                                                        "");
          h2HitsMultROC[indexP]->getTProfile2D()->SetOption("colztext");
          h2HitsMultROC[indexP]->getTProfile2D()->SetMinimum(1.e-10);

          ibooker.setCurrentFolder(rpd + "/latency");
          hRPotActivBX[indexP] =
              ibooker.book1D("5 fired planes per BX", rpTitle + ";Event.BX", 4002, -1.5, 4000. + 0.5);

          hRPotActivBXroc[indexP] =
              ibooker.book1D("4 fired ROCs per BX", rpTitle + ";Event.BX", 4002, -1.5, 4000. + 0.5);
          hRPotActivBXroc_3[indexP] =
              ibooker.book1D("3 fired ROCs per BX", rpTitle + ";Event.BX", 4002, -1.5, 4000. + 0.5);
          hRPotActivBXroc_2[indexP] =
              ibooker.book1D("2 fired ROCs per BX", rpTitle + ";Event.BX", 4002, -1.5, 4000. + 0.5);

          hRPotActivBXall[indexP] = ibooker.book1D("hits per BX", rpTitle + ";Event.BX", 4002, -1.5, 4000. + 0.5);
        }
        int nbins = rpixValues::defaultDetSizeInX / pixBinW;

        for (int p = 0; p < NplaneMAX; p++) {
          if (isPlanePlotsTurnedOff[arm][stn][rp][p])
            continue;
          sprintf(s, "plane_%d", p);
          string pd = rpd + "/" + string(s);
          ibooker.setCurrentFolder(pd);
          string st1 = ": " + rpTitle + "_" + string(s);

          st = "adc average value";
          hp2xyADC[indexP][p] = ibooker.bookProfile2D(st,
                                                      st1 + ";pix col;pix row",
                                                      nbins,
                                                      0,
                                                      rpixValues::defaultDetSizeInX,
                                                      nbins,
                                                      0,
                                                      rpixValues::defaultDetSizeInX,
                                                      0.,
                                                      512.,
                                                      "");
          hp2xyADC[indexP][p]->getTProfile2D()->SetOption("colz");

          if (onlinePlots) {
            st = "hits position";
            h2xyHits[indexP][p] = ibooker.book2DD(st,
                                                  st1 + ";pix col;pix row",
                                                  rpixValues::defaultDetSizeInX,
                                                  0,
                                                  rpixValues::defaultDetSizeInX,
                                                  rpixValues::defaultDetSizeInX,
                                                  0,
                                                  rpixValues::defaultDetSizeInX);
            h2xyHits[indexP][p]->getTH2D()->SetOption("colz");

            st = "hits multiplicity";
            hHitsMult[indexP][p] =
                ibooker.book1DD(st, st1 + ";number of hits;N / 1 hit", hitMultMAX + 1, -0.5, hitMultMAX + 0.5);
          }

          if (offlinePlots) {
            st = "plane efficiency";
            h2Efficiency[indexP][p] = ibooker.bookProfile2D(
                st, st1 + ";x0;y0", mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax, 0, 1, "");
            h2Efficiency[indexP][p]->getTProfile2D()->SetOption("colz");
          }
        }  // end of for(int p=0; p<NplaneMAX;..

      }  // end for(int rp=0; rp<NRPotsMAX;...
    }    // end of for(int stn=0; stn<
  }      // end of for(int arm=0; arm<2;...

  return;
}

//-------------------------------------------------------------------------------

void CTPPSPixelDQMSource::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {
  ++nEvents;
  int lumiId = event.getLuminosityBlock().id().luminosityBlock();
  if (lumiId < 0)
    lumiId = 0;

  int RPactivity[RPotsTotalNumber], RPdigiSize[RPotsTotalNumber];
  int pixRPTracks[RPotsTotalNumber];

  for (int rp = 0; rp < RPotsTotalNumber; rp++) {
    RPactivity[rp] = RPdigiSize[rp] = pixRPTracks[rp] = 0;
  }

  for (int ind = 0; ind < RPotsTotalNumber; ind++) {
    for (int p = 0; p < NplaneMAX; p++) {
      HitsMultPlane[ind][p] = 0;
    }
  }
  for (int ind = 0; ind < RPotsTotalNumber * NplaneMAX; ind++) {
    for (int roc = 0; roc < NROCsMAX; roc++) {
      HitsMultROC[ind][roc] = 0;
    }
  }
  Handle<DetSetVector<CTPPSPixelDigi>> pixDigi;
  event.getByToken(tokenDigi, pixDigi);

  Handle<DetSetVector<CTPPSPixelDataError>> pixError;
  event.getByToken(tokenError, pixError);

  Handle<DetSetVector<CTPPSPixelCluster>> pixClus;
  event.getByToken(tokenCluster, pixClus);

  Handle<DetSetVector<CTPPSPixelLocalTrack>> pixTrack;
  event.getByToken(tokenTrack, pixTrack);

  Handle<edm::TriggerResults> hltResults;
  event.getByToken(tokenTrigResults, hltResults);

  const CTPPSPixelDAQMapping *mapping = &eventSetup.getData(tokenPixelDAQMapping);

  if (onlinePlots) {
    hBX->Fill(event.bunchCrossing());
    hBXshort->Fill(event.bunchCrossing());
  }

  if (pixTrack.isValid()) {
    for (const auto &ds_tr : *pixTrack) {
      int idet = getDet(ds_tr.id);
      if (idet != DetId::VeryForward) {
        if (verbosity > 1)
          LogPrint("CTPPSPixelDQMSource") << "not CTPPS: ds_tr.id" << ds_tr.id;
        continue;
      }
      CTPPSDetId theId(ds_tr.id);
      int arm = theId.arm() & 0x1;
      int station = theId.station() & 0x3;
      int rpot = theId.rp() & 0x7;
      int rpInd = getRPindex(arm, station, rpot);

      for (DetSet<CTPPSPixelLocalTrack>::const_iterator dit = ds_tr.begin(); dit != ds_tr.end(); ++dit) {
        ++pixRPTracks[rpInd];
        int nh_tr = (dit->ndf() + TrackFitDimension) / 2;
        if (onlinePlots) {
          for (int i = 0; i <= NplaneMAX; i++) {
            if (i == nh_tr)
              htrackHits[rpInd]->Fill(nh_tr, 1.);
            else
              htrackHits[rpInd]->Fill(i, 0.);
          }
        }
        float x0 = dit->x0();
        float y0 = dit->y0();
        h2trackXY0[rpInd]->Fill(x0, y0);

        if (x0_MAX < x0)
          x0_MAX = x0;
        if (y0_MAX < y0)
          y0_MAX = y0;
        if (x0_MIN > x0)
          x0_MIN = x0;
        if (y0_MIN > y0)
          y0_MIN = y0;

        if (offlinePlots) {
          edm::DetSetVector<CTPPSPixelFittedRecHit> fittedHits = dit->hits();

          std::map<int, int> numberOfPointPerPlaneEff;
          for (const auto &ds_frh : fittedHits) {
            int plane = getPixPlane(ds_frh.id);
            for (DetSet<CTPPSPixelFittedRecHit>::const_iterator frh_it = ds_frh.begin(); frh_it != ds_frh.end();
                 ++frh_it) {  // there should always be only one hit in each
                              // vector
              if (frh_it != ds_frh.begin())
                if (verbosity > 1)
                  LogPrint("CTPPSPixelDQMSource") << "More than one FittedRecHit found in plane " << plane;
              if (frh_it->isRealHit())
                for (int p = 0; p < NplaneMAX; p++) {
                  if (p != plane)
                    numberOfPointPerPlaneEff[p]++;
                }
            }
          }

          if (verbosity > 1)
            for (auto planeAndHitsOnOthers : numberOfPointPerPlaneEff) {
              LogPrint("CTPPSPixelDQMSource")
                  << "For plane " << planeAndHitsOnOthers.first << ", " << planeAndHitsOnOthers.second
                  << " hits on other planes were found" << endl;
            }

          for (const auto &ds_frh : fittedHits) {
            int plane = getPixPlane(ds_frh.id);
            if (isPlanePlotsTurnedOff[arm][station][rpot][plane])
              continue;
            for (DetSet<CTPPSPixelFittedRecHit>::const_iterator frh_it = ds_frh.begin(); frh_it != ds_frh.end();
                 ++frh_it) {
              float frhX0 = frh_it->globalCoordinates().x() + frh_it->xResidual();
              float frhY0 = frh_it->globalCoordinates().y() + frh_it->yResidual();
              if (numberOfPointPerPlaneEff[plane] >= 3) {
                if (frh_it->isRealHit())
                  h2Efficiency[rpInd][plane]->Fill(frhX0, frhY0, 1);
                else
                  h2Efficiency[rpInd][plane]->Fill(frhX0, frhY0, 0);
              }
            }
          }
        }
      }
    }
  }  // end  if(pixTrack.isValid())

  bool valid = false;
  valid |= pixDigi.isValid();
  //  valid |= Clus.isValid();

  if (!valid && verbosity)
    LogPrint("CTPPSPixelDQMSource") << "No valid data in Event " << nEvents;

  if (pixDigi.isValid()) {
    for (const auto &ds_digi : *pixDigi) {
      int idet = getDet(ds_digi.id);
      if (idet != DetId::VeryForward) {
        if (verbosity > 1)
          LogPrint("CTPPSPixelDQMSource") << "not CTPPS: ds_digi.id" << ds_digi.id;
        continue;
      }
      //   int subdet = getSubdet(ds_digi.id);

      int plane = getPixPlane(ds_digi.id);

      CTPPSDetId theId(ds_digi.id);
      int arm = theId.arm() & 0x1;
      int station = theId.station() & 0x3;
      int rpot = theId.rp() & 0x7;
      int rpInd = getRPindex(arm, station, rpot);
      RPactivity[rpInd] = 1;
      ++RPdigiSize[rpInd];

      if (StationStatus[station] && RPstatus[station][rpot]) {
        if (onlinePlots) {
          h2HitsMultipl[arm][station]->Fill(prIndex(rpot, plane), ds_digi.data.size());
          h2AllPlanesActive->Fill(plane, getRPglobalBin(arm, station));
        }
        int index = getRPindex(arm, station, rpot);
        HitsMultPlane[index][plane] += ds_digi.data.size();
        if (RPindexValid[index]) {
          int nh = ds_digi.data.size();
          if (nh > hitMultMAX)
            nh = hitMultMAX;
          if (!isPlanePlotsTurnedOff[arm][station][rpot][plane])
            if (onlinePlots)
              hHitsMult[index][plane]->Fill(nh);
        }
        int rocHistIndex = getPlaneIndex(arm, station, rpot, plane);

        for (DetSet<CTPPSPixelDigi>::const_iterator dit = ds_digi.begin(); dit != ds_digi.end(); ++dit) {
          int row = dit->row();
          int col = dit->column();
          int adc = dit->adc();

          if (RPindexValid[index]) {
            if (!isPlanePlotsTurnedOff[arm][station][rpot][plane]) {
              if (onlinePlots)
                h2xyHits[index][plane]->Fill(col, row);
              hp2xyADC[index][plane]->Fill(col, row, adc);
            }
            int colROC, rowROC;
            int trocId;
            if (!thePixIndices.transformToROC(col, row, trocId, colROC, rowROC)) {
              if (trocId >= 0 && trocId < NROCsMAX) {
                ++HitsMultROC[rocHistIndex][trocId];
              }
            }
          }  // end if(RPindexValid[index]) {
        }
      }  // end  if(StationStatus[station]) {
    }    // end for(const auto &ds_digi : *pixDigi)
  }      // if(pixDigi.isValid()) {

  if (pixError.isValid()) {
    std::map<CTPPSPixelFramePosition, CTPPSPixelROCInfo> rocMapping = mapping->ROCMapping;
    for (const auto &ds_error : *pixError) {
      int idet = getDet(ds_error.id);
      if (idet != DetId::VeryForward) {
        if (idet == 15) {  //dummy det id: store in a plot with fed info

          for (DetSet<CTPPSPixelDataError>::const_iterator dit = ds_error.begin(); dit != ds_error.end(); ++dit) {
            // recover fed channel number
            int chanNmbr = -1;
            if (dit->errorType() == 32 || dit->errorType() == 33 || dit->errorType() == 34) {
              long long errorWord = dit->errorWord64();  // for 64-bit error words
              chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
            } else {
              uint32_t errorWord = dit->errorWord32();
              chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
            }

            // recover detector Id from chanNmbr and fedId
            CTPPSPixelFramePosition fPos(dit->fedId(), 0, chanNmbr, 0);  // frame position for ROC 0
            std::map<CTPPSPixelFramePosition, CTPPSPixelROCInfo>::const_iterator mit;
            int index = -1;
            int plane = -1;
            bool goodRecovery = false;
            mit = rocMapping.find(fPos);
            if (mit != rocMapping.end()) {
              CTPPSPixelROCInfo rocInfo = (*mit).second;
              CTPPSPixelDetId recoveredDetId(rocInfo.iD);
              plane = recoveredDetId.plane();
              index = getRPindex(recoveredDetId.arm(), recoveredDetId.station(), recoveredDetId.rp());
              if (RPindexValid[index] && !isPlanePlotsTurnedOff[recoveredDetId.arm()][recoveredDetId.station()]
                                                               [recoveredDetId.rp()][recoveredDetId.plane()])
                goodRecovery = true;
            }  // if (mit != rocMapping.end())

            if (goodRecovery) {
              h2ErrorCodeRP[index]->Fill(dit->errorType(), plane);
            } else {
              h2ErrorCodeUnidDet->Fill(dit->errorType(), dit->fedId());
            }

            bool fillFED = false;
            int iFed = dit->fedId() - minFedNumber;
            if (iFed >= 0 && iFed < numberOfFeds)
              fillFED = true;

            if (dit->errorType() == 28) {  //error 28 = FIFO nearly full: identify FIFO and fill histogram
              int fullType = -1;
              uint32_t errorWord = dit->errorWord32();
              int NFa = (errorWord >> DB0_shift) & DataBit_mask;
              int NFb = (errorWord >> DB1_shift) & DataBit_mask;
              int NFc = (errorWord >> DB2_shift) & DataBit_mask;
              int NFd = (errorWord >> DB3_shift) & DataBit_mask;
              int NFe = (errorWord >> DB4_shift) & DataBit_mask;
              int NF2 = (errorWord >> DB6_shift) & DataBit_mask;
              int L1A = (errorWord >> DB7_shift) & DataBit_mask;
              if (NFa == 1) {
                fullType = 1;
                h2FullType->Fill((int)fullType, dit->fedId());
              }
              if (NFb == 1) {
                fullType = 2;
                h2FullType->Fill((int)fullType, dit->fedId());
              }
              if (NFc == 1) {
                fullType = 3;
                h2FullType->Fill((int)fullType, dit->fedId());
              }
              if (NFd == 1) {
                fullType = 4;
                h2FullType->Fill((int)fullType, dit->fedId());
              }
              if (NFe == 1) {
                fullType = 5;
                h2FullType->Fill((int)fullType, dit->fedId());
              }
              if (NF2 == 1) {
                fullType = 6;
                h2FullType->Fill((int)fullType, dit->fedId());
              }
              if (L1A == 1) {
                fullType = 7;
                h2FullType->Fill((int)fullType, dit->fedId());
              }
            }
            if (dit->errorType() == 30) {  //error 30 = TBM error trailer
              uint32_t errorWord = dit->errorWord32();
              int t0 = (errorWord >> DB0_shift) & DataBit_mask;
              int t1 = (errorWord >> DB1_shift) & DataBit_mask;
              int t2 = (errorWord >> DB2_shift) & DataBit_mask;
              int t3 = (errorWord >> DB3_shift) & DataBit_mask;
              int t4 = (errorWord >> DB4_shift) & DataBit_mask;
              int t5 = (errorWord >> DB5_shift) & DataBit_mask;
              int t6 = (errorWord >> DB6_shift) & DataBit_mask;
              int t7 = (errorWord >> DB7_shift) & DataBit_mask;
              if (t0 == 1) {
                if (goodRecovery) {
                  h2TBMMessageRP[index]->Fill(0, plane);
                } else {
                  h2TBMMessageUnidDet->Fill(0, dit->fedId());
                }
                if (fillFED)
                  h2TBMMessageFED[iFed]->Fill(0, chanNmbr);
              }
              if (t1 == 1) {
                if (goodRecovery) {
                  h2TBMMessageRP[index]->Fill(1, plane);
                } else {
                  h2TBMMessageUnidDet->Fill(1, dit->fedId());
                }
                if (fillFED)
                  h2TBMMessageFED[iFed]->Fill(1, chanNmbr);
              }
              if (t2 == 1) {
                if (goodRecovery) {
                  h2TBMMessageRP[index]->Fill(2, plane);
                } else {
                  h2TBMMessageUnidDet->Fill(2, dit->fedId());
                }
                if (fillFED)
                  h2TBMMessageFED[iFed]->Fill(2, chanNmbr);
              }
              if (t3 == 1) {
                if (goodRecovery) {
                  h2TBMMessageRP[index]->Fill(3, plane);
                } else {
                  h2TBMMessageUnidDet->Fill(3, dit->fedId());
                }
                if (fillFED)
                  h2TBMMessageFED[iFed]->Fill(3, chanNmbr);
              }
              if (t4 == 1) {
                if (goodRecovery) {
                  h2TBMMessageRP[index]->Fill(4, plane);
                } else {
                  h2TBMMessageUnidDet->Fill(4, dit->fedId());
                }
                if (fillFED)
                  h2TBMMessageFED[iFed]->Fill(4, chanNmbr);
              }
              if (t5 == 1) {
                if (goodRecovery) {
                  h2TBMMessageRP[index]->Fill(5, plane);
                } else {
                  h2TBMMessageUnidDet->Fill(5, dit->fedId());
                }
                if (fillFED)
                  h2TBMMessageFED[iFed]->Fill(5, chanNmbr);
              }
              if (t6 == 1) {
                if (goodRecovery) {
                  h2TBMMessageRP[index]->Fill(6, plane);
                } else {
                  h2TBMMessageUnidDet->Fill(6, dit->fedId());
                }
                if (fillFED)
                  h2TBMMessageFED[iFed]->Fill(6, chanNmbr);
              }
              if (t7 == 1) {
                if (goodRecovery) {
                  h2TBMMessageRP[index]->Fill(7, plane);
                } else {
                  h2TBMMessageUnidDet->Fill(7, dit->fedId());
                }
                if (fillFED)
                  h2TBMMessageFED[iFed]->Fill(7, chanNmbr);
              }
              int stateMach_bits = 4;
              int stateMach_shift = 8;
              uint32_t stateMach_mask = ~(~uint32_t(0) << stateMach_bits);
              uint32_t stateMach = (errorWord >> stateMach_shift) & stateMach_mask;
              if (stateMach == 0) {
                if (goodRecovery) {
                  h2TBMTypeRP[index]->Fill(0, plane);
                } else {
                  h2TBMTypeUnidDet->Fill(0, dit->fedId());
                }
                if (fillFED)
                  h2TBMTypeFED[iFed]->Fill(0, chanNmbr);
              } else {
                if (((stateMach >> DB0_shift) & DataBit_mask) == 1) {
                  if (goodRecovery) {
                    h2TBMTypeRP[index]->Fill(1, plane);
                  } else {
                    h2TBMTypeUnidDet->Fill(1, dit->fedId());
                  }
                  if (fillFED)
                    h2TBMTypeFED[iFed]->Fill(1, chanNmbr);
                }
                if (((stateMach >> DB1_shift) & DataBit_mask) == 1) {
                  if (goodRecovery) {
                    h2TBMTypeRP[index]->Fill(2, plane);
                  } else {
                    h2TBMTypeUnidDet->Fill(2, dit->fedId());
                  }
                  if (fillFED)
                    h2TBMTypeFED[iFed]->Fill(2, chanNmbr);
                }
                if (((stateMach >> DB2_shift) & DataBit_mask) == 1) {
                  if (goodRecovery) {
                    h2TBMTypeRP[index]->Fill(3, plane);
                  } else {
                    h2TBMTypeUnidDet->Fill(3, dit->fedId());
                  }
                  if (fillFED)
                    h2TBMTypeFED[iFed]->Fill(3, chanNmbr);
                }
                if (((stateMach >> DB3_shift) & DataBit_mask) == 1) {
                  if (goodRecovery) {
                    h2TBMTypeRP[index]->Fill(4, plane);
                  } else {
                    h2TBMTypeUnidDet->Fill(4, dit->fedId());
                  }
                  if (fillFED)
                    h2TBMTypeFED[iFed]->Fill(4, chanNmbr);
                }
              }
            }
            if (fillFED)
              h2ErrorCodeFED[iFed]->Fill(dit->errorType(), chanNmbr);
          }
          continue;
        }  // if idet == 15
        if (verbosity > 1)
          LogPrint("CTPPSPixelDQMSource") << "not CTPPS: ds_error.id" << ds_error.id;
        continue;
      }  // end of dummy detId block

      int plane = getPixPlane(ds_error.id);
      CTPPSDetId theId(ds_error.id);
      int arm = theId.arm() & 0x1;
      int station = theId.station() & 0x3;
      int rpot = theId.rp() & 0x7;
      int rpInd = getRPindex(arm, station, rpot);
      RPactivity[rpInd] = 1;
      CTPPSPixelDetId IDD(ds_error.id);

      if (StationStatus[station] && RPstatus[station][rpot]) {
        int index = getRPindex(arm, station, rpot);
        for (DetSet<CTPPSPixelDataError>::const_iterator dit = ds_error.begin(); dit != ds_error.end(); ++dit) {
          if (RPindexValid[index]) {
            if (!isPlanePlotsTurnedOff[arm][station][rpot][plane]) {
              h2ErrorCodeRP[index]->Fill(dit->errorType(), plane);
              // recover fed channel number
              int chanNmbr = -1;
              if (dit->errorType() == 32 || dit->errorType() == 33 || dit->errorType() == 34) {
                long long errorWord = dit->errorWord64();  // for 64-bit error words
                chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
              } else {
                uint32_t errorWord = dit->errorWord32();
                chanNmbr = (errorWord >> LINK_shift) & LINK_mask;
              }
              bool fillFED = false;
              int iFed = dit->fedId() - minFedNumber;
              if (iFed >= 0 && iFed < numberOfFeds)
                fillFED = true;
              if (dit->errorType() == 28) {  //error 28 = FIFO nearly full: identify FIFO and fill histogram
                int fullType = -1;
                uint32_t errorWord = dit->errorWord32();
                int NFa = (errorWord >> DB0_shift) & DataBit_mask;
                int NFb = (errorWord >> DB1_shift) & DataBit_mask;
                int NFc = (errorWord >> DB2_shift) & DataBit_mask;
                int NFd = (errorWord >> DB3_shift) & DataBit_mask;
                int NFe = (errorWord >> DB4_shift) & DataBit_mask;
                int NF2 = (errorWord >> DB6_shift) & DataBit_mask;
                int L1A = (errorWord >> DB7_shift) & DataBit_mask;
                if (NFa == 1) {
                  fullType = 1;
                  h2FullType->Fill((int)fullType, dit->fedId());
                }
                if (NFb == 1) {
                  fullType = 2;
                  h2FullType->Fill((int)fullType, dit->fedId());
                }
                if (NFc == 1) {
                  fullType = 3;
                  h2FullType->Fill((int)fullType, dit->fedId());
                }
                if (NFd == 1) {
                  fullType = 4;
                  h2FullType->Fill((int)fullType, dit->fedId());
                }
                if (NFe == 1) {
                  fullType = 5;
                  h2FullType->Fill((int)fullType, dit->fedId());
                }
                if (NF2 == 1) {
                  fullType = 6;
                  h2FullType->Fill((int)fullType, dit->fedId());
                }
                if (L1A == 1) {
                  fullType = 7;
                  h2FullType->Fill((int)fullType, dit->fedId());
                }
              }

              if (dit->errorType() == 30) {  //error 30 = TBM error trailer
                uint32_t errorWord = dit->errorWord32();
                int t0 = (errorWord >> DB0_shift) & DataBit_mask;
                int t1 = (errorWord >> DB1_shift) & DataBit_mask;
                int t2 = (errorWord >> DB2_shift) & DataBit_mask;
                int t3 = (errorWord >> DB3_shift) & DataBit_mask;
                int t4 = (errorWord >> DB4_shift) & DataBit_mask;
                int t5 = (errorWord >> DB5_shift) & DataBit_mask;
                int t6 = (errorWord >> DB6_shift) & DataBit_mask;
                int t7 = (errorWord >> DB7_shift) & DataBit_mask;
                if (t0 == 1) {
                  h2TBMMessageRP[index]->Fill(0, plane);
                  if (fillFED)
                    h2TBMMessageFED[iFed]->Fill(0, chanNmbr);
                }
                if (t1 == 1) {
                  h2TBMMessageRP[index]->Fill(1, plane);
                  if (fillFED)
                    h2TBMMessageFED[iFed]->Fill(1, chanNmbr);
                }
                if (t2 == 1) {
                  h2TBMMessageRP[index]->Fill(2, plane);
                  if (fillFED)
                    h2TBMMessageFED[iFed]->Fill(2, chanNmbr);
                }
                if (t3 == 1) {
                  h2TBMMessageRP[index]->Fill(3, plane);
                  if (fillFED)
                    h2TBMMessageFED[iFed]->Fill(3, chanNmbr);
                }
                if (t4 == 1) {
                  h2TBMMessageRP[index]->Fill(4, plane);
                  if (fillFED)
                    h2TBMMessageFED[iFed]->Fill(4, chanNmbr);
                }
                if (t5 == 1) {
                  h2TBMMessageRP[index]->Fill(5, plane);
                  if (fillFED)
                    h2TBMMessageFED[iFed]->Fill(5, chanNmbr);
                }
                if (t6 == 1) {
                  h2TBMMessageRP[index]->Fill(6, plane);
                  if (fillFED)
                    h2TBMMessageFED[iFed]->Fill(6, chanNmbr);
                }
                if (t7 == 1) {
                  h2TBMMessageRP[index]->Fill(7, plane);
                  if (fillFED)
                    h2TBMMessageFED[iFed]->Fill(7, chanNmbr);
                }
                int stateMach_bits = 4;
                int stateMach_shift = 8;
                uint32_t stateMach_mask = ~(~uint32_t(0) << stateMach_bits);
                uint32_t stateMach = (errorWord >> stateMach_shift) & stateMach_mask;
                if (stateMach == 0) {
                  h2TBMTypeRP[index]->Fill(0, plane);
                  if (fillFED)
                    h2TBMTypeFED[iFed]->Fill(0, chanNmbr);
                } else {
                  if (((stateMach >> DB0_shift) & DataBit_mask) == 1) {
                    h2TBMTypeRP[index]->Fill(1, plane);
                    if (fillFED)
                      h2TBMTypeFED[iFed]->Fill(1, chanNmbr);
                  }
                  if (((stateMach >> DB1_shift) & DataBit_mask) == 1) {
                    h2TBMTypeRP[index]->Fill(2, plane);
                    if (fillFED)
                      h2TBMTypeFED[iFed]->Fill(2, chanNmbr);
                  }
                  if (((stateMach >> DB2_shift) & DataBit_mask) == 1) {
                    h2TBMTypeRP[index]->Fill(3, plane);
                    if (fillFED)
                      h2TBMTypeFED[iFed]->Fill(3, chanNmbr);
                  }
                  if (((stateMach >> DB3_shift) & DataBit_mask) == 1) {
                    h2TBMTypeRP[index]->Fill(4, plane);
                    if (fillFED)
                      h2TBMTypeFED[iFed]->Fill(4, chanNmbr);
                  }
                }
              }
              if (fillFED)
                h2ErrorCodeFED[iFed]->Fill(dit->errorType(), chanNmbr);
            }
          }  // end if(RPindexValid[index]) {
        }
      }  // end  if(StationStatus[station]) {
    }    // end for(const auto &ds_error : *pixDigi)
  }      // if(pixError.isValid())

  if (pixClus.isValid() && onlinePlots)
    for (const auto &ds : *pixClus) {
      int idet = getDet(ds.id);
      if (idet != DetId::VeryForward && verbosity > 1) {
        LogPrint("CTPPSPixelDQMSource") << "not CTPPS: cluster.id" << ds.id;
        continue;
      }

      CTPPSDetId theId(ds.id);
      int plane = getPixPlane(ds.id);
      int arm = theId.arm() & 0x1;
      int station = theId.station() & 0x3;
      int rpot = theId.rp() & 0x7;

      if ((StationStatus[station] == 0) || (RPstatus[station][rpot] == 0))
        continue;

      for (const auto &p : ds) {
        int clusize = p.size();

        if (clusize > ClusterSizeMax)
          clusize = ClusterSizeMax;

        h2CluSize[arm][station]->Fill(prIndex(rpot, plane), clusize);
      }
    }  // end if(pixClus.isValid()) for(const auto &ds : *pixClus)

  bool allRPactivity = false;
  for (int rp = 0; rp < RPotsTotalNumber; rp++)
    if (RPactivity[rp] > 0)
      allRPactivity = true;
  for (int arm = 0; arm < 2; arm++) {
    for (int stn = 0; stn < NStationMAX; stn++) {
      for (int rp = 0; rp < NRPotsMAX; rp++) {
        int index = getRPindex(arm, stn, rp);
        if (RPindexValid[index] == 0)
          continue;

        if (onlinePlots)
          hpRPactive->Fill(getRPglobalBin(arm, stn), RPactivity[index]);
        //        if(RPactivity[index]==0) continue;
        if (!allRPactivity)
          continue;
        if (onlinePlots)
          hpixLTrack->Fill(getRPglobalBin(arm, stn), pixRPTracks[index]);
        int ntr = pixRPTracks[index];
        if (ntr > NLocalTracksMAX)
          ntr = NLocalTracksMAX;
        for (int i = 0; i <= NLocalTracksMAX; i++) {
          if (i == ntr)
            htrackMult[index]->Fill(ntr, 1.);
          else
            htrackMult[index]->Fill(i, 0.);
        }

        int np = 0;
        for (int p = 0; p < NplaneMAX; p++)
          if (HitsMultPlane[index][p] > 0)
            np++;
        for (int p = 0; p <= NplaneMAX; p++) {
          if (p == np)
            hRPotActivPlanes[index]->Fill(p, 1.);
          else
            hRPotActivPlanes[index]->Fill(p, 0.);
        }
        if (onlinePlots) {
          if (np >= 5)
            hRPotActivBX[index]->Fill(event.bunchCrossing());
          hRPotActivBXall[index]->Fill(event.bunchCrossing(), float(RPdigiSize[index]));
        }

        // Select only events from the desired random trigger and fill the histogram
        const edm::TriggerNames &trigNames = event.triggerNames(*hltResults);
        for (int p = 0; p < NplaneMAX; p++) {
          for (unsigned int i = 0; i < trigNames.size(); i++) {
            const std::string &triggerName = trigNames.triggerName(i);
            if ((hltResults->accept(i) > 0) && (triggerName == randomHLTPath))
              h2HitsVsBXRandoms[index]->Fill(event.bunchCrossing(), p, HitsMultPlane[index][p]);
          }
        }

        int planesFiredAtROC[NROCsMAX];  // how many planes registered hits on ROC r
        for (int r = 0; r < NROCsMAX; r++)
          planesFiredAtROC[r] = 0;
        for (int p = 0; p < NplaneMAX; p++) {
          int indp = getPlaneIndex(arm, stn, rp, p);
          for (int r = 0; r < NROCsMAX; r++)
            if (HitsMultROC[indp][r] > 0)
              ++planesFiredAtROC[r];
          for (int r = 0; r < NROCsMAX; r++) {
            if (onlinePlots)
              h2HitsMultROC[index]->Fill(p, r, HitsMultROC[indp][r]);
            hp2HitsMultROC_LS[index]->Fill(lumiId, p * NROCsMAX + r, HitsMultROC[indp][r]);
          }
        }
        int max = 0;
        for (int r = 0; r < NROCsMAX; r++)
          if (max < planesFiredAtROC[r])
            max = planesFiredAtROC[r];
        if (max >= 4 && onlinePlots)  // fill only if there are at least 4 aligned ROCs firing
          hRPotActivBXroc[index]->Fill(event.bunchCrossing());
        if (max >= 3 && onlinePlots)  // fill only if there are at least 3 aligned ROCs firing
          hRPotActivBXroc_3[index]->Fill(event.bunchCrossing());
        if (max >= 2 && onlinePlots)  // fill only if there are at least 2 aligned ROCs firing
          hRPotActivBXroc_2[index]->Fill(event.bunchCrossing());
      }  // end for(int rp=0; rp<NRPotsMAX; rp++) {
    }
  }  // end for(int arm=0; arm<2; arm++) {

  if ((nEvents % 100))
    return;
  if (verbosity)
    LogPrint("CTPPSPixelDQMSource") << "analyze event " << nEvents;
}

//---------------------------------------------------------------------------
DEFINE_FWK_MODULE(CTPPSPixelDQMSource);
