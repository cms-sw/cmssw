/******************************************
 *
 * This is a part of CTPPSDQM software.
 * Authors:
 *   A. Bellora (Universita' e INFN Torino)
 *
 *******************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

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

#include <string>

//-----------------------------------------------------------------------------

class CTPPSRandomDQMSource : public DQMEDAnalyzer {
public:
  CTPPSRandomDQMSource(const edm::ParameterSet &ps);
  ~CTPPSRandomDQMSource() {};

protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

private:
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDigi>> tokenDigi;

  static constexpr int NArms = 2;
  static constexpr int NStationMAX = 3;  // in an arm
  static constexpr int NRPotsMAX = 6;    // per station
  static constexpr int NplaneMAX = 6;    // per RPot
  static constexpr int RPn_first = 3, RPn_last = 4;
  static constexpr int StationIDMAX = 4;  // possible range of ID
  static constexpr int RPotsIDMAX = 8;    // possible range of ID

  unsigned int rpStatusWord = 0x8008;      // 220_fr_hr(stn2rp3)+ 210_fr_hr
  int RPstatus[StationIDMAX][RPotsIDMAX];  // symmetric in both arms
  int StationStatus[StationIDMAX];         // symmetric in both arms
  const int IndexNotValid = 0;

  MonitorElement *hBX;

  static constexpr int RPotsTotalNumber = NArms * NStationMAX * NRPotsMAX;

  int RPindexValid[RPotsTotalNumber];
  MonitorElement *h2HitsVsBXRandoms[RPotsTotalNumber];

  int getRPindex(int arm, int station, int rp) {
    if (arm < 0 || station < 0 || rp < 0)
      return (IndexNotValid);
    if (arm > 1 || station >= NStationMAX || rp >= NRPotsMAX)
      return (IndexNotValid);
    int rc = (arm * NStationMAX + station) * NRPotsMAX + rp;
    return (rc);
  }

};

//----------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//-------------------------------------------------------------------------------

CTPPSRandomDQMSource::CTPPSRandomDQMSource(const edm::ParameterSet &ps)
  : rpStatusWord(ps.getUntrackedParameter<unsigned int>("RPStatusWord", 0x8008)){
  tokenDigi = consumes<DetSetVector<CTPPSPixelDigi>>(ps.getUntrackedParameter<edm::InputTag>("tagRPixDigi"));
}

//--------------------------------------------------------------------------

void CTPPSRandomDQMSource::dqmBeginRun(edm::Run const &run, edm::EventSetup const &) {

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

  for (int index = 0; index < 2 * 3 * NRPotsMAX; index++)
    RPindexValid[index] = 0;

}

//-------------------------------------------------------------------------------------

void CTPPSRandomDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {

  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS/TrackingPixel");

  hBX = ibooker.book1D("events per BX", "ctpps_pixel;Event.BX", 4002, -1.5, 4000. + 0.5);
  
  for (int arm = 0; arm < NArms; arm++) {
    CTPPSDetId ID(CTPPSDetId::sdTrackingPixel, arm, 0);
    string sd;
    ID.armName(sd, CTPPSDetId::nPath);

    ibooker.setCurrentFolder(sd);

    for (int stn = 0; stn < NStationMAX; stn++) {
      if (StationStatus[stn] == 0)
        continue;
      ID.setStation(stn);
      string stnd;
      CTPPSDetId(ID.stationId()).stationName(stnd, CTPPSDetId::nPath);

      ibooker.setCurrentFolder(stnd);

      for (int rp = RPn_first; rp < RPn_last; rp++) {  // only installed pixel pots
        ID.setRP(rp);
        string rpd, rpTitle;
        CTPPSDetId(ID.rpId()).rpName(rpTitle, CTPPSDetId::nFull);
        CTPPSDetId(ID.rpId()).rpName(rpd, CTPPSDetId::nPath);

        ibooker.setCurrentFolder(rpd);
    
        int indexP = getRPindex(arm, stn, rp);
        RPindexValid[indexP] = 1;

        h2HitsVsBXRandoms[indexP] = ibooker.book2D(
            "Digi per plane per BX - random triggers", rpTitle + ";Event.BX;Plane", 4002, -1.5, 4000. + 0.5, NplaneMAX, 0, NplaneMAX);

      }  // end for(int rp=0; rp<NRPotsMAX;...
    }    // end of for(int stn=0; stn<
  }      // end of for(int arm=0; arm<2;...

  return;
}

//-------------------------------------------------------------------------------

void CTPPSRandomDQMSource::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {

  Handle<DetSetVector<CTPPSPixelDigi>> pixDigi;
  event.getByToken(tokenDigi, pixDigi);

  cout << "Before checking digi" << endl;
  if (!pixDigi.isValid())
    return;
  cout << "After checking digi" << endl;
  
  hBX->Fill(event.bunchCrossing());

  for (int arm = 0; arm < 2; arm++) {
    for (int stn = 0; stn < NStationMAX; stn++) {
      if (!StationStatus[stn])
        continue;
      for (int rp = 0; rp < NRPotsMAX; rp++) {
        if (!RPstatus[stn][rp])
          continue;        
        int index = getRPindex(arm, stn, rp);
        if (RPindexValid[index] == 0)
          continue;

        for (int p = 0; p < NplaneMAX; p++) {
          CTPPSPixelDetId planeId(arm,stn,rp,p);
          if((*pixDigi).find(planeId.rawId()) != (*pixDigi).end()){
            int n_digis = (*pixDigi)[planeId.rawId()].size();           
            h2HitsVsBXRandoms[index]->Fill(event.bunchCrossing(), p, n_digis);
          }
        }
      }  // end for (int rp=0; rp<NRPotsMAX; rp++) {
    }    // end for (int stn = 0; stn < NStationMAX; stn++) {
  }      // end for (int arm=0; arm<2; arm++) {

}

//---------------------------------------------------------------------------
DEFINE_FWK_MODULE(CTPPSRandomDQMSource);
