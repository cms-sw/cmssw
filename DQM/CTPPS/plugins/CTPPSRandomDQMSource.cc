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
  ~CTPPSRandomDQMSource() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

private:
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelDigi>> const tokenDigi_;

  static constexpr int kNArms_ = 2;
  static constexpr int kNStationMAX_ = 3;  // in an arm
  static constexpr int kNRPotsMAX_ = 6;    // per station
  static constexpr int kNplaneMAX_ = 6;    // per RPot
  static constexpr int kFirstRPn_ = 3, kLastRPn_ = 4;
  static constexpr int kStationIDMAX_ = 4;  // possible range of ID
  static constexpr int kRPotsIDMAX_ = 8;    // possible range of ID

  const std::string folderName_ = "PPSRANDOM/RandomPixel";

  unsigned int rpStatusWord_ = 0x8008;          // 220_fr_hr(stn2rp3)+ 210_fr_hr
  int rpStatus_[kStationIDMAX_][kRPotsIDMAX_];  // symmetric in both arms
  int stationStatus_[kStationIDMAX_];           // symmetric in both arms
  const int kIndexNotValid = 0;

  MonitorElement *hBX_;

  static constexpr int kRPotsTotalNumber_ = kNArms_ * kNStationMAX_ * kNRPotsMAX_;

  int RPindexValid_[kRPotsTotalNumber_];
  MonitorElement *h2HitsVsBXRandoms_[kRPotsTotalNumber_];

  int getRPindex(int arm, int station, int rp) const {
    if (arm < 0 || station < 0 || rp < 0)
      return (kIndexNotValid);
    if (arm > 1 || station >= kNStationMAX_ || rp >= kNRPotsMAX_)
      return (kIndexNotValid);
    int rc = (arm * kNStationMAX_ + station) * kNRPotsMAX_ + rp;
    return (rc);
  }
};

//-------------------------------------------------------------------------------

CTPPSRandomDQMSource::CTPPSRandomDQMSource(const edm::ParameterSet &ps)
    : tokenDigi_(consumes<edm::DetSetVector<CTPPSPixelDigi>>(ps.getParameter<edm::InputTag>("tagRPixDigi"))),
      folderName_(ps.getUntrackedParameter<std::string>("folderName", "PPSRANDOM/RandomPixel")),
      rpStatusWord_(ps.getUntrackedParameter<unsigned int>("RPStatusWord", 0x8008)) {
  for (int stn = 0; stn < kStationIDMAX_; stn++) {
    stationStatus_[stn] = 0;
    for (int rp = 0; rp < kRPotsIDMAX_; rp++)
      rpStatus_[stn][rp] = 0;
  }

  unsigned int rpSts = rpStatusWord_ << 1;
  for (int stn = 0; stn < kNStationMAX_; stn++) {
    int stns = 0;
    for (int rp = 0; rp < kNRPotsMAX_; rp++) {
      rpSts = (rpSts >> 1);
      rpStatus_[stn][rp] = rpSts & 1;
      if (rpStatus_[stn][rp] > 0)
        stns = 1;
    }
    stationStatus_[stn] = stns;
  }

  for (int index = 0; index < 2 * 3 * kNRPotsMAX_; index++)
    RPindexValid_[index] = 0;
}

//--------------------------------------------------------------------------

void CTPPSRandomDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {
  ibooker.cd();
  ibooker.setCurrentFolder(folderName_);

  hBX_ = ibooker.book1D("events per BX", "ctpps_pixel;Event.BX", 4002, -1.5, 4000. + 0.5);

  for (int arm = 0; arm < kNArms_; arm++) {
    CTPPSDetId ID(CTPPSDetId::sdTrackingPixel, arm, 0);
    std::string sd;
    ID.armName(sd, CTPPSDetId::nShort);
    sd = folderName_ + "/sector " + sd;

    ibooker.setCurrentFolder(sd);

    for (int stn = 0; stn < kNStationMAX_; stn++) {
      if (stationStatus_[stn] == 0)
        continue;
      ID.setStation(stn);
      std::string stnd;
      CTPPSDetId(ID.stationId()).stationName(stnd, CTPPSDetId::nShort);
      stnd = sd + "/station " + stnd;

      ibooker.setCurrentFolder(stnd);

      for (int rp = kFirstRPn_; rp < kLastRPn_; rp++) {  // only installed pixel pots
        ID.setRP(rp);
        std::string rpd, rpTitle;
        CTPPSDetId(ID.rpId()).rpName(rpTitle, CTPPSDetId::nFull);
        CTPPSDetId(ID.rpId()).rpName(rpd, CTPPSDetId::nShort);
        rpd = stnd + "/" + rpd;

        ibooker.setCurrentFolder(rpd);

        int indexP = getRPindex(arm, stn, rp);
        RPindexValid_[indexP] = 1;

        h2HitsVsBXRandoms_[indexP] = ibooker.book2D("Digi per plane per BX - random triggers",
                                                    rpTitle + ";Event.BX;Plane",
                                                    4002,
                                                    -1.5,
                                                    4000. + 0.5,
                                                    kNplaneMAX_,
                                                    0,
                                                    kNplaneMAX_);
        h2HitsVsBXRandoms_[indexP]->getTH2F()->SetOption("colz");

      }  // end for(int rp=0; rp<kNRPotsMAX_;...
    }    // end of for(int stn=0; stn<
  }      // end of for(int arm=0; arm<2;...

  return;
}

//-------------------------------------------------------------------------------

void CTPPSRandomDQMSource::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {
  auto const pixDigi = event.getHandle(tokenDigi_);

  if (!pixDigi.isValid())
    return;

  hBX_->Fill(event.bunchCrossing());

  for (int arm = 0; arm < 2; arm++) {
    for (int stn = 0; stn < kNStationMAX_; stn++) {
      if (!stationStatus_[stn])
        continue;
      for (int rp = 0; rp < kNRPotsMAX_; rp++) {
        if (!rpStatus_[stn][rp])
          continue;
        int index = getRPindex(arm, stn, rp);
        if (RPindexValid_[index] == 0)
          continue;

        for (int p = 0; p < kNplaneMAX_; p++) {
          CTPPSPixelDetId planeId(arm, stn, rp, p);
          auto pix_d = pixDigi->find(planeId.rawId());
          if (pix_d != pixDigi->end()) {
            int n_digis = pix_d->size();
            h2HitsVsBXRandoms_[index]->Fill(event.bunchCrossing(), p, n_digis);
          }
        }
      }  // end for (int rp=0; rp<kNRPotsMAX_; rp++) {
    }    // end for (int stn = 0; stn < kNStationMAX_; stn++) {
  }      // end for (int arm=0; arm<2; arm++) {
}

//---------------------------------------------------------------------------

void CTPPSRandomDQMSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tagRPixDigi", edm::InputTag("ctppsPixelDigisAlCaRecoProducer"));
  desc.addUntracked<std::string>("folderName", "PPSRANDOM/RandomPixel");
  desc.addUntracked<unsigned int>("RPStatusWord", 0x8008);
  descriptions.add("ctppsRandomDQMSource", desc);
}

//---------------------------------------------------------------------------
DEFINE_FWK_MODULE(CTPPSRandomDQMSource);
