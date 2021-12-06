#include <string>

#include "DQM/L1TMonitor/interface/L1TdeCSCTPGShower.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"

L1TdeCSCTPGShower::L1TdeCSCTPGShower(const edm::ParameterSet& ps)
    : dataALCTShower_token_(consumes<CSCShowerDigiCollection>(ps.getParameter<edm::InputTag>("dataALCTShower"))),
      emulALCTShower_token_(consumes<CSCShowerDigiCollection>(ps.getParameter<edm::InputTag>("emulALCTShower"))),
      dataCLCTShower_token_(consumes<CSCShowerDigiCollection>(ps.getParameter<edm::InputTag>("dataCLCTShower"))),
      emulCLCTShower_token_(consumes<CSCShowerDigiCollection>(ps.getParameter<edm::InputTag>("emulCLCTShower"))),
      dataLCTShower_token_(consumes<CSCShowerDigiCollection>(ps.getParameter<edm::InputTag>("dataLCTShower"))),
      emulLCTShower_token_(consumes<CSCShowerDigiCollection>(ps.getParameter<edm::InputTag>("emulLCTShower"))),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")) {}



L1TdeCSCTPGShower::~L1TdeCSCTPGShower() {}

void L1TdeCSCTPGShower::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run&, const edm::EventSetup&) {
  iBooker.setCurrentFolder(monitorDir_);

  // 2D summary plots
  lctShowerDataSummary_denom_ = iBooker.book2D("lct_cscshower_data_summary_denom", "Data LCT Shower All", 36, 1, 37, 18, 0, 18);
  lctShowerDataSummary_num_ = iBooker.book2D("lct_cscshower_data_summary_num", "Data LCT Shower Emul Matched", 36, 1, 37, 18, 0, 18);
  alctShowerDataSummary_denom_ = iBooker.book2D("alct_cscshower_data_summary_denom", "Data ALCT Shower All", 36, 1, 37, 18, 0, 18);
  alctShowerDataSummary_num_ = iBooker.book2D("alct_cscshower_data_summary_num", "Data ALCT Shower Emul Matched", 36, 1, 37, 18, 0, 18);
  clctShowerDataSummary_denom_ = iBooker.book2D("clct_cscshower_data_summary_denom", "Data CLCT Shower All", 36, 1, 37, 18, 0, 18);
  clctShowerDataSummary_num_ = iBooker.book2D("clct_cscshower_data_summary_num", "Data CLCT Shower Emul Matched", 36, 1, 37, 18, 0, 18);

  lctShowerEmulSummary_denom_ = iBooker.book2D("lct_cscshower_emul_summary_denom", "Emul LCT Shower All", 36, 1, 37, 18, 0, 18);
  lctShowerEmulSummary_num_ = iBooker.book2D("lct_cscshower_emul_summary_num", "Emul LCT Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);
  alctShowerEmulSummary_denom_ = iBooker.book2D("alct_cscshower_emul_summary_denom", "Emul ALCT Shower All", 36, 1, 37, 18, 0, 18);
  alctShowerEmulSummary_num_ = iBooker.book2D("alct_cscshower_emul_summary_num", "Emul ALCT Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);
  clctShowerEmulSummary_denom_ = iBooker.book2D("clct_cscshower_emul_summary_denom", "Emul CLCT Shower All", 36, 1, 37, 18, 0, 18);
  clctShowerEmulSummary_num_ = iBooker.book2D("clct_cscshower_emul_summary_num", "Emul CLCT Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);

  // x labels
  lctShowerDataSummary_denom_->setAxisTitle("Chamber", 1);
  lctShowerDataSummary_num_->setAxisTitle("Chamber", 1);
  alctShowerDataSummary_denom_->setAxisTitle("Chamber", 1);
  alctShowerDataSummary_num_->setAxisTitle("Chamber", 1);
  clctShowerDataSummary_denom_->setAxisTitle("Chamber", 1);
  clctShowerDataSummary_num_->setAxisTitle("Chamber", 1);

  lctShowerEmulSummary_denom_->setAxisTitle("Chamber", 1);
  lctShowerEmulSummary_num_->setAxisTitle("Chamber", 1);
  alctShowerEmulSummary_denom_->setAxisTitle("Chamber", 1);
  alctShowerEmulSummary_num_->setAxisTitle("Chamber", 1);
  clctShowerEmulSummary_denom_->setAxisTitle("Chamber", 1);
  clctShowerEmulSummary_num_->setAxisTitle("Chamber", 1);

  // plotting option
  lctShowerDataSummary_denom_->setOption("colz");
  lctShowerDataSummary_num_->setOption("colz");
  alctShowerDataSummary_denom_->setOption("colz");
  alctShowerDataSummary_num_->setOption("colz");
  clctShowerDataSummary_denom_->setOption("colz");
  clctShowerDataSummary_num_->setOption("colz");

  lctShowerEmulSummary_denom_->setOption("colz");
  lctShowerEmulSummary_num_->setOption("colz");
  alctShowerEmulSummary_denom_->setOption("colz");
  alctShowerEmulSummary_num_->setOption("colz");
  clctShowerEmulSummary_denom_->setOption("colz");
  clctShowerEmulSummary_num_->setOption("colz");

  const std::array<std::string, 9> suffix_label{{"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1"}};

  // y labels
  for (int ybin = 1; ybin <= 9; ++ybin) {
    lctShowerDataSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    lctShowerDataSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerEmulSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    lctShowerEmulSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerDataSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    lctShowerDataSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctShowerEmulSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    lctShowerEmulSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
  }
}

void L1TdeCSCTPGShower::analyze(const edm::Event& e, const edm::EventSetup& c) {
  // handles
  edm::Handle<CSCShowerDigiCollection> dataALCTshs;
  edm::Handle<CSCShowerDigiCollection> emulALCTshs;
  edm::Handle<CSCShowerDigiCollection> dataCLCTshs;
  edm::Handle<CSCShowerDigiCollection> emulCLCTshs;
  edm::Handle<CSCShowerDigiCollection> dataLCTshs;
  edm::Handle<CSCShowerDigiCollection> emulLCTshs;

  e.getByToken(dataALCTShower_token_, dataALCTshs);
  e.getByToken(emulALCTShower_token_, emulALCTshs);
  e.getByToken(dataCLCTShower_token_, dataCLCTshs);
  e.getByToken(emulCLCTShower_token_, emulCLCTshs);
  e.getByToken(dataLCTShower_token_, dataLCTshs);
  e.getByToken(emulLCTShower_token_, emulLCTshs);

  const std::map<std::pair<int, int>, int> histIndexCSC = {{{1, 1}, 8},
                                                           {{1, 2}, 7},
                                                           {{1, 3}, 6},
                                                           {{2, 1}, 5},
                                                           {{2, 2}, 4},
                                                           {{3, 1}, 3},
                                                           {{3, 2}, 2},
                                                           {{4, 1}, 1},
                                                           {{4, 2}, 0}};

  const int min_endcap = CSCDetId::minEndcapId();
  const int max_endcap = CSCDetId::maxEndcapId();
  const int min_station = CSCDetId::minStationId();
  const int max_station = CSCDetId::maxStationId();
  const int min_sector = CSCTriggerNumbering::minTriggerSectorId();
  const int max_sector = CSCTriggerNumbering::maxTriggerSectorId();
  const int min_subsector = CSCTriggerNumbering::minTriggerSubSectorId();
  const int max_subsector = CSCTriggerNumbering::maxTriggerSubSectorId();
  const int min_chamber = CSCTriggerNumbering::minTriggerCscId();
  const int max_chamber = CSCTriggerNumbering::maxTriggerCscId();

  for (int endc = min_endcap; endc <= max_endcap; endc++) {
    // loop on all stations
    for (int stat = min_station; stat <= max_station; stat++) {
      int numsubs = ((stat == 1) ? max_subsector : 1);
      // loop on sectors and subsectors
      for (int sect = min_sector; sect <= max_sector; sect++) {
        for (int subs = min_subsector; subs <= numsubs; subs++) {
          // loop on all chambers
          for (int cham = min_chamber; cham <= max_chamber; cham++) {
            // extract the ring number
            int ring = CSCTriggerNumbering::ringFromTriggerLabels(stat, cham);

            // actual chamber number =/= trigger chamber number
            int chid = CSCTriggerNumbering::chamberFromTriggerLabels(sect, subs, stat, cham);

            // 0th layer means whole chamber.
            CSCDetId detid(endc, stat, ring, chid, 0);

            int chamber = detid.chamber();

            int sr = histIndexCSC.at({stat, ring});
            if (endc == 1)
              sr = 17 - sr;
            bool chamber20 = (sr == 1 or sr == 3 or sr == 5 or sr == 12 or sr == 14 or sr == 16);

            // ALCT analysis
            auto range_dataALCT = dataALCTshs->get(detid);
            auto range_emulALCT = emulALCTshs->get(detid);

            for (auto dalct = range_dataALCT.first; dalct != range_dataALCT.second; dalct++) {
              if (dalct->isValid()) {
                if (chamber20) {
                  alctShowerDataSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  alctShowerDataSummary_denom_->Fill(chamber * 2, sr, 0.5);
                }
                else
                  alctShowerDataSummary_denom_->Fill(chamber, sr);
                // check for least one matching ALCT
                for (auto ealct = range_emulALCT.first; ealct != range_emulALCT.second; ealct++) {
                  if (ealct->isValid() and areSameShowers(*dalct, *ealct)) {
                    if (chamber20) {
                      alctShowerDataSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      alctShowerDataSummary_num_->Fill(chamber * 2, sr, 0.5);
                    }
                    else
                      alctShowerDataSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }

            for (auto ealct = range_emulALCT.first; ealct != range_emulALCT.second; ealct++) {
              bool isMatched = false;
              if (ealct->isValid()) {
                if (chamber20) {
                  alctShowerEmulSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  alctShowerEmulSummary_denom_->Fill(chamber * 2, sr, 0.5);
                }
                else
                  alctShowerEmulSummary_denom_->Fill(chamber, sr);
                // check for least one matching ALCT
                for (auto dalct = range_dataALCT.first; dalct != range_dataALCT.second; dalct++) {
                  if (areSameShowers(*dalct, *ealct))
                    isMatched = true;
                }
                // only fill when it is not matched to an ALCT
                // to understand if the emulator is producing too many ALCTs
                if (!isMatched) {
                  if (chamber20) {
                    alctShowerEmulSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                    alctShowerEmulSummary_num_->Fill(chamber * 2, sr, 0.5);
                  }
                  else
                    alctShowerEmulSummary_num_->Fill(chamber, sr);
                }
              }
            }

            // CLCT analysis
            auto range_dataCLCT = dataCLCTshs->get(detid);
            auto range_emulCLCT = emulCLCTshs->get(detid);

            for (auto dclct = range_dataCLCT.first; dclct != range_dataCLCT.second; dclct++) {
              if (dclct->isValid()) {
                if (chamber20) {
                  clctShowerDataSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  clctShowerDataSummary_denom_->Fill(chamber * 2, sr, 0.5);
                }
                else
                  clctShowerDataSummary_denom_->Fill(chamber, sr);
                // check for least one matching CLCT
                for (auto eclct = range_emulCLCT.first; eclct != range_emulCLCT.second; eclct++) {
                  if (eclct->isValid() and areSameShowers(*dclct, *eclct)) {
                    if (chamber20) {
                      clctShowerDataSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      clctShowerDataSummary_num_->Fill(chamber * 2, sr, 0.5);
                    }
                    else
                      clctShowerDataSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }

            for (auto eclct = range_emulCLCT.first; eclct != range_emulCLCT.second; eclct++) {
              bool isMatched = false;
              if (eclct->isValid()) {
                if (chamber20) {
                  clctShowerEmulSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  clctShowerEmulSummary_denom_->Fill(chamber * 2, sr, 0.5);
                }
                else
                  clctShowerEmulSummary_denom_->Fill(chamber, sr);
                // check for least one matching CLCT
                for (auto dclct = range_dataCLCT.first; dclct != range_dataCLCT.second; dclct++) {
                  if (areSameShowers(*dclct, *eclct))
                    isMatched = true;
                }
                // only fill when it is not matched to an CLCT
                // to understand if the emulator is producing too many CLCTs
                if (!isMatched) {
                  if (chamber20) {
                    clctShowerEmulSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                    clctShowerEmulSummary_num_->Fill(chamber * 2, sr, 0.5);
                  }
                  else
                    clctShowerEmulSummary_num_->Fill(chamber, sr);
                }
              }
            }

            // LCT analysis
            auto range_dataLCT = dataLCTshs->get(detid);
            auto range_emulLCT = emulLCTshs->get(detid);

            for (auto dlct = range_dataLCT.first; dlct != range_dataLCT.second; dlct++) {
              if (dlct->isValid()) {
                if (chamber20) {
                  lctShowerDataSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  lctShowerDataSummary_denom_->Fill(chamber * 2, sr, 0.5);
                }
                else
                  lctShowerDataSummary_denom_->Fill(chamber, sr);
                // check for least one matching LCT
                for (auto elct = range_emulLCT.first; elct != range_emulLCT.second; elct++) {
                  if (elct->isValid() and areSameShowers(*dlct, *elct)) {
                    if (chamber20) {
                      lctShowerDataSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      lctShowerDataSummary_num_->Fill(chamber * 2, sr, 0.5);
                    }
                    else
                      lctShowerDataSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }
            
            for (auto elct = range_emulLCT.first; elct != range_emulLCT.second; elct++) {
              bool isMatched = false;
              if (elct->isValid()) {
                if (chamber20) {
                  lctShowerEmulSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  lctShowerEmulSummary_denom_->Fill(chamber * 2, sr, 0.5);
                }
                else
                  lctShowerEmulSummary_denom_->Fill(chamber, sr);
                // check for least one matching LCT
                for (auto dlct = range_dataLCT.first; dlct != range_dataLCT.second; dlct++) {
                  if (areSameShowers(*dlct, *elct))
                    isMatched = true;
                }
                // only fill when it is not matched to an LCT
                // to understand if the emulator is producing too many LCTs
                if (!isMatched) {
                  if (chamber20) {
                    lctShowerEmulSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                    lctShowerEmulSummary_num_->Fill(chamber * 2, sr, 0.5);
                  }
                  else
                    lctShowerEmulSummary_num_->Fill(chamber, sr);
                }
              }
            }
          }
        }
      }
    }
  }
}

bool L1TdeCSCTPGShower::areSameShowers(const CSCShowerDigi& lhs, const CSCShowerDigi& rhs) const {
  bool returnValue = false;
  if (lhs.isValid() == rhs.isValid() && lhs.getCSCID() == rhs.getCSCID() &&
      lhs.bitsInTime() == rhs.bitsInTime() && lhs.bitsOutOfTime() == rhs.bitsOutOfTime()) {
    returnValue = true;
  }
  return returnValue;
}
