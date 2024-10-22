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
  lctShowerDataNomSummary_denom_ =
      iBooker.book2D("lct_cscshower_data_nom_summary_denom", "Data LCT Nominal Shower All", 36, 1, 37, 18, 0, 18);
  lctShowerDataNomSummary_num_ = iBooker.book2D(
      "lct_cscshower_data_nom_summary_num", "Data LCT Nominal Shower Emul Matched", 36, 1, 37, 18, 0, 18);
  alctShowerDataNomSummary_denom_ =
      iBooker.book2D("alct_cscshower_data_nom_summary_denom", "Data ALCT Nominal Shower All", 36, 1, 37, 18, 0, 18);
  alctShowerDataNomSummary_num_ = iBooker.book2D(
      "alct_cscshower_data_nom_summary_num", "Data ALCT Nominal Shower Emul Matched", 36, 1, 37, 18, 0, 18);
  clctShowerDataNomSummary_denom_ =
      iBooker.book2D("clct_cscshower_data_nom_summary_denom", "Data CLCT Nominal Shower All", 36, 1, 37, 18, 0, 18);
  clctShowerDataNomSummary_num_ = iBooker.book2D(
      "clct_cscshower_data_nom_summary_num", "Data CLCT Nominal Shower Emul Matched", 36, 1, 37, 18, 0, 18);

  lctShowerEmulNomSummary_denom_ =
      iBooker.book2D("lct_cscshower_emul_nom_summary_denom", "Emul LCT Nominal Shower All", 36, 1, 37, 18, 0, 18);
  lctShowerEmulNomSummary_num_ = iBooker.book2D(
      "lct_cscshower_emul_nom_summary_num", "Emul LCT Nominal Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);
  alctShowerEmulNomSummary_denom_ =
      iBooker.book2D("alct_cscshower_emul_nom_summary_denom", "Emul ALCT Nominal Shower All", 36, 1, 37, 18, 0, 18);
  alctShowerEmulNomSummary_num_ = iBooker.book2D(
      "alct_cscshower_emul_nom_summary_num", "Emul ALCT Nominal Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);
  clctShowerEmulNomSummary_denom_ =
      iBooker.book2D("clct_cscshower_emul_nom_summary_denom", "Emul CLCT Nominal Shower All", 36, 1, 37, 18, 0, 18);
  clctShowerEmulNomSummary_num_ = iBooker.book2D(
      "clct_cscshower_emul_nom_summary_num", "Emul CLCT Nominal Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);

  lctShowerDataTightSummary_denom_ =
      iBooker.book2D("lct_cscshower_data_tight_summary_denom", "Data LCT Tight Shower All", 36, 1, 37, 18, 0, 18);
  lctShowerDataTightSummary_num_ = iBooker.book2D(
      "lct_cscshower_data_tight_summary_num", "Data LCT Tight Shower Emul Matched", 36, 1, 37, 18, 0, 18);
  alctShowerDataTightSummary_denom_ =
      iBooker.book2D("alct_cscshower_data_tight_summary_denom", "Data ALCT Tight Shower All", 36, 1, 37, 18, 0, 18);
  alctShowerDataTightSummary_num_ = iBooker.book2D(
      "alct_cscshower_data_tight_summary_num", "Data ALCT Tight Shower Emul Matched", 36, 1, 37, 18, 0, 18);
  clctShowerDataTightSummary_denom_ =
      iBooker.book2D("clct_cscshower_data_tight_summary_denom", "Data CLCT Tight Shower All", 36, 1, 37, 18, 0, 18);
  clctShowerDataTightSummary_num_ = iBooker.book2D(
      "clct_cscshower_data_tight_summary_num", "Data CLCT Tight Shower Emul Matched", 36, 1, 37, 18, 0, 18);

  lctShowerEmulTightSummary_denom_ =
      iBooker.book2D("lct_cscshower_emul_tight_summary_denom", "Emul LCT Tight Shower All", 36, 1, 37, 18, 0, 18);
  lctShowerEmulTightSummary_num_ = iBooker.book2D(
      "lct_cscshower_emul_tight_summary_num", "Emul LCT Tight Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);
  alctShowerEmulTightSummary_denom_ =
      iBooker.book2D("alct_cscshower_emul_tight_summary_denom", "Emul ALCT Tight Shower All", 36, 1, 37, 18, 0, 18);
  alctShowerEmulTightSummary_num_ = iBooker.book2D(
      "alct_cscshower_emul_tight_summary_num", "Emul ALCT Tight Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);
  clctShowerEmulTightSummary_denom_ =
      iBooker.book2D("clct_cscshower_emul_tight_summary_denom", "Emul CLCT Tight Shower All", 36, 1, 37, 18, 0, 18);
  clctShowerEmulTightSummary_num_ = iBooker.book2D(
      "clct_cscshower_emul_tight_summary_num", "Emul CLCT Tight Shower Not Matched to Data", 36, 1, 37, 18, 0, 18);

  // x labels
  lctShowerDataNomSummary_denom_->setAxisTitle("Chamber", 1);
  lctShowerDataNomSummary_num_->setAxisTitle("Chamber", 1);
  alctShowerDataNomSummary_denom_->setAxisTitle("Chamber", 1);
  alctShowerDataNomSummary_num_->setAxisTitle("Chamber", 1);
  clctShowerDataNomSummary_denom_->setAxisTitle("Chamber", 1);
  clctShowerDataNomSummary_num_->setAxisTitle("Chamber", 1);

  lctShowerEmulNomSummary_denom_->setAxisTitle("Chamber", 1);
  lctShowerEmulNomSummary_num_->setAxisTitle("Chamber", 1);
  alctShowerEmulNomSummary_denom_->setAxisTitle("Chamber", 1);
  alctShowerEmulNomSummary_num_->setAxisTitle("Chamber", 1);
  clctShowerEmulNomSummary_denom_->setAxisTitle("Chamber", 1);
  clctShowerEmulNomSummary_num_->setAxisTitle("Chamber", 1);

  lctShowerDataTightSummary_denom_->setAxisTitle("Chamber", 1);
  lctShowerDataTightSummary_num_->setAxisTitle("Chamber", 1);
  alctShowerDataTightSummary_denom_->setAxisTitle("Chamber", 1);
  alctShowerDataTightSummary_num_->setAxisTitle("Chamber", 1);
  clctShowerDataTightSummary_denom_->setAxisTitle("Chamber", 1);
  clctShowerDataTightSummary_num_->setAxisTitle("Chamber", 1);

  lctShowerEmulTightSummary_denom_->setAxisTitle("Chamber", 1);
  lctShowerEmulTightSummary_num_->setAxisTitle("Chamber", 1);
  alctShowerEmulTightSummary_denom_->setAxisTitle("Chamber", 1);
  alctShowerEmulTightSummary_num_->setAxisTitle("Chamber", 1);
  clctShowerEmulTightSummary_denom_->setAxisTitle("Chamber", 1);
  clctShowerEmulTightSummary_num_->setAxisTitle("Chamber", 1);

  // plotting option
  lctShowerDataNomSummary_denom_->setOption("colz");
  lctShowerDataNomSummary_num_->setOption("colz");
  alctShowerDataNomSummary_denom_->setOption("colz");
  alctShowerDataNomSummary_num_->setOption("colz");
  clctShowerDataNomSummary_denom_->setOption("colz");
  clctShowerDataNomSummary_num_->setOption("colz");

  lctShowerEmulNomSummary_denom_->setOption("colz");
  lctShowerEmulNomSummary_num_->setOption("colz");
  alctShowerEmulNomSummary_denom_->setOption("colz");
  alctShowerEmulNomSummary_num_->setOption("colz");
  clctShowerEmulNomSummary_denom_->setOption("colz");
  clctShowerEmulNomSummary_num_->setOption("colz");

  lctShowerDataTightSummary_denom_->setOption("colz");
  lctShowerDataTightSummary_num_->setOption("colz");
  alctShowerDataTightSummary_denom_->setOption("colz");
  alctShowerDataTightSummary_num_->setOption("colz");
  clctShowerDataTightSummary_denom_->setOption("colz");
  clctShowerDataTightSummary_num_->setOption("colz");

  lctShowerEmulTightSummary_denom_->setOption("colz");
  lctShowerEmulTightSummary_num_->setOption("colz");
  alctShowerEmulTightSummary_denom_->setOption("colz");
  alctShowerEmulTightSummary_num_->setOption("colz");
  clctShowerEmulTightSummary_denom_->setOption("colz");
  clctShowerEmulTightSummary_num_->setOption("colz");

  const std::array<std::string, 9> suffix_label{{"4/2", "4/1", "3/2", "3/1", " 2/2", "2/1", "1/3", "1/2", "1/1"}};

  // y labels
  for (int ybin = 1; ybin <= 9; ++ybin) {
    lctShowerDataNomSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    lctShowerDataNomSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataNomSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataNomSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataNomSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataNomSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerEmulNomSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    lctShowerEmulNomSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulNomSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulNomSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulNomSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulNomSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerDataNomSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    lctShowerDataNomSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataNomSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataNomSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataNomSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataNomSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctShowerEmulNomSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    lctShowerEmulNomSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulNomSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulNomSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulNomSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulNomSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctShowerDataTightSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    lctShowerDataTightSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataTightSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerDataTightSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataTightSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerDataTightSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerEmulTightSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    lctShowerEmulTightSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulTightSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    alctShowerEmulTightSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulTightSummary_denom_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);
    clctShowerEmulTightSummary_num_->setBinLabel(ybin, "ME-" + suffix_label[ybin - 1], 2);

    lctShowerDataTightSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    lctShowerDataTightSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataTightSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerDataTightSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataTightSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerDataTightSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);

    lctShowerEmulTightSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    lctShowerEmulTightSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulTightSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    alctShowerEmulTightSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulTightSummary_denom_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
    clctShowerEmulTightSummary_num_->setBinLabel(19 - ybin, "ME+" + suffix_label[ybin - 1], 2);
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
              if (dalct->isValid() and dalct->isNominalInTime()) {
                if (dalct->isTightInTime()) {
                  if (chamber20) {
                    alctShowerDataTightSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                    alctShowerDataTightSummary_denom_->Fill(chamber * 2, sr, 0.5);
                  } else
                    alctShowerDataTightSummary_denom_->Fill(chamber, sr);
                }
                if (chamber20) {
                  alctShowerDataNomSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  alctShowerDataNomSummary_denom_->Fill(chamber * 2, sr, 0.5);
                } else
                  alctShowerDataNomSummary_denom_->Fill(chamber, sr);
                // check for least one matching ALCT
                for (auto ealct = range_emulALCT.first; ealct != range_emulALCT.second; ealct++) {
                  if (ealct->isValid() and areSameShowers(*dalct, *ealct)) {
                    if (dalct->isTightInTime()) {
                      if (chamber20) {
                        alctShowerDataTightSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                        alctShowerDataTightSummary_num_->Fill(chamber * 2, sr, 0.5);
                      } else
                        alctShowerDataTightSummary_num_->Fill(chamber, sr);
                    }
                    if (chamber20) {
                      alctShowerDataNomSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      alctShowerDataNomSummary_num_->Fill(chamber * 2, sr, 0.5);
                    } else
                      alctShowerDataNomSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }  // End of for (auto dalct = range_dataALCT.first; dalct != range_dataALCT.second; dalct++)

            for (auto ealct = range_emulALCT.first; ealct != range_emulALCT.second; ealct++) {
              bool isMatched = false;
              if (ealct->isValid() and ealct->isNominalInTime()) {
                if (ealct->isTightInTime()) {
                  if (chamber20) {
                    alctShowerEmulTightSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                    alctShowerEmulTightSummary_denom_->Fill(chamber * 2, sr, 0.5);
                  } else
                    alctShowerEmulTightSummary_denom_->Fill(chamber, sr);
                }
                if (chamber20) {
                  alctShowerEmulNomSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  alctShowerEmulNomSummary_denom_->Fill(chamber * 2, sr, 0.5);
                } else
                  alctShowerEmulNomSummary_denom_->Fill(chamber, sr);
                // check for least one matching ALCT
                for (auto dalct = range_dataALCT.first; dalct != range_dataALCT.second; dalct++) {
                  if (areSameShowers(*dalct, *ealct))
                    isMatched = true;
                }
                // only fill when it is not matched to an ALCT
                // to understand if the emulator is producing too many ALCTs
                if (!isMatched) {
                  if (ealct->isTightInTime()) {
                    if (chamber20) {
                      alctShowerEmulTightSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      alctShowerEmulTightSummary_num_->Fill(chamber * 2, sr, 0.5);
                    } else
                      alctShowerEmulTightSummary_num_->Fill(chamber, sr);
                  }
                  if (chamber20) {
                    alctShowerEmulNomSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                    alctShowerEmulNomSummary_num_->Fill(chamber * 2, sr, 0.5);
                  } else
                    alctShowerEmulNomSummary_num_->Fill(chamber, sr);
                }
              }
            }  // End of for (auto ealct = range_emulALCT.first; ealct != range_emulALCT.second; ealct++)

            // CLCT analysis
            auto range_dataCLCT = dataCLCTshs->get(detid);
            auto range_emulCLCT = emulCLCTshs->get(detid);

            for (auto dclct = range_dataCLCT.first; dclct != range_dataCLCT.second; dclct++) {
              if (dclct->isValid() and dclct->isNominalInTime()) {
                if (dclct->isTightInTime()) {
                  if (chamber20) {
                    clctShowerDataTightSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                    clctShowerDataTightSummary_denom_->Fill(chamber * 2, sr, 0.5);
                  } else
                    clctShowerDataTightSummary_denom_->Fill(chamber, sr);
                }
                if (chamber20) {
                  clctShowerDataNomSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  clctShowerDataNomSummary_denom_->Fill(chamber * 2, sr, 0.5);
                } else
                  clctShowerDataNomSummary_denom_->Fill(chamber, sr);
                // check for least one matching CLCT
                for (auto eclct = range_emulCLCT.first; eclct != range_emulCLCT.second; eclct++) {
                  if (eclct->isValid() and areSameShowers(*dclct, *eclct)) {
                    if (dclct->isTightInTime()) {
                      if (chamber20) {
                        clctShowerDataTightSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                        clctShowerDataTightSummary_num_->Fill(chamber * 2, sr, 0.5);
                      } else
                        clctShowerDataTightSummary_num_->Fill(chamber, sr);
                    }
                    if (chamber20) {
                      clctShowerDataNomSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      clctShowerDataNomSummary_num_->Fill(chamber * 2, sr, 0.5);
                    } else
                      clctShowerDataNomSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }  // End of for (auto dclct = range_dataCLCT.first; dclct != range_dataCLCT.second; dclct++)

            for (auto eclct = range_emulCLCT.first; eclct != range_emulCLCT.second; eclct++) {
              bool isMatched = false;
              if (eclct->isValid() and eclct->isNominalInTime()) {
                if (eclct->isTightInTime()) {
                  if (chamber20) {
                    clctShowerEmulTightSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                    clctShowerEmulTightSummary_denom_->Fill(chamber * 2, sr, 0.5);
                  } else
                    clctShowerEmulTightSummary_denom_->Fill(chamber, sr);
                }
                if (chamber20) {
                  clctShowerEmulNomSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  clctShowerEmulNomSummary_denom_->Fill(chamber * 2, sr, 0.5);
                } else
                  clctShowerEmulNomSummary_denom_->Fill(chamber, sr);
                // check for least one matching CLCT
                for (auto dclct = range_dataCLCT.first; dclct != range_dataCLCT.second; dclct++) {
                  if (areSameShowers(*dclct, *eclct))
                    isMatched = true;
                }
                // only fill when it is not matched to an CLCT
                // to understand if the emulator is producing too many CLCTs
                if (!isMatched) {
                  if (eclct->isTightInTime()) {
                    if (chamber20) {
                      clctShowerEmulTightSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      clctShowerEmulTightSummary_num_->Fill(chamber * 2, sr, 0.5);
                    } else
                      clctShowerEmulTightSummary_num_->Fill(chamber, sr);
                  }
                  if (chamber20) {
                    clctShowerEmulNomSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                    clctShowerEmulNomSummary_num_->Fill(chamber * 2, sr, 0.5);
                  } else
                    clctShowerEmulNomSummary_num_->Fill(chamber, sr);
                }
              }
            }  // End of for (auto eclct = range_emulCLCT.first; eclct != range_emulCLCT.second; eclct++)

            // LCT analysis
            auto range_dataLCT = dataLCTshs->get(detid);
            auto range_emulLCT = emulLCTshs->get(detid);

            for (auto dlct = range_dataLCT.first; dlct != range_dataLCT.second; dlct++) {
              if (dlct->isValid() and dlct->isNominalInTime()) {
                if (dlct->isTightInTime()) {
                  if (chamber20) {
                    lctShowerDataTightSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                    lctShowerDataTightSummary_denom_->Fill(chamber * 2, sr, 0.5);
                  } else
                    lctShowerDataTightSummary_denom_->Fill(chamber, sr);
                }
                if (chamber20) {
                  lctShowerDataNomSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  lctShowerDataNomSummary_denom_->Fill(chamber * 2, sr, 0.5);
                } else
                  lctShowerDataNomSummary_denom_->Fill(chamber, sr);
                // check for least one matching LCT
                for (auto elct = range_emulLCT.first; elct != range_emulLCT.second; elct++) {
                  if (elct->isValid() and areSameShowers(*dlct, *elct)) {
                    if (dlct->isTightInTime()) {
                      if (chamber20) {
                        lctShowerDataTightSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                        lctShowerDataTightSummary_num_->Fill(chamber * 2, sr, 0.5);
                      } else
                        lctShowerDataTightSummary_num_->Fill(chamber, sr);
                    }
                    if (chamber20) {
                      lctShowerDataNomSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      lctShowerDataNomSummary_num_->Fill(chamber * 2, sr, 0.5);
                    } else
                      lctShowerDataNomSummary_num_->Fill(chamber, sr);
                  }
                }
              }
            }  // End of for (auto dlct = range_dataLCT.first; dlct != range_dataLCT.second; dlct++)

            for (auto elct = range_emulLCT.first; elct != range_emulLCT.second; elct++) {
              bool isMatched = false;
              if (elct->isValid() and elct->isNominalInTime()) {
                if (elct->isTightInTime()) {
                  if (chamber20) {
                    lctShowerEmulTightSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                    lctShowerEmulTightSummary_denom_->Fill(chamber * 2, sr, 0.5);
                  } else
                    lctShowerEmulTightSummary_denom_->Fill(chamber, sr);
                }
                if (chamber20) {
                  lctShowerEmulNomSummary_denom_->Fill(chamber * 2 - 1, sr, 0.5);
                  lctShowerEmulNomSummary_denom_->Fill(chamber * 2, sr, 0.5);
                } else
                  lctShowerEmulNomSummary_denom_->Fill(chamber, sr);
                // check for least one matching LCT
                for (auto dlct = range_dataLCT.first; dlct != range_dataLCT.second; dlct++) {
                  if (areSameShowers(*dlct, *elct))
                    isMatched = true;
                }
                // only fill when it is not matched to an LCT
                // to understand if the emulator is producing too many LCTs
                if (!isMatched) {
                  if (elct->isTightInTime()) {
                    if (chamber20) {
                      lctShowerEmulTightSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                      lctShowerEmulTightSummary_num_->Fill(chamber * 2, sr, 0.5);
                    } else
                      lctShowerEmulTightSummary_num_->Fill(chamber, sr);
                  }
                  if (chamber20) {
                    lctShowerEmulNomSummary_num_->Fill(chamber * 2 - 1, sr, 0.5);
                    lctShowerEmulNomSummary_num_->Fill(chamber * 2, sr, 0.5);
                  } else
                    lctShowerEmulNomSummary_num_->Fill(chamber, sr);
                }
              }
            }  // End of for (auto elct = range_emulLCT.first; elct != range_emulLCT.second; elct++) {
          }
        }
      }
    }
  }
}

bool L1TdeCSCTPGShower::areSameShowers(const CSCShowerDigi& lhs, const CSCShowerDigi& rhs) const {
  bool returnValue = false;
  if (lhs.isValid() == rhs.isValid() && lhs.getCSCID() == rhs.getCSCID() && lhs.bitsInTime() == rhs.bitsInTime() &&
      lhs.bitsOutOfTime() == rhs.bitsOutOfTime()) {
    returnValue = true;
  }
  return returnValue;
}
