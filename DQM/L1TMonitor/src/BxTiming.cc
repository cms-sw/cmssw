#include "DQM/L1TMonitor/interface/BxTiming.h"
#include <cstdio>

BxTiming::BxTiming(const edm::ParameterSet &iConfig) {
  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag", 0);
  if (verbose())
    std::cout << "BxTiming::BxTiming()...\n" << std::flush;

  fedRef_ = iConfig.getUntrackedParameter<int>("ReferenceFedId", 813);
  fedSource_ = iConfig.getUntrackedParameter<edm::InputTag>("FedSource", edm::InputTag("source"));
  fedSource_token_ = consumes<FEDRawDataCollection>(
      iConfig.getUntrackedParameter<edm::InputTag>("FedSource", edm::InputTag("source")));
  gtSource_ = iConfig.getUntrackedParameter<edm::InputTag>("GtSource", edm::InputTag("gtUnpack"));
  gtSource_token_ = consumes<L1GlobalTriggerReadoutRecord>(
      iConfig.getUntrackedParameter<edm::InputTag>("GtSource", edm::InputTag("gtUnpack")));
  histFile_ = iConfig.getUntrackedParameter<std::string>("HistFile", "");
  histFolder_ = iConfig.getUntrackedParameter<std::string>("HistFolder", "L1T/BXSynch");

  runInFF_ = iConfig.getUntrackedParameter<bool>("RunInFilterFarm", true);
  if (runInFF_)
    histFolder_ = "L1T/BXSynch_EvF";
  if (verbose())
    std::cout << "Filter farm run setting?" << runInFF_ << "\n" << std::flush;

  listGtBits_ = iConfig.getUntrackedParameter<std::vector<int> >("GtBitList", std::vector<int>(1, 0));
  if (listGtBits_.size() == 1 && listGtBits_.at(0) == -1) {
    int ngtbits = 128;
    listGtBits_.reserve(ngtbits);
    for (int i = 0; i < ngtbits; i++)
      listGtBits_[i] = i;
  }

  if (verbose()) {
    std::cout << "BxTiming: gt bits set for timing dqm:";
    std::cout << "nbits:" << listGtBits_.size() << " list: ";
    for (size_t i = 0; i != listGtBits_.size(); i++)
      std::cout << listGtBits_.at(i) << " ";
    std::cout << "\n" << std::flush;
  }

  nEvt_ = 0;

  if (verbose())
    std::cout << "BxTiming::BxTiming constructor...done.\n" << std::flush;
}

BxTiming::~BxTiming() {}

void BxTiming::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {
  ibooker.setCurrentFolder(histFolder_);

  /// initialize counters
  for (int i = 0; i < nfed_; i++) {
    nBxDiff[i][0] = 0;
    nBxDiff[i][1] = nbig_;
    nBxDiff[i][2] = -1 * nbig_;
    nBxOccy[i][0] = 0;
    nBxOccy[i][1] = nbig_;
    nBxOccy[i][2] = -1 * nbig_;
  }

  std::string lbl("");
  std::string SysLabel[NSYS] = {"PreShower", "ECAL", "HCAL", "GCT", "CSCTPG", "CSCTF", "DTTPG", "DTTF", "RPC", "GT"};

  typedef std::pair<int, int> FEDRange;

  std::pair<int, int> fedRange[NSYS] = {
      FEDRange(FEDNumbering::MINPreShowerFEDID, FEDNumbering::MAXPreShowerFEDID),     //520..575
      FEDRange(FEDNumbering::MINECALFEDID, FEDNumbering::MAXECALFEDID),               //600..670
      FEDRange(FEDNumbering::MINHCALFEDID, FEDNumbering::MAXHCALFEDID),               //700..731
      FEDRange(FEDNumbering::MINTriggerGCTFEDID, FEDNumbering::MAXTriggerEGTPFEDID),  //745..749
      FEDRange(FEDNumbering::MINCSCFEDID, FEDNumbering::MAXCSCFEDID),                 //750..757
      FEDRange(FEDNumbering::MINCSCTFFEDID, FEDNumbering::MAXCSCTFFEDID),             //760..760
      FEDRange(FEDNumbering::MINDTFEDID, FEDNumbering::MAXDTFEDID),                   //770..775
      FEDRange(FEDNumbering::MINDTTFFEDID, FEDNumbering::MAXDTTFFEDID),               //780..780
      FEDRange(FEDNumbering::MINRPCFEDID, FEDNumbering::MAXRPCFEDID),                 //790..795
      FEDRange(FEDNumbering::MINTriggerGTPFEDID, FEDNumbering::MAXTriggerGTPFEDID)    //812..813
  };
  for (int i = 0; i < NSYS; i++)
    fedRange_[i] = fedRange[i];

  int fedRefSys = -1;
  for (int i = 0; i < NSYS; i++)
    if (fedRef_ >= fedRange_[i].first && fedRef_ <= fedRange_[i].second) {
      fedRefSys = i;
      break;
    }
  std::string refName("");
  std::string spreadLabel[nspr_] = {"Spread", "Min", "Max"};
  if (fedRefSys >= 0)
    refName += SysLabel[fedRefSys];
  else
    refName += fedRef_;

  /// book the histograms

  const int dbx = nbig_;

  ibooker.setCurrentFolder(histFolder_);

  hBxDiffAllFed = ibooker.bookProfile(
      "BxDiffAllFed", "BxDiffAllFed", nfed_ + 1, -0.5, nfed_ + 0.5, 2 * dbx + 1, -1 * dbx - 0.5, dbx + 0.5);

  for (int i = 0; i < nspr_; i++) {
    lbl.clear();
    lbl += "BxDiffAllFed";
    lbl += spreadLabel[i];
    hBxDiffAllFedSpread[i] = ibooker.book1D(lbl.data(), lbl.data(), nfed_ + 1, -0.5, nfed_ + 0.5);
    lbl.clear();
    lbl += "BxOccyAllFed";
    lbl += spreadLabel[i];
    hBxOccyAllFedSpread[i] = ibooker.book1D(lbl.data(), lbl.data(), nfed_ + 1, -0.5, nfed_ + 0.5);

    lbl.clear();
    lbl += "BxOccyAllFed";
    hBxOccyAllFed = ibooker.book1D(lbl.data(), lbl.data(), norb_ + 1, -0.5, norb_ + 0.5);
  }

  // following histos defined only when not runing in the ff
  if (!runInFF_) {
    ibooker.setCurrentFolder(histFolder_);

    for (int i = 0; i < NSYS; i++) {
      lbl.clear();
      lbl += SysLabel[i];
      lbl += "FedBxDiff";
      int nfeds = fedRange_[i].second - fedRange_[i].first + 1;
      nfeds = (nfeds > 0) ? nfeds : 1;
      hBxDiffSysFed[i] = ibooker.bookProfile(lbl.data(),
                                             lbl.data(),
                                             nfeds,
                                             fedRange_[i].first - 0.5,
                                             fedRange_[i].second + 0.5,
                                             2 * dbx + 1,
                                             -1 * dbx - 0.5,
                                             dbx + 0.5);
    }

    lbl.clear();
    lbl += "BxOccyAllFed";
    hBxOccyOneFed = new MonitorElement *[nfed_];
    ibooker.setCurrentFolder(histFolder_ + "/SingleFed");
    for (int i = 0; i < nfed_; i++) {
      lbl.clear();
      lbl += "BxOccyOneFed";
      char *ii = new char[1000];
      std::sprintf(ii, "%d", i);
      lbl += ii;
      hBxOccyOneFed[i] = ibooker.book1D(lbl.data(), lbl.data(), norb_ + 1, -0.5, norb_ + 0.5);
      delete[] ii;
    }

    ibooker.setCurrentFolder(histFolder_);
    for (int i = 0; i < nttype_; i++) {
      lbl.clear();
      lbl += "BxOccyGtTrigType";
      char *ii = new char[16];
      std::sprintf(ii, "%d", i + 1);
      lbl += ii;
      hBxOccyGtTrigType[i] = ibooker.book1D(lbl.data(), lbl.data(), norb_ + 1, -0.5, norb_ + 0.5);
      delete[] ii;
    }

    ibooker.setCurrentFolder(histFolder_ + "/SingleBit");
    for (int i = 0; i < NSYS; i++) {
      hBxOccyTrigBit[i] = new MonitorElement *[listGtBits_.size()];
      for (size_t j = 0; j < listGtBits_.size(); j++) {
        lbl.clear();
        lbl += SysLabel[i];
        lbl += "BxOccyGtBit";
        char *ii = new char[1000];
        std::sprintf(ii, "%d", listGtBits_.at(j));
        lbl += ii;
        hBxOccyTrigBit[i][j] = ibooker.book1D(lbl.data(), lbl.data(), norb_ + 1, -0.5, norb_ + 0.5);
        delete[] ii;
      }
    }
  }

  /// labeling (cosmetics added here)
  hBxDiffAllFed->setAxisTitle("FED ID", 1);
  lbl.clear();
  lbl += "BX(fed)-BX(";
  lbl += refName;
  lbl += ")";
  hBxDiffAllFed->setAxisTitle(lbl, 2);
  for (int i = 0; i < nspr_; i++) {
    lbl.clear();
    lbl += "BX(fed)-BX(";
    lbl += refName;
    lbl += ") " + spreadLabel[i];
    hBxDiffAllFedSpread[i]->setAxisTitle("FED ID", 1);
    hBxDiffAllFedSpread[i]->setAxisTitle(lbl, 2);
    lbl.clear();
    lbl += "Bx FED occupancy";
    lbl += " ";
    lbl += spreadLabel[i];
    hBxOccyAllFedSpread[i]->setAxisTitle("FED ID", 1);
    hBxOccyAllFedSpread[i]->setAxisTitle(lbl, 2);
  }

  hBxOccyAllFed->setAxisTitle("bx", 1);
  lbl.clear();
  lbl += "Combined FED occupancy";
  hBxOccyAllFed->setAxisTitle(lbl, 2);

  // skip next if running in filter farm
  if (runInFF_)
    return;

  for (int i = 0; i < NSYS; i++) {
    lbl.clear();
    lbl += SysLabel[i];
    lbl += " FED ID";
    hBxDiffSysFed[i]->setAxisTitle(lbl, 1);
    lbl.clear();
    lbl += "BX(";
    lbl += SysLabel[i];
    lbl += ")-BX(";
    lbl += refName;
    lbl += ")";
    hBxDiffSysFed[i]->setAxisTitle(lbl, 2);
  }

  for (int i = 0; i < nfed_; i++) {
    hBxOccyOneFed[i]->setAxisTitle("bx", 1);
    lbl.clear();
    lbl += " FED ";
    char *ii = new char[1000];
    std::sprintf(ii, "%d", i);
    lbl += ii;
    lbl += " occupancy";
    hBxOccyOneFed[i]->setAxisTitle(lbl, 2);
    delete[] ii;
  }
  for (int i = 0; i < nttype_; i++) {
    hBxOccyGtTrigType[i]->setAxisTitle("bx", 1);
    lbl.clear();
    lbl += "GT occupancy for trigger type ";
    char *ii = new char[16];
    std::sprintf(ii, "%d", i + 1);
    lbl += ii;
    hBxOccyGtTrigType[i]->setAxisTitle(lbl, 2);
    delete[] ii;
  }

  for (int i = 0; i < NSYS; i++) {
    for (size_t j = 0; j < listGtBits_.size(); j++) {
      hBxOccyTrigBit[i][j]->setAxisTitle("bx", 1);
      lbl.clear();
      lbl += SysLabel[i];
      lbl += " Bx occupancy for Trigger bit ";
      char *ii = new char[10];
      std::sprintf(ii, "%d", listGtBits_.at(j));
      lbl += ii;
      hBxOccyTrigBit[i][j]->setAxisTitle(lbl, 2);
      delete[] ii;
    }
  }
}

// ------------ method called to for each event  ------------
void BxTiming::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (verbose())
    std::cout << "BxTiming::analyze()  start\n" << std::flush;

  nEvt_++;

  /// get the raw data - if not found, return
  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByToken(fedSource_token_, rawdata);

  if (!rawdata.isValid()) {
    if (verbose())
      std::cout << "BxTiming::analyze() | FEDRawDataCollection with input tag " << fedSource_ << " not found.";

    return;
  }

  // get the GT bits
  edm::Handle<L1GlobalTriggerReadoutRecord> gtdata;
  iEvent.getByToken(gtSource_token_, gtdata);
  std::vector<bool> gtbits;
  int ngtbits = 128;
  gtbits.reserve(ngtbits);
  for (int i = 0; i < ngtbits; i++)
    gtbits[i] = false;
  if (gtdata.isValid())
    gtbits = gtdata->decisionWord();

  if (gtbits.empty()) {
    gtbits.push_back(true);  // gtdata->decision();
    if (verbose())
      std::cout << "BxTiming::analyze() | unexpected empty decision bits!";
  }

  if (verbose()) {
    std::cout << "BxTiming::analyze()  gt data valid:" << (int)(gtdata.isValid() ? 0 : 1)
              << " decision word size:" << (int)(gtbits.size()) << "  bits: ";
    for (size_t i = 0; i != gtbits.size(); i++) {
      int ii = gtbits.at(i) ? 1 : 0;
      std::cout << ii;
    }
    std::cout << ".\n" << std::flush;
  }

  // get reference bx
  int bxRef = FEDHeader(rawdata->FEDData(fedRef_).data()).bxID();

  // triggerType
  // trigger types: physics (1), calibration (2), random (3), traced physics (5),  test (6)
  int ttype = static_cast<double>(iEvent.eventAuxiliary().experimentType());

  // loop over feds
  for (int i = 0; i < FEDNumbering::MAXFEDID + 1; i++) {
    const FEDRawData &data = rawdata->FEDData(i);
    size_t size = data.size();

    if (!size)
      continue;
    FEDHeader header(data.data());
    //int lvl1id = header.lvl1ID(); //Level-1 event number generated by the TTC system
    int bx = header.bxID();  // The bunch crossing number

    int bxDiff = calcBxDiff(bx, bxRef);  // deviation from reference bx

    //min
    if (nBxDiff[i][1] > bxDiff)
      nBxDiff[i][1] = bxDiff;
    if (nBxOccy[i][1] > bx)
      nBxOccy[i][1] = bx;
    //max
    if (nBxDiff[i][2] < bxDiff)
      nBxDiff[i][2] = bxDiff;
    if (nBxOccy[i][2] < bx)
      nBxOccy[i][2] = bx;

    if (verbose())
      std::cout << " fed:" << i << " bx:" << bx << " bxRef:" << bxRef << " diff:" << bxDiff << " nBxDiff"
                << " del:" << nBxDiff[i][0] << " min:" << nBxDiff[i][1] << " max:" << nBxDiff[i][2] << " nBxOccy"
                << " del:" << nBxOccy[i][0] << " min:" << nBxOccy[i][1] << " max:" << nBxOccy[i][2] << "\n"
                << std::flush;

    hBxDiffAllFed->Fill(i, bxDiff);

    //if(ttype==1) //skip if not a physics trigger
    hBxOccyAllFed->Fill(bx);

    // done if running in filter farm
    if (runInFF_)
      continue;

    for (int j = 0; j < NSYS; j++)
      if (i >= fedRange_[j].first && i <= fedRange_[j].second)
        hBxDiffSysFed[j]->Fill(i, bxDiff);

    for (size_t k = 0; k != listGtBits_.size(); k++) {
      if ((int)gtbits.size() <= listGtBits_.at(k)) {
        if (verbose())
          std::cout << "BxTiming analyze | problem with vector size!\n" << std::endl;
        continue;
      } else if (!gtbits.at(listGtBits_.at(k)))
        continue;
      for (int j = 0; j < NSYS; j++) {
        if (i >= fedRange_[j].first && i <= fedRange_[j].second) {
          hBxOccyTrigBit[j][k]->Fill(bx);
        }
      }
    }

    if (i >= fedRange_[GLT].first && i <= fedRange_[GLT].second)  //GT fed
      if (ttype < nttype_)
        hBxOccyGtTrigType[ttype - 1]->Fill(bx);

    if (ttype != 1)
      continue;  //skip if not a physics trigger
    //hBxOccyAllFed->Fill(bx);
    hBxOccyOneFed[i]->Fill(bx);
  }

  for (int i = 0; i < nfed_; i++) {
    nBxDiff[i][0] = nBxDiff[i][2] - nBxDiff[i][1];
    nBxOccy[i][0] = nBxOccy[i][2] - nBxOccy[i][1];
    if (nBxDiff[i][0] < 0 || nBxOccy[i][0] < 0)
      continue;
    for (int j = 0; j < nspr_; j++) {
      hBxDiffAllFedSpread[j]->setBinContent(i, nBxDiff[i][j]);
      hBxOccyAllFedSpread[j]->setBinContent(i, nBxOccy[i][j]);
    }
    if (verbose())
      std::cout << "BxTiming fed:" << i << " Bx-Bx(" << fedRef_ << ")::"
                << " del:" << nBxDiff[i][0] << " min:" << nBxDiff[i][1] << " max:" << nBxDiff[i][2] << " Occy: "
                << " del:" << nBxOccy[i][0] << " min:" << nBxOccy[i][1] << " max:" << nBxOccy[i][2] << "\n"
                << std::flush;
  }

  if (verbose())
    std::cout << "BxTiming::analyze() end.\n" << std::flush;
}

//----------------------------------------------------------------------

int BxTiming::calcBxDiff(int bx1, int bx2) {
  int diff = bx1 - bx2;

  while (diff < -half_norb_)
    diff += norb_;

  while (diff > half_norb_)
    diff -= norb_;

  return diff;
}

//----------------------------------------------------------------------
