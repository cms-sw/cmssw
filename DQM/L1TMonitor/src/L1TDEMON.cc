#include "DQM/L1TMonitor/interface/L1TDEMON.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <bitset>

using namespace dedefs;

L1TDEMON::L1TDEMON(const edm::ParameterSet& iConfig) {
  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag", 0);

  if (verbose())
    std::cout << "L1TDEMON::L1TDEMON()...\n" << std::flush;

  DEsource_ = iConfig.getParameter<edm::InputTag>("DataEmulCompareSource");
  histFolder_ = iConfig.getUntrackedParameter<std::string>("HistFolder", "L1TEMU");
  histFile_ = iConfig.getUntrackedParameter<std::string>("HistFile", "");

  runInFF_ = iConfig.getUntrackedParameter<bool>("RunInFilterFarm", false);
  if (verbose())
    std::cout << "Filter farm run setting? " << runInFF_ << "\n" << std::flush;

  if (iConfig.getUntrackedParameter<bool>("disableROOToutput", true))
    histFile_ = "";

  if (!histFile_.empty()) {
    edm::LogInfo("OutputRootFile") << "L1TEmulator Monitoring histograms will be saved to " << histFile_.c_str()
                                   << std::endl;
  }

  // copied from HardwareValidation/L1Comparator
  // check on size are done there
  std::vector<unsigned int> dosys(0, DEnsys);
  dosys = iConfig.getUntrackedParameter<std::vector<unsigned int> >("COMPARE_COLLS", dosys);

  for (int i = 0; i < DEnsys; i++) {
    m_doSys[i] = dosys[i];
  }

  //

  nEvt_ = 0;
  for (int i = 0; i < DEnsys; i++) {
    deSysCount[i] = 0;
    nEvtWithSys[i] = 0;
  }

  hasRecord_ = true;

  if (verbose())
    std::cout << "L1TDEMON::L1TDEMON constructor...done.\n" << std::flush;

  //set Tokens(-s)
  DEsourceToken_ = consumes<L1DataEmulRecord>(iConfig.getParameter<edm::InputTag>("DataEmulCompareSource"));
}

L1TDEMON::~L1TDEMON() {}

void L1TDEMON::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  if (verbose())
    std::cout << "L1TDEMON::beginJob()  start\n" << std::flush;

  ibooker.setCurrentFolder(histFolder_);

  //physical values disabled now, waiting for scale procedure
  //const double tpi = 6.2832;
  //const double amin=   -0.5;
  //const double amax=tpi+0.5;

  //                           ETP,  HTP,  RCT, GCT, DTP, DTF,  CTP, CTF, RPC,LTC, GMT,GLT
  int phiNBins[DEnsys] = {71, 71, 18, 18, 12, 255, 160, 255, 144, 0, 255, 0};
  double phiMinim[DEnsys] = {0.5, 0.5, -0.5, -0.5, -0.5, 0, -0.5, 0, -0.5, 0, 0, 0};
  double phiMaxim[DEnsys] = {71.5, 71.5, 17.5, 17.5, 11.5, 255, 159.5, 255, 143.5, 0, 255, 0};

  int etaNBins[DEnsys] = {35, 35, 22, 22, 5, 20, 120, 20, 64, 0, 20, 0};
  double etaMinim[DEnsys] = {-17.5, -17.5, -0.5, -0.5, -2.5, 0, -0.5, 0, -0.5, 0, 0, 0};
  double etaMaxim[DEnsys] = {17.5, 17.5, 21.5, 21.5, 2.5, 63, 119.5, 63, 63.5, 0, 63, 0};

  int x3NBins[DEnsys] = {0, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0};
  double x3Minim[DEnsys] = {0, 0, -0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0};
  double x3Maxim[DEnsys] = {0, 0, 6.5, 0, 4.5, 0, 0, 0, 0, 0, 0, 0};

  int rnkNBins[DEnsys] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double rnkMinim[DEnsys] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double rnkMaxim[DEnsys] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  //assume for
  for (int i = 0; i < DEnsys; i++) {
    rnkNBins[i] = 63;
    rnkMinim[i] = 0.5;
    rnkMaxim[i] = 63.5;
  }  //rank 0x3f->63
  rnkNBins[DTP] = 7;
  rnkMinim[DTP] = -0.5;
  rnkMaxim[DTP] = 6.5;  //rank 0-6
  rnkNBins[CTP] = 16;
  rnkMinim[CTP] = -0.5;
  rnkMaxim[CTP] = 15.5;  //quality 0-15

  /*--notes
     RCT: global index ieta (0-21)=[22,-0.5,21.5] , iphi (0-17)=[18,-0.5,17.5]; card (0-6)
     GCT: phi index (0-17); eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z -> 0.17
     DTP: usc 0..11; uwh -2..2; ust 1..4;
     CTP: rank is quality 0..15
     */

  if (!runInFF_)
    ibooker.setCurrentFolder(std::string(histFolder_ + "/common"));

  for (int j = 0; j < 2; j++) {
    std::string lbl("sysncand");
    lbl += (j == 0 ? "Data" : "Emul");
    sysncand[j] = ibooker.book1D(lbl.data(), lbl.data(), DEnsys, 0, DEnsys);
  }

  sysrates = ibooker.book1D("sysrates", "sysrates", DEnsys, 0, DEnsys);
  const int nerr = 5;
  errordist = ibooker.book1D("errorflag", "errorflag", nerr, 0, nerr);

  for (int j = 0; j < DEnsys; j++) {
    // skip if system disabled
    if (!m_doSys[j]) {
      continue;
    }

    if (!runInFF_) {
      ibooker.setCurrentFolder(std::string(histFolder_ + "/" + SystLabelExt[j]));
    }

    std::string lbl("");
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "ErrorFlag";
    errortype[j] = ibooker.book1D(lbl.data(), lbl.data(), nerr, 0, nerr);

    // skip next histos if running in filter farm
    if (runInFF_) {
      continue;
    }

    //
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "eta";
    eta[j] = ibooker.book1D(lbl.data(), lbl.data(), etaNBins[j], etaMinim[j], etaMaxim[j]);
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "phi";
    phi[j] = ibooker.book1D(lbl.data(), lbl.data(), phiNBins[j], phiMinim[j], phiMaxim[j]);
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "x3";
    x3[j] = ibooker.book1D(lbl.data(), lbl.data(), x3NBins[j], x3Minim[j], x3Maxim[j]);
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "etaphi";
    etaphi[j] = ibooker.book2D(
        lbl.data(), lbl.data(), etaNBins[j], etaMinim[j], etaMaxim[j], phiNBins[j], phiMinim[j], phiMaxim[j]);
    //
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "eta";
    lbl += "Data";
    etaData[j] = ibooker.book1D(lbl.data(), lbl.data(), etaNBins[j], etaMinim[j], etaMaxim[j]);
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "phi";
    lbl += "Data";
    phiData[j] = ibooker.book1D(lbl.data(), lbl.data(), phiNBins[j], phiMinim[j], phiMaxim[j]);
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "x3";
    lbl += "Data";
    x3Data[j] = ibooker.book1D(lbl.data(), lbl.data(), x3NBins[j], x3Minim[j], x3Maxim[j]);
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "rank";
    lbl += "Data";
    rnkData[j] = ibooker.book1D(lbl.data(), lbl.data(), rnkNBins[j], rnkMinim[j], rnkMaxim[j]);

    const int nbit = (j == GLT) ? 128 : 32;
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "dword";
    dword[j] = ibooker.book1D(lbl.data(), lbl.data(), nbit, 0, nbit);
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "eword";
    eword[j] = ibooker.book1D(lbl.data(), lbl.data(), nbit, 0, nbit);
    lbl.clear();
    lbl += SystLabel[j];
    lbl += "deword";
    deword[j] = ibooker.book1D(lbl.data(), lbl.data(), nbit, 0, nbit);
    //lbl.clear();
    //lbl+=SystLabel[j];lbl+="masked";
    //masked[j] = dbe->book1D(lbl.data(),lbl.data(),nbit,0,nbit);
  }

  /// labeling (temporary cosmetics added here)
  for (int i = 0; i < DEnsys; i++) {
    sysrates->setBinLabel(i + 1, SystLabel[i]);
    sysncand[0]->setBinLabel(i + 1, SystLabel[i]);
    sysncand[1]->setBinLabel(i + 1, SystLabel[i]);
  }
  //const int nerr = 5;
  std::string errLabel[nerr] = {"Agree", "Loc. Agree", "L.Disagree", "Data only", "Emul only"};
  for (int j = 0; j < nerr; j++) {
    errordist->setBinLabel(j + 1, errLabel[j]);
  }
  for (int i = 0; i < DEnsys; i++) {
    // skip if system disabled
    if (!m_doSys[i]) {
      continue;
    }

    for (int j = 0; j < nerr; j++) {
      errortype[i]->setBinLabel(j + 1, errLabel[j]);
    }
  }

  // done if running in filter farm
  if (runInFF_)
    return;

  for (int i = 0; i < DEnsys; i++) {
    // skip if system disabled
    if (!m_doSys[i]) {
      continue;
    }

    etaphi[i]->setAxisTitle("eta", 1);
    etaphi[i]->setAxisTitle("phi", 2);
    eta[i]->setAxisTitle("eta");
    phi[i]->setAxisTitle("phi");
    x3[i]->setAxisTitle("x3");
    etaData[i]->setAxisTitle("eta");
    phiData[i]->setAxisTitle("phi");
    x3Data[i]->setAxisTitle("x3");
    rnkData[i]->setAxisTitle("rank");
    dword[i]->setAxisTitle("trigger data word bit");
    eword[i]->setAxisTitle("trigger data word bit");
    deword[i]->setAxisTitle("trigger data word bit");
    //masked [i]->setAxisTitle("trigger data word bit");
  }

  // assertions/temporary
  assert(ETP == 0);
  assert(HTP == 1);
  assert(RCT == 2);
  assert(GCT == 3);
  assert(DTP == 4);
  assert(DTF == 5);
  assert(CTP == 6);
  assert(CTF == 7);
  assert(RPC == 8);
  assert(LTC == 9);
  assert(GMT == 10);
  assert(GLT == 11);

  if (verbose())
    std::cout << "L1TDEMON::bookHistograms()  end.\n" << std::flush;
}

// ------------ method called to for each event  ------------
void L1TDEMON::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (!hasRecord_)
    return;

  if (verbose())
    std::cout << "L1TDEMON::analyze()  start\n" << std::flush;

  nEvt_++;

  /// get the comparison results
  edm::Handle<L1DataEmulRecord> deRecord;
  iEvent.getByToken(DEsourceToken_, deRecord);

  if (!deRecord.isValid()) {
    edm::LogInfo("DataNotFound") << "Cannot find L1DataEmulRecord with label " << DEsource_.label()
                                 << " Please verify that comparator was successfully executed."
                                 << " Emulator DQM will be skipped!" << std::endl;
    hasRecord_ = false;
    return;
  }

  bool deMatch[DEnsys];
  deRecord->get_status(deMatch);
  if (verbose()) {
    std::cout << "[L1TDEMON] verbose sys match?: ";
    for (int i = 0; i < DEnsys; i++)
      std::cout << deMatch[i] << " ";
    std::cout << std::endl;
  }

  bool isComp[DEnsys];
  for (int i = 0; i < DEnsys; i++)
    isComp[i] = deRecord->get_isComp(i);
  if (verbose()) {
    std::cout << "[L1TDEMON] verbose dosys?: ";
    for (int i = 0; i < DEnsys; i++)
      std::cout << isComp[i];
    std::cout << std::endl;
  }

  int DEncand[DEnsys][2];
  for (int i = 0; i < DEnsys; i++)
    for (int j = 0; j < 2; j++)
      DEncand[i][j] = deRecord->getNCand(i, j);

  if (verbose()) {
    std::cout << "[L1TDEMON] ncands d: ";
    for (int i = 0; i < DEnsys; i++)
      std::cout << DEncand[i][0] << " ";
    std::cout << "\n[L1TDEMON] ncands e: ";
    for (int i = 0; i < DEnsys; i++)
      std::cout << DEncand[i][1] << " ";
    std::cout << std::endl;
  }

  const int nullVal = L1DataEmulDigi().reset();

  /// get the de candidates
  L1DEDigiCollection deColl;
  deColl = deRecord->getColl();

  // global counters
  int hasSys[DEnsys] = {0};
  for (L1DEDigiCollection::const_iterator it = deColl.begin(); it != deColl.end(); it++)
    if (!it->empty())
      hasSys[it->sid()]++;
  for (int i = 0; i < DEnsys; i++) {
    if (!hasSys[i])
      continue;
    nEvtWithSys[i]++;
    if (deMatch[i])
      deSysCount[i]++;
  }

  if (verbose()) {
    std::cout << "[L1TDEMON] digis: \n";
    for (L1DEDigiCollection::const_iterator it = deColl.begin(); it != deColl.end(); it++)
      std::cout << "\t" << *it << std::endl;
  }

  /// --- Fill histograms(me) ---

  // global, sub-systems d|e match, ncands
  for (int i = 0; i < DEnsys; i++) {
    // skip if system disabled
    if (!m_doSys[i]) {
      continue;
    }

    if (!isComp[i])
      continue;

    for (int j = 0; j < 2; j++)
      sysncand[j]->Fill(i, DEncand[i][j]);

    //sysrates->Fill(i,(int)!deMatch[i]);
    int ibin = i + 1;
    double rate = nEvtWithSys[i] ? (nEvtWithSys[i] - 1. * deSysCount[i]) / nEvtWithSys[i] : 0.;
    sysrates->setBinContent(ibin, rate);

    if (verbose()) {
      std::cout << "[L1TDEMON] analyze "
                << " sysid:" << i << " nEvt:" << nEvt_ << " match?" << deMatch[i] << " ncands:" << hasSys[i]
                << " nevtwsys:" << nEvtWithSys[i] << " nevtgood:" << deSysCount[i]
                << " rate:" << sysrates->getBinContent(ibin) << "\n";

      if (rate > 1)
        std::cout << "problem, error rate for " << SystLabel[i] << " is " << sysrates->getBinContent(ibin) << std::endl;
    }
  }

  // container for subsystem's leading candidate
  const int ncorr = 3;
  float LeadCandVal[DEnsys][ncorr] = {{(float)nullVal}};
  for (int i = 0; i < DEnsys; i++)
    for (int j = 0; j < ncorr; j++)
      LeadCandVal[i][j] = nullVal;

  // d|e candidate loop
  for (L1DEDigiCollection::const_iterator it = deColl.begin(); it != deColl.end(); it++) {
    int sid = it->sid();
    int cid = it->cid();

    if (it->empty())
      continue;
    assert(isComp[sid]);

    int type = it->type();
    double phiv = it->x1();
    double etav = it->x2();
    double x3v = it->x3();

    float rankarr[2];
    it->rank(rankarr);
    float rnkv = rankarr[0];

    double wei = 1.;

    unsigned int mask = (~0x0);

    if (sid == RCT) {
      //if(cid!=RCTem)  continue;
      //if(cid!=RCTrgn) continue;
    }
    //     if(sid==GCT) {
    //       // For the moment just include EM and jet objects, energy sums being debugged in the expert DQM first.
    //       if(cid!=GCTisolaem &&
    //          cid!=GCTnoisoem &&
    //          cid!=GCTcenjets &&
    //          cid!=GCTforjets &&
    //          cid!=GCTtaujets) continue;
    //     }
    if (sid == DTP) {
      //tbd cols:th,ph; plots per wheel
      //if(it->x3()!=0) continue;
    }
    if (sid == GMT) {
      //select gmt cands only for GMT sys
      if (cid != GMTcnd)
        continue;
      //masking: gres -- I.Mikulec: mask bits 0,5,16,21,22,23
      //mask = (~(0x0e10021));
    }
    if (sid == DTF) {
      if (cid != DTtftrk)
        continue;
    }
    if (sid == RPC || sid == CTF || sid == RPC) {
      //select mu regional cands only for dtf,ctf,rpc
      //if(cid!=MUrtf) continue;
      //masking: gres dttf only -- I.Mikulec: lowest 16 bits only
      //if(sid==DTF) mask = 0xffff;
    }

    errordist->Fill(type);
    errortype[sid]->Fill(type);

    // skip next if running in filter farm
    if (runInFF_)
      continue;

    //exclude agreeing cands
    wei = 1.;
    if (!type)
      wei = 0.;
    if (etav != nullVal && phiv != nullVal)
      etaphi[sid]->Fill(etav, phiv, wei);
    if (etav != nullVal)
      eta[sid]->Fill(etav, wei);
    if (phiv != nullVal)
      phi[sid]->Fill(phiv, wei);
    if (sid == DTP)
      if (x3v != nullVal)
        x3[sid]->Fill(x3v, wei);

    unsigned int word[2];
    it->data(word);
    std::bitset<32> dbits(word[0]);
    std::bitset<32> ebits(word[1]);
    unsigned int dexor = ((word[0]) ^ (word[1]));
    //disagreeing bits
    std::bitset<32> debits(dexor);
    //disagreeing bits after masking
    std::bitset<32> dembits(((dexor) & (mask)));

    if (verbose())
      std::cout << "l1demon"
                << " sid:" << sid << " cid:" << cid << "\n"
                << " data:0x" << std::hex << word[0] << std::dec << " bitset:" << dbits << "\n"
                << " emul:0x" << std::hex << word[1] << std::dec << " bitset:" << ebits << "\n"
                << "  xor:0x" << std::hex << dexor << std::dec << " bitset:" << debits
                << " bitset:" << ((dbits) ^ (ebits)) << "\n"
                << std::flush;

    ///bitset loop
    for (int ibit = 0; ibit < 32; ibit++) {
      wei = 1.;
      //comparison gives no info if there's only 1 candidate
      if (type == 3 || type == 4)
        wei = 0.;
      if (dbits[ibit])
        dword[sid]->Fill(ibit, wei);
      if (ebits[ibit])
        eword[sid]->Fill(ibit, wei);
      if (debits[ibit])
        deword[sid]->Fill(ibit, wei);
      //if(dembits[ibit])masked[sid]->Fill(ibit,wei);
    }

    //exclude e-only cands (only data)
    wei = 1.;
    if (type == 4)
      wei = 0.;
    if (etav != nullVal)
      etaData[sid]->Fill(etav, wei);
    if (phiv != nullVal)
      phiData[sid]->Fill(phiv, wei);
    if (sid == DTP)
      if (x3v != nullVal)
        x3Data[sid]->Fill(x3v, wei);
    rnkData[sid]->Fill(rnkv, wei);

    //correlations: store leading candidate
    if (type == 4)
      continue;  //exclude e-only cands
    bool islead = false;
    if (rnkv > LeadCandVal[sid][2])
      islead = true;
    else if (rnkv == LeadCandVal[sid][2]) {
      if (phiv > LeadCandVal[sid][0])
        islead = true;
      else if (phiv == LeadCandVal[sid][0])
        if (etav > LeadCandVal[sid][1])
          islead = true;
    }

    if (islead) {
      LeadCandVal[sid][0] = phiv;
      LeadCandVal[sid][1] = etav;
      LeadCandVal[sid][2] = rnkv;
    }

  }  //close loop over dedigi-cands
}
