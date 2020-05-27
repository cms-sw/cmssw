#include "L1Trigger/TrackFindingTracklet/interface/TrackletEventProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include "L1Trigger/TrackFindingTracklet/interface/Sector.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/Track.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/Cabling.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <iomanip>

using namespace trklet;
using namespace std;

TrackletEventProcessor::TrackletEventProcessor() = default;

TrackletEventProcessor::~TrackletEventProcessor() {
  if (settings_->bookHistos()) {
    histbase_->close();
  }
}

void TrackletEventProcessor::init(const Settings* theSettings) {
  settings_ = theSettings;

  globals_ = make_unique<Globals>(*settings_);

  //Verify consistency
  if (settings_->kphi0pars() != globals_->ITC_L1L2()->phi0_final.K()) {
    throw cms::Exception("Inconsistency") << "phi0 conversion parameter inconsistency\n";
  }

  if (settings_->krinvpars() != globals_->ITC_L1L2()->rinv_final.K()) {
    throw cms::Exception("Inconsistency") << "ring conversion parameter inconsistency\n";
  }

  if (settings_->ktpars() != globals_->ITC_L1L2()->t_final.K()) {
    throw cms::Exception("Inconsistency") << "t conversion parameter inconsistency\n";
  }

  if (settings_->debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "========================================================= \n"
                                 << "Conversion factors for global coordinates: \n"
                                 << "z    kz            = " << settings_->kz() << "\n"
                                 << "r    kr            = " << settings_->kr() << "\n"
                                 << "phi  kphi1         = " << settings_->kphi1() << "\n"
                                 << "========================================================= \n"
                                 << "Conversion factors for track(let) parameters: \n"
                                 << "rinv krinvpars     = " << settings_->krinvpars() << "\n"
                                 << "phi0 kphi0pars     = " << settings_->kphi0pars() << "\n"
                                 << "d0   kd0pars       = " << settings_->kd0pars() << "\n"
                                 << "t    ktpars        = " << settings_->ktpars() << "\n"
                                 << "z0   kz0pars       = " << settings_->kz0pars() << "\n"
                                 << "========================================================= \n"
                                 << "phi0bitshift = " << settings_->phi0bitshift() << "\n"
                                 << "d0bitshift   = ??? \n"
                                 << "=========================================================";
  }

  //option to write out tables for HLS code, not used in production
  /*
  const Settings& settings = *settings_;
  Globals* globals = globals_;

  if (settings_->writeVerilog() || settings_->writeHLS()) {
#include "../test/WriteInvTables.icc"
#include "../test/WriteDesign.icc"
  }
  */

  if (settings_->bookHistos()) {
    histbase_ = new HistBase;
    histbase_->open();
    histbase_->bookLayerResidual();
    histbase_->bookDiskResidual();
    histbase_->bookTrackletParams();
    histbase_->bookSeedEff();

    globals_->histograms() = histbase_;
  }

  // create the sector processors (1 sector processor = 1 board)
  sectors_.resize(N_SECTOR);

  for (unsigned int i = 0; i < N_SECTOR; i++) {
    sectors_[i] = make_unique<Sector>(i, *settings_, globals_.get());
  }

  // get the memory modules
  if (settings_->debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "Will read memory modules file";
  }

  ifstream inmem(settings_->memoryModulesFile().c_str());
  assert(inmem.good());

  while (inmem.good()) {
    string memType, memName, size;
    inmem >> memType >> memName >> size;
    if (!inmem.good())
      continue;
    if (settings_->writetrace()) {
      edm::LogVerbatim("Tracklet") << "Read memory: " << memType << " " << memName;
    }
    for (auto& sector : sectors_) {
      sector->addMem(memType, memName);
    }
  }

  // get the processing modules
  if (settings_->debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "Will read processing modules file";
  }

  ifstream inproc(settings_->processingModulesFile().c_str());
  assert(inproc.good());

  while (inproc.good()) {
    string procType, procName;
    inproc >> procType >> procName;
    if (!inproc.good())
      continue;
    if (settings_->writetrace()) {
      edm::LogVerbatim("Tracklet") << "Read process: " << procType << " " << procName;
    }
    for (auto& sector : sectors_) {
      sector->addProc(procType, procName);
    }
  }

  // get the wiring information
  if (settings_->debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "Will read wiring information";
  }

  ifstream inwire(settings_->wiresFile().c_str());
  assert(inwire.good());

  while (inwire.good()) {
    string line;
    getline(inwire, line);
    if (!inwire.good())
      continue;
    if (settings_->writetrace()) {
      edm::LogVerbatim("Tracklet") << "Line : " << line;
    }
    stringstream ss(line);
    string mem, tmp1, procin, tmp2, procout;
    ss >> mem >> tmp1 >> procin;
    if (procin == "output=>") {
      procin = "";
      ss >> procout;
    } else {
      ss >> tmp2 >> procout;
    }

    for (auto& sector : sectors_) {
      sector->addWire(mem, procin, procout);
    }
  }

  // get the DTC/cabling information
  ifstream indtc(settings_->DTCLinkLayerDiskFile());
  assert(indtc.good());
  string dtc;
  indtc >> dtc;
  while (indtc.good()) {
    vector<int> tmp;
    dtclayerdisk_[dtc] = tmp;
    int layerdisk;
    indtc >> layerdisk;
    while (layerdisk > 0) {
      dtclayerdisk_[dtc].push_back(layerdisk);
      indtc >> layerdisk;
    }
    indtc >> dtc;
  }

  cabling_ = make_unique<Cabling>(settings_->DTCLinkFile(), settings_->moduleCablingFile(), *settings_);
}

void TrackletEventProcessor::event(SLHCEvent& ev) {
  globals_->event() = &ev;

  tracks_.clear();

  eventnum_++;
  bool first = (eventnum_ == 1);

  cleanTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->clean();
  }
  cleanTimer_.stop();

  addStubTimer_.start();

  for (int j = 0; j < ev.nstubs(); j++) {
    L1TStub stub = ev.stub(j);

    int layer = stub.layer() + 1;
    int ladder = stub.ladder();
    int module = stub.module();

    string dtc = cabling_->dtc(layer, ladder, module);
    string dtcbase = dtc.substr(2, dtc.size() - 2);
    if (dtc[0] == 'n') {
      dtcbase = dtc.substr(0, 4) + dtc.substr(6, dtc.size() - 6);
    }

    cabling_->addphi(dtc, stub.phi(), layer, module);

    double phi = angle0to2pi::make0To2pi(stub.phi() + 0.5 * settings_->dphisectorHG());

    unsigned int isector = N_SECTOR * phi / (2 * M_PI);

    for (unsigned int k = 0; k < N_SECTOR; k++) {
      int diff = k - isector;
      if (diff > (int)N_SECTOR / 2)
        diff -= (int)N_SECTOR;
      if (diff < (-1) * (int)N_SECTOR / 2)
        diff += (int)N_SECTOR;
      if (abs(diff) > 1)
        continue;
      double phiminsect =
          k * 2 * M_PI / N_SECTOR - 0.5 * (settings_->dphisectorHG() - 2 * M_PI / N_SECTOR) - M_PI / N_SECTOR;
      double dphi = stub.phi() - phiminsect;
      if (dphi > M_PI)
        dphi -= 2 * M_PI;
      while (dphi < 0.0)
        dphi += 2 * M_PI;
      if (dphi > settings_->dphisectorHG())
        continue;
      bool add = sectors_[k]->addStub(stub, dtcbase);

      static std::map<string, ofstream*> dtcstubs;

      if (settings_->writeMem()) {
        vector<string> dtcs = cabling_->DTCs();
        for (const auto& dtc : dtcs) {
          string dtcbase = dtc.substr(2, dtc.size() - 2);
          if (dtc[0] == 'n') {
            dtcbase = dtc.substr(0, 4) + dtc.substr(6, dtc.size() - 6);
          }

          string fname = "../data/MemPrints/InputStubs/Link_";
          fname += dtcbase;
          if (dtcstubs.find(dtcbase + "A") != dtcstubs.end())
            continue;
          fname += "_A.dat";
          ofstream* out = new ofstream;
          out->open(fname.c_str());
          dtcstubs[dtcbase + "A"] = out;

          fname = "../data/MemPrints/InputStubs/Link_";
          fname += dtcbase;
          if (dtcstubs.find(dtcbase + "B") != dtcstubs.end())
            continue;
          fname += "_B.dat";
          out = new ofstream;
          out->open(fname.c_str());
          dtcstubs[dtcbase + "B"] = out;
        }

        static int oldevent = -1;
        if (eventnum_ != oldevent) {
          oldevent = eventnum_;
          for (auto& dtcstub : dtcstubs) {
            FPGAWord tmp;
            tmp.set(eventnum_ % 8, 3);
            (*(dtcstub.second)) << "BX " << tmp.str() << " Event : " << eventnum_ + 1 << endl;
          }
        }
      }

      if (add && settings_->writeMem() && k == settings_->writememsect()) {
        Stub fpgastub(stub, *settings_, sectors_[k]->phimin(), sectors_[k]->phimax());
        FPGAWord phi = fpgastub.phi();
        int topbit = phi.value() >> (phi.nbits() - 1);
        std::vector<int> tmp = dtclayerdisk_[dtcbase];
        int layerdisk = stub.layer() + 1;
        if (layerdisk > 999) {
          layerdisk = 10 + abs(stub.disk());
        }
        int layerdiskcode = -1;
        for (unsigned int i = 0; i < tmp.size(); i++) {
          if (tmp[i] == layerdisk)
            layerdiskcode = i;
        }
        if (layerdiskcode == -1) {
          edm::LogVerbatim("Tracklet") << "dtcbase layerdisk layer disk : " << dtcbase << " " << layerdisk << " "
                                       << stub.layer() + 1 << " " << stub.disk();
        }
        assert(layerdiskcode >= 0);
        assert(layerdiskcode < 4);
        FPGAWord ldcode;
        ldcode.set(layerdiskcode, 2);
        string dataword = ldcode.str() + "|" + fpgastub.str();
        if (topbit == 0) {
          (*dtcstubs[dtcbase + "A"]) << dataword << " " << trklet::hexFormat(dataword) << endl;
        } else {
          (*dtcstubs[dtcbase + "B"]) << dataword << " " << trklet::hexFormat(dataword) << endl;
        }
      }
    }
  }

  if (settings_->writeMem()) {
    for (unsigned int k = 0; k < N_SECTOR; k++) {
      if (k == settings_->writememsect())
        sectors_[k]->writeInputStubs(first);
    }
  }

  addStubTimer_.stop();

  // ----------------------------------------------------------------------------------------
  // Now start the tracklet processing

  // VM router
  VMRouterTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeVMR();
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeInputStubs(first);
      sectors_[k]->writeVMSTE(first);
      sectors_[k]->writeVMSME(first);
      sectors_[k]->writeAS(first);
    }
  }
  VMRouterTimer_.stop();

  // tracklet engine
  TETimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeTE();
  }
  TETimer_.stop();

  // tracklet engine displaced
  TEDTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeTED();
  }
  TEDTimer_.stop();

  // triplet engine
  TRETimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeTRE();
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeST(first);
    }
  }
  TRETimer_.stop();

  // tracklet processor (alternative implementation to TE+TC)
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeTP();
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeTPAR(first);
    }
  }

  for (unsigned int k = 0; k < N_SECTOR; k++) {
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeSP(first);
    }
  }

  // tracklet calculator
  TCTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeTC();
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeTPAR(first);
    }
  }
  TCTimer_.stop();

  int nTP = globals_->event()->nsimtracks();
  for (int iTP = 0; iTP < nTP; iTP++) {
    L1SimTrack simtrk = globals_->event()->simtrack(iTP);
    if (simtrk.pt() < 2.0)
      continue;
    if (std::abs(simtrk.vz()) > 15.0)
      continue;
    if (hypot(simtrk.vx(), simtrk.vy()) > 0.1)
      continue;
    bool electron = (abs(simtrk.type()) == 11);
    bool muon = (abs(simtrk.type()) == 13);
    bool pion = (abs(simtrk.type()) == 211);
    bool kaon = (abs(simtrk.type()) == 321);
    bool proton = (abs(simtrk.type()) == 2212);
    if (!(electron || muon || pion || kaon || proton))
      continue;
    int nlayers = 0;
    int ndisks = 0;
    int simtrackid = simtrk.trackid();
    unsigned int hitmask = ev.layersHit(simtrackid, nlayers, ndisks);
    if (nlayers + ndisks < 4)
      continue;

    if (settings_->writeMonitorData("HitEff")) {
      static ofstream outhit("hiteff.txt");
      outhit << simtrk.eta() << " " << (hitmask & 1) << " " << (hitmask & 2) << " " << (hitmask & 4) << " "
             << (hitmask & 8) << " " << (hitmask & 16) << " " << (hitmask & 32) << " " << (hitmask & 64) << " "
             << (hitmask & 128) << " " << (hitmask & 256) << " " << (hitmask & 512) << " " << (hitmask & 1024) << endl;
    }

    std::unordered_set<int> matchseed;
    for (unsigned int k = 0; k < N_SECTOR; k++) {
      std::unordered_set<int> matchseedtmp = sectors_[k]->seedMatch(iTP);
      matchseed.insert(matchseedtmp.begin(), matchseedtmp.end());
    }
    if (settings_->bookHistos()) {
      for (int iseed = 0; iseed < 8; iseed++) {
        bool eff = matchseed.find(iseed) != matchseed.end();
        globals_->histograms()->fillSeedEff(iseed, simtrk.eta(), eff);
      }
    }
  }

  // tracklet calculator displaced
  TCDTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeTCD();
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeTPAR(first);
      sectors_[k]->writeTPROJ(first);
    }
  }
  TCDTimer_.stop();

  // projection router
  PRTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executePR();
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeVMPROJ(first);
      sectors_[k]->writeAP(first);
    }
  }
  PRTimer_.stop();

  // match engine
  METimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeME();
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeCM(first);
    }
  }
  METimer_.stop();

  // match calculator
  MCTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeMC();
  }
  MCTimer_.stop();

  // match processor (alternative to ME+MC)
  MPTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeMP();
  }
  MPTimer_.stop();

  for (unsigned int k = 0; k < N_SECTOR; k++) {
    if (settings_->writeMem() && k == settings_->writememsect()) {
      sectors_[k]->writeMC(first);
    }
  }

  // fit track
  FTTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executeFT();
    if ((settings_->writeMem() || settings_->writeMonitorData("IFit")) && k == settings_->writememsect()) {
      sectors_[k]->writeTF(first);
    }
  }
  FTTimer_.stop();

  // purge duplicate
  PDTimer_.start();
  for (unsigned int k = 0; k < N_SECTOR; k++) {
    sectors_[k]->executePD(tracks_);
    if (((settings_->writeMem() || settings_->writeMonitorData("IFit")) && k == settings_->writememsect()) ||
        settings_->writeMonitorData("CT")) {
      sectors_[k]->writeCT(first);
    }
  }
  PDTimer_.stop();
}

void TrackletEventProcessor::printSummary() {
  if (settings_->writeMonitorData("Cabling")) {
    cabling_->writephirange();
  }

  if (settings_->bookHistos()) {
    globals_->histograms()->close();
  }

  edm::LogVerbatim("Tracklet")
      << "Process             Times called   Average time (ms)      Total time (s) \n"
      << "Cleaning              " << setw(10) << cleanTimer_.ntimes() << setw(20) << setprecision(3)
      << cleanTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << cleanTimer_.tottime() << "\n"
      << "Add Stubs             " << setw(10) << addStubTimer_.ntimes() << setw(20) << setprecision(3)
      << addStubTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << addStubTimer_.tottime() << "\n"
      << "VMRouter              " << setw(10) << VMRouterTimer_.ntimes() << setw(20) << setprecision(3)
      << VMRouterTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << VMRouterTimer_.tottime() << "\n"
      << "TrackletEngine        " << setw(10) << TETimer_.ntimes() << setw(20) << setprecision(3)
      << TETimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << TETimer_.tottime() << "\n"
      << "TrackletEngineDisplaced" << setw(10) << TEDTimer_.ntimes() << setw(20) << setprecision(3)
      << TEDTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << TEDTimer_.tottime() << "\n"
      << "TripletEngine         " << setw(10) << TRETimer_.ntimes() << setw(20) << setprecision(3)
      << TRETimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << TRETimer_.tottime() << "\n"
      << "TrackletCalculator    " << setw(10) << TCTimer_.ntimes() << setw(20) << setprecision(3)
      << TCTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << TCTimer_.tottime() << "\n"
      << "TrackletCalculatorDisplaced" << setw(10) << TCDTimer_.ntimes() << setw(20) << setprecision(3)
      << TCDTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << TCDTimer_.tottime() << "\n"
      << "ProjectionRouter      " << setw(10) << PRTimer_.ntimes() << setw(20) << setprecision(3)
      << PRTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << PRTimer_.tottime() << "\n"
      << "MatchEngine           " << setw(10) << METimer_.ntimes() << setw(20) << setprecision(3)
      << METimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << METimer_.tottime() << "\n"
      << "MatchCalculator       " << setw(10) << MCTimer_.ntimes() << setw(20) << setprecision(3)
      << MCTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << MCTimer_.tottime() << "\n"
      << "MatchProcessor        " << setw(10) << MPTimer_.ntimes() << setw(20) << setprecision(3)
      << MPTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << MPTimer_.tottime() << "\n"
      << "FitTrack              " << setw(10) << FTTimer_.ntimes() << setw(20) << setprecision(3)
      << FTTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << FTTimer_.tottime() << "\n"
      << "PurgeDuplicate        " << setw(10) << PDTimer_.ntimes() << setw(20) << setprecision(3)
      << PDTimer_.avgtime() * 1000.0 << setw(20) << setprecision(3) << PDTimer_.tottime();
}
