#include <iostream>
#include <cassert>
#include <iostream>
#include <fstream>
using namespace std;

#include "MusEcal.hh"
#include "../../interface/MEChannel.h"
#include "MERun.hh"
#include "MERunManager.hh"
#include "MEVarVector.hh"
#include "MECanvasHolder.hh"
#include "../../interface/MEGeom.h"

#include <TMath.h>

ClassImp(MusEcal);

bool MusEcal::verbose = true;

TString MusEcal::historyVarTitle[MusEcal::iSizeLV] = {"Normalized LASER Signal",
                                                      "Corrected Normalized LASER Signal",
                                                      "Ratio APD over PNA",
                                                      "Ratio APD over PNB",
                                                      "Ratio APD over PN",
                                                      "APD",
                                                      "APD Time",
                                                      "Normalization PNA",
                                                      "Normalization PNB",
                                                      "Ratio PNB over PNA",
                                                      "Time Rise = alpha*beta",
                                                      "alpha*beta (used)",
                                                      "Shape Correction",
                                                      "MATACQ Time Rise",
                                                      "MATACQ Amplitude",
                                                      "MATACQ Full Width at Half Maximum",
                                                      "MATACQ Full Width at 20\%",
                                                      "MATACQ Full Width at 80\%",
                                                      "MATACQ Time"};

TString MusEcal::historyVarName[MusEcal::iSizeLV] = {"NLS",
                                                     "CorNLS",
                                                     "APDoPNA",
                                                     "APDoPNB",
                                                     "APDoPN",
                                                     "APD",
                                                     "APDTime",
                                                     "PNA",
                                                     "PNB",
                                                     "PNBoPNA",
                                                     "alpha*beta",
                                                     "alpha*beta used",
                                                     "Shape Correction",
                                                     "MATACQ time rise",
                                                     "MATACQ amplitude",
                                                     "MATACQ fwhm",
                                                     "MATACQ fw20",
                                                     "MATACQ fw80",
                                                     "MATACQ time"};

int MusEcal::iGVar[MusEcal::iSizeLV] = {
    ME::iCrystal,   // NLS
    ME::iCrystal,   // CorNLS
    ME::iCrystal,   // APDoPNA
    ME::iCrystal,   // APDoPNB
    ME::iCrystal,   // APDoPN
    ME::iCrystal,   // APD
    ME::iCrystal,   // APDTime
    ME::iLMModule,  // PNA
    ME::iLMModule,  // PNB
    ME::iLMModule,  // PNBoPNA
    ME::iCrystal,   // alpha*beta
    ME::iCrystal,   // alpha*beta used
    ME::iCrystal,   // shape correction
    ME::iLMRegion,  // MATACQ time rise
    ME::iLMRegion,  // MATACQ amplitude
    ME::iLMRegion,  // MATACQ fwhm
    ME::iLMRegion,  // MATACQ fw20
    ME::iLMRegion,  // MATACQ fw80
    ME::iLMRegion   // MATACQ time
};

int MusEcal::historyVarZoom[ME::iSizeC][MusEcal::iSizeLV] = {{
                                                                 MusEcal::iZero,           // NLS
                                                                 MusEcal::iZero,           // CorNLS
                                                                 MusEcal::iZero,           // APDoPNA
                                                                 MusEcal::iZero,           // APDoPNB
                                                                 MusEcal::iZero,           // APDoPN
                                                                 MusEcal::iThirtyPercent,  // APD
                                                                 MusEcal::iTenPercent,     // APDTime
                                                                 MusEcal::iZero,           // PNA
                                                                 MusEcal::iZero,           // PNB
                                                                 MusEcal::iZero,           // PNBoPNA
                                                                 MusEcal::iTenPercent,     // AlphaBeta
                                                                 MusEcal::iZero,           // AlphaBeta used
                                                                 MusEcal::iZero,           // Shape correction
                                                                 MusEcal::iZero,           // MTQ_Trise
                                                                 MusEcal::iZero,           // MTQ_Ampl
                                                                 MusEcal::iZero,           // MTQ_Fwhm
                                                                 MusEcal::iZero,           // MTQ_Fw20
                                                                 MusEcal::iZero,           // MTQ_Fw80
                                                                 MusEcal::iZero            // MTQ_time
                                                             },
                                                             {
                                                                 MusEcal::iZero,           // NLS
                                                                 MusEcal::iZero,           // CorNLS
                                                                 MusEcal::iZero,           // APDoPNA
                                                                 MusEcal::iZero,           // APDoPNB
                                                                 MusEcal::iZero,           // APDoPN
                                                                 MusEcal::iThirtyPercent,  // APD
                                                                 MusEcal::iTenPercent,     // APDTime
                                                                 MusEcal::iZero,           // PNA
                                                                 MusEcal::iZero,           // PNB
                                                                 MusEcal::iZero,           // PNBoPNA
                                                                 MusEcal::iTenPercent,     // AlphaBeta
                                                                 MusEcal::iZero,           // AlphaBeta used
                                                                 MusEcal::iZero,           // Shape correction
                                                                 MusEcal::iZero,           // MTQ_Trise
                                                                 MusEcal::iZero,           // MTQ_Ampl
                                                                 MusEcal::iZero,           // MTQ_Fwhm
                                                                 MusEcal::iZero,           // MTQ_Fw20
                                                                 MusEcal::iZero,           // MTQ_Fw80
                                                                 MusEcal::iZero            // MTQ_time
                                                             }};

int MusEcal::historyVarColor[MusEcal::iSizeLV] = {
    kBlue,     // NLS
    kRed,      // CorNLS
    kBlue,     // APDoPNA
    kBlue,     // APDoPNB
    kBlue,     // APDoPN
    kBlue,     // APD
    kBlue,     // APDTime
    kRed,      // PNA
    kRed,      // PNB
    kRed,      // PNBoPNA
    kMagenta,  // AlphaBeta
    kMagenta,  // AlphaBeta used
    kMagenta,  // Shape correction
    kGreen,    // MTQ_Trise
    kGreen,    // MTQ_Ampl
    kGreen,    // MTQ_Fwhm
    kGreen,    // MTQ_Fw20
    kGreen,    // MTQ_Fw80
    kGreen     // MTQ_time
};

TString MusEcal::historyTPVarName[MusEcal::iSizeTPV] = {"Test-Pulse APD gain 0",
                                                        "Test-Pulse APD gain 1",
                                                        "Test-Pulse APD gain 2",
                                                        "Test-Pulse PNA gain 0",
                                                        "Test-Pulse PNA gain 1",
                                                        "Test-Pulse PNB gain 0",
                                                        "Test-Pulse PNB gain 1"};

TString MusEcal::historyTPVarTitle[MusEcal::iSizeTPV] = {"Test-Pulse APD for Gain 0",
                                                         "Test-Pulse APD for Gain 1",
                                                         "Test-Pulse APD for Gain 2",
                                                         "Test-Pulse PNA for Gain 0",
                                                         "Test-Pulse PNA for Gain 1",
                                                         "Test-Pulse PNB for Gain 0",
                                                         "Test-Pulse PNB for Gain 1"};

int MusEcal::iGTPVar[MusEcal::iSizeTPV] = {
    ME::iCrystal,   // Test-Pulse APD gain 0
    ME::iCrystal,   // Test-Pulse APD gain 1
    ME::iCrystal,   // Test-Pulse APD gain 2
    ME::iLMModule,  // Test-Pulse PNA gain 0
    ME::iLMModule,  // Test-Pulse PNA gain 1
    ME::iLMModule,  // Test-Pulse PNB gain 0
    ME::iLMModule   // Test-Pulse PNB gain 1
};

int MusEcal::historyTPVarColor[MusEcal::iSizeTPV] = {kCyan, kCyan + 150, kBlue, kBlue, kRed, kRed + 150, kRed + 100};

int MusEcal::historyTPVarZoom[MusEcal::iSizeTPV] = {MusEcal::iThreePerMil,
                                                    MusEcal::iThreePerMil,
                                                    MusEcal::iThreePerMil,
                                                    MusEcal::iZero,
                                                    MusEcal::iZero,
                                                    MusEcal::iZero,
                                                    MusEcal::iZero};

TString MusEcal::zoomName[MusEcal::iZero] = {"Range +/- 100%",
                                             "Range +/- 50%",
                                             "Range +/- 30%",
                                             "Range +/- 10%",
                                             "Range +/- 5%",
                                             "Range +/- 3%",
                                             "Range +/- 1%",
                                             "Range +/- 0.5%",
                                             "Range +/- 0.3%",
                                             "Range +/- 0.1%"};

double MusEcal::zoomRange[MusEcal::iZero] = {1., 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001};

int MusEcal::firstRun = 0;
int MusEcal::lastRun = 999999;

MusEcal::MusEcal(int type, int color) {
  if (verbose)
    cout << "Welcome to MusEcal" << endl;

  // set type and color
  setType(type, color);

  // set default crystal: first crystal of the first monitoring region
  // leaf: selected first crystal by default

  _lmr = -1;

  map<TString, MERunManager*>::iterator it = _runMgr.begin();
  for (; it != _runMgr.end(); ++it) {
    MERunManager* mgr_ = it->second;
    if (mgr_->size() == 0)
      continue;
    int ilmr = mgr_->LMRegion();
    setLMRegion(ilmr);
    break;
  }
  assert(it != _runMgr.end());

  setChannel(ME::iCrystal, 0, 0, false);

  // set default time
  setDefaultTime();

  // set default variable
  setVar();

  // no debug by default
  _debug = true;

  // no GUI by default
  _isGUI = false;

  // histograms
  _histoBooked = false;
  _ebHistoBooked = false;
  _eeHistoBooked = false;
  _seq_t = 0;

  if (verbose)
    cout << "Enjoy!" << endl;
}

MusEcal::~MusEcal() {}

void MusEcal::setType(int type, int color) {
  _type = type;
  if (verbose)
    cout << "Current type is: " << ME::type[_type] << endl;

  if (_type == ME::iLaser) {
    // default color
    _color = color;
    if (verbose)
      cout << "Current color is: " << ME::color[_color] << endl;
  }

  // Barrel: Laser monitoring regions between 1 and 92
  for (unsigned int lmr = 1; lmr <= 92; lmr++) {
    TString str_ = MusEcal::mgrName(lmr, _type, _color);
    if (_runMgr.count(str_) == 0) {
      MERunManager* runMgr_ = new MERunManager(lmr, _type, _color);
      if (runMgr_->size() == 0) {
        delete runMgr_;
        continue;
      }
      _runMgr[str_] = runMgr_;
    }
  }
}

void MusEcal::setLMRegion(int lmr) {
  if (lmr == _lmr)
    return;
  if (curMgr() != 0)
    curMgr()->refresh();

  _lmr = lmr;
  cout << "Current Laser monitoring region is: "
       //       << MusEcal::mgrName( _lmr, _type, _color )
       << _lmr << " (" << ME::smName(_lmr) << ")" << endl;

  // filling maps
  curMgr()->fillMaps();
}

//
// access to run managers
//
MERunManager* MusEcal::runMgr(int lmr, int type, int color) {
  TString str_ = MusEcal::mgrName(lmr, type, color);
  if (_runMgr.count(str_) == 0) {
    //      cout << "Non existent Run manager" << endl;
    return 0;
  } else
    return _runMgr[str_];
}

void MusEcal::setTime(ME::Time time) {
  time_t t = time;
  if (verbose)
    cout << "Reference date: " << ctime(&t);
  map<TString, MERunManager*>::iterator it = _runMgr.begin();
  unsigned int currun_(0);
  unsigned int curlb_(0);
  for (; it != _runMgr.end(); ++it) {
    MERunManager* mgr_ = it->second;
    bool ok = mgr_->setCurrentRun(time);
    short int sign_(0);
    vector<ME::Time> dtvec;
    int nmaxlb = 7;
    if (ok) {
      MERun* run_ = mgr_->curRun();
      if (currun_ == 0) {
        currun_ = run_->run();
        curlb_ = run_->lb();
      }
      if (run_->run() != currun_)
        ok = false;
      else if (run_->lb() < curlb_ || run_->lb() > curlb_ + nmaxlb)
        ok = false;
      else {
        curlb_ = run_->lb();
        ME::Time t_ = run_->time();
        dtvec = ME::timeDiff(t_, time, sign_);
        if (dtvec[ME::iDay] != 0)
          ok = false;
        else if (dtvec[ME::iHour] != 0)
          ok = false;
        else if (dtvec[ME::iMinute] > 30)
          ok = false;
      }
    }
    if (!ok) {
      mgr_->setNoCurrent();
    }
    if (verbose) {
      int lmr_ = mgr_->LMRegion();
      cout << "LMR=" << lmr_ << " (" << ME::smName(lmr_) << ")";
      if (ok) {
        cout << " --> " << mgr_->curRun()->header().rundir << " d=" << dtvec[ME::iDay] << "/"
             << " h=" << dtvec[ME::iHour] << "/"
             << " m=" << dtvec[ME::iMinute] << "/"
             << " s=" << dtvec[ME::iSecond];
        if (sign_ != 0)
          cout << ((sign_ > 0) ? " in future" : " -------");
      } else {
        cout << "--> not in sequence ";
      }
      cout << endl;
    }
  }
  //  _time = time;
  //  _time = _runMgr.begin()->second->curRun()->time();
  _time = runMgr()->curRun()->time();
}

bool MusEcal::nextSequence() {
  // fixme: this assumes that the current LMR is always present
  //  MERunManager* mgr_ = _runMgr.begin()->second;
  MERunManager* mgr_ = runMgr();
  MusEcal::RunIterator it = mgr_->cur();
  if (it == mgr_->end())
    return false;
  ++it;
  if (it == mgr_->end())
    return false;
  ME::Time time = it->first;
  setTime(time);
  return true;
}

void MusEcal::setRunAndSequence(unsigned int irun, int iseq) {
  MERunManager* mgr_ = _runMgr.begin()->second;
  // MERunManager* mgr_ = runMgr();
  MusEcal::RunIterator runit = mgr_->it();
  for (; runit != mgr_->end(); ++runit) {
    MERun* run_ = runit->second;
    if (run_->run() == irun)
      iseq--;
    if (iseq == 0)
      break;
  }
  ME::Time time = runit->second->time();
  setTime(time);
}

void MusEcal::setDefaultTime() {
  MERunManager* mgr_ = _runMgr.begin()->second;
  // MERunManager* mgr_ = runMgr();
  MERun* run_ = mgr_->firstRun();
  if (run_ == 0) {
    cout << "run pointer is null " << endl;
    return;
  }
  ME::Time time = run_->time();
  setTime(time);
}

void MusEcal::dumpVector(int ivar) {
  // dummy !!!
  //  cout << "Dump current vector as ascii file" << endl;
  if (_leaf == 0) {
    cout << "Please select a channel first " << endl;
    return;
  }
  // test
  MEChannel* leaf_ = _leaf;
  MEVarVector* apdVector_ = curMgr()->apdVector(leaf_);
  vector<ME::Time> time;
  vector<float> val;
  vector<bool> flag;
  int ii = ivar;
  apdVector_->getTimeValAndFlag(ii, time, val, flag);
  unsigned int nrun = time.size();
  //  cout << endl;
  //  cout << leaf_->oneLine() << endl;
  for (unsigned int irun = 0; irun < nrun; irun++) {
    float dt_ = ME::timeDiff(time[irun], _time, ME::iHour);
    float val_ = val[irun];
    bool flag_ = flag[irun];
    TString str_ = ME::APDPrimVar[ii];
    time_t t = time[irun];
    struct tm* tb;
    tb = localtime(&t);
    //      cout << ctime(&t);
    int mon_ = tb->tm_mon + 1;
    int day_ = tb->tm_mday;
    int hr_ = tb->tm_hour + 1;
    int min_ = tb->tm_min;
    int sec_ = tb->tm_sec;
    cout << tb->tm_year + 1900 << "/";
    if (mon_ < 10)
      cout << "0";
    cout << mon_ << "/";
    if (day_ < 10)
      cout << "0";
    cout << day_ << "-";
    if (hr_ < 10)
      cout << "0";
    cout << hr_ << ":";
    if (min_ < 10)
      cout << "0";
    cout << min_ << ":";
    if (sec_ < 10)
      cout << "0";
    cout << sec_;
    cout << "\tt(sec)=" << t;
    cout << "\tdt(hrs)=" << dt_;
    cout << "\t\t" << str_ << "=" << val_ << "\tok=" << flag_ << endl;
  }
}

void MusEcal::setChannel(int ig, int ix, int iy, bool useGlobal) {
  int reg_(0);
  int sect_(0);
  int dcc_(0);
  int side_(0);
  ME::regionAndSector(_lmr, reg_, sect_, dcc_, side_);
  MEChannel* leaf_(0);
  if (reg_ == ME::iEBM || reg_ == ME::iEBP) {
    if (!useGlobal) {
      if (_lmr < 0)
        _lmr = 1;
      MEEBGeom::EtaPhiCoord etaphi_ = MEEBGeom::globalCoord(sect_, ix, iy);
      ix = etaphi_.first;
      iy = etaphi_.second;
    }
    // FIXME: ONLY BARREL FOR THE MOMENT
    assert(ix != 0 && std::abs(ix) <= 85);
    assert(iy >= 1 && iy <= 360);
    if (ix < 0)
      reg_ = ME::iEBM;
    if (ix > 0)
      reg_ = ME::iEBP;
    leaf_ = ME::regTree(reg_)->getChannel(ig, ix, iy);
  } else {
    if (!useGlobal) {
      leaf_ = ME::regTree(reg_)->getDescendant(ME::iLMRegion, _lmr)->getFirstDescendant(ME::iCrystal);

    } else {
      leaf_ = ME::regTree(reg_)->getChannel(ig, ix, iy);
    }
  }
  assert(leaf_ != 0);
  setChannel(leaf_);
}

void MusEcal::setChannel(MEChannel* leaf) {
  if (leaf == 0) {
    cout << "Wrong selection of channel" << endl;
    return;
  }
  if (leaf->ig() < ME::iLMRegion)
    return;
  int lmr_ = leaf->getAncestor(ME::iLMRegion)->id();
  if (lmr_ != _lmr)
    setLMRegion(lmr_);
  _leaf = leaf;
  cout << "\nCurrent channel: " << _leaf->oneLine() << endl;
}

void MusEcal::oneLevelUp() {
  MEChannel* leaf_ = _leaf->m();
  if (leaf_->ig() < ME::iLMRegion) {
    cout << "Already at Laser Monitoring Region level" << endl;
    return;
  }
  setChannel(leaf_);
}

void MusEcal::refresh() {}

bool MusEcal::isBarrel() {
  int reg_ = ME::ecalRegion(_lmr);
  return (reg_ == ME::iEBM || reg_ == ME::iEBP);
}

void MusEcal::bookHistograms() {
  if (_histoBooked)
    return;

  if (!_ebHistoBooked) {
    cout << "Book histograms for barrel" << endl;
    bookEBAPDHistograms();
    bookEBPNHistograms();
    _ebHistoBooked = true;
  }
  if (!_eeHistoBooked) {
    cout << "Book histograms for endcaps" << endl;
    bookEEAPDHistograms();
    _eeHistoBooked = true;
  }
  _histoBooked = _ebHistoBooked && _eeHistoBooked;

  if (_seq_t == 0) {
    _seq_t = new TTree("Sequences", "Sequences");
    _seq_t->Branch("run", &_seq_run, "run/I");
    _seq_t->Branch("lb", &_seq_lb, "lb/I");
    _seq_t->Branch("tbeg", &_seq_tbeg, "tbeg/I");
    _seq_t->Branch("tlmr", &_seq_tlmr, "tlmr[92]/I");

    _seq_t->SetBranchAddress("run", &_seq_run);
    _seq_t->SetBranchAddress("lb", &_seq_lb);
    _seq_t->SetBranchAddress("tbeg", &_seq_tbeg);
    _seq_t->SetBranchAddress("tlmr", &_seq_tlmr);
  }
}

void MusEcal::histConfig() {
  //
  // if there is a config file, modify the specified histogram limits
  //

  TString fileroot = TString(std::getenv("MECONFIG"));
  fileroot += "/";
  fileroot += TString(std::getenv("MEPERIOD"));
  TString ext_[2] = {"EB", "EE"};

  for (int ii = 0; ii < 2; ii++) {
    TString filename = fileroot;
    filename += "_";
    filename += ext_[ii];
    filename += ".config";

    FILE* test;
    test = fopen(filename, "r");
    char c;
    if (test) {
      std::ifstream fin(filename);
      fclose(test);
      while ((c = fin.peek()) != EOF) {
        TString ss;
        fin >> ss;
        int nbin;
        fin >> nbin;
        float min;
        fin >> min;
        float max;
        fin >> max;

        if (ii == 0) {
          _eb_nbin[ss] = nbin;
          _eb_min[ss] = min;
          _eb_max[ss] = max;
        } else {
          _ee_nbin[ss] = nbin;
          _ee_min[ss] = min;
          _ee_max[ss] = max;
        }
      }
    } else
      cout << "WARNING -- Histogram Configuration File " << filename << " NOT FOUND " << endl;
    //	abort();
  }
}

int MusEcal::hist_nbin(TString& str) {
  if (isBarrel()) {
    if (_eb_nbin.count(str) != 0)
      return _eb_nbin[str];
  } else {
    if (_ee_nbin.count(str) != 0)
      return _ee_nbin[str];
  }
  return 0;
}

float MusEcal::hist_min(TString& str) {
  assert(hist_nbin(str) != 0);
  if (isBarrel())
    return _eb_min[str];
  else
    return _ee_min[str];
}

float MusEcal::hist_max(TString& str) {
  assert(hist_nbin(str) != 0);
  if (isBarrel())
    return _eb_max[str];
  else
    return _ee_max[str];
}

void MusEcal::bookEBAPDHistograms() {
  _febgeom = TFile::Open(ME::path() + "geom/ebgeom.root");
  assert(_febgeom != 0);
  _eb_h = (TH2*)_febgeom->Get("eb");
  _eb_h->SetStats(kFALSE);
  _eb_h->GetXaxis()->SetTitle("ieta");
  _eb_h->GetXaxis()->CenterTitle();
  _eb_h->GetYaxis()->SetTitle("iphi");
  _eb_h->GetYaxis()->CenterTitle();

  _eb_loc_h = (TH2*)_febgeom->Get("eb_loc");
  _eb_loc_h->SetStats(kFALSE);
  _eb_loc_h->GetXaxis()->SetTitle("ix");
  _eb_loc_h->GetXaxis()->CenterTitle();
  _eb_loc_h->GetYaxis()->SetTitle("iy");
  _eb_loc_h->GetYaxis()->CenterTitle();

  TH2* h2_;
  TH1* h1_;

  int type = _type;
  //  for( int type=ME::iLaser; type<=ME::iTestPulse; type++ )
  {
    unsigned size_(0);
    TString str0_;
    if (type == ME::iLaser) {
      size_ = ME::iSizeAPD;
      str0_ = "APD-";
    } else if (type == ME::iTestPulse) {
      size_ = ME::iSizeTPAPD;
      str0_ = "TPAPD-";
    }
    for (unsigned int ii = 0; ii < size_; ii++) {
      TString varName_;
      if (type == ME::iLaser)
        varName_ = ME::APDPrimVar[ii];
      else if (type == ME::iTestPulse)
        varName_ = ME::TPAPDPrimVar[ii];
      TString str_ = str0_ + varName_;
      h2_ = (TH2*)_eb_h->Clone(str_);
      _eb_m[str_] = h2_;
      if (_eb_nbin.count(str_) != 0) {
        h2_->SetMinimum(_eb_min[str_]);
        h2_->SetMaximum(_eb_max[str_]);
      }
      h2_ = (TH2*)_eb_loc_h->Clone(str_ + "_loc");
      _eb_loc_m[str_] = h2_;
      if (_eb_nbin.count(str_) != 0) {
        h2_->SetMinimum(_eb_min[str_]);
        h2_->SetMaximum(_eb_max[str_]);
      }

      TString ext_ = "_1D";
      h1_ = new TH1F(str_ + ext_, varName_, 2448, -0.5, 2447.5);
      MECanvasHolder::setHistoStyle(h1_);
      h1_->SetStats(kFALSE);
      h1_->GetXaxis()->SetTitle("SC number (LM numbering)");
      h1_->GetYaxis()->SetTitle(ME::APDPrimVar[ii]);
      if (_eb_nbin.count(str_) != 0) {
        h1_->SetMinimum(_eb_min[str_]);
        h1_->SetMaximum(1.1 * _eb_max[str_]);
      }
      _eb_m[str_ + ext_] = h1_;

      ext_ = "_VS_CHANNEL";
      h1_ = new TH1F(str_ + ext_, varName_, 900, -0.5, 899.5);
      MECanvasHolder::setHistoStyle(h1_);
      h1_->SetStats(kFALSE);
      h1_->GetXaxis()->SetTitle("Channel number (LM numbering)");
      h1_->GetYaxis()->SetTitle(ME::APDPrimVar[ii]);
      if (_eb_nbin.count(str_) != 0) {
        h1_->SetMinimum(_eb_min[str_]);
        h1_->SetMaximum(1.1 * _eb_max[str_]);
      }
      _eb_loc_m[str_ + ext_] = h1_;
      ext_ += "_sel";
      h1_ = (TH1*)h1_->Clone(str_ + ext_);
      h1_->SetLineColor(kRed);
      h1_->SetFillColor(46);
      _eb_loc_m[str_ + ext_] = h1_;

      if (_eb_nbin.count(str_) != 0) {
        ext_ = "_HIST";
        h1_ = new TH1F(str_ + ext_, varName_, _eb_nbin[str_], _eb_min[str_], _eb_max[str_]);
        MECanvasHolder::setHistoStyle(h1_);
        _eb_loc_m[str_ + ext_] = h1_;
        ext_ += "_sel";
        h1_ = (TH1*)h1_->Clone(str_ + ext_);
        h1_->SetLineColor(kRed);
        h1_->SetFillColor(46);
        _eb_loc_m[str_ + ext_] = h1_;
      }
    }
  }
}

void MusEcal::bookEEAPDHistograms() {
  _feegeom = TFile::Open(ME::path() + "geom/eegeom.root");
  assert(_feegeom != 0);
  _ee_h = (TH2*)_feegeom->Get("ee");
  _ee_h->SetStats(kFALSE);
  _ee_h->GetXaxis()->SetTitle("ix");
  _ee_h->GetXaxis()->CenterTitle();
  _ee_h->GetYaxis()->SetTitle("iy");
  _ee_h->GetYaxis()->CenterTitle();

  _ee_loc_h[0] = 0;
  for (int isect = 1; isect <= 9; isect++) {
    TString sect_("ee_S");
    sect_ += isect;
    _ee_loc_h[isect] = (TH2*)_feegeom->Get(sect_);
    _ee_loc_h[isect]->SetStats(kFALSE);
    _ee_loc_h[isect]->GetXaxis()->SetTitle("ix");
    _ee_loc_h[isect]->GetXaxis()->CenterTitle();
    _ee_loc_h[isect]->GetYaxis()->SetTitle("iy");
    _ee_loc_h[isect]->GetYaxis()->CenterTitle();
  }

  // get the master tree for EE-plus (sectors 1 to 9)
  MEChannel* tree = ME::regTree(ME::iEEP);

  TH2* h2_;
  TH1* h1_;

  int type = _type;
  //  for( int type=ME::iLaser; type<=ME::iTestPulse; type++ )
  unsigned size_(0);
  TString str0_;
  if (type == ME::iLaser) {
    size_ = ME::iSizeAPD;
    str0_ = "APD-";
  } else if (type == ME::iTestPulse) {
    size_ = ME::iSizeTPAPD;
    str0_ = "TPAPD-";
  }
  for (unsigned int ii = 0; ii < size_; ii++) {
    TString varName_;
    if (type == ME::iLaser)
      varName_ = ME::APDPrimVar[ii];
    else if (type == ME::iTestPulse)
      varName_ = ME::TPAPDPrimVar[ii];
    TString str_ = str0_ + varName_;
    h2_ = (TH2*)_ee_h->Clone(str_);
    _ee_m[str_] = h2_;
    if (_ee_nbin.count(str_) != 0) {
      h2_->SetMinimum(_ee_min[str_]);
      h2_->SetMaximum(_ee_max[str_]);
    }
    for (int isect = 1; isect <= 9; isect++) {
      TString str__ = str_;
      str__ += "_";
      str__ += isect;
      h2_ = (TH2*)_ee_loc_h[isect]->Clone(str__ + "_loc");
      _ee_loc_m[str__] = h2_;
      if (_ee_nbin.count(str_) != 0) {
        h2_->SetMinimum(_ee_min[str_]);
        h2_->SetMaximum(_ee_max[str_]);
      }

      TString ext_ = "_VS_CHANNEL";

      // trick to get the correct number of bins...
      MEChannel* tree_ = tree->getDescendant(ME::iSector, isect);
      vector<MEChannel*> vec_;
      tree_->getListOfDescendants(ME::iCrystal, vec_);
      int nbin_ = vec_.size();
      float xmin_ = -0.5;
      float xmax_ = nbin_ - 0.5;

      h1_ = new TH1F(str__ + ext_, varName_, nbin_, xmin_, xmax_);
      MECanvasHolder::setHistoStyle(h1_);
      h1_->SetStats(kFALSE);
      h1_->GetXaxis()->SetTitle("Channel number (LM numbering)");
      h1_->GetYaxis()->SetTitle(ME::APDPrimVar[ii]);
      if (_ee_nbin.count(str_) != 0) {
        h1_->SetMinimum(_ee_min[str_]);
        h1_->SetMaximum(1.1 * _ee_max[str_]);
      }
      _ee_loc_m[str__ + ext_] = h1_;
      ext_ += "_sel";
      h1_ = (TH1*)h1_->Clone(str__ + ext_);
      h1_->SetLineColor(kRed);
      h1_->SetFillColor(46);
      _ee_loc_m[str__ + ext_] = h1_;

      if (_ee_nbin.count(str_) != 0) {
        ext_ = "_HIST";
        h1_ = new TH1F(str__ + ext_, varName_, _ee_nbin[str_], _ee_min[str_], _ee_max[str_]);
        MECanvasHolder::setHistoStyle(h1_);
        _ee_loc_m[str__ + ext_] = h1_;
        ext_ += "_sel";
        h1_ = (TH1*)h1_->Clone(str__ + ext_);
        h1_->SetLineColor(kRed);
        h1_->SetFillColor(46);
        _ee_loc_m[str__ + ext_] = h1_;
      }
    }

    TString ext_ = "_1D";
    h1_ = new TH1F(str_ + ext_, varName_, 624, -0.5, 623.5);
    MECanvasHolder::setHistoStyle(h1_);
    h1_->SetStats(kFALSE);
    h1_->GetXaxis()->SetTitle("SC number (LM numbering)");
    h1_->GetYaxis()->SetTitle(ME::APDPrimVar[ii]);
    if (_ee_nbin.count(str_) != 0) {
      h1_->SetMinimum(_ee_min[str_]);
      h1_->SetMaximum(1.1 * _ee_max[str_]);
    }
    _ee_m[str_ + ext_] = h1_;
  }
}

void MusEcal::bookEBPNHistograms() {
  TH1* h1_;

  int type = _type;
  //  for( int type=ME::iLaser; type<=ME::iTestPulse; type++ )
  {
    unsigned size_(0);
    TString str0_;
    if (type == ME::iLaser) {
      size_ = ME::iSizePN;
      str0_ = "PN-";
    } else if (type == ME::iTestPulse) {
      size_ = ME::iSizeTPPN;
      str0_ = "TPPN-";
    }
    for (unsigned int ii = 0; ii < size_; ii++) {
      TString varName_, str_;
      if (type == ME::iLaser)
        varName_ = ME::PNPrimVar[ii];
      else if (type == ME::iTestPulse)
        varName_ = ME::TPPNPrimVar[ii];
      str_ = str0_ + varName_;

      // global histogram
      h1_ = new TH1F(str_, varName_, 648, -0.5, 647.5);
      MECanvasHolder::setHistoStyle(h1_);
      h1_->SetStats(kFALSE);
      h1_->GetXaxis()->SetTitle("PN number (LM numbering)");
      h1_->GetYaxis()->SetTitle(varName_);
      if (_ee_nbin.count(str_) != 0) {
        h1_->SetMinimum(_ee_min[str_]);
        h1_->SetMaximum(1.1 * _ee_max[str_]);
      }
      _eb_m[str_] = h1_;

      // local histogram
      TString ext_ = "_LOCAL";
      h1_ = new TH1F(str_ + ext_, varName_, 18, -0.5, 17.5);
      MECanvasHolder::setHistoStyle(h1_);
      h1_->SetStats(kFALSE);
      h1_->GetXaxis()->SetTitle("PN number (LM numbering)");
      h1_->GetYaxis()->SetTitle(varName_);
      if (_ee_nbin.count(str_) != 0) {
        h1_->SetMinimum(_ee_min[str_]);
        h1_->SetMaximum(1.1 * _ee_max[str_]);
      }
      _eb_loc_m[str_ + ext_] = h1_;
    }
  }
}
void MusEcal::bookEEPNHistograms() {}

void MusEcal::fillHistograms() {
  if (!_histoBooked)
    bookHistograms();
  fillEBGlobalHistograms();
  fillEEGlobalHistograms();
  if (isBarrel()) {
    fillEBLocalHistograms();
  } else {
    fillEELocalHistograms();
  }
}

void MusEcal::fillEBGlobalHistograms() {
  if (!_histoBooked)
    bookHistograms();
  cout << "Filling EB Global Histograms";
  // cleaning
  for (map<TString, TH1*>::iterator it = _eb_m.begin(); it != _eb_m.end(); ++it) {
    it->second->Reset();
  }

  // filling the 2D histogram
  //  for( int ilmr=1; ilmr<=72; ilmr++ )
  TString rundir_;
  MERunManager* firstMgr_ = _runMgr.begin()->second;
  if (firstMgr_ != 0) {
    MERun* firstRun_ = firstMgr_->curRun();
    if (firstRun_ != 0) {
      rundir_ += firstRun_->rundir();
    }
  }
  TString titleW;

  titleW = ME::type[_type];
  if (_type == ME::iLaser) {
    titleW += " ";
    titleW += ME::color[_color];
  }
  titleW += " ECAL Barrel";
  titleW += " APD XXXXX";
  titleW += " ";
  titleW += rundir_;

  unsigned size_(0);
  TString str0_;
  int table_(0);

  vector<MEChannel*> vec;
  for (int ism = 1; ism <= 36; ism++) {
    cout << "." << flush;
    int idcc = MEEBGeom::dccFromSm(ism);
    int ilmr;
    MERunManager* mgr_;
    MERun* run_;
    for (int side = 0; side < 2; side++) {
      ilmr = ME::lmr(idcc, side);
      mgr_ = runMgr(ilmr);
      if (mgr_ == 0)
        continue;
      run_ = mgr_->curRun();
      if (run_ == 0)
        continue;

      // first APD
      if (_type == ME::iLaser) {
        size_ = ME::iSizeAPD;
        str0_ = "APD-";
        table_ = ME::iLmfLaserPrim;
      } else if (_type == ME::iTestPulse) {
        size_ = ME::iSizeTPAPD;
        str0_ = "TPAPD-";
        table_ = ME::iLmfTestPulsePrim;
      }

      vec.clear();
      mgr_->tree()->getListOfChannels(vec);
      for (unsigned int jj = 0; jj < size_; jj++) {
        TString varName_;
        if (_type == ME::iLaser) {
          varName_ = ME::APDPrimVar[jj];
        } else if (_type == ME::iTestPulse) {
          varName_ = ME::TPAPDPrimVar[jj];
        }
        for (unsigned int ii = 0; ii < vec.size(); ii++) {
          MEChannel* leaf_ = vec[ii];
          int ieta = leaf_->ix();
          int iphi = leaf_->iy();
          MEEBGeom::XYCoord ixy = MEEBGeom::localCoord(ieta, iphi);
          int ix = ixy.first;
          int iy = ixy.second;
          float val = run_->getVal(table_, jj, ix, iy);
          TString str_ = str0_ + varName_;
          TH2* h2_;
          h2_ = (TH2*)_eb_m[str_];
          TString title_ = titleW;
          title_.ReplaceAll("XXXXX", str_);
          h2_->SetTitle(title_);
          h2_->Fill(ieta, iphi, val);
          TH1* h1_;
          TString ext_ = "_1D";
          h1_ = (TH1*)_eb_m[str_ + ext_];
          h1_->SetTitle(title_);
          int ival = ii + ((ilmr - 1) / 2) * 1700;
          if (side == 1)
            ival += 900;
          h1_->Fill(ival / 25, val / 25);
        }
      }

      // then PNs
      if (_type == ME::iLaser) {
        size_ = ME::iSizePN;
        str0_ = "PN-";
        table_ = ME::iLmfLaserPnPrim;
      } else if (_type == ME::iTestPulse) {
        size_ = ME::iSizeTPPN;
        str0_ = "TPPN-";
        table_ = ME::iLmfTestPulsePnPrim;
      }
      vec.clear();
      mgr_->tree()->getListOfDescendants(ME::iLMModule, vec);
      for (unsigned int jj = 0; jj < size_; jj++) {
        TString varName_;
        if (_type == ME::iLaser) {
          varName_ = ME::PNPrimVar[jj];
        } else if (_type == ME::iTestPulse) {
          varName_ = ME::TPPNPrimVar[jj];
        }
        for (unsigned int ii = 0; ii < vec.size(); ii++) {
          MEChannel* leaf_ = vec[ii];
          int ilm = leaf_->id();
          TString str_ = str0_ + varName_;
          TH1* h1_;
          h1_ = (TH1*)_eb_m[str_];
          TString title_ = titleW;
          title_.ReplaceAll("APD", "PN");
          title_.ReplaceAll("XXXXX", str_);
          h1_->SetTitle(title_);
          for (int ipn = 0; ipn < 2; ipn++) {
            float val = run_->getVal(table_, jj, ilm, ipn);
            int ival = 2 * ((ilm - 1) / 2) + ipn + ((ilmr - 1) / 2) * 18;
            if (ilm % 2 == 0)
              ival += 10;
            h1_->Fill(ival, val);
          }
        }
      }
    }
  }
  cout << " Done." << endl;
}

void MusEcal::fillEBLocalHistograms() {
  if (!_histoBooked)
    bookHistograms();
  cout << "Filling EB Local Histograms";
  for (map<TString, TH1*>::iterator it = _eb_loc_m.begin(); it != _eb_loc_m.end(); ++it) {
    it->second->Reset();
  }
  MERunManager* mgr = curMgr();
  if (mgr == 0)
    return;
  MERun* run_ = mgr->curRun();
  TString rundir = "No Data";
  if (run_ != 0)
    rundir = run_->rundir();

  TString titleW;
  titleW = ME::type[_type];
  if (_type == ME::iLaser) {
    titleW += " ";
    titleW += ME::color[_color];
  }
  titleW += " ";
  titleW += "YYYYY";
  titleW += " APD XXXXX";
  titleW += " ";
  titleW += rundir;
  TString title_;

  MEChannel* l_[2] = {0, 0};
  l_[0] = mgr->tree();
  l_[1] = _leaf;

  unsigned size_(0);
  TString str0_;
  int table_(0);

  if (_type == ME::iLaser) {
    size_ = ME::iSizeAPD;
    str0_ = "APD-";
    table_ = ME::iLmfLaserPrim;
  } else if (_type == ME::iTestPulse) {
    size_ = ME::iSizeTPAPD;
    str0_ = "TPAPD-";
    table_ = ME::iLmfTestPulsePrim;
  }

  for (unsigned int jj = 0; jj < size_; jj++) {
    TString varName_;
    if (_type == ME::iLaser) {
      varName_ = ME::APDPrimVar[jj];
    } else if (_type == ME::iTestPulse) {
      varName_ = ME::TPAPDPrimVar[jj];
    }
    TH2* h2_(0);
    TH1* h1_[2] = {0, 0};
    TH1* h1_chan[2] = {0, 0};
    TString str_ = str0_ + varName_;
    h2_ = (TH2*)_eb_loc_m[str_];
    title_ = titleW;
    title_.ReplaceAll("XXXXX", varName_);
    //  title_.ReplaceAll("YYYYY",_leaf->oneWord( ME::iLMRegion ) );
    title_.ReplaceAll("YYYYY", _leaf->oneWord());
    h2_->SetTitle(title_);

    title_ = titleW;
    title_.ReplaceAll("XXXXX", varName_);
    title_.ReplaceAll("YYYYY", _leaf->oneWord());
    TString ext_ = "_HIST";
    if (_eb_loc_m.count(str_ + ext_) != 0) {
      h1_[0] = (TH1*)_eb_loc_m[str_ + ext_];
      h1_[0]->SetTitle(title_);
      ext_ += "_sel";
      h1_[1] = (TH1*)_eb_loc_m[str_ + ext_];
      h1_[1]->SetTitle(title_);
    }

    ext_ = "_VS_CHANNEL";
    h1_chan[0] = (TH1*)_eb_loc_m[str_ + ext_];
    h1_chan[0]->SetTitle(title_);
    ext_ += "_sel";
    h1_chan[1] = (TH1*)_eb_loc_m[str_ + ext_];
    h1_chan[1]->SetTitle(title_);

    if (run_ != 0) {
      vector<MEChannel*> vec[2];
      l_[0]->getListOfChannels(vec[0]);
      l_[1]->getListOfChannels(vec[1]);
      for (unsigned int ii = 0; ii < vec[0].size(); ii++) {
        MEChannel* leaf_ = vec[0][ii];
        int ieta = leaf_->ix();
        int iphi = leaf_->iy();
        MEEBGeom::XYCoord ixy = MEEBGeom::localCoord(ieta, iphi);
        int ix = ixy.first;
        int iy = ixy.second;
        float val = run_->getVal(table_, jj, ix, iy);
        if (h1_chan[0] != 0)
          h1_chan[0]->Fill(ii, val);
        if (h1_[0] != 0)
          h1_[0]->Fill(val);
        vector<MEChannel*>::iterator it_ = find(vec[1].begin(), vec[1].end(), leaf_);
        if (it_ != vec[1].end()) {
          if (h1_chan[1] != 0)
            h1_chan[1]->Fill(ii, val);
          if (h1_[1] != 0)
            h1_[1]->Fill(val);
          // test
          if (_leaf->ig() > ME::iLMRegion) {
            TAxis* xaxis_ = h2_->GetXaxis();
            TAxis* yaxis_ = h2_->GetYaxis();
            int a1_ = xaxis_->GetFirst();
            int a2_ = xaxis_->GetLast();
            int b1_ = yaxis_->GetFirst();
            int b2_ = yaxis_->GetLast();
            float max_ = h2_->GetMaximum();
            for (int ix_ = -1; ix_ >= xaxis_->GetBinCenter(a1_); ix_--) {
              h2_->Fill(ix_, iy, max_);
            }
            for (int ix_ = 85; ix_ <= xaxis_->GetBinCenter(a2_); ix_++) {
              h2_->Fill(ix_, iy, max_);
            }
            for (int iy_ = -1; iy_ >= yaxis_->GetBinCenter(b1_); iy_--) {
              h2_->Fill(ix, iy_, max_);
            }
            for (int iy_ = 20; iy_ <= yaxis_->GetBinCenter(b2_); iy_++) {
              h2_->Fill(ix, iy_, max_);
            }
          }
        }

        h2_->Fill(ix, iy, val);
      }
    }
  }
  // Now PN
  if (_type == ME::iLaser) {
    size_ = ME::iSizePN;
    str0_ = "PN-";
    table_ = ME::iLmfLaserPnPrim;
  } else if (_type == ME::iTestPulse) {
    size_ = ME::iSizeTPPN;
    str0_ = "TPPN-";
    table_ = ME::iLmfTestPulsePnPrim;
  }

  for (unsigned int jj = 0; jj < size_; jj++) {
    TString varName_;
    if (_type == ME::iLaser) {
      varName_ = ME::PNPrimVar[jj];
    } else if (_type == ME::iTestPulse) {
      varName_ = ME::TPPNPrimVar[jj];
    }
    TH1* h1_;
    TString str_ = str0_ + varName_;
    TString ext_ = "_LOCAL";
    h1_ = (TH1*)_eb_loc_m[str_ + ext_];
    title_ = titleW;
    title_.ReplaceAll("APD", "PN");
    title_.ReplaceAll("XXXXX", varName_);
    title_.ReplaceAll("YYYYY", _leaf->oneWord(ME::iLMRegion));
    h1_->SetTitle(title_);

    if (run_ != 0) {
      vector<MEChannel*> vec;
      mgr->tree()->getListOfDescendants(ME::iLMModule, vec);
      for (unsigned int ii = 0; ii < vec.size(); ii++) {
        MEChannel* leaf_ = vec[ii];
        int ilm = leaf_->id();
        for (int ipn = 0; ipn < 2; ipn++) {
          float val = run_->getVal(table_, jj, ilm, ipn);
          int ival = 2 * ((ilm - 1) / 2) + ipn;
          if (ilm % 2 == 0)
            ival += 10;
          h1_->Fill(ival, val);
        }
      }
    }
  }
  cout << "...Done." << endl;
}

void MusEcal::fillEEGlobalHistograms() {
  if (!_histoBooked)
    bookHistograms();
  cout << "Filling EE Global Histograms";
  // cleaning
  for (map<TString, TH1*>::iterator it = _ee_m.begin(); it != _ee_m.end(); ++it) {
    it->second->Reset();
  }

  // filling the 2D histogram
  TString rundir_;
  MERunManager* firstMgr_ = _runMgr.begin()->second;
  if (firstMgr_ != 0) {
    MERun* firstRun_ = firstMgr_->curRun();
    if (firstRun_ != 0) {
      rundir_ += firstRun_->rundir();
    }
  }
  TString titleW;

  titleW = ME::type[_type];
  if (_type == ME::iLaser) {
    titleW += " ";
    titleW += ME::color[_color];
  }
  titleW += " ECAL EndCap";
  titleW += " APD XXXXX";
  titleW += " ";
  titleW += rundir_;

  unsigned size_(0);
  TString str0_;
  int table_(0);

  vector<MEChannel*> vec;
  int ilmr, ireg, idcc, isect, side;
  for (ilmr = 73; ilmr <= 92; ilmr++) {
    cout << "." << flush;
    int iz = 1;
    if (ilmr > 82)
      iz = -1;
    ME::regionAndSector(ilmr, ireg, isect, idcc, side);
    MERunManager* mgr_;
    MERun* run_;
    mgr_ = runMgr(ilmr);
    if (mgr_ == 0)
      continue;
    //    int ism = MEEEGeom::smFromDcc( idcc );
    run_ = mgr_->curRun();
    if (run_ == 0)
      continue;

    // first APD
    if (_type == ME::iLaser) {
      size_ = ME::iSizeAPD;
      str0_ = "APD-";
      table_ = ME::iLmfLaserPrim;
    } else if (_type == ME::iTestPulse) {
      size_ = ME::iSizeTPAPD;
      str0_ = "TPAPD-";
      table_ = ME::iLmfTestPulsePrim;
    }

    vec.clear();
    mgr_->tree()->getListOfChannels(vec);

    for (unsigned int jj = 0; jj < size_; jj++) {
      TString varName_;
      if (_type == ME::iLaser) {
        varName_ = ME::APDPrimVar[jj];
      } else if (_type == ME::iTestPulse) {
        varName_ = ME::TPAPDPrimVar[jj];
      }
      TString str_ = str0_ + varName_;
      TH2* h2_;
      h2_ = (TH2*)_ee_m[str_];
      if (h2_ == 0) {
        cout << "non existing histogram " << str_ << endl;
        continue;
      }
      TString title_ = titleW;
      title_.ReplaceAll("XXXXX", str_);
      h2_->SetTitle(title_);
      TH1* h1_;
      TString ext_ = "_1D";
      h1_ = (TH1*)_ee_m[str_ + ext_];
      if (h1_ == 0) {
        cout << "non existing histogram " << str_ << endl;
        continue;
      }
      h1_->SetTitle(title_);

      for (unsigned int ii = 0; ii < vec.size(); ii++) {
        MEChannel* leaf_ = vec[ii];
        int ix = leaf_->ix();
        int iy = leaf_->iy();
        float val = run_->getVal(table_, jj, ix, iy);
        h2_->Fill(ix, iz * iy, val);
      }
    }

    //       // then PNs
    //       if( _type==ME::iLaser )
    // 	{
    // 	  size_=ME::iSizePN;
    // 	  str0_="PN-";
    // 	  table_=ME::iLmfLaserPnPrim;
    // 	}
    //       else if( _type==ME::iTestPulse )
    // 	{
    // 	  size_=ME::iSizeTPPN;
    // 	  str0_="TPPN-";
    // 	  table_=ME::iLmfTestPulsePnPrim;
    // 	}
    //       vec.clear();
    //       mgr_->tree()->getListOfDescendants( ME::iLMModule, vec );
    //       for( unsigned int jj=0; jj<size_; jj++ )
    // 	{
    // 	  TString varName_;
    // 	  if( _type==ME::iLaser )
    // 	    {
    // 	      varName_=ME::PNPrimVar[jj];
    // 	    }
    // 	  else if( _type==ME::iTestPulse )
    // 	    {
    // 	      varName_=ME::TPPNPrimVar[jj];
    // 	    }
    // 	  for( unsigned int ii=0; ii<vec.size(); ii++ )
    // 	    {
    // 	      MEChannel* leaf_ = vec[ii];
    // 	      int ilm = leaf_->id();
    // 	      TString str_=str0_+varName_;
    // 	      TH1* h1_;
    // 	      h1_ = (TH1*) _ee_m[str_];
    // 	      TString title_ = titleW;
    // 	      title_.ReplaceAll("APD","PN");
    // 	      title_.ReplaceAll("XXXXX",str_);
    // 	      h1_->SetTitle(title_);
    // 	      for( int ipn=0; ipn<2; ipn++ )
    // 		{
    // 		  float val = run_->getVal( table_, jj, ilm, ipn );
    // 		  int ival = 2*((ilm-1)/2) + ipn + ((ilmr-1)/2)*18;
    // 		  if( ilm%2==0 ) ival+=10;
    // 		  h1_->Fill( ival, val );
    // 		}
    // 	    }
    // 	}
  }
  cout << " Done." << endl;
}

void MusEcal::fillEELocalHistograms() {
  if (!_histoBooked)
    bookHistograms();
  if (isBarrel())
    return;
  //  int reg_= ME::ecalRegion( _lmr );
  //  int iz=1;
  //  if( reg_==ME::iEEM ) iz=-1;

  cout << "Filling EE Local Histograms";
  for (map<TString, TH1*>::iterator it = _ee_loc_m.begin(); it != _ee_loc_m.end(); ++it) {
    it->second->Reset();
  }
  MERunManager* mgr = curMgr();
  if (mgr == 0)
    return;
  MERun* run_ = mgr->curRun();
  TString rundir = "No Data";
  if (run_ != 0)
    rundir = run_->rundir();

  TString titleW;
  titleW = ME::type[_type];
  if (_type == ME::iLaser) {
    titleW += " ";
    titleW += ME::color[_color];
  }
  titleW += " ";
  titleW += "YYYYY";
  titleW += " APD XXXXX";
  titleW += " ";
  titleW += rundir;
  TString title_;

  cout << "GHM DBG -- " << titleW << endl;

  unsigned size_(0);
  TString str0_;
  int table_(0);

  if (_type == ME::iLaser) {
    size_ = ME::iSizeAPD;
    str0_ = "APD-";
    table_ = ME::iLmfLaserPrim;
  } else if (_type == ME::iTestPulse) {
    size_ = ME::iSizeTPAPD;
    str0_ = "TPAPD-";
    table_ = ME::iLmfTestPulsePrim;
  }

  cout << "GHM DBG -- " << _leaf->oneLine() << endl;

  MEChannel* l_[2] = {0, 0};
  l_[1] = _leaf;
  //  l_[0] = mgr->tree();
  l_[0] = _leaf->getAncestor(ME::iSector);

  int isect = l_[0]->id();
  if (isect > 9)
    isect -= 9;
  cout << "GHM DBG isect= " << isect << endl;

  for (unsigned int jj = 0; jj < size_; jj++) {
    TString varName_;
    if (_type == ME::iLaser) {
      varName_ = ME::APDPrimVar[jj];
    } else if (_type == ME::iTestPulse) {
      varName_ = ME::TPAPDPrimVar[jj];
    }
    TH2* h2_(0);
    TH1* h1_[2] = {0, 0};
    TH1* h1_chan[2] = {0, 0};
    TString str_ = str0_;
    str_ += varName_;
    str_ += "_";
    str_ += isect;
    h2_ = (TH2*)_ee_loc_m[str_];
    title_ = titleW;
    title_.ReplaceAll("XXXXX", varName_);
    //  title_.ReplaceAll("YYYYY",_leaf->oneWord( ME::iLMRegion ) );
    title_.ReplaceAll("YYYYY", _leaf->oneWord());
    h2_->SetTitle(title_);

    title_ = titleW;
    title_.ReplaceAll("XXXXX", varName_);
    title_.ReplaceAll("YYYYY", _leaf->oneWord());
    TString ext_ = "_HIST";
    if (_ee_loc_m.count(str_ + ext_) != 0) {
      h1_[0] = (TH1*)_ee_loc_m[str_ + ext_];
      h1_[0]->SetTitle(title_);
      ext_ += "_sel";
      h1_[1] = (TH1*)_ee_loc_m[str_ + ext_];
      h1_[1]->SetTitle(title_);
    }

    ext_ = "_VS_CHANNEL";
    if (_ee_loc_m.count(str_ + ext_) != 0) {
      h1_chan[0] = (TH1*)_ee_loc_m[str_ + ext_];
      h1_chan[0]->SetTitle(title_);
      ext_ += "_sel";
      h1_chan[1] = (TH1*)_ee_loc_m[str_ + ext_];
      h1_chan[1]->SetTitle(title_);
    }

    if (run_ != 0) {
      vector<MEChannel*> vec[2];
      l_[0]->getListOfChannels(vec[0]);
      l_[1]->getListOfChannels(vec[1]);
      for (unsigned int ii = 0; ii < vec[0].size(); ii++) {
        MEChannel* leaf_ = vec[0][ii];
        //	      int ieta = leaf_->ix();
        //	      int iphi = leaf_->iy();
        //	      MEEBGeom::XYCoord ixy = MEEBGeom::localCoord( ieta, iphi );
        int ix = leaf_->ix();
        int iy = leaf_->iy();
        float val = run_->getVal(table_, jj, ix, iy);
        if (h1_chan[0] != 0)
          h1_chan[0]->Fill(ii, val);
        if (h1_[0] != 0)
          h1_[0]->Fill(val);
        vector<MEChannel*>::iterator it_ = find(vec[1].begin(), vec[1].end(), leaf_);
        if (it_ != vec[1].end()) {
          if (h1_chan[1] != 0)
            h1_chan[1]->Fill(ii, val);
          if (h1_[1] != 0)
            h1_[1]->Fill(val);
          // test
          // 		  if(_leaf->ig()>ME::iLMRegion )
          // 		    {
          // 		      TAxis* xaxis_ = h2_->GetXaxis();
          // 		      TAxis* yaxis_ = h2_->GetYaxis();
          // 		      int a1_ = xaxis_->GetFirst();
          // 		      int a2_ = xaxis_->GetLast();
          // 		      int b1_ = yaxis_->GetFirst();
          // 		      int b2_ = yaxis_->GetLast();
          // 		      float max_ = h2_->GetMaximum();
          // 		      for( int ix_=-1; ix_>=xaxis_->GetBinCenter(a1_); ix_-- )
          // 			{
          // 			  h2_->Fill( ix_, iy, max_ );
          // 			}
          // 		      for( int ix_=85; ix_<=xaxis_->GetBinCenter(a2_); ix_++ )
          // 			{
          // 			  h2_->Fill( ix_, iy, max_ );
          // 			}
          // 		      for( int iy_=-1; iy_>=yaxis_->GetBinCenter(b1_); iy_-- )
          // 			{
          // 			  h2_->Fill( ix, iy_, max_ );
          // 			}
          // 		      for( int iy_=20; iy_<=yaxis_->GetBinCenter(b2_); iy_++ )
          // 			{
          // 			  h2_->Fill( ix, iy_, max_ );
          // 			}
          // 		    }
        }

        //	      cout << "GHM DBG ix/iy " << ix << "/" << iy << "/" << val << endl;
        h2_->Fill(ix, iy, val);
      }
    }
  }
  // Now PN
  if (_type == ME::iLaser) {
    size_ = ME::iSizePN;
    str0_ = "PN-";
    table_ = ME::iLmfLaserPnPrim;
  } else if (_type == ME::iTestPulse) {
    size_ = ME::iSizeTPPN;
    str0_ = "TPPN-";
    table_ = ME::iLmfTestPulsePnPrim;
  }

  for (unsigned int jj = 0; jj < size_; jj++) {
    TString varName_;
    if (_type == ME::iLaser) {
      varName_ = ME::PNPrimVar[jj];
    } else if (_type == ME::iTestPulse) {
      varName_ = ME::TPPNPrimVar[jj];
    }
    TH1* h1_;
    TString str_ = str0_ + varName_;
    TString ext_ = "_LOCAL";
    h1_ = (TH1*)_eb_loc_m[str_ + ext_];
    title_ = titleW;
    title_.ReplaceAll("APD", "PN");
    title_.ReplaceAll("XXXXX", varName_);
    title_.ReplaceAll("YYYYY", _leaf->oneWord(ME::iLMRegion));
    h1_->SetTitle(title_);

    if (run_ != 0) {
      vector<MEChannel*> vec;
      mgr->tree()->getListOfDescendants(ME::iLMModule, vec);
      for (unsigned int ii = 0; ii < vec.size(); ii++) {
        MEChannel* leaf_ = vec[ii];
        int ilm = leaf_->id();
        for (int ipn = 0; ipn < 2; ipn++) {
          float val = run_->getVal(table_, jj, ilm, ipn);
          int ival = 2 * ((ilm - 1) / 2) + ipn;
          if (ilm % 2 == 0)
            ival += 10;
          h1_->Fill(ival, val);
        }
      }
    }
  }

  //   for( map<TString,TH1*>::iterator it=_ee_loc_m.begin();
  //        it!=_ee_loc_m.end(); ++it )
  //     {
  //       cout << "ee_loc_m " << it->first << endl;
  //     }
  cout << "...Done." << endl;
}

void MusEcal::writeGlobalHistograms() {
  map<TString, TH1*>::iterator it;
  for (it = _eb_m.begin(); it != _eb_m.end(); ++it) {
    it->second->Write();
  }
  for (it = _ee_m.begin(); it != _ee_m.end(); ++it) {
    it->second->Write();
  }
}

TString MusEcal::mgrName(int lmr, int type, int color) {
  TString out_;
  out_ += type;
  out_ += "_";
  out_ += color;
  out_ += "_";
  if (lmr < 10)
    out_ += "0";
  out_ += lmr;
  return out_;
}
