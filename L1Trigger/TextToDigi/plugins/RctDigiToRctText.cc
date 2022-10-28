
#include "RctDigiToRctText.h"
#include <iomanip>

using std::dec;
using std::endl;
using std::hex;
using std::setfill;
using std::setw;

RctDigiToRctText::RctDigiToRctText(const edm::ParameterSet &iConfig)
    : m_rctInputLabel(iConfig.getParameter<edm::InputTag>("RctInputLabel")),
      m_textFileName(iConfig.getParameter<std::string>("TextFileName")),
      m_hexUpperCase(iConfig.getParameter<bool>("HexUpperCase")) {
  /// open output text files
  for (unsigned i = 0; i < NUM_RCT_CRATES; i++) {
    std::stringstream fileStream;
    fileStream << m_textFileName << std::setw(2) << std::setfill('0') << i << ".txt";
    std::string fileName(fileStream.str());
    m_file[i].open(fileName.c_str(), std::ios::out);

    if (!m_file[i].good()) {
      throw cms::Exception("RctDigiToRctTextTextFileOpenError")
          << "RctDigiToRctText::RctDigiToRctText : "
          << " couldn't create the file " << fileName << std::endl;
    }
  }

  /// open info|debug file
  fdebug.open("rctdigitorcttext_debug.txt", std::ios::out);
}

RctDigiToRctText::~RctDigiToRctText() {
  /// close  files
  for (unsigned i = 0; i < NUM_RCT_CRATES; i++)
    m_file[i].close();
  fdebug.close();
}

void RctDigiToRctText::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  /// count bunch crossing
  nevt++;

  /// get the RCT data
  edm::Handle<L1CaloEmCollection> em;
  edm::Handle<L1CaloRegionCollection> rgn;
  iEvent.getByLabel(m_rctInputLabel, em);
  iEvent.getByLabel(m_rctInputLabel, rgn);

  /// debug flags and stream
  bool ldebug = false;
  bool debug_NOTEMPTY[18] = {false};
  for (int i = 0; i < 18; i++)
    debug_NOTEMPTY[i] = false;
  std::stringstream dstrm;

  /// --- ELECTRONS ---

  unsigned long int CAND[18][8];
  int n_iso[18] = {0};
  int n_niso[18] = {0};
  bool iso;
  int id;

  for (L1CaloEmCollection::const_iterator iem = em->begin(); iem != em->end(); iem++) {
    int crate = iem->rctCrate();
    iso = iem->isolated();
    unsigned data = iem->raw();

    id = iso ? n_iso[crate]++ : 4 + n_niso[crate]++;

    CAND[crate][id] = data;

    /// debug
    if (crate > 17 || id > 7)
      throw cms::Exception("RctDigiToRctTextElectronIndexOutBounds")
          << "out of bounds indices  crate:" << crate << "id:" << id << std::endl;
    if (ldebug && data != 0)
      debug_NOTEMPTY[crate] = true;
    dstrm.str("");
    dstrm << "electron "
          << " bx:" << nevt << " crate:" << crate << " iso:" << iso << " raw:" << data << " \t cand:" << *iem;
    if (debug_NOTEMPTY[crate])
      fdebug << dstrm.str() << std::endl;
  }

  /// --- REGIONS ---

  unsigned short MIPbits[18][7][2] = {{{0}}};
  unsigned short QIEbits[18][7][2] = {{{0}}};
  unsigned short RC[18][7][2] = {{{0}}};
  unsigned short RCof[18][7][2] = {{{0}}};
  unsigned short RCtau[18][7][2] = {{{0}}};
  unsigned short HF[18][4][2] = {{{0}}};

  for (L1CaloRegionCollection::const_iterator irgn = rgn->begin(); irgn != rgn->end(); irgn++) {
    int crate = irgn->rctCrate();
    int card = irgn->rctCard();
    int rgnidx = irgn->rctRegionIndex();

    dstrm.str("");
    if (!irgn->id().isHf()) {
      RC[crate][card][rgnidx] = irgn->et();
      RCof[crate][card][rgnidx] = irgn->overFlow();
      RCtau[crate][card][rgnidx] = irgn->tauVeto();
      MIPbits[crate][card][rgnidx] = irgn->mip();
      QIEbits[crate][card][rgnidx] = irgn->quiet();
      // debug info
      dstrm << hex << "Et=" << irgn->et() << " OverFlow=" << irgn->overFlow() << " tauVeto=" << irgn->tauVeto()
            << " mip=" << irgn->mip() << " quiet=" << irgn->quiet() << " Card=" << irgn->rctCard()
            << " Region=" << irgn->rctRegionIndex() << " Crate=" << irgn->rctCrate() << dec;
      if (ldebug)
        LogDebug("Regions") << dstrm.str();
    } else {
      HF[crate][irgn->id().rctEta() - 7][irgn->id().rctPhi()] = irgn->et();
      // debug info
      dstrm << hex << "Et=" << irgn->et() << " FGrain=" << irgn->fineGrain() << " Eta=" << irgn->id().rctEta()
            << " Phi=" << irgn->id().rctPhi() << " Crate=" << irgn->rctCrate() << dec;
      if (ldebug)
        LogDebug("HFRegions") << dstrm.str();
    }

    if (ldebug && irgn->et() != 0)
      debug_NOTEMPTY[crate] = true;  // debug
    if (debug_NOTEMPTY[crate]) {
      fdebug << "region"
             << " bx:" << nevt << " crate:" << crate << "\t";
      fdebug << dstrm.str() << std::endl;
    }
  }

  std::stringstream sstrm;
  if (m_hexUpperCase)
    sstrm << std::uppercase;
  else
    sstrm.unsetf(std::ios::uppercase);

  /// print electrons

  for (unsigned crate = 0; crate < NUM_RCT_CRATES; crate++) {
    sstrm.str("");
    sstrm << "Crossing " << nevt << std::endl;

    for (int j = 0; j < 8; j++) {
      sstrm << setw(3) << setfill('0') << hex << (CAND[crate][j] & 0x3ff);
      if (j < 7)
        sstrm << " ";
    }
    sstrm << setfill(' ') << dec;
    m_file[crate] << sstrm.str() << std::endl;

    // debug
    if (debug_NOTEMPTY[crate])
      fdebug << sstrm.str() << std::endl;
    if (ldebug)
      LogDebug("Electrons") << sstrm.str() << std::endl;
  }

  /// print regions

  for (unsigned crate = 0; crate < NUM_RCT_CRATES; crate++) {
    /// mip bits
    sstrm.str("");
    for (int card = 0; card < 7; card++) {
      for (int j = 0; j < 2; j++) {
        sstrm << " " << MIPbits[crate][card][j];
      }
    }
    m_file[crate] << sstrm.str() << std::endl;
    if (debug_NOTEMPTY[crate])
      fdebug << sstrm.str() << std::endl;  // debug

    /// quiet bits
    sstrm.str("");
    for (int card = 0; card < 7; card++) {
      for (int j = 0; j < 2; j++) {
        sstrm << " " << QIEbits[crate][card][j];
      }
    }
    m_file[crate] << sstrm.str() << std::endl;
    if (debug_NOTEMPTY[crate])
      fdebug << sstrm.str() << std::endl;  // debug

    /// region info
    sstrm.str("");
    for (int card = 0; card < 7; card++) {
      for (int j = 0; j < 2; j++) {
        unsigned long int tmp;
        unsigned et = RC[crate][card][j];
        unsigned ovf = RCof[crate][card][j];
        unsigned tau = RCtau[crate][card][j];
        // ovf = ovf || (et>=0x400);
        tmp = ((tau & 0x1) << 11) | ((ovf & 0x1) << 10) | ((et & 0x3ff));
        sstrm << " " << setw(3) << setfill('0') << hex << tmp;
      }
    }
    m_file[crate] << sstrm.str() << std::endl;
    if (debug_NOTEMPTY[crate])
      fdebug << sstrm.str() << std::endl << std::endl;  // debug

    /// HF
    sstrm.str("");
    for (int ip = 0; ip < 2; ip++) {
      for (int ie = 0; ie < 4; ie++) {
        unsigned et = HF[crate][ie][ip] & 0xff;
        sstrm << " " << setw(2) << setfill('0') << hex << et;
      }
    }
    m_file[crate] << sstrm.str() << std::endl;
    if (debug_NOTEMPTY[crate])
      fdebug << sstrm.str() << std::endl;  // debug
    sstrm << setfill(' ') << dec;

  }  // end crate loop

  /// flush data to files
  for (unsigned i = 0; i < NUM_RCT_CRATES; i++)
    m_file[i] << std::flush;

  fdebug << std::flush;
}
