//-------------------------------------------------
//
//   Class: DTConfigTraco
//
//   Description: Configurable parameters and constants for Level1 Mu DT Trigger - TRACO chip
//
//
//   Author List:
//   S.Vanini
//   Modifications:
//   April,10th 2008: set TRACO parameters from string
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTraco.h"

//---------------
// C++ Headers --
//---------------
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Utilities/interface/Exception.h"

//----------------
// Constructors --
//----------------
DTConfigTraco::DTConfigTraco(const edm::ParameterSet& ps) { setDefaults(ps); }

DTConfigTraco::DTConfigTraco(int debugTRACO, unsigned short int* buffer) {
  m_debug = debugTRACO;

  // check if this is a TRACO configuration string
  if (buffer[2] != 0x15) {
    throw cms::Exception("DTTPG") << "===> ConfigTraco constructor : not a TRACO string!" << std::endl;
  }

  // decode
  unsigned short int memory_traco[38];

  for (int i = 0; i < 38; i++) {
    memory_traco[i] = buffer[i + 5];
    //std::cout << hex << memory_traco[i];
  }
  int btic = memory_traco[0] & 0x3f;
  int rad = ((memory_traco[0] & 0xc0) >> 6) | ((memory_traco[1] & 0x7) << 2);
  int dd = (memory_traco[1] & 0xf8) >> 3;
  int fprgcomp = memory_traco[2] & 0x3;
  int sprgcomp = memory_traco[3] & 0x3;
  int fhism = (memory_traco[2] & 0x4) != 0;
  int fhtprf = (memory_traco[2] & 0x8) != 0;
  int fslmsk = (memory_traco[2] & 0x10) != 0;
  int fltmsk = (memory_traco[2] & 0x20) != 0;
  int fhtmsk = (memory_traco[2] & 0x40) != 0;
  int shism = (memory_traco[3] & 0x4) != 0;
  int shtprf = (memory_traco[3] & 0x8) != 0;
  int sslmsk = (memory_traco[3] & 0x10) != 0;
  int sltmsk = (memory_traco[3] & 0x20) != 0;
  int shtmsk = (memory_traco[3] & 0x40) != 0;
  int reusei = (memory_traco[2] & 0x80) != 0;
  int reuseo = (memory_traco[3] & 0x80) != 0;
  int ltf = (memory_traco[4] & 1) != 0;
  int lts = (memory_traco[4] & 2) != 0;
  int prgdel = (memory_traco[4] & 0x1c) >> 2;
  int snapcor = (memory_traco[4] & 0xe0) >> 5;
  int trgenb[16];
  for (int it = 0; it < 2; it++) {
    trgenb[0 + it * 8] = memory_traco[5 + it] & 0x01;
    trgenb[1 + it * 8] = (memory_traco[5 + it] >> 1) & 0x01;
    trgenb[2 + it * 8] = (memory_traco[5 + it] >> 1) & 0x01;
    trgenb[3 + it * 8] = (memory_traco[5 + it] >> 1) & 0x01;
    trgenb[4 + it * 8] = (memory_traco[5 + it] >> 1) & 0x01;
    trgenb[5 + it * 8] = (memory_traco[5 + it] >> 1) & 0x01;
    trgenb[6 + it * 8] = (memory_traco[5 + it] >> 1) & 0x01;
    trgenb[7 + it * 8] = (memory_traco[5 + it] >> 1) & 0x01;
  }
  int trgadel = memory_traco[7] & 0x3;
  int ibtioff = (memory_traco[7] & 0xfc) >> 2;
  int kprgcom = (memory_traco[8] & 0xff);
  int testmode = (memory_traco[9] & 1) != 0;
  int starttest = (memory_traco[9] & 2) != 0;
  int prvsignmux = (memory_traco[9] & 4) != 0;
  int lth = (memory_traco[9] & 8) != 0;

  if (debug() == 1) {
    std::cout << "btic=" << btic << " rad=" << rad << " dd=" << dd << " fprgcomp=" << fprgcomp
              << " sprgcomp=" << sprgcomp << " fhism=" << fhism << " fhtprf=" << fhtprf << " fslmsk=" << fslmsk
              << " fltmsk=" << fltmsk << " fhtmsk=" << fhtmsk << " shism=" << shism << " shtprf=" << shtprf
              << " sslmsk=" << sslmsk << " sltmsk=" << sltmsk << " shtmsk=" << shtmsk << " reusei=" << reusei
              << " reuseo=" << reuseo << " ltf=" << ltf << " lts=" << lts << " prgdel=" << prgdel
              << " snapcor=" << snapcor << " trgenb=";
    for (int t = 0; t < 16; t++)
      std::cout << trgenb[t] << " ";
    std::cout << " trgadel=" << trgadel << " ibtioff=" << ibtioff << " kprgcom=" << kprgcom << " testmode=" << testmode
              << " starttest=" << starttest << " prvsignmux=" << prvsignmux << " lth=" << lth << std::endl;
  }

  // set parameters
  setBTIC(btic);
  setKRAD(rad);
  setDD(dd);
  setTcKToll(0, fprgcomp);
  setTcKToll(1, sprgcomp);
  setSortKascend(0, fhism);
  setSortKascend(1, shism);
  setPrefHtrig(0, fhtprf);
  setPrefHtrig(1, shtprf);
  setPrefInner(0, fslmsk);
  setPrefInner(1, sslmsk);
  setSingleLflag(0, fltmsk);
  setSingleLflag(1, sltmsk);
  setSingleHflag(0, fhtmsk);
  setSingleHflag(1, shtmsk);
  setTcReuse(0, reusei);
  setTcReuse(1, reuseo);
  setSingleLenab(0, ltf);
  setSingleLenab(1, ltf);
  setTcBxLts(lts);
  setIBTIOFF(ibtioff);
  setBendingAngleCut(kprgcom);
  setLVALIDIFH(lth);
  for (int t = 0; t < 16; t++)
    setUsedBti(t + 1, trgenb[t]);

  // the following are not relevant for simulation
  // prgdel, snapcor, trgadel, testmode, starttest, prvsignmux
}

//--------------
// Destructor --
//--------------
DTConfigTraco::~DTConfigTraco() {}

//--------------
// Operations --
//--------------

void DTConfigTraco::setDefaults(const edm::ParameterSet& ps) {
  // Debug flag
  m_debug = ps.getUntrackedParameter<int>("Debug");

  // KRAD traco parameter
  m_krad = ps.getParameter<int>("KRAD");

  // BTIC traco parameter
  m_btic = ps.getParameter<int>("BTIC");

  // DD traco parameter: this is fixed
  m_dd = ps.getParameter<int>("DD");

  // recycling of TRACO cand. in inner/outer SL : REUSEI/REUSEO
  m_reusei = ps.getParameter<int>("REUSEI");
  m_reuseo = ps.getParameter<int>("REUSEO");

  // single HTRIG enabling on first/second tracks F(S)HTMSK
  m_fhtmsk = ps.getParameter<int>("FHTMSK");
  m_shtmsk = ps.getParameter<int>("SHTMSK");

  // single LTRIG enabling on first/second tracks: F(S)LTMSK
  m_fltmsk = ps.getParameter<int>("FLTMSK");
  m_sltmsk = ps.getParameter<int>("SLTMSK");

  // preference to inner on first/second tracks: F(S)SLMSK
  m_fslmsk = ps.getParameter<int>("FSLMSK");
  m_sslmsk = ps.getParameter<int>("SSLMSK");

  // preference to HTRIG on first/second tracks: F(S)HTPRF
  m_fhtprf = ps.getParameter<int>("FHTPRF");
  m_shtprf = ps.getParameter<int>("SHTPRF");

  // ascend. order for K sorting first/second tracks: F(S)HISM
  m_fhism = ps.getParameter<int>("FHISM");
  m_shism = ps.getParameter<int>("SHISM");

  // K tollerance for correlation in TRACO: F(S)PRGCOMP
  m_fprgcomp = ps.getParameter<int>("FPRGCOMP");
  m_sprgcomp = ps.getParameter<int>("SPRGCOMP");

  // suppr. of LTRIG in 4 BX before HTRIG: LTS
  m_lts = ps.getParameter<int>("LTS");

  // single LTRIG accept enabling on first/second tracks LTF
  m_ltf = ps.getParameter<int>("LTF");

  // Connected bti in traco: bti mask
  for (int b = 0; b < 16; b++) {
    std::string label = "TRGENB";
    char p0 = (b / 10) + '0';
    char p1 = (b % 10) + '0';
    if (p0 != '0')
      label = label + p0;
    label = label + p1;

    m_trgenb.set(b, ps.getParameter<int>(label));
  }

  // IBTIOFF traco parameter
  m_ibtioff = ps.getParameter<int>("IBTIOFF");

  // bending angle cut for all stations and triggers : KPRGCOM
  m_kprgcom = ps.getParameter<int>("KPRGCOM");

  // flag for Low validation parameter
  m_lvalidifh = ps.getParameter<int>("LVALIDIFH");
}

void DTConfigTraco::print() const {
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : TRACO chips                                 *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;
  std::cout << "Debug flag : " << debug() << std::endl;
  std::cout << "KRAD traco parameter : " << KRAD() << std::endl;
  std::cout << "BTIC traco parameter : " << BTIC() << std::endl;
  std::cout << "DD traco parameter : " << DD() << std::endl;
  std::cout << "REUSEI, REUSEO : " << TcReuse(0) << ", " << TcReuse(1) << std::endl;
  std::cout << "FHTMSK, SHTMSK : " << singleHflag(0) << ", " << singleHflag(1) << std::endl;
  std::cout << "FLTMSK, SLTMSK: " << singleLflag(0) << ", " << singleLflag(1) << std::endl;
  std::cout << "FSLMSK, SSLMSK : " << prefInner(0) << ", " << prefInner(1) << std::endl;
  std::cout << "FHTPRF, SHTPRF : " << prefHtrig(0) << ", " << prefHtrig(1) << std::endl;
  std::cout << "FHISM, SHISM : " << sortKascend(0) << ", " << sortKascend(1) << std::endl;
  std::cout << "FPRGCOMP, SPRGCOMP : " << TcKToll(0) << ", " << TcKToll(1) << std::endl;
  std::cout << "LTS : " << TcBxLts() << std::endl;
  std::cout << "LTF : " << singleLenab(0) << std::endl;
  std::cout << "Connected bti in traco - bti mask : ";
  for (int b = 1; b <= 16; b++)
    std::cout << usedBti(b) << " ";
  std::cout << std::endl;
  std::cout << "IBTIOFF : " << IBTIOFF() << std::endl;
  std::cout << "bending angle cut : " << BendingAngleCut() << std::endl;
  std::cout << "flag for Low validation parameter : " << LVALIDIFH() << std::endl;
  std::cout << "******************************************************************************" << std::endl;
}
