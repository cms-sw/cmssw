#ifndef L1Trigger_Phase2L1ParticleFlow_FIRMWARE_DATA_H
#define L1Trigger_Phase2L1ParticleFlow_FIRMWARE_DATA_H

#include <ap_int.h>

typedef ap_int<16> pt_t;
typedef ap_int<10> etaphi_t;
typedef ap_int<5> vtx_t;
typedef ap_uint<3> particleid_t;
typedef ap_int<10> z0_t;  // 40cm / 0.1
typedef ap_uint<14> tk2em_dr_t;
typedef ap_uint<14> tk2calo_dr_t;
typedef ap_uint<10> em2calo_dr_t;
typedef ap_uint<13> tk2calo_dq_t;

enum PID { PID_Charged = 0, PID_Neutral = 1, PID_Photon = 2, PID_Electron = 3, PID_Muon = 4 };

// DEFINE MULTIPLICITIES
#if defined(REG_HGCal)
#define NTRACK 25
#define NCALO 20
#define NMU 4
#define NSELCALO 15
#define NALLNEUTRALS NSELCALO
// dummy
#define NEMCALO 1
#define NPHOTON NEMCALO
// not used but must be there because used in header files
#define NNEUTRALS 1
//--------------------------------
#elif defined(REG_HGCalNoTK)
#define NCALO 12
#define NNEUTRALS 8
#define NALLNEUTRALS NCALO
// dummy
#define NMU 1
#define NTRACK 1
#define NEMCALO 1
#define NPHOTON NEMCALO
#define NSELCALO 1
//--------------------------------
#elif defined(REG_HF)
#define NCALO 18
#define NNEUTRALS 10
#define NALLNEUTRALS NCALO
// dummy
#define NMU 1
#define NTRACK 1
#define NEMCALO 1
#define NPHOTON NEMCALO
#define NSELCALO 1
//--------------------------------
#else  // BARREL
#ifndef REG_Barrel
#ifndef CMSSW_GIT_HASH
#warning "No region defined, assuming it's barrel (#define REG_Barrel to suppress this)"
#endif
#endif
#if defined(BOARD_MP7)
#warning "MP7 NOT SUPPORTED ANYMORE"
#define NTRACK 14
#define NCALO 10
#define NMU 2
#define NEMCALO 10
#define NPHOTON NEMCALO
#define NSELCALO 10
#define NALLNEUTRALS (NPHOTON + NSELCALO)
#define NNEUTRALS 15
#elif defined(BOARD_CTP7)
#error "NOT SUPPORTED ANYMORE"
#elif defined(BOARD_KU15P)
#define NTRACK 14
#define NCALO 10
#define NMU 2
#define NEMCALO 10
#define NPHOTON NEMCALO
#define NSELCALO 10
#define NALLNEUTRALS (NPHOTON + NSELCALO)
#define NNEUTRALS 15
#elif defined(BOARD_VCU118)
#define NTRACK 22
#define NCALO 15
#define NEMCALO 13
#define NMU 2
#define NPHOTON NEMCALO
#define NSELCALO 10
#define NALLNEUTRALS (NPHOTON + NSELCALO)
#define NNEUTRALS 25
#else
#define NTRACK 22
#define NCALO 15
#define NEMCALO 13
#define NMU 2
#define NPHOTON NEMCALO
#define NSELCALO 10
#define NALLNEUTRALS (NPHOTON + NSELCALO)
#define NNEUTRALS 25
#endif

#endif  // region

#if defined(BOARD_MP7)
#define PACKING_DATA_SIZE 32
#define PACKING_NCHANN 72
#elif defined(BOARD_KU15P)
#define PACKING_DATA_SIZE 64
#define PACKING_NCHANN 42
#elif defined(BOARD_VCU118)
#define PACKING_DATA_SIZE 64
#define PACKING_NCHANN 96
#elif defined(BOARD_APD1)
#define PACKING_DATA_SIZE 64
#define PACKING_NCHANN 96
#endif

struct CaloObj {
  pt_t hwPt;
  etaphi_t hwEta, hwPhi;  // relative to the region center, at calo
};
struct HadCaloObj : public CaloObj {
  pt_t hwEmPt;
  bool hwIsEM;
};
inline void clear(HadCaloObj& c) {
  c.hwPt = 0;
  c.hwEta = 0;
  c.hwPhi = 0;
  c.hwEmPt = 0;
  c.hwIsEM = false;
}

struct EmCaloObj {
  pt_t hwPt, hwPtErr;
  etaphi_t hwEta, hwPhi;  // relative to the region center, at calo
};
inline void clear(EmCaloObj& c) {
  c.hwPt = 0;
  c.hwPtErr = 0;
  c.hwEta = 0;
  c.hwPhi = 0;
}

struct TkObj {
  pt_t hwPt, hwPtErr;
  etaphi_t hwEta, hwPhi;  // relative to the region center, at calo
  z0_t hwZ0;
  bool hwTightQuality;
};
inline void clear(TkObj& c) {
  c.hwPt = 0;
  c.hwPtErr = 0;
  c.hwEta = 0;
  c.hwPhi = 0;
  c.hwZ0 = 0;
  c.hwTightQuality = false;
}

struct MuObj {
  pt_t hwPt, hwPtErr;
  etaphi_t hwEta, hwPhi;  // relative to the region center, at vtx(?)
};
inline void clear(MuObj& c) {
  c.hwPt = 0;
  c.hwPtErr = 0;
  c.hwEta = 0;
  c.hwPhi = 0;
}

struct PFChargedObj {
  pt_t hwPt;
  etaphi_t hwEta, hwPhi;  // relative to the region center, at calo
  particleid_t hwId;
  z0_t hwZ0;
};
inline void clear(PFChargedObj& c) {
  c.hwPt = 0;
  c.hwEta = 0;
  c.hwPhi = 0;
  c.hwId = 0;
  c.hwZ0 = 0;
}

struct PFNeutralObj {
  pt_t hwPt;
  etaphi_t hwEta, hwPhi;  // relative to the region center, at calo
  particleid_t hwId;
  pt_t hwPtPuppi;
};
inline void clear(PFNeutralObj& c) {
  c.hwPt = 0;
  c.hwEta = 0;
  c.hwPhi = 0;
  c.hwId = 0;
  c.hwPtPuppi = 0;
}

//TMUX
#define NETA_TMUX 2
#define NPHI_TMUX 1
/* #define TMUX_IN 36 */
/* #define TMUX_OUT 18 */
#define TMUX_IN 18
#define TMUX_OUT 6
#define NTRACK_TMUX (NTRACK * TMUX_OUT * NETA_TMUX * NPHI_TMUX)
#define NCALO_TMUX (NCALO * TMUX_OUT * NETA_TMUX * NPHI_TMUX)
#define NEMCALO_TMUX (NEMCALO * TMUX_OUT * NETA_TMUX * NPHI_TMUX)
#define NMU_TMUX (NMU * TMUX_OUT * NETA_TMUX * NPHI_TMUX)

#endif
