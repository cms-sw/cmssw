#ifndef L1Trigger_Phase2L1ParticleFlow_PFALGO3_REF_H
#define L1Trigger_Phase2L1ParticleFlow_PFALGO3_REF_H

#include "../firmware/pfalgo3.h"
#include "pfalgo_common_ref.h"

struct pfalgo3_config : public pfalgo_config {
  unsigned int nEMCALO, nPHOTON, nALLNEUTRAL;
  unsigned int dR2MAX_TK_EM;
  unsigned int dR2MAX_EM_CALO;

  pfalgo3_config(unsigned int nTrack,
                 unsigned int nEmCalo,
                 unsigned int nCalo,
                 unsigned int nMu,
                 unsigned int nPhoton,
                 unsigned int nSelCalo,
                 unsigned int nAllNeutral,
                 unsigned int dR2Max_Tk_Mu,
                 unsigned int dR2Max_Tk_Em,
                 unsigned int dR2Max_Em_Calo,
                 unsigned int dR2Max_Tk_Calo,
                 unsigned int tk_MaxInvPt_Loose,
                 unsigned int tk_MaxInvPt_Tight)
      : pfalgo_config(nTrack, nCalo, nMu, nSelCalo, dR2Max_Tk_Mu, dR2Max_Tk_Calo, tk_MaxInvPt_Loose, tk_MaxInvPt_Tight),
        nEMCALO(nEmCalo),
        nPHOTON(nPhoton),
        nALLNEUTRAL(nAllNeutral),
        dR2MAX_TK_EM(dR2Max_Tk_Em),
        dR2MAX_EM_CALO(dR2Max_Em_Calo) {}
  ~pfalgo3_config() override {}
};

void pfalgo3_em_ref(const pfalgo3_config &cfg,
                    const EmCaloObj emcalo[/*cfg.nEMCALO*/],
                    const HadCaloObj hadcalo[/*cfg.nCALO*/],
                    const TkObj track[/*cfg.nTRACK*/],
                    const bool isMu[/*cfg.nTRACK*/],
                    bool isEle[/*cfg.nTRACK*/],
                    PFNeutralObj outpho[/*cfg.nPHOTON*/],
                    HadCaloObj hadcalo_out[/*cfg.nCALO*/],
                    bool debug);
void pfalgo3_ref(const pfalgo3_config &cfg,
                 const EmCaloObj emcalo[/*cfg.nEMCALO*/],
                 const HadCaloObj hadcalo[/*cfg.nCALO*/],
                 const TkObj track[/*cfg.nTRACK*/],
                 const MuObj mu[/*cfg.nMU*/],
                 PFChargedObj outch[/*cfg.nTRACK*/],
                 PFNeutralObj outpho[/*cfg.nPHOTON*/],
                 PFNeutralObj outne[/*cfg.nSELCALO*/],
                 PFChargedObj outmu[/*cfg.nMU*/],
                 bool debug);

void pfalgo3_merge_neutrals_ref(const pfalgo3_config &cfg,
                                const PFNeutralObj pho[/*cfg.nPHOTON*/],
                                const PFNeutralObj ne[/*cfg.nSELCALO*/],
                                PFNeutralObj allne[/*cfg.nALLNEUTRALS*/]);
#endif
