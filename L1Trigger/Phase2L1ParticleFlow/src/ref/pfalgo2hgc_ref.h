#ifndef L1Trigger_Phase2L1ParticleFlow_PFALGO2HGC_REF_H
#define L1Trigger_Phase2L1ParticleFlow_PFALGO2HGC_REF_H

#include "../firmware/pfalgo2hgc.h"
#include "pfalgo_common_ref.h"

void pfalgo2hgc_ref(const pfalgo_config &cfg,
                    const HadCaloObj calo[/*cfg.nCALO*/],
                    const TkObj track[/*cfg.nTRACK*/],
                    const MuObj mu[/*cfg.nMU*/],
                    PFChargedObj outch[/*cfg.nTRACK*/],
                    PFNeutralObj outne[/*cfg.nSELCALO*/],
                    PFChargedObj outmu[/*cfg.nMU*/],
                    bool debug);

#endif
