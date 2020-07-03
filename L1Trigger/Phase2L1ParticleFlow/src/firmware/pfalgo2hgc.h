#ifndef L1Trigger_Phase2L1ParticleFlow_FIRMWARE_PFALGO2HGC_H
#define L1Trigger_Phase2L1ParticleFlow_FIRMWARE_PFALGO2HGC_H

#include "pfalgo_common.h"

void pfalgo2hgc(const HadCaloObj calo[NCALO],
                const TkObj track[NTRACK],
                const MuObj mu[NMU],
                PFChargedObj outch[NTRACK],
                PFNeutralObj outne[NSELCALO],
                PFChargedObj outmu[NMU]);

#if defined(PACKING_DATA_SIZE) && defined(PACKING_NCHANN)
void packed_pfalgo2hgc(const ap_uint<PACKING_DATA_SIZE> input[PACKING_NCHANN],
                       ap_uint<PACKING_DATA_SIZE> output[PACKING_NCHANN]);
void pfalgo2hgc_pack_in(const HadCaloObj calo[NCALO],
                        const TkObj track[NTRACK],
                        const MuObj mu[NMU],
                        ap_uint<PACKING_DATA_SIZE> input[PACKING_NCHANN]);
void pfalgo2hgc_unpack_in(const ap_uint<PACKING_DATA_SIZE> input[PACKING_NCHANN],
                          HadCaloObj calo[NCALO],
                          TkObj track[NTRACK],
                          MuObj mu[NMU]);
void pfalgo2hgc_pack_out(const PFChargedObj outch[NTRACK],
                         const PFNeutralObj outne[NSELCALO],
                         const PFChargedObj outmu[NMU],
                         ap_uint<PACKING_DATA_SIZE> output[PACKING_NCHANN]);
void pfalgo2hgc_unpack_out(const ap_uint<PACKING_DATA_SIZE> output[PACKING_NCHANN],
                           PFChargedObj outch[NTRACK],
                           PFNeutralObj outne[NSELCALO],
                           PFChargedObj outmu[NMU]);
#endif

#ifndef CMSSW_GIT_HASH
#define PFALGO_DR2MAX_TK_CALO 525
#define PFALGO_TK_MAXINVPT_LOOSE 40
#define PFALGO_TK_MAXINVPT_TIGHT 80
#endif

#endif
