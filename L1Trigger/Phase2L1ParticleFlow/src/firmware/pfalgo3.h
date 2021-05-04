#ifndef L1Trigger_Phase2L1ParticleFlow_FIRMWARE_PFALGO3_H
#define L1Trigger_Phase2L1ParticleFlow_FIRMWARE_PFALGO3_H

#include "pfalgo_common.h"

void pfalgo3(const EmCaloObj emcalo[NEMCALO],
             const HadCaloObj hadcalo[NCALO],
             const TkObj track[NTRACK],
             const MuObj mu[NMU],
             PFChargedObj outch[NTRACK],
             PFNeutralObj outpho[NPHOTON],
             PFNeutralObj outne[NSELCALO],
             PFChargedObj outmu[NMU]);

#if defined(PACKING_DATA_SIZE) && defined(PACKING_NCHANN)
void packed_pfalgo3(const ap_uint<PACKING_DATA_SIZE> input[PACKING_NCHANN],
                    ap_uint<PACKING_DATA_SIZE> output[PACKING_NCHANN]);
void pfalgo3_pack_in(const EmCaloObj emcalo[NEMCALO],
                     const HadCaloObj hadcalo[NCALO],
                     const TkObj track[NTRACK],
                     const MuObj mu[NMU],
                     ap_uint<PACKING_DATA_SIZE> input[PACKING_NCHANN]);
void pfalgo3_unpack_in(const ap_uint<PACKING_DATA_SIZE> input[PACKING_NCHANN],
                       EmCaloObj emcalo[NEMCALO],
                       HadCaloObj hadcalo[NCALO],
                       TkObj track[NTRACK],
                       MuObj mu[NMU]);
void pfalgo3_pack_out(const PFChargedObj outch[NTRACK],
                      const PFNeutralObj outpho[NPHOTON],
                      const PFNeutralObj outne[NSELCALO],
                      const PFChargedObj outmu[NMU],
                      ap_uint<PACKING_DATA_SIZE> output[PACKING_NCHANN]);
void pfalgo3_unpack_out(const ap_uint<PACKING_DATA_SIZE> output[PACKING_NCHANN],
                        PFChargedObj outch[NTRACK],
                        PFNeutralObj outpho[NPHOTON],
                        PFNeutralObj outne[NSELCALO],
                        PFChargedObj outmu[NMU]);
#endif

void pfalgo3_set_debug(bool debug);

#ifndef CMSSW_GIT_HASH
#define PFALGO_DR2MAX_TK_CALO 1182
#define PFALGO_DR2MAX_EM_CALO 525
#define PFALGO_DR2MAX_TK_EM 84
#define PFALGO_TK_MAXINVPT_LOOSE 40
#define PFALGO_TK_MAXINVPT_TIGHT 80
#endif

#endif
