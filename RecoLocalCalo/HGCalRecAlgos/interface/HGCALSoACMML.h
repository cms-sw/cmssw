#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCALSoACMML_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCALSoACMML_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcalcmml {

  // Column order determines DNN input tensor layout (all 21 columns are adjacent float,
  // so TensorCollection can register them as a single contiguous [ndigis,21] tensor).
  //
  // cm0..cm11: pedestal-subtracted CM sum per eRx (digi.cm() of first channel in each eRx),
  //            filled with 0 for eRx slots beyond the module's nErx.
  // unconn0..3: pedestal-subtracted ADC of the 4 unconnected channels in this digi's eRx
  //             (within-eRx positions 8, 17, 19, 28).
  // All 21 float columns must remain adjacent in this definition for TensorCollection
  // contiguity; do not reorder or insert non-float columns between them.
  GENERATE_SOA_LAYOUT(HGCalCMMLSoALayout,
                      SOA_COLUMN(float, cm0),        // pedestal-subtracted CM average eRx 0:  0.5*digi.cm() - CM_ped
                      SOA_COLUMN(float, cm1),        // pedestal-subtracted CM average eRx 1:  0.5*digi.cm() - CM_ped
                      SOA_COLUMN(float, cm2),        // pedestal-subtracted CM average eRx 2:  0.5*digi.cm() - CM_ped
                      SOA_COLUMN(float, cm3),        // pedestal-subtracted CM average eRx 3:  0.5*digi.cm() - CM_ped
                      SOA_COLUMN(float, cm4),        // pedestal-subtracted CM average eRx 4:  0.5*digi.cm() - CM_ped
                      SOA_COLUMN(float, cm5),        // pedestal-subtracted CM average eRx 5:  0.5*digi.cm() - CM_ped
                      SOA_COLUMN(float, cm6),        // pedestal-subtracted CM average eRx 6:  0.0 if eRx inactive
                      SOA_COLUMN(float, cm7),        // pedestal-subtracted CM average eRx 7:  0.0 if eRx inactive
                      SOA_COLUMN(float, cm8),        // pedestal-subtracted CM average eRx 8:  0.0 if eRx inactive
                      SOA_COLUMN(float, cm9),        // pedestal-subtracted CM average eRx 9:  0.0 if eRx inactive
                      SOA_COLUMN(float, cm10),       // pedestal-subtracted CM average eRx 10: 0.0 if eRx inactive
                      SOA_COLUMN(float, cm11),       // pedestal-subtracted CM average eRx 11: 0.0 if eRx inactive
                      SOA_COLUMN(float, msubchidx),  // mean-subtracted channel index, unique per channel, float(chIdx) - (float(nErx) * 37.0f - 1.0f) / 2.0f;
                      SOA_COLUMN(float, msuberxidx), // mean-subtracted ERX index, same within ERX,.float(erxIdx) - (float(nErx) - 1.0f) / 2.0f;
                      SOA_COLUMN(float, cellfrac),   // cell area fraction: ML_F.SF or MH_F.SF from cellareas.json, indexed by chIdx
                      SOA_COLUMN(float, unconn0),    // pedestal-subtracted ADC of unconnected channel at eRx pos 8
                      SOA_COLUMN(float, unconn1),    // pedestal-subtracted ADC of unconnected channel at eRx pos 17
                      SOA_COLUMN(float, unconn2),    // pedestal-subtracted ADC of unconnected channel at eRx pos 19
                      SOA_COLUMN(float, unconn3),    // pedestal-subtracted ADC of unconnected channel at eRx pos 28
                      SOA_COLUMN(float, ntoa),       // module-level count of digis with TOA > 0
                      SOA_COLUMN(float, ntot))       // module-level count of digis with TOT > 0

  using HGCalCMMLSoA = HGCalCMMLSoALayout<>;

}  // namespace hgcalcmml

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCALSoACMML_h
