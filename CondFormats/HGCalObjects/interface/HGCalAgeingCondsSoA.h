#ifndef CondFormats_HGCalObjects_interface_HGCalAgeingCondsSoA_h
#define CondFormats_HGCalObjects_interface_HGCalAgeingCondsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcal {

  // Generate structure of module-level (ECON-D) arrays (SoA) layout with module mapping information
  GENERATE_SOA_LAYOUT(HGCalAgeingCondsSoALayout,
                      SOA_COLUMN(float, signal),         // signal (reference)
                      SOA_COLUMN(float, signalScale),    // signal ageing scale factor
                      SOA_COLUMN(float, enc),            // total noise
                      SOA_COLUMN(float, enc_s),          // series noise
                      SOA_COLUMN(float, enc_p),          // parallel noise
                      SOA_COLUMN(float, ileak),          // leakage current
                      SOA_COLUMN(float, xtalk),          // cross-talk or common-mode scaling factor
                      SOA_COLUMN(float, ntotalPE),       // total number of photo-electrons (SiPM-only)
                      SOA_COLUMN(unsigned short, gain),  // gain range most appropriate gain for given signal and noise
                      SOA_COLUMN(float, adc_lsb),        // ADC least significant bit (LSB) for the given gain range
                      SOA_COLUMN(float, tot_lsb),        // TOT least significant bit (LSB)
                      SOA_COLUMN(float, tot_thr),        // TOT threshold
                      SOA_COLUMN(float, toa_thr),        // TOA threshold
                      SOA_COLUMN(float, ln_f),           // ln(fluence)
                      SOA_COLUMN(float, ln_dose)         // ln(dose)
  )

  using HGCalAgeingCondsSoA = HGCalAgeingCondsSoALayout<>;

}  // namespace hgcal

#endif  // CondFormats_HGCalObjects_interface_HGCalAgeingCondsSoA_h
