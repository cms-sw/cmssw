#ifndef RecoLocalCalo_HcalRecAlgos_HcalCorrectionFunctions_h_
#define RecoLocalCalo_HcalRecAlgos_HcalCorrectionFunctions_h_

///Timeshift correction for HPDs based on the position of the peak ADC measurement.
///  Allows for an accurate determination of the relative phase of the pulse shape from
///  the HPD.  Calculated based on a weighted sum of the -1,0,+1 samples relative to the peak
///  as follows:  wpksamp = (0*sample[0] + 1*sample[1] + 2*sample[2]) / (sample[0] + sample[1] + sample[2])
///  where sample[1] is the maximum ADC sample value.
float timeshift_ns_hbheho(float wpksamp);

/// Special energy correction for some HB- cells
float hbminus_special_ecorr(int ieta, int iphi, double energy, int runnum);

#endif  // RecoLocalCalo_HcalRecAlgos_HcalCorrectionFunctions_h_
