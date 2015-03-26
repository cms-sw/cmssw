#ifndef CalibCalorimetry_HcalAlgos_equalADCSignalTime_h_
#define CalibCalorimetry_HcalAlgos_equalADCSignalTime_h_

class QIE8Simulator;

//
// Determine the start time of the simulator input signal which
// produces equal ADC counts in time slices N and N+1. If the start
// time is set to "ttry" then ADC[N] should be larger than ADC[N+1].
// If the start time is set to "ttry"+25 then ADC[N] should be smaller
// than ADC[N+1]. Note that N should be positive.
//
double equalADCSignalTime(QIE8Simulator& sim, double dt,
                          double tDigitize, unsigned N, double ttry);

#endif // CalibCalorimetry_HcalAlgos_equalADCSignalTime_h_
