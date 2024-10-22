#ifndef DATAFORMATS_ECALUNCALIBRATEDRECHIT
#define DATAFORMATS_ECALUNCALIBRATEDRECHIT

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

class EcalUncalibratedRecHit {
public:
  typedef DetId key_type;

  enum Flags {
    kGood = -1,  // channel is good (mutually exclusive with other states)  setFlagBit(kGood) reset flags_ to zero
    kPoorReco,   // channel has been badly reconstructed (e.g. bad shape, bad chi2 etc.)
    kSaturated,  // saturated channel
    kOutOfTime,  // channel out of time
    kLeadingEdgeRecovered,  // saturated channel: energy estimated from the leading edge before saturation
    kHasSwitchToGain6,      // at least one data frame is in G6
    kHasSwitchToGain1       // at least one data frame is in G1

  };

  EcalUncalibratedRecHit();

  EcalUncalibratedRecHit(
      const DetId& id, float ampl, float ped, float jit, float chi2, uint32_t flags = 0, uint32_t aux = 0);

  float amplitude() const { return amplitude_; }
  float amplitudeError() const { return amplitudeError_; }
  float pedestal() const { return pedestal_; }
  float jitter() const { return jitter_; }
  float chi2() const { return chi2_; }
  float outOfTimeAmplitude(int bx) const { return OOTamplitudes_[bx]; }

  uint32_t flags() const { return flags_; }
  float jitterError() const;
  uint8_t jitterErrorBits() const;
  DetId id() const { return id_; }

  void setAmplitude(float amplitude) { amplitude_ = amplitude; }
  void setAmplitudeError(float amplitudeerror) { amplitudeError_ = amplitudeerror; }
  void setPedestal(float pedestal) { pedestal_ = pedestal; }
  void setJitter(float jitter) { jitter_ = jitter; }
  void setChi2(float chi2) { chi2_ = chi2; }
  void setOutOfTimeAmplitude(int bx, float amplitude) { OOTamplitudes_[bx] = amplitude; }

  void setJitterError(float jitterErr);
  void setFlags(uint32_t flags) { flags_ = flags; }
  void setId(DetId id) { id_ = id; }
  void setAux(uint32_t aux) { aux_ = aux; }
  void setFlagBit(Flags flag);
  bool checkFlag(Flags flag) const;

  bool isSaturated() const;
  bool isJitterValid() const;
  bool isJitterErrorValid() const;

  // For CC Timing reco
  float nonCorrectedTime() const;
  void setNonCorrectedTime(const float correctedJittter, const float nonCorrectedJitter);

private:
  float amplitude_;       //< Reconstructed amplitude
  float amplitudeError_;  //< Reconstructed amplitude uncertainty
  float pedestal_;        //< Reconstructed pedestal
  float jitter_;          //< Reconstructed time jitter
  float chi2_;            //< Chi2 of the pulse
  //< Out-Of-Time reconstructed amplitude, one for each active BX, from readout sample 0 to 9
  float OOTamplitudes_[EcalDataFrame::MAXSAMPLES];
  float OOTchi2_;   //< Out-Of-Time Chi2
  uint32_t flags_;  //< flag to be propagated to RecHit
  uint32_t aux_;    //< aux word; first 8 bits contain time (jitter) error
  DetId id_;        //< Detector ID
};

#endif
