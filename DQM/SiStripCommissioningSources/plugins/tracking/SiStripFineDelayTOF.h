#ifndef CalibTracker_SiSitripLorentzAngle_SiStripFineDelayTOF_h
#define CalibTracker_SiSitripLorentzAngle_SiStripFineDelayTOF_h

namespace reco {
  class Track;
}

class TrackingRecHit;

class SiStripFineDelayTOF {
public:
  static double timeOfFlight(bool cosmics, bool field, double* trackParameters, double* hit, double* phit, bool onDisk);
  static void trackParameters(const reco::Track& tk, double* trackParameters);

  SiStripFineDelayTOF() = delete;
  virtual ~SiStripFineDelayTOF() = delete;

private:
  static double timeOfFlightCosmic(double* hit, double* phit);
  static double timeOfFlightCosmicB(double* trackParameters, double* hit, double* phit, bool onDisk);
  static double timeOfFlightBeam(double* hit, double* phit);
  static double timeOfFlightBeamB(double* trackParameters, double* hit, double* phit, bool onDisk);
  static double x(double* trackParameters, double phi);
  static double y(double* trackParameters, double phi);
  static double z(double* trackParameters, double phi);
  static double getPhi(double* trackParameters, double* hit, bool onDisk);
};

#endif
