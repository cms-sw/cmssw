#ifndef CALIBTRACKER_SISTRIPCHANNELGAIN_STAPVGAIN_H
#define CALIBTRACKER_SISTRIPCHANNELGAIN_STAPVGAIN_H

class TH1F;
#include <string>

struct stAPVGain {
  unsigned int Index;
  int Bin;
  unsigned int DetId;
  unsigned int APVId;
  unsigned int SubDet;
  float x;
  float y;
  float z;
  float Eta;
  float R;
  float Phi;
  float Thickness;
  double FitMPV;
  double FitMPVErr;
  double FitWidth;
  double FitWidthErr;
  double FitChi2;
  double FitNorm;
  double Gain;
  double CalibGain;
  double PreviousGain;
  double PreviousGainTick;
  double NEntries;
  TH1F* HCharge;
  TH1F* HChargeN;
  bool isMasked;
};

struct APVloc {
public:
  APVloc(int v0, int v1, int v2, int v3, const std::string& s)
      : m_thickness(v0), m_subdetectorId(v1), m_subdetectorSide(v2), m_subdetectorPlane(v3), m_string(s) {}

  int m_thickness;
  int m_subdetectorId;
  int m_subdetectorSide;
  int m_subdetectorPlane;
  std::string m_string;

  bool operator==(const APVloc& a) const {
    return (m_subdetectorId == a.m_subdetectorId && m_subdetectorSide == a.m_subdetectorSide &&
            m_subdetectorPlane == a.m_subdetectorPlane && m_thickness == a.m_thickness);
  }
};

enum statistic_type { None = -1, StdBunch, StdBunch0T, FaABunch, FaABunch0T, IsoBunch, IsoBunch0T, Harvest };

#endif
