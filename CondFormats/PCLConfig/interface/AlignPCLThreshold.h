#ifndef CondFormats_PCLConfig_AlignPCLThreshold_h
#define CondFormats_PCLConfig_AlignPCLThreshold_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <array>

class AlignPCLThreshold {
public:
  struct coordThresholds {
    coordThresholds() {
      m_Cut = 5.;
      m_sigCut = 2.5;
      m_errorCut = 10.;
      m_maxMoveCut = 200;
      m_label = "default";
    }
    ~coordThresholds() {}
    void setThresholds(
        float theCut, float theSigCut, float theErrorCut, float theMaxMoveCut, const std::string &theLabel) {
      m_Cut = theCut;
      m_sigCut = theSigCut;
      m_errorCut = theErrorCut;
      m_maxMoveCut = theMaxMoveCut;
      m_label = theLabel;
    }

    float m_Cut;
    float m_sigCut;
    float m_errorCut;
    float m_maxMoveCut;
    std::string m_label;

    COND_SERIALIZABLE;
  };

  virtual ~AlignPCLThreshold() {}

  AlignPCLThreshold(coordThresholds X = coordThresholds(),
                    coordThresholds tX = coordThresholds(),
                    coordThresholds Y = coordThresholds(),
                    coordThresholds tY = coordThresholds(),
                    coordThresholds Z = coordThresholds(),
                    coordThresholds tZ = coordThresholds(),
                    std::vector<coordThresholds> extraDOF = std::vector<coordThresholds>());

  float getXcut() const { return m_xCoord.m_Cut; }
  float getYcut() const { return m_yCoord.m_Cut; }
  float getZcut() const { return m_zCoord.m_Cut; }
  float getThetaXcut() const { return m_thetaXCoord.m_Cut; }
  float getThetaYcut() const { return m_thetaYCoord.m_Cut; }
  float getThetaZcut() const { return m_thetaZCoord.m_Cut; }

  float getSigXcut() const { return m_xCoord.m_sigCut; }
  float getSigYcut() const { return m_yCoord.m_sigCut; }
  float getSigZcut() const { return m_zCoord.m_sigCut; }
  float getSigThetaXcut() const { return m_thetaXCoord.m_sigCut; }
  float getSigThetaYcut() const { return m_thetaYCoord.m_sigCut; }
  float getSigThetaZcut() const { return m_thetaZCoord.m_sigCut; }

  float getErrorXcut() const { return m_xCoord.m_errorCut; }
  float getErrorYcut() const { return m_yCoord.m_errorCut; }
  float getErrorZcut() const { return m_zCoord.m_errorCut; }
  float getErrorThetaXcut() const { return m_thetaXCoord.m_errorCut; }
  float getErrorThetaYcut() const { return m_thetaYCoord.m_errorCut; }
  float getErrorThetaZcut() const { return m_thetaZCoord.m_errorCut; }

  float getMaxMoveXcut() const { return m_xCoord.m_maxMoveCut; }
  float getMaxMoveYcut() const { return m_yCoord.m_maxMoveCut; }
  float getMaxMoveZcut() const { return m_zCoord.m_maxMoveCut; }
  float getMaxMoveThetaXcut() const { return m_thetaXCoord.m_maxMoveCut; }
  float getMaxMoveThetaYcut() const { return m_thetaYCoord.m_maxMoveCut; }
  float getMaxMoveThetaZcut() const { return m_thetaZCoord.m_maxMoveCut; }

  bool hasExtraDOF() const { return (!m_extraDOF.empty()); }
  unsigned int extraDOFSize() const { return m_extraDOF.size(); }
  std::array<float, 4> getExtraDOFCuts(const unsigned int i) const;
  std::string getExtraDOFLabel(const unsigned int i) const;

private:
  coordThresholds m_xCoord;
  coordThresholds m_yCoord;
  coordThresholds m_zCoord;
  coordThresholds m_thetaXCoord;
  coordThresholds m_thetaYCoord;
  coordThresholds m_thetaZCoord;
  std::vector<coordThresholds> m_extraDOF;

  COND_SERIALIZABLE;
};

#endif
