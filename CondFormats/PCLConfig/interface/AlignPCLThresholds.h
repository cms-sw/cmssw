#ifndef CondFormats_PCLConfig_AlignPCLThresholds_h
#define CondFormats_PCLConfig_AlignPCLThresholds_h

#include "CondFormats/PCLConfig/interface/AlignPCLThreshold.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <array>
#include <map>
#include <string>
#include <vector>

class AlignPCLThresholds {
public:
  typedef std::map<std::string, AlignPCLThreshold> threshold_map;
  enum coordType { X, Y, Z, theta_X, theta_Y, theta_Z, extra_DOF, endOfTypes };

  AlignPCLThresholds() {}
  virtual ~AlignPCLThresholds() {}

  void setAlignPCLThreshold(const std::string &AlignableId, const AlignPCLThreshold &Threshold);
  void setAlignPCLThresholds(const int &Nrecords, const threshold_map &Thresholds);
  void setNRecords(const int &Nrecords);

  const threshold_map &getThreshold_Map() const { return m_thresholds; }
  const int &getNrecords() const { return m_nrecords; }

  AlignPCLThreshold getAlignPCLThreshold(const std::string &AlignableId) const;
  AlignPCLThreshold &getAlignPCLThreshold(const std::string &AlignableId);

  float getSigCut(const std::string &AlignableId, const coordType &type) const;
  float getCut(const std::string &AlignableId, const coordType &type) const;
  float getMaxMoveCut(const std::string &AlignableId, const coordType &type) const;
  float getMaxErrorCut(const std::string &AlignableId, const coordType &type) const;

  // overloaded methods to get all the coordinates
  std::array<float, 6> getSigCut(const std::string &AlignableId) const;
  std::array<float, 6> getCut(const std::string &AlignableId) const;
  std::array<float, 6> getMaxMoveCut(const std::string &AlignableId) const;
  std::array<float, 6> getMaxErrorCut(const std::string &AlignableId) const;

  std::array<float, 4> getExtraDOFCutsForAlignable(const std::string &AlignableId, const unsigned int i) const;
  std::string getExtraDOFLabelForAlignable(const std::string &AlignableId, const unsigned int i) const;

  double size() const { return m_thresholds.size(); }
  std::vector<std::string> getAlignableList() const;

  void printAll() const;

protected:
  threshold_map m_thresholds;
  int m_nrecords;

  COND_SERIALIZABLE;
};

#endif
