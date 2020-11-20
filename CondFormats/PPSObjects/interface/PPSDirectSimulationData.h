#ifndef CondFormats_PPSObjects_PPSDirectSimulationData_h
#define CondFormats_PPSObjects_PPSDirectSimulationData_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH2F.h"

class PPSDirectSimulationData {
public:
  PPSDirectSimulationData();
  ~PPSDirectSimulationData();

  // Getters
  bool getUseEmpiricalApertures() const;
  const std::string& getEmpiricalAperture45() const;
  const std::string& getEmpiricalAperture56() const;
  const std::string& getTimeResolutionDiamonds45() const;
  const std::string& getTimeResolutionDiamonds56() const;
  bool getUseTimeEfficiencyCheck() const;
  const std::string& getEffTimePath() const;
  const std::string& getEffTimeObject45() const;
  const std::string& getEffTimeObject56() const;

  // Setters
  void setUseEmpiricalApertures(bool b);
  void setEmpiricalAperture45(std::string s);
  void setEmpiricalAperture56(std::string s);
  void setTimeResolutionDiamonds45(std::string s);
  void setTimeResolutionDiamonds56(std::string s);
  void setUseTimeEfficiencyCheck(bool b);
  void setEffTimePath(std::string s);
  void setEffTimeObject45(std::string s);
  void setEffTimeObject56(std::string s);

  void printInfo(std::stringstream& s);

private:
  std::string empiricalAperture45_;
  std::string empiricalAperture56_;

  std::string timeResolutionDiamonds45_;
  std::string timeResolutionDiamonds56_;

  std::string effTimePath_;
  std::string effTimeObject45_;
  std::string effTimeObject56_;

  COND_SERIALIZABLE
};

std::ostream& operator<<(std::ostream&, PPSDirectSimulationData);

#endif