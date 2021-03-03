#ifndef CondFormats_PPSObjects_PPSDirectSimulationData_h
#define CondFormats_PPSObjects_PPSDirectSimulationData_h

#include <string>
#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH2F.h"

class PPSDirectSimulationData {
public:
  PPSDirectSimulationData();
  ~PPSDirectSimulationData();

  typedef std::pair<std::string, std::string> FileObject;

  // Getters
  const std::string& getEmpiricalAperture45() const;
  const std::string& getEmpiricalAperture56() const;

  const std::string& getTimeResolutionDiamonds45() const;
  const std::string& getTimeResolutionDiamonds56() const;

  std::map<unsigned int, FileObject>& getEfficienciesPerRP();
  std::map<unsigned int, FileObject>& getEfficienciesPerPlane();

  // Setters
  void setEmpiricalAperture45(std::string s);
  void setEmpiricalAperture56(std::string s);

  void setTimeResolutionDiamonds45(std::string s);
  void setTimeResolutionDiamonds56(std::string s);

  // utility methods
  std::map<unsigned int, std::unique_ptr<TH2F>> loadEffeciencyHistogramsPerRP() const;
  std::map<unsigned int, std::unique_ptr<TH2F>> loadEffeciencyHistogramsPerPlane() const;

private:
  std::string empiricalAperture45_;
  std::string empiricalAperture56_;

  std::string timeResolutionDiamonds45_;
  std::string timeResolutionDiamonds56_;

  std::map<unsigned int, FileObject> efficienciesPerRP_, efficienciesPerPlane_;

  static std::unique_ptr<TH2F> loadObject(const std::string& file, const std::string& object);
  static std::string replace(std::string input, const std::string& from, const std::string& to);

  COND_SERIALIZABLE
};

#endif