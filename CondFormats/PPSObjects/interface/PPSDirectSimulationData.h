#ifndef CondFormats_PPSObjects_PPSDirectSimulationData_h
#define CondFormats_PPSObjects_PPSDirectSimulationData_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH2F.h"
class PPSDirectSimulationData {
public:
    // Constructor
    PPSDirectSimulationData();
    // Destructor
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

    void printInfo(std::stringstream &s);

private:
    bool useEmpiricalApertures;
    std::string empiricalAperture45;
    std::string empiricalAperture56;

    std::string timeResolutionDiamonds45;
    std::string timeResolutionDiamonds56;

    bool useTimeEfficiencyCheck;
    std::string effTimePath;
    std::string effTimeObject45;
    std::string effTimeObject56;


    COND_SERIALIZABLE
};
std::ostream &operator<<(std::ostream &, PPSDirectSimulationData);
#endif
