#ifndef MuonReco_DYTInfo_h
#define MuonReco_DYTInfo_h

#include <vector>
#include <map>
#include <iostream>
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {
 
  class DYTInfo {
  public:
    /// Constructor - Destructor
    DYTInfo();
    ~DYTInfo();

    /// copy from another DYTInfo
    void CopyFrom(const DYTInfo&);
    
    /// number of stations used by DYT
    const int  NStUsed() const { return NStUsed_; };
    void setNStUsed( int NStUsed ) { NStUsed_=NStUsed; };
       
    /// estimator values for all station
    const std::vector<double> DYTEstimators() const { return DYTEstimators_; };
    void setDYTEstimators( std::map<int, double> &dytEstMap ) {
      DYTEstimators_.clear();
      for (int st = 1; st <= 4; st++) {
	if (dytEstMap.count(st) > 0) DYTEstimators_.push_back(dytEstMap[st]);
        else DYTEstimators_.push_back(-1);
      }
    };
    void setDYTEstimators( const std::vector<double> &EstValues ) { DYTEstimators_ = EstValues; }
    
    /// number of segments tested per muon station
    const std::vector<bool> UsedStations() const { return UsedStations_; };
    void setUsedStations( std::map<int, bool> &ustMap ) {
      UsedStations_.clear();
      for (int st = 1; st <= 4; st++) 
        UsedStations_.push_back(ustMap[st]);
    };
    void setUsedStations( std::vector<bool> ustVal ) { UsedStations_ = ustVal; };

    /// DetId vector of chamber with valid estimator
    const std::vector<DetId> IdChambers() const { return IdChambers_; };
    void setIdChambers( std::map<int, DetId> &IdChambersMap ) {
      IdChambers_.clear();
      for (int st = 1; st <= 4; st++)
        IdChambers_.push_back(IdChambersMap[st]);
    };
    void setIdChambers( const std::vector<DetId> &IdChambersVal ) { IdChambers_ = IdChambersVal; };
    
    /// vector of thresholds                                                                                                                                                        
    const std::vector<double> Thresholds() const { return Thresholds_; };
    void setThresholds( std::map<int, double> &ThresholdsMap ) {
      Thresholds_.clear();
      for (int st = 1; st <= 4; st++)
	Thresholds_.push_back(ThresholdsMap[st]);
    };
    void setThresholds( const std::vector<double> &ThresholdsVal ) { Thresholds_ = ThresholdsVal; };
    
  private:
    
    int  NStUsed_;
    std::vector<bool> UsedStations_;
    std::vector<double> DYTEstimators_;
    std::vector<DetId> IdChambers_;
    std::vector<double> Thresholds_;
  };
}
#endif
