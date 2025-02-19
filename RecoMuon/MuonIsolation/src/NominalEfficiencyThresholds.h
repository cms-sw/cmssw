#ifndef MuonIsolation_NominalEfficiencyThresholds_H
#define MuonIsolation_NominalEfficiencyThresholds_H

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace muonisolation {
class NominalEfficiencyThresholds {
public:
  NominalEfficiencyThresholds() { }
  NominalEfficiencyThresholds(const std::string & infile);
  ~NominalEfficiencyThresholds() { }

  /// threshold location
  struct ThresholdLocation { float eta; int cone; };

  float thresholdValueForEfficiency(ThresholdLocation location, float eff_thr) const;

  std::vector<double> bins() const;
  void dump();

private:


  /// compare to efficiencies
  struct EfficiencyBin {
    float eff;
    float eff_previous;
    bool operator() (const EfficiencyBin & e1,
                 const EfficiencyBin & e2) const;
  };


  class EtaBounds {
  public:
    enum { NumberOfTowers = 32 };
    EtaBounds();
    int    towerFromEta(double eta) const;
    float  operator()(unsigned int i) const { return theBounds[i]; }
  private:
    float theBounds[NumberOfTowers+1]; //max eta of towers 1-32 (indx 1-32) and 0. for indx 0
  };


  /// compare two locations
  struct locless {
    bool operator()(const ThresholdLocation & l1,
                const ThresholdLocation & l2) const;
    EtaBounds etabounds;
  };



  typedef std::pair<EfficiencyBin,float> ThresholdConstituent;
  typedef std::map<EfficiencyBin,float,EfficiencyBin> ThresholdConstituents;
  typedef std::map<ThresholdLocation,ThresholdConstituents,locless> MapType;

  void add(ThresholdLocation location, ThresholdConstituent threshold);
  MapType thresholds;

  EtaBounds etabounds;
};
}
#endif

