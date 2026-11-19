#ifndef Alignment_MuonAlignmentAlgorithms_FlatOccupancy_H
#define Alignment_MuonAlignmentAlgorithms_FlatOccupancy_H

#include <TH1F.h>
#include <TString.h>

#include <string>
#include <unordered_map>

class FlatOccupancy {
public:
  FlatOccupancy() = default;
  ~FlatOccupancy() = default;

  void LoadWeigths(TString FileName);
  float GiveCorrection(int Wheel, int Station, int Sector, float positionX, float positionY);

private:
  std::unordered_map<std::string, TH1F *> Occup_weights;
  bool map_created{false};
};

#endif
