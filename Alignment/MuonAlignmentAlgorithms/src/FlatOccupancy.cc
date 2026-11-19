#include "Alignment/MuonAlignmentAlgorithms/interface/FlatOccupancy.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TFile.h>

#include <cmath>

void FlatOccupancy::LoadWeigths(TString FileName) {
  if (FileName != "") {
    if (const auto file = std::make_unique<TFile>(FileName.Data()); file) {
      edm::LogInfo("FlatOccupancy::LoadWeigths") << "Constructing FlatOccupancy using file: " << FileName;
      map_created = true;
      for (int nW = -2; nW <= 2; nW++) {
        for (int nSt = 1; nSt <= 4; nSt++) {
          for (int nSe = 1; nSe <= 14; nSe++) {
            if (nSt < 4 && (nSe == 13 || nSe == 14))
              continue;
            std::string name =
                "Occupancy_XYweight_" + std::to_string(nW) + "_" + std::to_string(nSt) + "_" + std::to_string(nSe);
            if (TH1F *h1 = file->Get<TH1F>(name.data()); h1) {
              edm::LogInfo("FlatOccupancy::LoadWeigths") << "Init Weights for: " << name;
              Occup_weights[name] = dynamic_cast<TH1F *>(h1->Clone());
            } else
              edm::LogWarning("FlatOccupancy::LoadWeigths") << name << " not found in " << FileName;
          }
        }
      }
      edm::LogInfo("FlatOccupancy::LoadWeigths") << "Weights applied!";
    } else {
      edm::LogWarning("FlatOccupancy::LoadWeigths")
          << FileName << " not found! Weights to have flat occupancy will nor be created.";
      map_created = false;
    }
  } else {
    edm::LogWarning("FlatOccupancy::LoadWeigths")
        << "FileName is empty. Weights to have flat occupancy will nor be created.";
    map_created = false;
  }
}

float FlatOccupancy::GiveCorrection(int Wheel, int Station, int Sector, float positionX, float positionY) {
  if (map_created) {
    std::string Name =
        "Occupancy_XYweight_" + std::to_string(Wheel) + "_" + std::to_string(Station) + "_" + std::to_string(Sector);
    const int BinX =
        std::floor((positionX + 210) / 4.2) + 1;  //Assuming the histrograms are from -210 to 210 with 100 bins
    const int BinY =
        std::floor((positionY + 210) / 4.2) + 1;  //Assuming the histrograms are from -210 to 210 with 100 bins
    if (auto it = Occup_weights.find(Name); it != Occup_weights.end() && it->second)
      return it->second->GetBinContent(BinX, BinY);
  }
  return 1.;
}
