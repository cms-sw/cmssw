#ifndef Alignment_MuonAlignmentAlgorithms_FlatOccupancy_H
#define Alignment_MuonAlignmentAlgorithms_FlatOccupancy_H

#include <TFile.h>
#include <TH1F.h>
#include <TMath.h>
#include <TFormula.h>
#include <TPad.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TROOT.h>
#include <TMarker.h>
#include <TChain.h>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include "TMath.h"
#include "TBranch.h"
#include "TString.h"
#include "TEventList.h"
#include "TArrow.h"
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

class FlatOccupancy {
public:
  FlatOccupancy();
  ~FlatOccupancy();
  void LoadWeigths(TString FileName);
  float GiveCorrection(int Wheel, int Station, int Sector, float positionX, float positionY);

private:
  std::map<TString, TH1F *> Occup_weights;
  bool map_created;
};

inline FlatOccupancy::FlatOccupancy() { map_created = false; }
inline FlatOccupancy::~FlatOccupancy() {}

inline void FlatOccupancy::LoadWeigths(TString FileName) {
  if (FileName != "") {
    TFile *f = new TFile(FileName.Data());
    if (f) {
      std::cout << "Constructing FlatOccupancy using file: " << FileName << std::endl;
      map_created = true;
      for (int nW = -2; nW <= 2; nW++) {
        for (int nSt = 1; nSt <= 4; nSt++) {
          for (int nSe = 1; nSe <= 14; nSe++) {
            string wheel = std::to_string(nW);
            string stat = std::to_string(nSt);
            string sect = std::to_string(nSe);
            if (nSt < 4 && (nSe == 13 || nSe == 14))
              continue;
            TString name = "Occupancy_XYweight_" + wheel + "_" + stat + "_" + sect;
            TH1F *h1 = (TH1F *)f->Get(name.Data());
            if (h1)
              std::cout << "Init Weights for: " << name << std::endl;
            if (!h1)
              std::cout << "Warning!!! " << name << " not found in " << FileName << std::endl;
            Occup_weights[name] = h1;
          }
        }
      }
      std::cout << "Weights applied!" << std::endl;
    } else {
      std::cout << "Warning!!! " << FileName << " not found! Weights to have flat occupancy will nor be created."
                << std::endl;
      map_created = false;
    }
  } else {
    std::cout << "Warning!!! FileName is empty. Weights to have flat occupancy will nor be created." << std::endl;
    map_created = false;
  }
}

inline float FlatOccupancy::GiveCorrection(int Wheel, int Station, int Sector, float positionX, float positionY) {
  TString Name =
      "Occupancy_XYweight_" + std::to_string(Wheel) + "_" + std::to_string(Station) + "_" + std::to_string(Sector);
  int BinX = floor((positionX + 210) / 4.2) + 1;  //Assuming the histrograms are from -210 to 210 with 100 bins
  int BinY = floor((positionY + 210) / 4.2) + 1;  //Assuming the histrograms are from -210 to 210 with 100 bins
  if (map_created) {
    float weight = Occup_weights[Name]->GetBinContent(BinX, BinY);
    return weight;
  } else
    return 1.;
}

#endif
