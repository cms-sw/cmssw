#ifndef PhysicsTools_Utilities_interface_Lumi3DReWeighting_h
#define PhysicsTools_Utilities_interface_Lumi3DReWeighting_h

/**
  \class    Lumi3DReWeighting Lumi3DReWeighting.h "PhysicsTools/Utilities/interface/Lumi3DReWeighting.h"
  \brief    Class to provide lumi weighting for analyzers to weight "flat-to-N" MC samples to data

  This class will trivially take two histograms:
  1. The generated "flat-to-N" distributions from a given processing
  2. A histogram generated from the "estimatePileup" macro here:

  https://twiki.cern.ch/twiki/bin/view/CMS/LumiCalc#How_to_use_script_estimatePileup

  \author Mike Hildreth
*/

#include "TH1.h"
#include "TFile.h"
#include <cmath>
#include <string>

#include <vector>

namespace edm {
  class EventBase;
  class Lumi3DReWeighting {
  public:
    Lumi3DReWeighting(std::string generatedFile,
                      std::string dataFile,
                      std::string GenHistName,
                      std::string DataHistName,
                      std::string WeightOutputFile);

    Lumi3DReWeighting(const std::vector<float>& MC_distr,
                      const std::vector<float>& Lumi_distr,
                      std::string WeightOutputFile);

    Lumi3DReWeighting(){};

    double weight3D(const edm::EventBase& e);

    double weight3D(int, int, int);

    void weight3D_set(std::string generatedFile,
                      std::string dataFile,
                      std::string GenHistName,
                      std::string DataHistName,
                      std::string WeightOutputFile);

    void weight3D_init(float Scale);

    void weight3D_init(std::string WeightFileName);  // initialize from root file

    void weight3D_init(std::string MCFileName, std::string DataFileName);  // initialize from root files

  protected:
    std::string generatedFileName_;
    std::string dataFileName_;
    std::string GenHistName_;
    std::string DataHistName_;
    std::string weightFileName_;
    std::shared_ptr<TFile> generatedFile_;
    std::shared_ptr<TFile> dataFile_;
    std::shared_ptr<TH1> weights_;

    //keep copies of normalized distributions:

    std::shared_ptr<TH1> MC_distr_;
    std::shared_ptr<TH1> Data_distr_;

    double Weight3D_[50][50][50];
  };
}  // namespace edm

#endif
