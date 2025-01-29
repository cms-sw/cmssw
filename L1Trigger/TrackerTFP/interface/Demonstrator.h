#ifndef L1Trigger_TrackerTFP_Demonstrator_h
#define L1Trigger_TrackerTFP_Demonstrator_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackerTFP/interface/DemonstratorRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <vector>
#include <string>

namespace trackerTFP {

  /*! \class  trackerTFP::Demonstrator
   *  \brief  ESProduct providing the algorithm to run input data through modelsim
   *          and to compares results with expected output data
   *  \date   2021, April
   */
  class Demonstrator {
  public:
    // configuration
    struct Config {
      std::string dirIPBB_;
      double runTime_;
      std::vector<int> linkMappingIn_;
      std::vector<int> linkMappingOut_;
    };
    Demonstrator() {}
    Demonstrator(const Config& iConfig, const tt::Setup* setup);
    //~Demonstrator() {}
    // plays input through modelsim and compares result with output
    bool analyze(const std::vector<std::vector<tt::Frame>>& input,
                 const std::vector<std::vector<tt::Frame>>& output) const;

  private:
    // converts streams of bv into stringstream
    void convert(const std::vector<std::vector<tt::Frame>>& bits,
                 std::stringstream& ss,
                 const std::vector<int>& mapping) const;
    // plays stringstream through modelsim
    void sim(const std::stringstream& ss) const;
    // compares stringstream with modelsim output
    bool compare(std::stringstream& ss) const;
    // creates emp file header
    std::string header(const std::vector<int>& links) const;
    // creates 6 frame gap between packets
    std::string infraGap(int& nFrame, int numLinks) const;
    // creates frame number
    std::string frame(int& nFrame) const;
    // converts bv into hex
    std::string hex(const tt::Frame& bv, bool first = false) const;

    // path to ipbb proj area
    std::string dirIPBB_;
    // runtime in ms
    double runTime_;
    //
    std::vector<int> linkMappingIn_;
    //
    std::vector<int> linkMappingOut_;
    // path to input text file
    std::string dirIn_;
    // path to output text file
    std::string dirOut_;
    // path to expected output text file
    std::string dirPre_;
    // path to diff text file
    std::string dirDiff_;
    // number of frames per event (161)
    int numFrames_;
    // number of emp reset frames per event (6)
    int numFramesInfra_;
    // number of TFPs per time node (9)
    int numRegions_;
  };

}  // namespace trackerTFP

EVENTSETUP_DATA_DEFAULT_RECORD(trackerTFP::Demonstrator, trackerTFP::DemonstratorRcd);

#endif
