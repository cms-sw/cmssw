#ifndef MLClient_H
#define MLClient_H

#include "DQWorkerClient.h"
#include <vector>
#include <valarray>
#include <deque>

namespace ecaldqm {

  class MLClient : public DQWorkerClient {
  public:
    MLClient();
    ~MLClient() override {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    //Each Ecal Barrel occupancy map is plotted at the tower level
    //34 towers in the eta and 72 towers in the phi directions
    static const int nEtaTowers = 34;
    static const int nPhiTowers = 72;
    //After padding with two rows above and below to prevent the edge effect, 36 towers in eta direction
    static const int nEtaTowersPad = 36;
    float MLThreshold_;
    float PUcorr_slope_;
    float PUcorr_intercept_;
    size_t nLS = 3;      //No.of lumisections to add the occupancy over
    size_t nLSloss = 6;  //No.of lumisections to multiply the loss over

    std::deque<int> NEventQ;                       //To keep the no.of events in each occupancy plot
    std::deque<std::valarray<float>> ebOccMap1dQ;  //To keep the input occupancy plots to be summed
    std::vector<double> avgOcc_;                   //To keep the average occupancy to do response correction
    std::deque<std::valarray<std::valarray<float>>> lossMap2dQ;  //To keep the ML losses to be multiplied
  };

}  // namespace ecaldqm

#endif
