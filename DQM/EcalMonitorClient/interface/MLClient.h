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

    //Occcupancy map is plotted at the tower level
    //For the EB: 34 towers in the eta and 72 towers in the phi directions
    static const int nEBEtaTowers = 34;
    static const int nEBPhiTowers = 72;
    //For the EE: 20 towers in the eta and 20 towers in the phi directions
    static const int nEEEtaTowers = 20;
    static const int nEEPhiTowers = 20;
    //After padding with two rows above and below to prevent the edge effect
    //For EB: 36 towers in eta direction
    //For EE: padding on all four sides, 22 towers in both eta and phi directions.
    static const int nEBEtaTowersPad = 36;
    static const int nEETowersPad = 22;
    float EBThreshold_;
    float EEpThreshold_;
    float EEmThreshold_;
    float EB_PUcorr_slope_;
    float EEp_PUcorr_slope_;
    float EEm_PUcorr_slope_;
    float EB_PUcorr_intercept_;
    float EEp_PUcorr_intercept_;
    float EEm_PUcorr_intercept_;

    size_t nLS = 4;      //No.of lumisections to add the occupancy over
    size_t nLSloss = 6;  //No.of lumisections to multiply the loss over
    int nbadtowerEB;     //count the no.of bad towers flagged by the ML model.
    int nbadtowerEE;
    int LScount = 0;  //count no.of lumisections over which the MLquality is made.

    std::deque<int> NEventQ;  //To keep the no.of events in each occupancy plot

    //To keep the input occupancy plots to be summed
    std::deque<std::valarray<float>> ebOccMap1dQ;
    std::deque<std::valarray<float>> eepOccMap1dQ;
    std::deque<std::valarray<float>> eemOccMap1dQ;
    //To keep the average occupancy to do response correction
    std::vector<double> EBavgOcc;
    std::vector<double> EEpavgOcc;
    std::vector<double> EEmavgOcc;
    //To keep the ML losses to be multiplied
    std::deque<std::valarray<std::valarray<float>>> EBlossMap2dQ;
    std::deque<std::valarray<std::valarray<float>>> EEplossMap2dQ;
    std::deque<std::valarray<std::valarray<float>>> EEmlossMap2dQ;
  };

}  // namespace ecaldqm

#endif
