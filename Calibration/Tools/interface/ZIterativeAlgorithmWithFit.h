#ifndef ZITERATIVEALGORITHMWITHFIT_H
#define ZITERATIVEALGORITHMWITHFIT_H

/* ******************************************
 * ZIterativeAlgorithmWithFit.h 
 *
 * Paolo Meridiani 06/03/2003
 ********************************************/

#include <TROOT.h>
#include <TClonesArray.h>
#include <vector>
#include <string>
#include <TH1.h>

#include "Calibration/Tools/interface/CalibElectron.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
/// Class that implements an iterative in situ calibration algorithm
/// using Z events 
/** \class ZIterativeAlgorithmWithFit 
    Author: paolo.meridiani@roma1.infn.it
*/

#define nMaxIterations 50
#define nMaxChannels 250



class ZIterativeAlgorithmWithFit 
{
 public:
  struct ZIterativeAlgorithmWithFitPlots {
    TH1* weightedRescaleFactor[nMaxIterations][nMaxChannels];
    TH1* unweightedRescaleFactor[nMaxIterations][nMaxChannels];
    TH1* weight[nMaxIterations][nMaxChannels];
  };

  /// Default constructor
  ZIterativeAlgorithmWithFit();
  
  /// Constructor with explicit iterations & exponent
  ZIterativeAlgorithmWithFit(const edm::ParameterSet&  ps);
  //, unsigned int events);
  
  /// Assignment operator
  ZIterativeAlgorithmWithFit & operator=(const ZIterativeAlgorithmWithFit &r){
    return *this;
  }
  
  /// Destructor
  virtual ~ZIterativeAlgorithmWithFit();
  
  bool resetIteration();
  
  bool iterate();

  bool addEvent(calib::CalibElectron*, calib::CalibElectron*, float);

  const ZIterativeAlgorithmWithFitPlots* getHistos() const { return thePlots_; }

  int getNumberOfIterations() const { return numberOfIterations_; }
  
  int getNumberOfChannels() const { return channels_; }
  
  const std::vector<float>& getOptimizedCoefficients() const { return optimizedCoefficients_; }

  const std::vector<float>& getOptimizedCoefficientsError() const { return optimizedCoefficientsError_; }

  const std::vector<float>& getOptimizedChiSquare() const { return optimizedChiSquare_; }

  const std::vector<int>& getOptimizedIterations() const { return optimizedIterations_; }

  const std::vector<float>& getWeightSum() const { return weight_sum_; }

  const std::vector<float>& getEpsilonSum() const { return calib_fac_; }

  //Helper Methods

  static inline float invMassCalc(float Energy1, float Eta1, float Phi1, float Energy2, float Eta2, float Phi2) {
    return (sqrt(2 * Energy1 * Energy2 * (1 - cosTheta12(Eta1, Phi1, Eta2, Phi2))));
  }

  static inline float cosTheta12(float Eta1, float Phi1, float Eta2, float Phi2) {
    return ((cos(Phi1 - Phi2) + sinh(Eta1) * sinh(Eta2)) / (cosh(Eta1) * cosh(Eta2)));
  }

  /*
  static TF1* gausfit(TH1F * histoou,double* par,double* errpar) {
    return gausfit(histoou,par,errpar,1.,2.);
  }
  */

  static void gausfit(TH1F * histoou,double* par,double* errpar,float nsigmalow, float nsigmaup, double* mychi2, int* iterations); 

 private:

  void addWeightsCorrections(unsigned int event_id);

  void getStatWeights(const std::string& file);

  float getEventWeight(unsigned int event_id);

  void recalculateWeightsEnergies();

  void recalculateMasses();

  void recalculateWeightsEnergies(calib::CalibElectron* electron); 

  void getWeight(unsigned int evid,std::pair<calib::CalibElectron*,calib::CalibElectron*>, float);

  void getWeight(unsigned int evid,calib::CalibElectron* ele,float);

  void bookHistograms();

  ZIterativeAlgorithmWithFitPlots* thePlots_;

  int nCrystalCut_;

  unsigned int channels_;
  unsigned int totalEvents_;
  unsigned int numberOfIterations_;

  unsigned int currentEvent_;
  unsigned int currentIteration_;

  std::vector< std::pair<calib::CalibElectron*,calib::CalibElectron*> > electrons_;

  std::vector<float> optimizedCoefficients_;
  std::vector<float> optimizedCoefficientsError_;
  std::vector<float> calib_fac_;
  std::vector<float> weight_sum_;
  std::vector<float> massReco_;
  std::vector<float> optimizedChiSquare_;
  std::vector<int> optimizedIterations_;

  std::string massMethod;

  bool UseStatWeights_;
  std::string WeightFileName_;
 
  std::vector<float> StatWeights_;
  std::vector<float> Event_Weight_;

  TString calibType_;
  
  static const double M_Z_;
};

#endif // ZIterativeAlgorithmWithFit_H


