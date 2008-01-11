#ifndef _CONTENTS_WITHIN_RANGE_ROOT_H
#define _CONTENTS_WITHIN_RANGE_ROOT_H

#include "DQMServices/Core/interface/QualTestBase.h"
#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>

/// check that histogram contents are between [Xmin, Xmax]
class ContentsXRangeROOT : public SimpleTest<TH1F>
{
 public:
  ContentsXRangeROOT(void) : SimpleTest<TH1F>(){rangeInitialized = false;}
  ~ContentsXRangeROOT(void){}
  /// run the test (result: fraction of entries [*not* bins!] within X-range)
  /// [0, 1] or <0 for failure
  float runTest(const TH1F * const h);
  /// set allowed range in X-axis (default values: histogram's X-range)
  void setAllowedXRange(float xmin, float xmax)
  {xmin_ = xmin; xmax_ = xmax; rangeInitialized = true;}

  ///get  algorithm name
  static std::string getAlgoName(void){return "ContentsXRange";}
  
 protected:
  /// allowed range in X-axis
  float xmin_;
  float xmax_;
  /// init-flag for xmin_, xmax_
  bool rangeInitialized;

};

/// check that histogram contents are between [Ymin, Ymax]
/// (class also used by DeadChannelROOT algorithm)
class ContentsYRangeROOT : public SimpleTest<TH1F>
{
 public:
  ContentsYRangeROOT(void) : SimpleTest<TH1F>()
  {rangeInitialized = false; deadChanAlgo_ = false;}
  ~ContentsYRangeROOT(void){}
  /// run the test (result: fraction of bins [*not* entries!] that passed test)
  /// [0, 1] or <0 for failure
  float runTest(const TH1F * const h);
  /// set allowed range in Y-axis (default values: histogram's FULL Y-range)
  void setAllowedYRange(float ymin, float ymax)
  {ymin_ = ymin; ymax_ = ymax; rangeInitialized = true;}

  ///get  algorithm name
  static std::string getAlgoName(void){return "ContentsYRange";}
  
 protected:
  /// allowed range in Y-axis
  float ymin_;
  float ymax_;
  /// to be used to run derived-class algorithm
  bool deadChanAlgo_;
  /// init-flag for ymin_, ymax_
  bool rangeInitialized;

};

/// the ContentsYRangeROOT algorithm w/o a check for Ymax and excluding Ymin
class DeadChannelROOT : public ContentsYRangeROOT
{
 public:
  DeadChannelROOT(void) : ContentsYRangeROOT()
  {
    setAllowedYRange(0, 0); /// ymax value is ignored
    deadChanAlgo_ = true;
  }
  ~DeadChannelROOT(void){}

  /// set Ymin (inclusive) threshold for "dead" channel (default: 0)
  void setThreshold(float ymin)
  { setAllowedYRange(ymin, 0);} /// ymax value is ignored
  ///get  algorithm name
  static std::string getAlgoName(void){return "DeadChannel";}
};


/// check if any channels are noisy compared to neighboring ones
class NoisyChannelROOT : public SimpleTest<TH1F>
{
 public:
  NoisyChannelROOT(void) : SimpleTest<TH1F>()
    {rangeInitialized = false; numNeighbors = 1;}
  ~NoisyChannelROOT(void){}
  /// run the test (result: fraction of channels not appearing noisy or "hot")
  /// [0, 1] or <0 for failure
  float runTest(const TH1F * const h);
  /// set # of neighboring channels for calculating average to be used 
  /// for comparison with channel under consideration;
  /// use 1 for considering bin+1 and bin-1 (default), 
  /// use 2 for considering bin+1,bin-1, bin+2,bin-2, etc;
  /// Will use rollover when bin+i or bin-i is beyond histogram limits (e.g.
  /// for histogram with N bins, bin N+1 corresponds to bin 1, 
  /// and bin -1 corresponds to bin N)
  void setNumNeighbors(unsigned N)
  {if (N>0) numNeighbors = N;}
  /// set (percentage) tolerance for considering a channel noisy;
  /// eg. if tolerance = 20%, a channel will be noisy 
  /// if (contents-average)/|average| > 20%; average is calculated from 
  /// neighboring channels (also see method setNumNeighbors)
  void setTolerance(float percentage)
  {if (percentage >=0){ tolerance = percentage; rangeInitialized = true;}}

  ///get  algorithm name
  static std::string getAlgoName(void){return "NoisyChannel";}
  
 protected:
  /// tolerance for considering a channel noisy
  float tolerance;
  /// # of neighboring channels for calculating average to be used 
  /// for comparison with channel under consideration;
  unsigned numNeighbors;
  /// init-flag for tolerance
  bool rangeInitialized;

  /// get average for bin under consideration
  /// (see description of method setNumNeighbors)
  Double_t getAverage(int bin, const TH1F * const h) const;
};

/// check that every TH2F channel has mean, RMS within allowed range;
/// implementation: Giuseppe Della Ricca
class ContentsTH2FWithinRangeROOT : public SimpleTest<TH2F>
{
 public:

  ContentsTH2FWithinRangeROOT(void) : SimpleTest<TH2F>(){
    checkMean_ = checkRMS_ = validMethod_ = false;
    minMean_ = maxMean_ = minRMS_ = maxRMS_ = 0.0;
    checkMeanTolerance_ = false;
    toleranceMean_ = -1.0;
  }
  virtual ~ContentsTH2FWithinRangeROOT(void){}

  /// set expected value for mean
  void setMeanRange(float xmin, float xmax){
    checkRange(xmin, xmax);
    minMean_ = xmin;
    maxMean_ = xmax;
    checkMean_ = true;
  }

  /// set expected value for mean
  void setRMSRange(float xmin, float xmax){
    checkRange(xmin, xmax);
    minRMS_ = xmin;
    maxRMS_ = xmax;
    checkRMS_ = true;
  }

  /// set (fractional) tolerance for mean
  void setMeanTolerance(float frac_tolerance) {
    if ( frac_tolerance >= 0.0 ) {
      toleranceMean_ = frac_tolerance;
      checkMeanTolerance_ = true;
    }
  }
  

  /// run the test
  float runTest(const TH2F * const h);

  ///get  algorithm name
  static std::string getAlgoName(void){
    return "ContentsWithinExpectedTH2F";
  }

  /// true if test cannot run
  bool isInvalid(void){
    return !validMethod_;
  }

 protected:

  /// if true, check the mean value
  bool checkMean_;

  /// if true, check the RMS value
  bool checkRMS_;

  /// if true, check mean tolerance
  bool checkMeanTolerance_;

  /// fractional tolerance on mean (use only if checkMeanTolerance_ = true)
  float toleranceMean_;
  
  /// allowed range for mean (use only if checkMean_ = true)
  float minMean_, maxMean_;

  /// allowed range for mean (use only if checkRMS_ = true)
  float minRMS_, maxRMS_;

  /// true if method has been chosen
  bool validMethod_;

  /// check that allowed range is logical
  void checkRange(const float xmin, const float xmax);

};

/// check that every TProf channel has mean, RMS within allowed range;
/// implementation: Giuseppe Della Ricca
class ContentsProfWithinRangeROOT : public SimpleTest<TProfile>
{
 public:

  ContentsProfWithinRangeROOT(void) : SimpleTest<TProfile>(){
    checkMean_ = checkRMS_ = validMethod_ = false;
    minMean_ = maxMean_ = minRMS_ = maxRMS_ = 0.0;
    checkMeanTolerance_ = false;
    toleranceMean_ = -1.0;
  }
  virtual ~ContentsProfWithinRangeROOT(void){}

  /// set expected value for mean
  void setMeanRange(float xmin, float xmax){
    checkRange(xmin, xmax);
    minMean_ = xmin;
    maxMean_ = xmax;
    checkMean_ = true;
  }

  /// set expected value for mean
  void setRMSRange(float xmin, float xmax){
    checkRange(xmin, xmax);
    minRMS_ = xmin;
    maxRMS_ = xmax;
    checkRMS_ = true;
  }

  /// set (fractional) tolerance for mean
  void setMeanTolerance(float frac_tolerance) {
    if ( frac_tolerance >= 0.0 ) {
      toleranceMean_ = frac_tolerance;
      checkMeanTolerance_ = true;
    }
  }

  /// run the test
  float runTest(const TProfile * const h);

  ///get  algorithm name
  static std::string getAlgoName(void){
    return "ContentsWithinExpectedProf";
  }

  /// true if test cannot run
  bool isInvalid(void){
    return !validMethod_;
  }

 protected:

  /// if true, check the mean value
  bool checkMean_;

  /// if true, check the RMS value
  bool checkRMS_;

  /// if true, check mean tolerance
  bool checkMeanTolerance_;

  /// fractional tolerance on mean (use only if checkMeanTolerance_ = true)
  float toleranceMean_;

  /// allowed range for mean (use only if checkMean_ = true)
  float minMean_, maxMean_;

  /// allowed range for mean (use only if checkRMS_ = true)
  float minRMS_, maxRMS_;

  /// true if method has been chosen
  bool validMethod_;

  /// check that allowed range is logical
  void checkRange(const float xmin, const float xmax);

};

/// check that every TProf2D channel has mean, RMS within allowed range;
/// implementation: Giuseppe Della Ricca
class ContentsProf2DWithinRangeROOT : public SimpleTest<TProfile2D>
{
 public:

  ContentsProf2DWithinRangeROOT(void) : SimpleTest<TProfile2D>(){
    checkMean_ = checkRMS_ = validMethod_ = false;
    minMean_ = maxMean_ = minRMS_ = maxRMS_ = 0.0;
    checkMeanTolerance_ = false;
    toleranceMean_ = -1.0;
  }
  virtual ~ContentsProf2DWithinRangeROOT(void){}

  /// set expected value for mean
  void setMeanRange(float xmin, float xmax){
    checkRange(xmin, xmax);
    minMean_ = xmin;
    maxMean_ = xmax;
    checkMean_ = true;
  }

  /// set expected value for mean
  void setRMSRange(float xmin, float xmax){
    checkRange(xmin, xmax);
    minRMS_ = xmin;
    maxRMS_ = xmax;
    checkRMS_ = true;
  }

  /// set (fractional) tolerance for mean
  void setMeanTolerance(float frac_tolerance) {
    if ( frac_tolerance >= 0.0 ) {
      toleranceMean_ = frac_tolerance;
      checkMeanTolerance_ = true;
    }
  }

  /// run the test
  float runTest(const TProfile2D * const h);

  ///get  algorithm name
  static std::string getAlgoName(void){
    return "ContentsWithinExpectedProf2D";
  }

  /// true if test cannot run
  bool isInvalid(void){
    return !validMethod_;
  }

 protected:

  /// if true, check the mean value
  bool checkMean_;

  /// if true, check the RMS value
  bool checkRMS_;

  /// if true, check mean tolerance
  bool checkMeanTolerance_;

  /// fractional tolerance on mean (use only if checkMeanTolerance_ = true)
  float toleranceMean_;

  /// allowed range for mean (use only if checkMean_ = true)
  float minMean_, maxMean_;

  /// allowed range for mean (use only if checkRMS_ = true)
  float minRMS_, maxRMS_;

  /// true if method has been chosen
  bool validMethod_;

  /// check that allowed range is logical
  void checkRange(const float xmin, const float xmax);

};

#endif
