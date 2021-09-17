#ifndef DQMSERVICES_CORE_Q_CRITERION_H
#define DQMSERVICES_CORE_Q_CRITERION_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "TProfile2D.h"
#include "TProfile.h"
#include "TH2F.h"
#include "TH1F.h"
#include <sstream>
#include <string>
#include <map>
#include <utility>

//#include "DQMServices/Core/interface/DQMStore.h"

using DQMChannel = MonitorElementData::QReport::DQMChannel;
using QReport = MonitorElementData::QReport;

/** Base class for quality tests run on Monitoring Elements;

    Currently supporting the following tests:
    - Comparison to reference (Chi2, Kolmogorov)
    - Contents within [Xmin, Xmax]
    - Contents within [Ymin, Ymax]
    - Identical contents
    - Mean value within expected value
    - Check for dead or noisy channels
    - Check that mean, RMS of bins are within allowed range
    (support for 2D histograms, 1D, 2D profiles)  */

class QCriterion {
  /// (class should be created by DQMStore class)

public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  /// get test status
  int getStatus() const { return status_; }
  /// get message attached to test
  std::string getMessage() const { return message_; }
  /// get name of quality test
  std::string getName() const { return qtname_; }
  /// get algorithm name
  std::string algoName() const { return algoName_; }
  /// set probability limit for warning and error (default: 90% and 50%)
  void setWarningProb(float prob) { warningProb_ = prob; }
  void setErrorProb(float prob) { errorProb_ = prob; }
  /// get vector of channels that failed test
  /// (not relevant for all quality tests!)
  virtual std::vector<DQMChannel> getBadChannels() const { return std::vector<DQMChannel>(); }

  QCriterion(std::string qtname) {
    qtname_ = std::move(qtname);
    init();
  }
  /// initialize values
  void init();

  virtual ~QCriterion() = default;

  /// default "probability" values for setting warnings & errors when running tests
  static const float WARNING_PROB_THRESHOLD;
  static const float ERROR_PROB_THRESHOLD;

  float runTest(const MonitorElement *me, QReport &qr, DQMNet::QValue &qv) {
    assert(qv.qtname == qtname_);

    prob_ = runTest(me);  // this runTest goes to SimpleTest derivates

    if (prob_ < errorProb_)
      status_ = dqm::qstatus::ERROR;
    else if (prob_ < warningProb_)
      status_ = dqm::qstatus::WARNING;
    else
      status_ = dqm::qstatus::STATUS_OK;

    setMessage();  // this goes to SimpleTest derivates

    if (verbose_ == 2)
      std::cout << " Message = " << message_ << std::endl;
    if (verbose_ == 2)
      std::cout << " Name = " << qtname_ << " / Algorithm = " << algoName_ << " / Status = " << status_
                << " / Prob = " << prob_ << std::endl;

    qv.code = status_;
    qv.message = message_;
    qv.qtname = qtname_;
    qv.algorithm = algoName_;
    qv.qtresult = prob_;
    qr.setBadChannels(getBadChannels());

    return prob_;
  }

protected:
  /// set algorithm name
  void setAlgoName(std::string name) { algoName_ = std::move(name); }

  virtual float runTest(const MonitorElement *me);

  /// set message after test has run
  virtual void setMessage() = 0;

  std::string qtname_;    /// name of quality test
  std::string algoName_;  /// name of algorithm
  float prob_;
  int status_;                     /// quality test status
  std::string message_;            /// message attached to test
  float warningProb_, errorProb_;  /// probability limits for warnings, errors
  void setVerbose(int verbose) { verbose_ = verbose; }
  int verbose_;

private:
  /// for running the test
  friend class dqm::legacy::MonitorElement;
  friend class dqm::impl::MonitorElement;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class SimpleTest : public QCriterion {
public:
  SimpleTest(const std::string &name, bool keepBadChannels = false)
      : QCriterion(name), minEntries_(0), keepBadChannels_(keepBadChannels) {}

  /// set minimum # of entries needed
  void setMinimumEntries(unsigned n) { minEntries_ = n; }
  /// get vector of channels that failed test (not always relevant!)
  std::vector<DQMChannel> getBadChannels() const override {
    return keepBadChannels_ ? badChannels_ : QCriterion::getBadChannels();
  }

protected:
  /// set status & message after test has run
  void setMessage() override { message_.clear(); }

  unsigned minEntries_;  //< minimum # of entries needed
  std::vector<DQMChannel> badChannels_;
  bool keepBadChannels_;
};

//===============================================================//
//========= Classes for particular QUALITY TESTS ================//
//===============================================================//

//==================== ContentsXRange =========================//
//== Check that histogram contents are between [Xmin, Xmax] ==//
class ContentsXRange : public SimpleTest {
public:
  ContentsXRange(const std::string &name) : SimpleTest(name) {
    rangeInitialized_ = false;
    setAlgoName(getAlgoName());
  }
  static std::string getAlgoName() { return "ContentsXRange"; }
  float runTest(const MonitorElement *me) override;

  /// set allowed range in X-axis (default values: histogram's X-range)
  virtual void setAllowedXRange(double xmin, double xmax) {
    xmin_ = xmin;
    xmax_ = xmax;
    rangeInitialized_ = true;
  }

protected:
  double xmin_;
  double xmax_;
  bool rangeInitialized_;
};

//==================== ContentsYRange =========================//
//== Check that histogram contents are between [Ymin, Ymax] ==//
class ContentsYRange : public SimpleTest {
public:
  ContentsYRange(const std::string &name) : SimpleTest(name, true) {
    rangeInitialized_ = false;
    setAlgoName(getAlgoName());
  }
  static std::string getAlgoName() { return "ContentsYRange"; }
  float runTest(const MonitorElement *me) override;

  void setUseEmptyBins(unsigned int useEmptyBins) { useEmptyBins_ = useEmptyBins; }
  virtual void setAllowedYRange(double ymin, double ymax) {
    ymin_ = ymin;
    ymax_ = ymax;
    rangeInitialized_ = true;
  }

protected:
  double ymin_;
  double ymax_;
  bool rangeInitialized_;
  unsigned int useEmptyBins_;
};

//============================== DeadChannel =================================//
/// test that histogram contents are above Ymin
class DeadChannel : public SimpleTest {
public:
  DeadChannel(const std::string &name) : SimpleTest(name, true) {
    rangeInitialized_ = false;
    setAlgoName(getAlgoName());
  }
  static std::string getAlgoName() { return "DeadChannel"; }
  float runTest(const MonitorElement *me) override;

  /// set Ymin (inclusive) threshold for "dead" channel (default: 0)
  void setThreshold(double ymin) {
    ymin_ = ymin;
    rangeInitialized_ = true;
  }  /// ymin - threshold

protected:
  double ymin_;
  bool rangeInitialized_;
};

//==================== NoisyChannel =========================//
/// Check if any channels are noisy compared to neighboring ones.
class NoisyChannel : public SimpleTest {
public:
  NoisyChannel(const std::string &name) : SimpleTest(name, true) {
    rangeInitialized_ = false;
    numNeighbors_ = 1;
    setAlgoName(getAlgoName());
  }
  static std::string getAlgoName() { return "NoisyChannel"; }
  float runTest(const MonitorElement *me) override;

  /// set # of neighboring channels for calculating average to be used
  /// for comparison with channel under consideration;
  /// use 1 for considering bin+1 and bin-1 (default),
  /// use 2 for considering bin+1,bin-1, bin+2,bin-2, etc;
  /// Will use rollover when bin+i or bin-i is beyond histogram limits (e.g.
  /// for histogram with N bins, bin N+1 corresponds to bin 1,
  /// and bin -1 corresponds to bin N)
  void setNumNeighbors(unsigned n) {
    if (n > 0)
      numNeighbors_ = n;
  }

  /// set (percentage) tolerance for considering a channel noisy;
  /// eg. if tolerance = 20%, a channel will be noisy
  /// if (contents-average)/|average| > 20%; average is calculated from
  /// neighboring channels (also see method setNumNeighbors)
  void setTolerance(float percentage) {
    if (percentage >= 0) {
      tolerance_ = percentage;
      rangeInitialized_ = true;
    }
  }

protected:
  /// get average for bin under consideration
  /// (see description of method setNumNeighbors)
  double getAverage(int bin, const TH1 *h) const;
  double getAverage2D(int binX, int binY, const TH2 *h) const;

  float tolerance_;       /*< tolerance for considering a channel noisy */
  unsigned numNeighbors_; /*< # of neighboring channels for calculating average to be used
			     for comparison with channel under consideration */
  bool rangeInitialized_; /*< init-flag for tolerance */
};

//===============ContentSigma (Emma Yeager and Chad Freer)=====================//
/// Check the sigma of each bin against the rest of the chamber by a factor of tolerance/
class ContentSigma : public SimpleTest {
public:
  ContentSigma(const std::string &name) : SimpleTest(name, true) {
    rangeInitialized_ = false;
    numXblocks_ = 1;
    numYblocks_ = 1;
    numNeighborsX_ = 1;
    numNeighborsY_ = 1;
    setAlgoName(getAlgoName());
  }
  static std::string getAlgoName() { return "ContentSigma"; }

  float runTest(const MonitorElement *me) override;
  /// set # of neighboring channels for calculating average to be used
  /// for comparison with channel under consideration;
  /// use 1 for considering bin+1 and bin-1 (default),
  /// use 2 for considering bin+1,bin-1, bin+2,bin-2, etc;
  /// Will use rollover when bin+i or bin-i is beyond histogram limits (e.g.
  /// for histogram with N bins, bin N+1 corresponds to bin 1,
  /// and bin -1 corresponds to bin N)
  void setNumXblocks(unsigned ncx) {
    if (ncx > 0)
      numXblocks_ = ncx;
  }
  void setNumYblocks(unsigned ncy) {
    if (ncy > 0)
      numYblocks_ = ncy;
  }
  void setNumNeighborsX(unsigned ncx) {
    if (ncx > 0)
      numNeighborsX_ = ncx;
  }
  void setNumNeighborsY(unsigned ncy) {
    if (ncy > 0)
      numNeighborsY_ = ncy;
  }

  /// set factor tolerance for considering a channel noisy or dead;
  /// eg. if tolerance = 1, channel will be noisy if (content - 1 x sigma) > chamber_avg
  /// or channel will be dead if (content - 1 x sigma) < chamber_avg
  void setToleranceNoisy(float factorNoisy) {
    if (factorNoisy >= 0) {
      toleranceNoisy_ = factorNoisy;
      rangeInitialized_ = true;
    }
  }
  void setToleranceDead(float factorDead) {
    if (factorDead >= 0) {
      toleranceDead_ = factorDead;
      rangeInitialized_ = true;
    }
  }
  void setNoisy(bool noisy) { noisy_ = noisy; }
  void setDead(bool dead) { dead_ = dead; }

  void setXMin(unsigned xMin) { xMin_ = xMin; }
  void setXMax(unsigned xMax) { xMax_ = xMax; }
  void setYMin(unsigned yMin) { yMin_ = yMin; }
  void setYMax(unsigned yMax) { yMax_ = yMax; }

protected:
  /// for each bin get sum of the surrounding neighbors
  // double getNeighborSum(int binX, int binY, unsigned Xblocks, unsigned Yblocks, unsigned neighborsX, unsigned neighborsY, const TH1 *h) const;
  double getNeighborSum(unsigned groupx,
                        unsigned groupy,
                        unsigned Xblocks,
                        unsigned Yblocks,
                        unsigned neighborsX,
                        unsigned neighborsY,
                        const TH1 *h) const;
  double getNeighborSigma(double average,
                          unsigned groupx,
                          unsigned groupy,
                          unsigned Xblocks,
                          unsigned Yblocks,
                          unsigned neighborsX,
                          unsigned neighborsY,
                          const TH1 *h) const;

  bool noisy_;
  bool dead_;            /*< declare if test will be checking for noisy channels, dead channels, or both */
  float toleranceNoisy_; /*< factor by which sigma is compared for noisy channels */
  float toleranceDead_;  /*< factor by which sigma is compared for dead channels*/
  unsigned numXblocks_;
  unsigned numYblocks_;
  unsigned numNeighborsX_; /*< # of neighboring channels along x-axis for calculating average to be used
			     for comparison with channel under consideration */
  unsigned numNeighborsY_; /*< # of neighboring channels along y-axis for calculating average to be used
			     for comparison with channel under consideration */
  bool rangeInitialized_;  /*< init-flag for tolerance */
  unsigned xMin_;
  unsigned xMax_;
  unsigned yMin_;
  unsigned yMax_;
};

//==================== ContentsWithinExpected  =========================//
// Check that every TH2 channel has mean, RMS within allowed range.
class ContentsWithinExpected : public SimpleTest {
public:
  ContentsWithinExpected(const std::string &name) : SimpleTest(name, true) {
    checkMean_ = checkRMS_ = false;
    minMean_ = maxMean_ = minRMS_ = maxRMS_ = 0.0;
    checkMeanTolerance_ = false;
    toleranceMean_ = -1.0;
    setAlgoName(getAlgoName());
  }
  static std::string getAlgoName() { return "ContentsWithinExpected"; }
  float runTest(const MonitorElement *me) override;

  void setUseEmptyBins(unsigned int useEmptyBins) { useEmptyBins_ = useEmptyBins; }
  void setMeanRange(double xmin, double xmax);
  void setRMSRange(double xmin, double xmax);

  /// set (fractional) tolerance for mean
  void setMeanTolerance(float fracTolerance) {
    if (fracTolerance >= 0.0) {
      toleranceMean_ = fracTolerance;
      checkMeanTolerance_ = true;
    }
  }

protected:
  bool checkMean_;           //< if true, check the mean value
  bool checkRMS_;            //< if true, check the RMS value
  bool checkMeanTolerance_;  //< if true, check mean tolerance
  float toleranceMean_;      //< fractional tolerance on mean (use only if checkMeanTolerance_ = true)
  float minMean_, maxMean_;  //< allowed range for mean (use only if checkMean_ = true)
  float minRMS_, maxRMS_;    //< allowed range for mean (use only if checkRMS_ = true)
  unsigned int useEmptyBins_;
};

//==================== MeanWithinExpected  =========================//
/// Algorithm for testing if histogram's mean value is near expected value.
class MeanWithinExpected : public SimpleTest {
public:
  MeanWithinExpected(const std::string &name) : SimpleTest(name) { setAlgoName(getAlgoName()); }
  static std::string getAlgoName() { return "MeanWithinExpected"; }
  float runTest(const MonitorElement *me) override;

  void setExpectedMean(double mean) { expMean_ = mean; }
  void useRange(double xmin, double xmax);
  void useSigma(double expectedSigma);
  void useRMS();

protected:
  bool useRMS_;         //< if true, will use RMS of distribution
  bool useSigma_;       //< if true, will use expected_sigma
  bool useRange_;       //< if true, will use allowed range
  double sigma_;        //< sigma to be used in probability calculation (use only if useSigma_ = true)
  double expMean_;      //< expected mean value (used only if useSigma_ = true or useRMS_ = true)
  double xmin_, xmax_;  //< allowed range for mean (use only if useRange_ = true)
};

//==================== AllContentWithinFixedRange   =========================//
/*class AllContentWithinFixedRange : public SimpleTest
{
public:
  AllContentWithinFixedRange(const std::string &name) : SimpleTest(name)
  { 
    setAlgoName(getAlgoName()); 
  }
  static std::string getAlgoName(void) { return "RuleAllContentWithinFixedRange"; }
  float runTest(const MonitorElement *me);

  void set_x_min(double x)             { x_min  = x; }
  void set_x_max(double x)             { x_max  = x; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S)	       { S_fail = S; }
  void set_S_pass(double S)	       { S_pass = S; }
  double get_epsilon_obs(void) 	       { return epsilon_obs; }
  double get_S_fail_obs(void)  	       { return S_fail_obs;  }
  double get_S_pass_obs(void)  	       { return S_pass_obs;  }
  int get_result(void)		       { return result; }

protected:
  TH1F *histogram ; //define Test histo
  double x_min, x_max;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};
*/
//==================== AllContentWithinFloatingRange  =========================//
/*class AllContentWithinFloatingRange : public SimpleTest
{
public:
  AllContentWithinFloatingRange(const std::string &name) : SimpleTest(name)
  { 
    setAlgoName(getAlgoName()); 
  }
  static std::string getAlgoName(void) { return "RuleAllContentWithinFloatingRange"; }

  void set_Nrange(int N)               { Nrange = N; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S)	       { S_fail = S; }
  void set_S_pass(double S)	       { S_pass = S; }
  double get_epsilon_obs(void) 	       { return epsilon_obs; }
  double get_S_fail_obs(void)  	       { return S_fail_obs;  }
  double get_S_pass_obs(void)	       { return S_pass_obs;  }
  int get_result(void)		       { return result; }

  float runTest(const MonitorElement *me );

protected:
  TH1F *histogram ; //define Test histo
  int Nrange;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};*/

//==================== FlatOccupancy1d   =========================//
#if 0  // FIXME: need to know what parameters to set before runTest!
class FlatOccupancy1d : public SimpleTest
{
public:
  FlatOccupancy1d(const std::string &name) : SimpleTest(name)
  {
    Nbins = 0;
    FailedBins[0] = 0;
    FailedBins[1] = 0;
    setAlgoName(getAlgoName());
  }

  ~FlatOccupancy1d(void)
  {
    delete [] FailedBins[0];
    delete [] FailedBins[1];
  }

  static std::string getAlgoName(void) { return "RuleFlatOccupancy1d"; }

  void set_ExclusionMask(double *mask) { ExclusionMask = mask; }
  void set_epsilon_min(double epsilon) { epsilon_min = epsilon; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S)            { S_fail = S; }
  void set_S_pass(double S)            { S_pass = S; }
  double get_FailedBins(void)          { return *FailedBins[1]; } // FIXME: WRONG! OFF BY ONE!?
  int get_result()                     { return result; }

  float runTest(const MonitorElement*me);

protected:
  double *ExclusionMask;
  double epsilon_min, epsilon_max;
  double S_fail, S_pass;
  double *FailedBins[2];
  int    Nbins;
  int    result;
};
#endif

//==================== FixedFlatOccupancy1d   =========================//
class FixedFlatOccupancy1d : public SimpleTest {
public:
  FixedFlatOccupancy1d(const std::string &name) : SimpleTest(name) {
    Nbins = 0;
    FailedBins[0] = nullptr;
    FailedBins[1] = nullptr;
    setAlgoName(getAlgoName());
  }

  ~FixedFlatOccupancy1d() override {
    if (Nbins > 0) {
      delete[] FailedBins[0];
      delete[] FailedBins[1];
    }
  }

  static std::string getAlgoName() { return "RuleFixedFlatOccupancy1d"; }

  void set_Occupancy(double level) { b = level; }
  void set_ExclusionMask(double *mask) { ExclusionMask = mask; }
  void set_epsilon_min(double epsilon) { epsilon_min = epsilon; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S) { S_fail = S; }
  void set_S_pass(double S) { S_pass = S; }
  double get_FailedBins() { return *FailedBins[1]; }  // FIXME: WRONG! OFF BY ONE!?
  int get_result() { return result; }

  float runTest(const MonitorElement *me) override;

protected:
  double b;
  double *ExclusionMask;
  double epsilon_min, epsilon_max;
  double S_fail, S_pass;
  double *FailedBins[2];
  int Nbins;
  int result;
};

//==================== CSC01   =========================//
class CSC01 : public SimpleTest {
public:
  CSC01(const std::string &name) : SimpleTest(name) { setAlgoName(getAlgoName()); }
  static std::string getAlgoName() { return "RuleCSC01"; }

  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S) { S_fail = S; }
  void set_S_pass(double S) { S_pass = S; }
  double get_epsilon_obs() { return epsilon_obs; }
  double get_S_fail_obs() { return S_fail_obs; }
  double get_S_pass_obs() { return S_pass_obs; }
  int get_result() { return result; }

  float runTest(const MonitorElement *me) override;

protected:
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};

//======================== CompareToMedian ====================//
class CompareToMedian : public SimpleTest {
public:
  //Initialize for TProfile, colRings
  CompareToMedian(const std::string &name) : SimpleTest(name, true) {
    this->_min = 0.2;
    this->_max = 3.0;
    this->_emptyBins = 0;
    this->_maxMed = 10;
    this->_minMed = 0;
    this->nBins = 0;
    this->_statCut = 0;
    reset();
    setAlgoName(getAlgoName());
  };

  ~CompareToMedian() override = default;
  ;

  static std::string getAlgoName() { return "CompareToMedian"; }

  float runTest(const MonitorElement *me) override;
  void setMin(float min) { _min = min; };
  void setMax(float max) { _max = max; };
  void setEmptyBins(int eB) { eB > 0 ? _emptyBins = 1 : _emptyBins = 0; };
  void setMaxMedian(float max) { _maxMed = max; };
  void setMinMedian(float min) { _minMed = min; };
  void setStatCut(float cut) { _statCut = (cut > 0) ? cut : 0; };

protected:
  void setMessage() override {
    std::ostringstream message;
    message << "Test " << qtname_ << " (" << algoName_ << "): Entry fraction within range = " << prob_;
    message_ = message.str();
  }

private:
  float _min, _max;        //Test values
  int _emptyBins;          //use empty bins
  float _maxMed, _minMed;  //Global max for median&mean
  float _statCut;          //Minimal number of non zero entries needed for the quality test

  int nBinsX, nBinsY;  //Dimensions of hystogram

  int nBins;  //Number of (non empty) bins

  //Vector contain bin values
  std::vector<float> binValues;

  void reset() { binValues.clear(); };
};
//======================== CompareLastFilledBin ====================//
class CompareLastFilledBin : public SimpleTest {
public:
  //Initialize for TProfile, colRings
  CompareLastFilledBin(const std::string &name) : SimpleTest(name, true) {
    this->_min = 0.0;
    this->_max = 1.0;
    this->_average = 0.0;
    setAlgoName(getAlgoName());
  };

  ~CompareLastFilledBin() override = default;
  ;

  static std::string getAlgoName() { return "CompareLastFilledBin"; }

  float runTest(const MonitorElement *me) override;
  void setAverage(float average) { _average = average; };
  void setMin(float min) { _min = min; };
  void setMax(float max) { _max = max; };

protected:
  void setMessage() override {
    std::ostringstream message;
    message << "Test " << qtname_ << " (" << algoName_ << "): Last Bin filled with desired value = " << prob_;
    message_ = message.str();
  }

private:
  float _min, _max;  //Test values
  float _average;
};

//==================== AllContentAlongDiagonal   =========================//
#if 0  // FIXME: need to know what parameters to set before runTest!
class AllContentAlongDiagonal : public SimpleTest

public:
  AllContentAlongDiagonal(const std::string &name) : SimpleTest(name)
  { 
    setAlgoName(getAlgoName()); 
  }
  static std::string getAlgoName(void) { return "RuleAllContentAlongDiagonal"; }

  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S)	       { S_fail = S; }
  void set_S_pass(double S)	       { S_pass = S; } 
  double get_epsilon_obs() 	       { return epsilon_obs; }
  double get_S_fail_obs()  	       { return S_fail_obs;  }
  double get_S_pass_obs()  	       { return S_pass_obs;  }
  int get_result()		       { return result; }

  //public:
  //using SimpleTest::runTest;
  float runTest(const MonitorElement*me); 

protected:
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};
#endif
//==================== CheckVariance =========================//
//== Check the variance of a TProfile//
class CheckVariance : public SimpleTest {
public:
  CheckVariance(const std::string &name) : SimpleTest(name) { setAlgoName(getAlgoName()); }
  /// get algorithm name
  static std::string getAlgoName() { return "CheckVariance"; }
  float runTest(const MonitorElement *me) override;
};
#endif  // DQMSERVICES_CORE_Q_CRITERION_H
