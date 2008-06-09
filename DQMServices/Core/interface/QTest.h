#ifndef DQMSERVICES_CORE_Q_CRITERION_H
# define DQMSERVICES_CORE_Q_CRITERION_H

# include "DQMServices/Core/interface/MonitorElement.h"
# include "TProfile2D.h"
# include "TProfile.h"
# include "TH2F.h"
# include "TH1F.h"
# include <sstream>
# include <string>
# include <map>

#include "DQMServices/Core/interface/DQMStore.h"


using namespace std;
//class MonitorElement;
class Comp2RefChi2;			typedef Comp2RefChi2 Comp2RefChi2ROOT;
class Comp2RefKolmogorov;		typedef Comp2RefKolmogorov Comp2RefKolmogorovROOT;

class Comp2RefEqualH;			typedef Comp2RefEqualH Comp2RefEqualHROOT;
//class Comp2RefEqualString;		typedef Comp2RefEqualString Comp2RefEqualStringROOT;
//class Comp2RefEqualInt;			typedef Comp2RefEqualInt Comp2RefEqualIntROOT;
//class Comp2RefEqualFloat;		typedef Comp2RefEqualFloat Comp2RefEqualFloatROOT;

class ContentsXRange;			typedef ContentsXRange ContentsXRangeROOT;
//class ContentsXRangeAS;                   typedef ContentsXRangeAS ContentsXRangeASROOT;
class ContentsYRange;			typedef ContentsYRange ContentsYRangeROOT;
//class ContentsYRangeAS;			typedef ContentsYRangeAS ContentsYRangeASROOT;
class NoisyChannel;			typedef NoisyChannel NoisyChannelROOT;
class DeadChannel;			typedef DeadChannel DeadChannelROOT;

class ContentsWithinExpected;		typedef ContentsWithinExpected ContentsWithinExpectedROOT;
//class ContentsWithinExpectedAS;		typedef ContentsWithinExpectedAS ContentsWithinExpectedASROOT;
//class ContentsWithinExpected;		typedef ContentsWithinExpected ContentsWithinExpectedROOT;
//class ContentsProfWithinRange;		typedef ContentsProfWithinRange ContentsProfWithinRangeROOT;
//class ContentsProf2DWithinRange;	typedef ContentsProf2DWithinRange ContentsProf2DWithinRangeROOT;

class MeanWithinExpected;		typedef MeanWithinExpected MeanWithinExpectedROOT;
//class MostProbableLandau;		typedef MostProbableLandau MostProbableLandauROOT;

class AllContentWithinFixedRange;	typedef AllContentWithinFixedRange RuleAllContentWithinFixedRange; typedef AllContentWithinFixedRange AllContentWithinFixedRangeROOT;
class AllContentWithinFloatingRange;	typedef AllContentWithinFloatingRange RuleAllContentWithinFloatingRange;	typedef AllContentWithinFloatingRange AllContentWithinFloatingRangeROOT;

class FlatOccupancy1d;			typedef FlatOccupancy1d RuleFlatOccupancy1d; typedef FlatOccupancy1d FlatOccupancy1dROOT;
class FixedFlatOccupancy1d;		typedef FixedFlatOccupancy1d RuleFixedFlatOccupancy1d; typedef FixedFlatOccupancy1d FixedFlatOccupancy1dROOT;
class CSC01;				typedef CSC01 RuleCSC01; typedef CSC01 CSC01ROOT;

class AllContentAlongDiagonal;		typedef AllContentAlongDiagonal RuleAllContentAlongDiagonal; typedef AllContentAlongDiagonal AllContentAlongDiagonalROOT;

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

class QCriterion
{
  /// (class should be created by DQMStore class)

public:

  /// true if QCriterion has been modified since last time it ran
  bool wasModified(void) const          { return wasModified_; }
  /// get test status (see Core/interface/DQMDefinitions.h)
  int getStatus(void) const             { return status_; }
  /// get message attached to test
  std::string getMessage(void) const    { return message_; }
  /// get name of quality test
  std::string getName(void) const       { return qtname_; }
  /// get algorithm name
  std::string algoName(void) const      { return algoName_; }
  /// set probability limit for test warning (default: 90%)
  void setWarningProb(float prob)       { if (validProb(prob)) warningProb_ = prob; }
  /// set probability limit for test error (default: 50%)
  void setErrorProb(float prob)         { if (validProb(prob)) errorProb_ = prob; }
  /// get vector of channels that failed test
  /// (not relevant for all quality tests!)
  virtual std::vector<DQMChannel> getBadChannels(void) const
                                        { return std::vector<DQMChannel>(); }

protected:
  QCriterion(std::string qtname)        { qtname_ = qtname; init(); }
  /// initialize values
  void init(void);

  virtual ~QCriterion(void)             {}

  virtual float runTest(const MonitorElement *me);
  /// set algorithm name
  void setAlgoName(std::string name)    { algoName_ = name; }

  float runTest(const MonitorElement *me, QReport &qr, DQMNet::QValue &qv)   {
      assert(qr.qcriterion_ == this);
      assert(qv.qtname == qtname_);
      //this runTest goes to SimpleTest
      prob_ = runTest(me);
      if (! validProb(prob_)) setInvalid();
      else if (prob_ < errorProb_) status_ = dqm::qstatus::ERROR;
      else if (prob_ < warningProb_) status_ = dqm::qstatus::WARNING;
      else status_ = dqm::qstatus::STATUS_OK;
      setMessage();
     
/* // debug output
      cout << " Message:    " << message_ << endl;
      cout << " Name = " << qtname_ << 
              " Algorithm = " << algoName_ << 
              "  Prob = " << prob_ << 
	      "  Status = " << status_ << endl;
*/

      qv.code = status_ ;
      qv.message = message_;
      qv.qtname = qtname_ ;
      qv.algorithm = algoName_;
      qr.badChannels_ = getBadChannels();

      return prob_;
    }

  /// call method when something in the algorithm changes
  void update(void)                 { wasModified_ = true; }
  /// true if probability value is valid
  bool validProb(float prob) const  { return prob >= 0 && prob <= 1; }
  /// set status & message for disabled tests
  void setDisabled(void);
  /// set status & message for invalid tests
  void setInvalid(void);
  /// set message after test has run
  virtual void setMessage(void) = 0;

  bool enabled_;  /// if true will run test
  int status_;  /// quality test status (see Core/interface/QTestStatus.h)
  std::string message_;  /// message attached to test
  std::string qtname_;  /// name of quality test
  bool wasModified_;  /// flag for indicating algorithm modifications since last time it ran
  std::string algoName_;  /// name of algorithm
  float warningProb_, errorProb_;  /// probability limits for warnings, errors
  /// test result [0, 1] ;
  /// (a) for comparison to reference :
  /// probability that histogram is consistent w/ reference
  /// (b) for "contents within range":
  /// fraction of entries that fall within allowed range
  float prob_;

private:
  /// default "probability" values for setting warnings & errors when running tests
  static const float WARNING_PROB_THRESHOLD;
  static const float ERROR_PROB_THRESHOLD;

  /// for creating and deleting class instances
  friend class DQMStore;
  /// for running the test
  friend class MonitorElement;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class SimpleTest : public QCriterion
{
 public:
  SimpleTest(const std::string &name, bool keepBadChannels = false) : QCriterion(name),
  minEntries_ (0),
  keepBadChannels_ (keepBadChannels)
  {}

  /// set minimum # of entries needed
  void setMinimumEntries(unsigned n)
  { minEntries_ = n; this->update(); }

  /// get vector of channels that failed test (not always relevant!)
  virtual std::vector<DQMChannel> getBadChannels(void) const
  { return keepBadChannels_ ? badChannels_ : QCriterion::getBadChannels(); }


 protected:

  /// set status & message after test has run

  virtual void setMessage(void) {
      std::ostringstream message;
      message << " Test " << this->qtname_ << " (" << this->algoName_
	      << "): prob = " << this->prob_;
      this->message_ = message.str();
    }

  unsigned minEntries_;  //< minimum # of entries needed
  std::vector<DQMChannel> badChannels_;
  bool keepBadChannels_;
 };


//===============================================================//
//========= Classes for particular QUALITY TESTS ================//
//===============================================================//

//===================== Comp2RefEqualH ===================//
//== Algorithm for comparing equality of histograms ==//
class Comp2RefEqualH : public SimpleTest
{
public:
  Comp2RefEqualH(const std::string &name) : SimpleTest(name)
  { setAlgoName( getAlgoName() ); }

  static std::string getAlgoName(void)
  { return "Comp2RefEqualH"; }

public:

  float runTest(const MonitorElement*me);

protected:
    TH1*h    ; //define test histogram
    TH1*ref_ ; // define ref histogram
  /// # of bins for test & reference histograms
  Int_t ncx1; Int_t ncx2;
};

//===================== Comp2RefChi2 ===================//
// comparison to reference using the  chi^2 algorithm
class Comp2RefChi2 : public SimpleTest
{
public:
   Comp2RefChi2(const std::string &name) :SimpleTest(name)
   { setAlgoName(getAlgoName()); }

   float runTest(const MonitorElement*me);
   static std::string getAlgoName(void)
   { return "Comp2RefChi2"; }
  
protected:

  void setMessage(void) {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): chi2/Ndof = " << chi2_ << "/" << Ndof_
	      << " prob = " << prob_
	      << ", minimum needed statistics = " << minEntries_
	      << " warning threshold = " << this->warningProb_
	      << " error threshold = " << this->errorProb_;
      message_ = message.str();
    }

  TH1*h    ; //define test histogram
  TH1*ref_ ; // define ref histogram

  // # of degrees of freedom and chi^2 for test
  int Ndof_; float chi2_;
  // # of bins for test & reference histogram
  Int_t ncx1; Int_t ncx2;
};

//===================== Comp2RefKolmogorov ===================//
/// Comparison to reference using the  Kolmogorov algorithm
class Comp2RefKolmogorov : public SimpleTest
{
public:
  Comp2RefKolmogorov(const std::string &name) : SimpleTest(name)
  { setAlgoName(getAlgoName()); }

  float runTest(const MonitorElement *me);

  static std::string getAlgoName(void)
  { return "Comp2RefKolmogorov"; }

protected:
  
   TH1*h    ; //define test histogram
   TH1*ref_ ; // define ref histogram

  /// # of bins for test & reference histograms
  Int_t ncx1; Int_t ncx2;
  static const Double_t difprec;
};





//==================== ContentsXRange =========================//
//== Check that histogram contents are between [Xmin, Xmax] ==//

class ContentsXRange : public SimpleTest
{
public:
  ContentsXRange(const std::string &name) : SimpleTest(name)
  {
      rangeInitialized_ = false;
      setAlgoName(getAlgoName());
  }

  /// set allowed range in X-axis (default values: histogram's X-range)
  virtual void setAllowedXRange(float xmin, float xmax)
  { xmin_ = xmin; xmax_ = xmax; rangeInitialized_ = true; }

  float runTest(const MonitorElement *me) ;

  static std::string getAlgoName(void)
  { return "ContentsXRange"; }

protected: 
  void setMessage(void) {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << algoName_
            << "): Entry fraction within X range = " << prob_;
    message_ = message.str();
    }

  /// allowed range in X-axis
  float xmin_;float xmax_;
  /// init-flag for xmin_, xmax_
  bool rangeInitialized_;
};

//==================== ContentsXRange =========================//
//== Check that histogram contents are between [Xmin, Xmax] ==//

class ContentsXRangeAS : public SimpleTest
{
public:
  ContentsXRangeAS(const std::string &name) : SimpleTest(name)
  {
      rangeInitialized_ = false;
      setAlgoName(getAlgoName());
  }

  /// set allowed range in X-axis (default values: histogram's X-range)
  virtual void setAllowedXRange(float xmin, float xmax)
  { xmin_ = xmin; xmax_ = xmax; rangeInitialized_ = true; }

  float runTest(const MonitorElement *me) ;

  static std::string getAlgoName(void)
  { return "ContentsXRangeAS"; }

protected:
  void setMessage(void) {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << algoName_
            << "): Entry fraction within X range = " << prob_;
    message_ = message.str();
    }

  /// allowed range in X-axis
  float xmin_;float xmax_;
  /// init-flag for xmin_, xmax_
  bool rangeInitialized_;
};


//==================== ContentsYRange =========================//
//== Check that histogram contents are between [Ymin, Ymax] ==//
/// (class also used by DeadChannel algorithm)
class ContentsYRange : public SimpleTest
{
public:
  ContentsYRange(const std::string &name) : SimpleTest(name,true)
  {
   rangeInitialized_ = false;
   deadChanAlgo_ = false;
   setAlgoName(getAlgoName());
  }

  float runTest(const MonitorElement *me);

  static std::string getAlgoName(void)
  { return "ContentsYRange"; }

  /// set allowed range in Y-axis (default values: histogram's FULL Y-range)
  virtual void setAllowedYRange(float ymin, float ymax)
  { ymin_ = ymin; ymax_ = ymax; rangeInitialized_ = true; }


protected:

  void setMessage(void) {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Bin fraction within Y range = " << prob_;
      message_ = message.str();
    }

  /// allowed range in Y-axis
  float ymin_; float ymax_;
  /// to be used to run derived-class algorithm
  bool deadChanAlgo_;
  /// init-flag for ymin_, ymax_
  bool rangeInitialized_;
};

//==================== ContentsYRangeAS =========================//
//== Check that histogram contents are between [Ymin, Ymax] ==//
class ContentsYRangeAS : public SimpleTest
{
public:
  ContentsYRangeAS(const std::string &name) : SimpleTest(name,true)
  {
   rangeInitialized_ = false;
   deadChanAlgo_ = false;
   setAlgoName(getAlgoName());
  }

  float runTest(const MonitorElement *me);

  static std::string getAlgoName(void)
  { return "ContentsYRangeAS"; }

  /// set allowed range in Y-axis (default values: histogram's FULL Y-range)
  virtual void setAllowedYRange(float ymin, float ymax)
  { ymin_ = ymin; ymax_ = ymax; rangeInitialized_ = true; }


protected:

  void setMessage(void) {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Bin fraction within Y range = " << prob_;
      message_ = message.str();
    }

  /// allowed range in Y-axis
  float ymin_; float ymax_;
  /// to be used to run derived-class algorithm
  bool deadChanAlgo_;
  /// init-flag for ymin_, ymax_
  bool rangeInitialized_;
};


//==================== NoisyChannel =========================//
/// Check if any channels are noisy compared to neighboring ones.
class NoisyChannel : public SimpleTest
{
public:
  NoisyChannel(const std::string &name) : SimpleTest(name,true)
  {
   rangeInitialized_ = false;
   numNeighbors_ = 1;
   setAlgoName(getAlgoName());
  }

   /// run the test (result: fraction of channels not appearing noisy or "hot")
  float runTest(const MonitorElement*me);

  static std::string getAlgoName(void)
  { return "NoisyChannel"; }

  /// set # of neighboring channels for calculating average to be used
  /// for comparison with channel under consideration;
  /// use 1 for considering bin+1 and bin-1 (default),
  /// use 2 for considering bin+1,bin-1, bin+2,bin-2, etc;
  /// Will use rollover when bin+i or bin-i is beyond histogram limits (e.g.
  /// for histogram with N bins, bin N+1 corresponds to bin 1,
  /// and bin -1 corresponds to bin N)
  void setNumNeighbors(unsigned n) { if (n > 0) numNeighbors_ = n; }

  /// set (percentage) tolerance for considering a channel noisy;
  /// eg. if tolerance = 20%, a channel will be noisy
  /// if (contents-average)/|average| > 20%; average is calculated from
  /// neighboring channels (also see method setNumNeighbors)
  void setTolerance(float percentage)
    {
      if (percentage >=0)
      {
	tolerance_ = percentage;
	rangeInitialized_ = true;
      }
    }


protected:
 
   void setMessage(void) {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Fraction of non-noisy channels = " << prob_;
      message_ = message.str();
    }

    TH1F*h    ; //define test histogram

  /// get average for bin under consideration
  /// (see description of method setNumNeighbors)
  Double_t getAverage(int bin, const TH1F *h) const;

  float tolerance_;        /*< tolerance for considering a channel noisy */
  unsigned numNeighbors_;  /*< # of neighboring channels for calculating average to be used
			     for comparison with channel under consideration */
  bool rangeInitialized_;  /*< init-flag for tolerance */
};

//----------------------------------------------------------------------------//
//============================== DeadChannel =================================//
//----------------------------------------------------------------------------//

/// the ContentsYRange algorithm w/o a check for Ymax and excluding Ymin
class DeadChannel : public ContentsYRange
{
public:
  DeadChannel(const std::string &name) : ContentsYRange(name)
    {
      setAllowedYRange(0, 0); /// ymax value is ignored
      deadChanAlgo_ = true;
      setAlgoName(getAlgoName());
    }

  /// set Ymin (inclusive) threshold for "dead" channel (default: 0)
  void setThreshold(float ymin)
  { setAllowedYRange(ymin, 0); } /// ymax value is ignored

  static std::string getAlgoName(void)
 { return "DeadChannel"; }

protected:
  void setMessage(void) {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Alive channel fraction = " << prob_;
      message_ = message.str();
    }
};


//==================== ContentsWithinExpectedAS  =========================//
// Check that every TH2F channels are within range
class ContentsWithinExpectedAS : public SimpleTest
{
public:
  ContentsWithinExpectedAS(const std::string &name) : SimpleTest(name,true)
    {
      rangeInitialized_ = false;
      minCont_ = maxCont_ = 0.0;
      setAlgoName(getAlgoName());
    }

  float runTest(const MonitorElement *me);

  static std::string getAlgoName(void)
  { return "ContentsWithinExpectedAS"; }

  /// set expected value for contents
  void setContentsRange(float xmin, float xmax)
    {
      minCont_ = xmin;
      maxCont_ = xmax;
      rangeInitialized_ = true;
    }

protected:

   bool isInvalid(const TH2F *h)
    { return false; } // any scenarios for invalid test?

  TH1*h    ; //define test histogram

  void setMessage(void) {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Entry fraction within range = " << prob_;
      message_ = message.str();
    }

  float minCont_, maxCont_; //< allowed range 
  bool rangeInitialized_;
};

//==================== ContentsWithinExpected  =========================//
// Check that every TH2F channel has mean, RMS within allowed range.
class ContentsWithinExpected : public SimpleTest
{
public:
  ContentsWithinExpected(const std::string &name) : SimpleTest(name,true)
    {
      checkMean_ = checkRMS_ = validMethod_ = false;
      minMean_ = maxMean_ = minRMS_ = maxRMS_ = 0.0;
      checkMeanTolerance_ = false;
      toleranceMean_ = -1.0;
      setAlgoName(getAlgoName());
    }

  float runTest(const MonitorElement *me);

  static std::string getAlgoName(void)
  { return "ContentsWithinExpected"; }

  /// set expected value for mean
  void setMeanRange(float xmin, float xmax)
    {
      checkRange(xmin, xmax);
      minMean_ = xmin;
      maxMean_ = xmax;
      checkMean_ = true;
    }

  /// set expected value for mean
  void setRMSRange(float xmin, float xmax)
    {
      checkRange(xmin, xmax);
      minRMS_ = xmin;
      maxRMS_ = xmax;
      checkRMS_ = true;
    }

  /// set (fractional) tolerance for mean
  void setMeanTolerance(float fracTolerance)
    {
      if (fracTolerance >= 0.0)
      {
	toleranceMean_ = fracTolerance;
	checkMeanTolerance_ = true;
      }
    }



protected:


  TH1*h    ; //define test histogram

  void setMessage(void) {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Entry fraction within range = " << prob_;
      message_ = message.str();
    }

  /// check that allowed range is logical
  void checkRange(const float xmin, const float xmax);

  bool checkMean_;	    //< if true, check the mean value
  bool checkRMS_;           //< if true, check the RMS value
  bool checkMeanTolerance_; //< if true, check mean tolerance
  float toleranceMean_;     //< fractional tolerance on mean (use only if checkMeanTolerance_ = true)
  float minMean_, maxMean_; //< allowed range for mean (use only if checkMean_ = true)
  float minRMS_, maxRMS_;   //< allowed range for mean (use only if checkRMS_ = true)
  bool validMethod_;        //< true if method has been chosen
};

//==================== MeanWithinExpected  =========================//
/// Algorithm for testing if histogram's mean value is near expected value.
class MeanWithinExpected : public SimpleTest
{
public:
  MeanWithinExpected(const std::string &name) : SimpleTest(name)
    {
      validMethod_ = validExpMean_ = false;
      setAlgoName(getAlgoName());
    }

  static std::string getAlgoName(void)
  { return "MeanWithinExpected"; }

  /// set expected value for mean
  void setExpectedMean(float expMean)
  {
   expMean_ = expMean;
   validExpMean_ = true;
  }

  void useRMS(void)
  {
   useRMS_ = true;
   useSigma_ = useRange_ = false;
   validMethod_ = true;
  }

  void useSigma(float expectedSigma)
  {
   useSigma_ = true;
   useRMS_ = useRange_ = false;
   sigma_ = expectedSigma;
   checkSigma();
  }

  void useRange(float xmin, float xmax)
  {
    useRange_ = true;
    useSigma_ = useRMS_ = false;
    xmin_ = xmin; xmax_ = xmax;
    checkRange();
   }


  
  /** run the test;
      (a) if useRange is called: 1 if mean within allowed range, 0 otherwise

      (b) is useRMS or useSigma is called: result is the probability
      Prob(chi^2, ndof=1) that the mean of histogram will be deviated by more than
      +/- delta from <expected_mean>, where delta = mean - <expected_mean>, and
      chi^2 = (delta/sigma)^2. sigma is the RMS of the histogram ("useRMS") or
      <expected_sigma> ("useSigma")
      e.g. for delta = 1, Prob = 31.7%
      for delta = 2, Prob = 4.55%

      (returns result in [0, 1] or <0 for failure) */
  float runTest(const MonitorElement*me);

protected:
  bool isInvalid(void);

  void setMessage(void) {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_ << "): ";
      if(useRange_)
      {
	message << "Mean within allowed range? ";
	if(prob_)
	  message << "Yes";
	else
	  message << "No";
      }
      else
	message << "prob = " << prob_;

      message_ = message.str();
    }

  /// check that exp_sigma_ is non-zero
  void checkSigma(void);

  /// check that allowed range is logical
  void checkRange(void);

  /// test for useRange_ = true case
  float doRangeTest(const TH1F *h);

  /// test assuming mean value is quantity with gaussian errors
  float doGaussTest(const TH1F *h, float sigma);


  TH1F*h;              //define Test histo
  bool useRMS_;       //< if true, will use RMS of distribution
  bool useSigma_;     //< if true, will use expected_sigma
  bool useRange_;     //< if true, will use allowed range
  float sigma_;       //< sigma to be used in probability calculation (use only if useSigma_ = true)
  float expMean_;     //< expected mean value (used only if useSigma_ = true or useRMS_ = true)
  float xmin_, xmax_; //< allowed range for mean (use only if useRange_ = true)
  bool validMethod_;  //< true if method has been chosen
  bool validExpMean_; //< true if expected mean has been chosen

};


// //==================== MostProbableBase   =========================//
// // QTest that should test Most Probable value for some Expected number
// 
// namespace edm {
//   namespace qtests {
//     namespace fits {
//       // Convert Significance into Probability value.
//       double erfc( const double &rdX);
//     }
//   }
// }
// 
// 
// /**
//  * @brief
//  *   Base for all MostProbables Children classes. Thus each child
//  *   implementation will concentrate on fit itself.
//  */
// class MostProbableBase : public SimpleTest
// {
// public:
//   MostProbableBase(const std::string &name);
// 
//   // Set/Get local variables methods
//   inline void   setMostProbable(double rdMP) { dMostProbable_ = rdMP;}
//   inline double getMostProbable(void) const  { return dMostProbable_;}
// 
//   inline void   setSigma(double rdSIGMA)     { dSigma_ = rdSIGMA; }
//   inline double getSigma(void) const         { return dSigma_; }
// 
//   inline void   setXMin(double rdMIN)        { dXMin_ = rdMIN; }
//   inline double getXMin(void) const          { return dXMin_;  }
// 
//   inline void   setXMax(double rdMAX)        { dXMax_ = rdMAX; }
//   inline double getXMax(void) const          { return dXMax_;  }
// 
//   /**
//    * @brief
//    *   Actual Run Test method. Should return: [0, 1] or <0 for failure.
//    *   [Note: See SimpleTest<class T> template for details]
//    *
//    * @param poPLOT  Plot for Which QTest to be run
//    *
//    * @return
//    *   -1      On Error
//    *   [0, 1]  Measurement of how Fit value is close to Expected one
//    */
//   float runTest(const MonitorElement*me);
// 
// protected:
//   /**
//    * @brief
//    *   Each Child should implement fit method which responsibility is to
//    *   perform actual fit and compare mean value with some additional
//    *   Cuts if any needed. The reason this task is put into separate method
//    *   is that a priory it is unknown what distribution QTest is dealing with.
//    *   It might be simple Gauss, Landau or something more sophisticated.
//    *   Each Plot needs special treatment (fitting) and extraction of
//    *   parameters. Children know about that but not Parent class.
//    *
//    * @param poPLOT  Plot to be fitted
//    *
//    * @return
//    *   -1     On Error
//    *   [0,1]  Measurement of how close Fit Value is to Expected one
//    */
// 
//   TH1F *poPLOT; //dine Test histo
//   virtual float fit(TH1F *poPLOT) = 0;
// 
//   /**
//    * @brief
//    *   Child should check test if it is valid and return corresponding value
//    *   Next common tests are performed here:
//    *     1. min < max
//    *     2. MostProbable is in (min, max)
//    *     3. Sigma > 0
//    *
//    * @return
//    *   True   Invalid QTest
//    *   False  Otherwise
//    */
//   bool isInvalid(void);
// 
//   /**
//    * @brief
//    *   General function that compares MostProbable value gotten from Fit and
//    *   Expected one.
//    *
//    * @param rdMP_FIT     MostProbable value gotten from Fit
//    * @param rdSIGMA_FIT  Sigma value gotten from Fit
//    *
//    * @return
//    *   Probability of found Value that measures how close is gotten one to
//    *   expected
//    */
//   double compareMostProbables(const double &rdMP_FIT, const double &rdSIGMA_FIT) const;
// 
//   void setMessage(void) {
//       std::ostringstream message;
//       message << " Test " << qtname_ << " (" << algoName_
// 	      << "): Fraction of Most Probable value match = " << prob_;
//       message_ = message.str();
//     }
// 
// private:
//   // Most common Fit values
//   double dMostProbable_;
//   double dSigma_;
//   double dXMin_;
//   double dXMax_;
// };
// 
// /** MostProbable QTest for Landau distributions */
// class MostProbableLandau : public MostProbableBase
// {
// public:
//   MostProbableLandau(const std::string &name);
// 
//   static std::string getAlgoName()
//   { return "MostProbableLandau"; }
// 
//   // Set/Get local variables methods
//   void setNormalization(const double &rdNORMALIZATION)
//   { dNormalization_ = rdNORMALIZATION; }
// 
//   double getNormalization(void) const
//    { return dNormalization_; }
// 
// protected:
//   //
//   // @brief
//   //   Perform Actual Fit
//   //
//   // @param poPLOT  Plot to be fitted
//   //
//   // @return
//   //   -1     On Error
//   //   [0,1]  Measurement of how close Fit Value is to Expected one
//   //
//   virtual float fit(TH1F *poPLOT);
// 
// private:
//   double dNormalization_;
// };
// 
// 


//==================== AllContentWithinFixedRange   =========================//
class AllContentWithinFixedRange : public SimpleTest
{
public:
  AllContentWithinFixedRange(const std::string &name) : SimpleTest(name)
 { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
  { return "RuleAllContentWithinFixedRange"; }

  void set_x_min(double x)             { x_min  = x; }
  void set_x_max(double x)             { x_max  = x; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S)	       { S_fail = S; }
  void set_S_pass(double S)	       { S_pass = S; }
  double get_epsilon_obs(void) 	       { return epsilon_obs; }
  double get_S_fail_obs(void)  	       { return S_fail_obs;  }
  double get_S_pass_obs(void)  	       { return S_pass_obs;  }
  int get_result(void)		       { return result; }

  float runTest(const MonitorElement *me);

protected:
  TH1F *histogram ; //define Test histo
  double x_min, x_max;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};


//==================== AllContentWithinFloatingRange  =========================//
class AllContentWithinFloatingRange : public SimpleTest
{
public:
  AllContentWithinFloatingRange(const std::string &name) : SimpleTest(name)
    { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
    { return "RuleAllContentWithinFloatingRange"; }

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
};




//==================== FlatOccupancy1d   =========================//
#if 0 // FIXME: need to know what parameters to set before runTest!
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

  static std::string getAlgoName(void)
    { return "RuleFlatOccupancy1d"; }

  void set_ExclusionMask(double *mask) { ExclusionMask = mask; }
  void set_epsilon_min(double epsilon) { epsilon_min = epsilon; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S)            { S_fail = S; }
  void set_S_pass(double S)            { S_pass = S; }
  double get_FailedBins(void)          { return *FailedBins[2]; } // FIXME: WRONG! OFF BY ONE!?
  int get_result()                     { return result; }

  float runTest(const MonitorElement*me);

protected:
  TH1F *histogram; //define Test histogram
  double *ExclusionMask;
  double epsilon_min, epsilon_max;
  double S_fail, S_pass;
  double *FailedBins[2];
  int    Nbins;
  int    result;
};
#endif

//==================== FixedFlatOccupancy1d   =========================//
class FixedFlatOccupancy1d : public SimpleTest
{
public:
  FixedFlatOccupancy1d(const std::string &name) : SimpleTest(name)
    {
      Nbins = 0;
      FailedBins[0] = 0;
      FailedBins[1] = 0;
      setAlgoName(getAlgoName());
    }

  ~FixedFlatOccupancy1d(void)
    {
      if( Nbins > 0 )
      {
	delete [] FailedBins[0];
	delete [] FailedBins[1];
      }
    }

  static std::string getAlgoName(void)
    { return "RuleFixedFlatOccupancy1d"; }

  void set_Occupancy(double level)     { b = level; }
  void set_ExclusionMask(double *mask) { ExclusionMask = mask; }
  void set_epsilon_min(double epsilon) { epsilon_min = epsilon; }
  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S)            { S_fail = S; }
  void set_S_pass(double S)            { S_pass = S; }
  double get_FailedBins(void)          { return *FailedBins[2]; } // FIXME: WRONG! OFF BY ONE!?
  int get_result()                     { return result; }

  float runTest(const MonitorElement*me);

protected:
  TH1F *histogram;
  double b;
  double *ExclusionMask;
  double epsilon_min, epsilon_max;
  double S_fail, S_pass;
  double *FailedBins[2];
  int    Nbins;
  int    result;
};

//==================== CSC01   =========================//
class CSC01 : public SimpleTest
{
public:
  CSC01(const std::string &name) : SimpleTest(name)
   { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
   { return "RuleCSC01"; }

  void set_epsilon_max(double epsilon) { epsilon_max = epsilon; }
  void set_S_fail(double S)	       { S_fail = S; }
  void set_S_pass(double S)	       { S_pass = S; }
  double get_epsilon_obs() 	       { return epsilon_obs; }
  double get_S_fail_obs()  	       { return S_fail_obs;  }
  double get_S_pass_obs()  	       { return S_pass_obs;  }
  int get_result()		       { return result; }

  float runTest(const MonitorElement*me);

protected:
  TH1F *histogram;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};

//==================== AllContentAlongDiagonal   =========================//
#if 0 // FIXME: need to know what parameters to set before runTest!
class AllContentAlongDiagonal : public SimpleTest

public:
  AllContentAlongDiagonal(const std::string &name) : SimpleTest(name)
  { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
  { return "RuleAllContentAlongDiagonal"; }

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
  TH2F *histogram;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};
#endif



#endif // DQMSERVICES_CORE_Q_CRITERION_H
