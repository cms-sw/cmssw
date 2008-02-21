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

class MonitorElement;
class Comp2RefChi2;			typedef Comp2RefChi2 Comp2RefChi2ROOT;
class Comp2RefKolmogorov;		typedef Comp2RefKolmogorov Comp2RefKolmogorovROOT;
class Comp2RefEqualString;		typedef Comp2RefEqualString Comp2RefEqualStringROOT;
class Comp2RefEqualInt;			typedef Comp2RefEqualInt Comp2RefEqualIntROOT;
class Comp2RefEqualFloat;		typedef Comp2RefEqualFloat Comp2RefEqualFloatROOT;
class Comp2RefEqualH1;			typedef Comp2RefEqualH1 Comp2RefEqualH1ROOT;
class Comp2RefEqualH2;			typedef Comp2RefEqualH2 Comp2RefEqualH2ROOT;
class Comp2RefEqualH3;			typedef Comp2RefEqualH3 Comp2RefEqualH3ROOT;
class ContentsXRange;			typedef ContentsXRange ContentsXRangeROOT;
class ContentsYRange;			typedef ContentsYRange ContentsYRangeROOT;
class DeadChannel;			typedef DeadChannel DeadChannelROOT;
class NoisyChannel;			typedef NoisyChannel NoisyChannelROOT;
class ContentsTH2FWithinRange;		typedef ContentsTH2FWithinRange ContentsTH2FWithinRangeROOT;
class ContentsProfWithinRange;		typedef ContentsProfWithinRange ContentsProfWithinRangeROOT;
class ContentsProf2DWithinRange;	typedef ContentsProf2DWithinRange ContentsProf2DWithinRangeROOT;
class MeanWithinExpected;		typedef MeanWithinExpected MeanWithinExpectedROOT;
class MostProbableLandau;		typedef MostProbableLandau MostProbableLandauROOT;
class AllContentWithinFixedRange;	typedef AllContentWithinFixedRange RuleAllContentWithinFixedRange;		typedef AllContentWithinFixedRange AllContentWithinFixedRangeROOT;
class AllContentWithinFloatingRange;	typedef AllContentWithinFloatingRange RuleAllContentWithinFloatingRange;	typedef AllContentWithinFloatingRange AllContentWithinFloatingRangeROOT;
class FlatOccupancy1d;			typedef FlatOccupancy1d RuleFlatOccupancy1d;					typedef FlatOccupancy1d FlatOccupancy1dROOT;
class FixedFlatOccupancy1d;		typedef FixedFlatOccupancy1d RuleFixedFlatOccupancy1d;				typedef FixedFlatOccupancy1d FixedFlatOccupancy1dROOT;
class CSC01;				typedef CSC01 RuleCSC01;							typedef CSC01 CSC01ROOT;
class AllContentAlongDiagonal;		typedef AllContentAlongDiagonal RuleAllContentAlongDiagonal;			typedef AllContentAlongDiagonal AllContentAlongDiagonalROOT;

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
  /// (class should be created by DaqMonitorBEInterface class)

public:
  /// enable test
  void enable(void)
    { enabled_ = true; }

  /// disable test
  void disable(void)
    { enabled_ = false; }

  /// true if test is enabled
  bool isEnabled(void) const
    { return enabled_; }

  /// true if QCriterion has been modified since last time it ran
  bool wasModified(void) const
    { return wasModified_; }

  /// get test status (see Core/interface/DQMDefinitions.h)
  int getStatus(void) const
    { return status_; }

  /// get message attached to test
  std::string getMessage(void) const
    { return message_; }

  /// get name of quality test
  std::string getName(void) const
    { return qtname_; }

  /// get algorithm name
  std::string algoName(void) const
    { return algoName_; }

  /// set probability limit for test warning (default: 90%)
  void setWarningProb(float prob)
    { if (validProb(prob)) warningProb_ = prob; }

  /// set probability limit for test error (default: 50%)
  void setErrorProb(float prob)
    { if (validProb(prob)) errorProb_ = prob; }

  /// get vector of channels that failed test
  /// (not relevant for all quality tests!)
  virtual std::vector<DQMChannel> getBadChannels(void) const
    { return std::vector<DQMChannel>(); }

protected:
  QCriterion(std::string qtname)
    { qtname_ = qtname; init(); }

  virtual ~QCriterion(void)
    {}

  /// initialize values
  void init(void);

  /// set algorithm name
  void setAlgoName(std::string name)
    { algoName_ = name; }

  /// run test (result: [0, 1])
  virtual float runTest(const MonitorElement *me) = 0;
  float runTest(const MonitorElement *me, QReport &qr, DQMNet::QValue &qv)
    {
      assert(qr.qcriterion_ == this);
      assert(qv.qtname == qtname_);

      float prob = runTest(me);

      qv.code = status_;
      qv.message = message_;
      qv.qtname = qtname_;
      qv.algorithm = algoName_;
      qr.badChannels_ = getBadChannels();

      return prob;
    }

  /// call method when something in the algorithm changes
  void update(void)
    { wasModified_ = true; }

  /// make sure algorithm can run (false: should not run)
  bool check(const MonitorElement *me);

  /// true if MonitorElement does not have enough entries to run test
  virtual bool notEnoughStats(const MonitorElement *me) const = 0;

  /// true if probability value is valid
  bool validProb(float prob) const
    { return prob >= 0 && prob <= 1; }

  /// set status & message for disabled tests
  void setDisabled(void);

  /// set status & message for invalid tests
  void setInvalid(void);

  /// set status & message for tests w/o enough statistics
  void setNotEnoughStats(void);

  /// set status & message for succesfull tests
  void setOk(void);

  /// set status & message for tests w/ warnings
  void setWarning(void);

  /// set status & message for tests w/ errors
  void setError(void);

  /// set message after test has run
  virtual void setMessage(void) = 0;

  bool enabled_;  /// if true will run test
  int status_;  /// quality test status (see Core/interface/QTestStatus.h)
  std::string message_;  /// message attached to test
  std::string qtname_;  /// name of quality test
  bool wasModified_;  /// flag for indicating algorithm modifications since last time it ran
  std::string algoName_;  /// name of algorithm
  float warningProb_, errorProb_;  /// probability limits for warnings, errors
  /// test result [0, 1];
  /// (a) for comparison to reference:
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
template <class INTO> inline const INTO *
QTestValueOf(const MonitorElement *me)
{ return dynamic_cast<const INTO *>(me->getTH1()); }

template <> inline const int *
QTestValueOf(const MonitorElement *me)
{ return &me->getIntValue(); }

template <> inline const double *
QTestValueOf(const MonitorElement *me)
{ return &me->getFloatValue(); }

template <> inline const std::string *
QTestValueOf(const MonitorElement *me)
{ return &me->getStringValue(); }

template <class T> inline bool
QTestNotEnoughStats(const T *, unsigned)
{ return false; }

template <> inline bool
QTestNotEnoughStats(const TH1F *h, unsigned minEntries)
{ return h && h->GetEntries() < minEntries; }

template <> inline bool
QTestNotEnoughStats(const TH2F *h, unsigned minEntries)
{ return h && h->GetEntries() < minEntries; }

template <> inline bool
QTestNotEnoughStats(const TH3F *h, unsigned minEntries)
{ return h && h->GetEntries() < minEntries; }

template <> inline bool
QTestNotEnoughStats(const TProfile *h, unsigned minEntries)
{ return h && h->GetEntries() < minEntries; }

template <> inline bool
QTestNotEnoughStats(const TProfile2D *h, unsigned minEntries)
{ return h && h->GetEntries() < minEntries; }

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
template <class T>
class QCriterionBase : public QCriterion
{
public:
  QCriterionBase(const std::string &name)
    : QCriterion(name)
    {}

public:
  /// (public only so it can be tested... sigh)
  virtual float runTest(const T *t) = 0;

protected:
  /// get () object from MonitorElement <me>
  /// (will redefine for scalars)
  virtual const T *getObject(const MonitorElement *me) const
    { return QTestValueOf<T>(me); }

  /// run the test on MonitorElement <me> (result: [0, 1] or <0 for failure)
  virtual float runTest(const MonitorElement *me)
    {
      if (! check(me))
	return -1;
      prob_ = runTest(getObject(me));
      setStatusMessage();
      return prob_;
    }

  /// true if object <t> does not have enough statistics
  virtual bool notEnoughStats(const T *t) const = 0;

  /// true if MonitorElement <me> does not have enough statistics
  bool notEnoughStats(const MonitorElement *me) const
    { return notEnoughStats(getObject(me)); }

  /// set status & message after test has run
  void setStatusMessage(void)
    {
      if (! validProb(prob_))
	setInvalid();
      else if (prob_ < errorProb_)
	setError();
      else if (prob_ < warningProb_)
	setWarning();
      else
	setOk();
    }
};

/// Class T must be one of the usual histogram/profile objects: THX
/// for method notEnoughStats to be used...
template <class T>
class SimpleTest : public QCriterionBase<T>
{
public:
  SimpleTest(const std::string &name, bool keepBadChannels = false)
    : QCriterionBase<T>(name),
      minEntries_ (0),
      keepBadChannels_ (keepBadChannels)
    {}

  /// set minimum # of entries needed
  void setMinimumEntries(unsigned n)
    { minEntries_ = n; this->update(); }

  /// get vector of channels that failed test (not always relevant!)
  virtual std::vector<DQMChannel> getBadChannels(void) const
    { return keepBadChannels_ ? badChannels_ : QCriterionBase<T>::getBadChannels(); }

protected:
  /// true if histogram does not have enough statistics
  virtual bool notEnoughStats(const T *h) const
    { return QTestNotEnoughStats(h, minEntries_); }

  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << this->qtname_ << " (" << this->algoName_
	      << "): prob = " << this->prob_;
      this->message_ = message.str();
    }

  unsigned minEntries_;  //< minimum # of entries needed
  std::vector<DQMChannel> badChannels_;
  bool keepBadChannels_;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// Class T must be one of the usual histogram/profile objects: THX.
template <class T>
class Comp2RefBase : public SimpleTest<T>
{
public:
  Comp2RefBase(const std::string &name)
    : SimpleTest<T>(name),
      ref_ (0)
    {}

  /// set reference object
  void setReference(const T *r)
    { ref_ = r; this->update(); }

  /// set reference object
  void setReference(const MonitorElement *r)
    { ref_ = QTestValueOf<T>(r); this->update(); }

  /// true if reference object is null
  bool hasNullReference(void) const
    { return ref_ == 0; }

protected:
  /// reference object
  const T *ref_;
};

// comparison to reference using the  chi^2 algorithm
class Comp2RefChi2 : public Comp2RefBase<TH1F>
{
public:
  Comp2RefChi2(const std::string &name)
    : Comp2RefBase<TH1F>(name)
    { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
    { return "Comp2RefChi2"; }

public:
  using Comp2RefBase<TH1F>::runTest;
  virtual float runTest(const TH1F *t);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const TH1F *h);

  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): chi2/Ndof = " << chi2_ << "/" << Ndof_
	      << " prob = " << prob_
	      << ", minimum needed statistics = " << minEntries_
	      << " warning threshold = " << this->warningProb_
	      << " error threshold = " << this->errorProb_;
      message_ = message.str();
    }

protected:
  void resetResults(void);

  // # of degrees of freedom and chi^2 for test
  int Ndof_;
  float chi2_;

  // # of bins for test & reference histogram
  Int_t nbins1;
  Int_t nbins2;
};

/// Comparison to reference using the  Kolmogorov algorithm
class Comp2RefKolmogorov : public Comp2RefBase<TH1F>
{
public:
  Comp2RefKolmogorov(const std::string &name)
    : Comp2RefBase<TH1F>(name)
    { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
    { return "Comp2RefKolmogorov"; }

public:
  using Comp2RefBase<TH1F>::runTest;
  virtual float runTest(const TH1F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const TH1F *me);

  /// # of bins for test & reference histograms
  Int_t ncx1;
  Int_t ncx2;
  static const Double_t difprec;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// template class for strings, integers, floats
template <class T>
class Comp2RefEqualT : public Comp2RefBase<T>
{
public:
  Comp2RefEqualT(const std::string &name)
    : Comp2RefBase<T>(name)
    {}

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const T *h)
    { return !h || !this->ref_; }

  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << this->qtname_
	      << " (" << this->algoName_ << "): Identical contents? "
	      << (this->prob_ ? "Yes" : "No");
      this->message_ = message.str();
    }
};

/// Algorithm for comparing equality of strings.
class Comp2RefEqualString : public Comp2RefEqualT<std::string>
{
public:
  Comp2RefEqualString(const std::string &name)
    : Comp2RefEqualT<std::string>(name)
    { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
    { return "Comp2RefEqualString"; }

public:
  using Comp2RefEqualT<std::string>::runTest;
  virtual float runTest(const std::string *t);

protected:
  virtual bool notEnoughStats(const std::string *t) const
    { return false; } // statistics not an issue for strings
};

/// Algorithm for comparing equality of integers.
class Comp2RefEqualInt : public Comp2RefEqualT<int>
{
public:
  Comp2RefEqualInt(const std::string &name)
    : Comp2RefEqualT<int>(name)
    { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
    { return "Comp2RefEqualInt"; }

public:
  using Comp2RefEqualT<int>::runTest;
  virtual float runTest(const int *t);

protected:
  virtual bool notEnoughStats(const int *t) const
    { return false; } // statistics not an issue for ints
};

/// Algorithm for comparing equality of floats.
class Comp2RefEqualFloat : public Comp2RefEqualT<double>
{
public:
  Comp2RefEqualFloat(const std::string &name)
    : Comp2RefEqualT<double>(name)
    { setAlgoName(getAlgoName()); }

  static std::string getAlgoName(void)
    { return "Comp2RefEqualFloat"; }

public:
  using Comp2RefEqualT<double>::runTest;
  virtual float runTest(const double *t);

protected:
  virtual bool notEnoughStats(const double *t) const
    { return false; } // statistics not an issue for floats
};

/// Algorithm for comparing equality of 1D histograms.
class Comp2RefEqualH1 : public Comp2RefEqualT<TH1F>
{
public:
  Comp2RefEqualH1(const std::string &name)
    : Comp2RefEqualT<TH1F>(name)
    {
      setAlgoName(getAlgoName());
      keepBadChannels_ = true;
    }

  static std::string getAlgoName(void)
    { return "Comp2RefEqualH1"; }

public:
  using Comp2RefEqualT<TH1F>::runTest;
  virtual float runTest(const TH1F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const TH1F *h);

  /// # of bins for test & reference histograms
  Int_t ncx1;
  Int_t ncx2;
};

/// Algorithm for comparing equality of 2D histograms.
class Comp2RefEqualH2 : public Comp2RefEqualT<TH2F>
{
public:
  Comp2RefEqualH2(const std::string &name)
    : Comp2RefEqualT<TH2F>(name)
    {
      setAlgoName(getAlgoName());
      keepBadChannels_ = true;
    }

  static std::string getAlgoName(void)
    { return "Comp2RefEqualH2"; }

public:
  using Comp2RefEqualT<TH2F>::runTest;
  virtual float runTest(const TH2F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const TH2F *h);

  /// # of bins for test & reference histograms
  Int_t ncx1;
  Int_t ncx2;
  Int_t ncy1;
  Int_t ncy2;
};

class Comp2RefEqualH3 : public Comp2RefEqualT<TH3F>
{
public:
  Comp2RefEqualH3(const std::string &name)
    : Comp2RefEqualT<TH3F>(name)
    {
      setAlgoName(getAlgoName());
      keepBadChannels_ = true;
    }

  static std::string getAlgoName(void)
    { return "Comp2RefEqualH3"; }

public:
  using Comp2RefEqualT<TH3F>::runTest;
  virtual float runTest(const TH3F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const TH3F *h);

  /// # of bins for test & reference histograms
  Int_t ncx1;
  Int_t ncx2;
  Int_t ncy1;
  Int_t ncy2;
  Int_t ncz1;
  Int_t ncz2;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// Check that histogram contents are between [Xmin, Xmax]
class ContentsXRange : public SimpleTest<TH1F>
{
public:
  ContentsXRange(const std::string &name)
    : SimpleTest<TH1F>(name)
    {
      rangeInitialized_ = false;
      setAlgoName(getAlgoName());
    }

  /// set allowed range in X-axis (default values: histogram's X-range)
  virtual void setAllowedXRange(float xmin, float xmax)
    {
      xmin_ = xmin;
      xmax_ = xmax;
      rangeInitialized_ = true;
    }

  static std::string getAlgoName(void)
    { return "ContentsXRange"; }

public:
  using SimpleTest<TH1F>::runTest;
  /// run the test (result: fraction of entries [*not* bins!] within X-range)
  virtual float runTest(const TH1F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const TH1F *h)
    { return false; } // any scenarios for invalid test?

  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Entry fraction within X range = " << prob_;
      message_ = message.str();
    }

  /// allowed range in X-axis
  float xmin_;
  float xmax_;
  /// init-flag for xmin_, xmax_
  bool rangeInitialized_;
};

/// check that histogram contents are between [Ymin, Ymax]
/// (class also used by DeadChannel algorithm)
class ContentsYRange : public SimpleTest<TH1F>
{
public:
  ContentsYRange(const std::string &name)
    : SimpleTest<TH1F>(name, true)
    {
      rangeInitialized_ = false;
      deadChanAlgo_ = false;
      setAlgoName(getAlgoName());
    }

  static std::string getAlgoName(void)
    { return "ContentsYRange"; }

  /// set allowed range in Y-axis (default values: histogram's FULL Y-range)
  virtual void setAllowedYRange(float ymin, float ymax)
    { ymin_ = ymin; ymax_ = ymax; rangeInitialized_ = true; }

public:
  using SimpleTest<TH1F>::runTest;
  /// run the test (result: fraction of bins [*not* entries!] that passed test)
  virtual float runTest(const TH1F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const TH1F *h)
    { return false; } // any scenarios for invalid test?

  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Bin fraction within Y range = " << prob_;
      message_ = message.str();
    }

  /// allowed range in Y-axis
  float ymin_;
  float ymax_;
  /// to be used to run derived-class algorithm
  bool deadChanAlgo_;
  /// init-flag for ymin_, ymax_
  bool rangeInitialized_;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// the ContentsYRange algorithm w/o a check for Ymax and excluding Ymin
class DeadChannel : public ContentsYRange
{
public:
  DeadChannel(const std::string &name)
    : ContentsYRange(name)
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
  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Alive channel fraction = " << prob_;
      message_ = message.str();
    }
};

/// Check if any channels are noisy compared to neighboring ones.
class NoisyChannel : public SimpleTest<TH1F>
{
public:
  NoisyChannel(const std::string &name)
    : SimpleTest<TH1F>(name, true)
    {
      rangeInitialized_ = false;
      numNeighbors_ = 1;
      setAlgoName(getAlgoName());
    }

  static std::string getAlgoName(void)
    { return "NoisyChannel"; }

  /// set # of neighboring channels for calculating average to be used
  /// for comparison with channel under consideration;
  /// use 1 for considering bin+1 and bin-1 (default),
  /// use 2 for considering bin+1,bin-1, bin+2,bin-2, etc;
  /// Will use rollover when bin+i or bin-i is beyond histogram limits (e.g.
  /// for histogram with N bins, bin N+1 corresponds to bin 1,
  /// and bin -1 corresponds to bin N)
  void setNumNeighbors(unsigned n)
    { if (n > 0) numNeighbors_ = n; }

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

public:
  using SimpleTest<TH1F>::runTest;
  /// run the test (result: fraction of channels not appearing noisy or "hot")
  virtual float runTest(const TH1F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(const TH1F *h)
    { return false; } // any scenarios for invalid test?

  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Fraction of non-noisy channels = " << prob_;
      message_ = message.str();
    }

  /// get average for bin under consideration
  /// (see description of method setNumNeighbors)
  Double_t getAverage(int bin, const TH1F *h) const;

  float tolerance_;        /*< tolerance for considering a channel noisy */
  unsigned numNeighbors_;  /*< # of neighboring channels for calculating average to be used
			     for comparison with channel under consideration */
  bool rangeInitialized_;  /*< init-flag for tolerance */
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/// Check that every TH2F channel has mean, RMS within allowed range.
/// Implementation: Giuseppe Della Ricca
class ContentsTH2FWithinRange : public SimpleTest<TH2F>
{
public:
  ContentsTH2FWithinRange(const std::string &name)
    : SimpleTest<TH2F>(name, true)
    {
      checkMean_ = checkRMS_ = validMethod_ = false;
      minMean_ = maxMean_ = minRMS_ = maxRMS_ = 0.0;
      checkMeanTolerance_ = false;
      toleranceMean_ = -1.0;
      setAlgoName(getAlgoName());
    }

  static std::string getAlgoName(void)
    { return "ContentsWithinExpectedTH2F"; }

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

public:
  using SimpleTest<TH2F>::runTest;
  virtual float runTest(const TH2F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(void)
    { return false; } // FIXME: or "!validMethod_";

  virtual void setMessage(void)
    {
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

/// Check that every TProf channel has mean, RMS within allowed range.
/// Implementation: Giuseppe Della Ricca.
class ContentsProfWithinRange : public SimpleTest<TProfile>
{
public:
  ContentsProfWithinRange(const std::string &name)
    : SimpleTest<TProfile>(name, true)
    {
      checkMean_ = checkRMS_ = validMethod_ = false;
      minMean_ = maxMean_ = minRMS_ = maxRMS_ = 0.0;
      checkMeanTolerance_ = false;
      toleranceMean_ = -1.0;
      setAlgoName(getAlgoName());
    }

  static std::string getAlgoName(void)
    { return "ContentsWithinExpectedProf"; }

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

public:
  using SimpleTest<TProfile>::runTest;
  virtual float runTest(const TProfile *t);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(void)
    { return false; } // any scenarios for invalid test?  FIXME: "return !validMethod_;"?

  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Entry fraction within range = " << prob_;
      message_ = message.str();
    }

  /// check that allowed range is logical
  void checkRange(const float xmin, const float xmax);

  bool checkMean_;           //< if true, check the mean value
  bool checkRMS_;            //< if true, check the RMS value
  bool checkMeanTolerance_;  //< if true, check mean tolerance
  float toleranceMean_;      //< fractional tolerance on mean (use only if checkMeanTolerance_ = true)
  float minMean_, maxMean_;  //< allowed range for mean (use only if checkMean_ = true)
  float minRMS_, maxRMS_;    //< allowed range for mean (use only if checkRMS_ = true)
  bool validMethod_;         //< true if method has been chosen
};

/// Check that every TProfile2D channel has mean, RMS within allowed
/// range.  Implementation: Giuseppe Della Ricca.
class ContentsProf2DWithinRange : public SimpleTest<TProfile2D>
{
public:
  ContentsProf2DWithinRange(const std::string &name)
    : SimpleTest<TProfile2D>(name, true)
    {
      checkMean_ = checkRMS_ = validMethod_ = false;
      minMean_ = maxMean_ = minRMS_ = maxRMS_ = 0.0;
      checkMeanTolerance_ = false;
      toleranceMean_ = -1.0;
      setAlgoName(getAlgoName());
    }

  static std::string getAlgoName(void)
    { return "ContentsWithinExpectedProf2D"; }

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

public:
  using SimpleTest<TProfile2D>::runTest;
  virtual float runTest(const TProfile2D *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(void)
    { return false; } // any scenarios for invalid test?  FIXME: "return !validMethod_;"?

  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Entry fraction within range = " << prob_;
      message_ = message.str();
    }

  /// check that allowed range is logical
  void checkRange(const float xmin, const float xmax);

  bool checkMean_;          //< if true, check the mean value
  bool checkRMS_;           //< if true, check the RMS value
  bool checkMeanTolerance_; //< if true, check mean tolerance
  float toleranceMean_;     //< fractional tolerance on mean (use only if checkMeanTolerance_ = true)
  float minMean_, maxMean_; //< allowed range for mean (use only if checkMean_ = true)
  float minRMS_, maxRMS_;   //< allowed range for mean (use only if checkRMS_ = true)
  bool validMethod_;        //< true if method has been chosen
};

/// Algorithm for testing if histogram's mean value is near expected value.
class MeanWithinExpected : public SimpleTest<TH1F>
{
public:
  MeanWithinExpected(const std::string &name)
    : SimpleTest<TH1F>(name)
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

public:
  using SimpleTest<TH1F>::runTest;
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
  virtual float runTest(const TH1F *h);

protected:
  /// true if algorithm is invalid (e.g. wrong type of reference object)
  bool isInvalid(void);

  virtual void setMessage(void)
    {
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

  bool useRMS_;       //< if true, will use RMS of distribution
  bool useSigma_;     //< if true, will use expected_sigma
  bool useRange_;     //< if true, will use allowed range
  float sigma_;       //< sigma to be used in probability calculation (use only if useSigma_ = true)
  float expMean_;     //< expected mean value (used only if useSigma_ = true or useRMS_ = true)
  float xmin_, xmax_; //< allowed range for mean (use only if useRange_ = true)
  bool validMethod_;  //< true if method has been chosen
  bool validExpMean_; //< true if expected mean has been chosen

};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// QTest that should test Most Probable value for some Expected number
// Author : Samvel Khalatian ( samvel at fnal dot gov )
// Created: 04/26/07

namespace edm {
  namespace qtests {
    namespace fits {
      // Convert Significance into Probability value.
      double erfc( const double &rdX);
    }
  }
}

/**
 * @brief
 *   Base for all MostProbables Children classes. Thus each child
 *   implementation will concentrate on fit itself.
 */
class MostProbableBase : public SimpleTest<TH1F>
{
public:
  MostProbableBase(const std::string &name);

  // Set/Get local variables methods
  inline void   setMostProbable(double rdMP) { dMostProbable_ = rdMP;}
  inline double getMostProbable(void) const  { return dMostProbable_;}

  inline void   setSigma(double rdSIGMA)     { dSigma_ = rdSIGMA; }
  inline double getSigma(void) const         { return dSigma_; }

  inline void   setXMin(double rdMIN)        { dXMin_ = rdMIN; }
  inline double getXMin(void) const          { return dXMin_;  }

  inline void   setXMax(double rdMAX)        { dXMax_ = rdMAX; }
  inline double getXMax(void) const          { return dXMax_;  }

public:
  using SimpleTest<TH1F>::runTest;
  /**
   * @brief
   *   Actual Run Test method. Should return: [0, 1] or <0 for failure.
   *   [Note: See SimpleTest<class T> template for details]
   *
   * @param poPLOT  Plot for Which QTest to be run
   *
   * @return
   *   -1      On Error
   *   [0, 1]  Measurement of how Fit value is close to Expected one
   */
  virtual float runTest(const TH1F *poPLOT);

protected:
  /**
   * @brief
   *   Each Child should implement fit method which responsibility is to
   *   perform actual fit and compare mean value with some additional
   *   Cuts if any needed. The reason this task is put into separate method
   *   is that a priory it is unknown what distribution QTest is dealing with.
   *   It might be simple Gauss, Landau or something more sophisticated.
   *   Each Plot needs special treatment (fitting) and extraction of
   *   parameters. Children know about that but not Parent class.
   *
   * @param poPLOT  Plot to be fitted
   *
   * @return
   *   -1     On Error
   *   [0,1]  Measurement of how close Fit Value is to Expected one
   */
  virtual float fit(TH1F *poPLOT) = 0;

  /**
   * @brief
   *   Child should check test if it is valid and return corresponding value
   *   Next common tests are performed here:
   *     1. min < max
   *     2. MostProbable is in (min, max)
   *     3. Sigma > 0
   *
   * @return
   *   True   Invalid QTest
   *   False  Otherwise
   */
  bool isInvalid(void);

  /**
   * @brief
   *   General function that compares MostProbable value gotten from Fit and
   *   Expected one.
   *
   * @param rdMP_FIT     MostProbable value gotten from Fit
   * @param rdSIGMA_FIT  Sigma value gotten from Fit
   *
   * @return
   *   Probability of found Value that measures how close is gotten one to
   *   expected
   */
  double compareMostProbables(const double &rdMP_FIT, const double &rdSIGMA_FIT) const;


  virtual void setMessage(void)
    {
      std::ostringstream message;
      message << " Test " << qtname_ << " (" << algoName_
	      << "): Fraction of Most Probable value match = " << prob_;
      message_ = message.str();
    }

private:
  // Most common Fit values
  double dMostProbable_;
  double dSigma_;
  double dXMin_;
  double dXMax_;
};

/** MostProbable QTest for Landau distributions */
class MostProbableLandau : public MostProbableBase
{
public:
  MostProbableLandau(const std::string &name);

  static std::string getAlgoName()
    { return "MostProbableLandau"; }

  // Set/Get local variables methods
  void setNormalization(const double &rdNORMALIZATION)
    { dNormalization_ = rdNORMALIZATION; }

  double getNormalization(void) const
    { return dNormalization_; }

protected:
  /**
   * @brief
   *   Perform Actual Fit
   *
   * @param poPLOT  Plot to be fitted
   *
   * @return
   *   -1     On Error
   *   [0,1]  Measurement of how close Fit Value is to Expected one
   */
  virtual float fit(TH1F *poPLOT);

private:
  double dNormalization_;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
class AllContentWithinFixedRange : public SimpleTest<TH1F>
{
public:
  AllContentWithinFixedRange(const std::string &name)
    : SimpleTest<TH1F>(name)
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

public:
  using SimpleTest<TH1F>::runTest;
  virtual float runTest(const TH1F *histogram);

protected:
  double x_min, x_max;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};

class AllContentWithinFloatingRange : public SimpleTest<TH1F>
{
public:
  AllContentWithinFloatingRange(const std::string &name)
    : SimpleTest<TH1F>(name)
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

public:
  using SimpleTest<TH1F>::runTest;
  virtual float runTest(const TH1F *histogram);

protected:
  int Nrange;
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};

#if 0 // FIXME: need to know what parameters to set before runTest!
class FlatOccupancy1d : public SimpleTest<TH1F>
{
public:
  FlatOccupancy1d(const std::string &name)
    : SimpleTest<TH1F>(name)
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

public:
  using SimpleTest<TH1F>::runTest;
  virtual float runTest(const TH1F *histogram);

protected:
  double *ExclusionMask;
  double epsilon_min, epsilon_max;
  double S_fail, S_pass;
  double *FailedBins[2];
  int    Nbins;
  int    result;
};
#endif

class FixedFlatOccupancy1d : public SimpleTest<TH1F>
{
public:
  FixedFlatOccupancy1d(const std::string &name)
    : SimpleTest<TH1F>(name)
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

public:
  using SimpleTest<TH1F>::runTest;
  virtual float runTest(const TH1F *histogram);

protected:
  double b;
  double *ExclusionMask;
  double epsilon_min, epsilon_max;
  double S_fail, S_pass;
  double *FailedBins[2];
  int    Nbins;
  int    result;
};

class CSC01 : public SimpleTest<TH1F>
{
public:
  CSC01(const std::string &name)
    : SimpleTest<TH1F>(name)
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

public:
  using SimpleTest<TH1F>::runTest;
  virtual float runTest(const TH1F *histogram);

protected:
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
#if 0 // FIXME: need to know what parameters to set before runTest!
class AllContentAlongDiagonal : public SimpleTest<TH2F>

public:
  AllContentAlongDiagonal(const std::string &name)
    : SimpleTest<TH2F>(name)
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

public:
  using SimpleTest<TH2F>::runTest;
  virtual float runTest(const TH2F *histogram); 

protected:
  double epsilon_max;
  double S_fail, S_pass;
  double epsilon_obs;
  double S_fail_obs, S_pass_obs;
  int result;
};
#endif

#endif // DQMSERVICES_CORE_Q_CRITERION_H
