#ifndef _QCRITERION_H_
#define _QCRITERION_H_

#include <string>
#include <vector>
#include <set>
#include <map>

class MonitorElement;

/* Base class for quality tests run on Monitoring Elements; 

   Currently supporting the following tests: 
   - Comparison to reference (Chi2, Kolmogorov)
   - Contents within [Xmin, Xmax]
   - Contents within [Ymin, Ymax]
   - Identical contents
   - Mean value within expected value
*/

class QCriterion
{
  // (class should be created by DaqMonitorBEInterface class)

 public:
  // enable test
  void enable(void){enabled_ = true;}
  // disable test
  void disable(void){enabled_ = false;}
  // true if test is enabled
  bool isEnabled(void) const {return enabled_;}
  // true if QCriterion has been modified since last time it ran
  bool wasModified(void) const {return wasModified_;}
  // get test status (see Core/interface/QTestStatus.h)
  int getStatus(void) const {return status_;}
  // get message attached to test
  std::string getMessage(void) const {return message_;} 
  // get name of quality test
  std::string getName(void) const {return qtname_;}
  //get  algorithm name
  virtual std::string getAlgoName(void) = 0;

  // set probability limit for test warning (default: 90%)
  void setWarningProb(float prob)
  {if (validProb(prob))warningProb_ = prob;}
  // set probability limit for test error (default: 50%)
  void setErrorProb(float prob)
  {if (validProb(prob))errorProb_ = prob;}

 protected:
  // 
  QCriterion(std::string qtname){qtname_ = qtname; init();}
  //
  virtual ~QCriterion(void){}
  // initialize values
  void init(void);
  // set algorithm name
  void setAlgoName(std::string name){algoName_ = name;}
  // run test (result: [0, 1])
  virtual float runTest(const MonitorElement * const me) = 0;
  // if true will run test
  bool enabled_;
 // quality test status (see Core/interface/QTestStatus.h)
  int status_;
  // message attached to test
  std::string message_;
  // name of quality test
  std::string qtname_;
  // flag for indicating algorithm modifications since last time it ran
  bool wasModified_;
  // name of algorithm
  std::string algoName_;
  // all search strings for MEs using this QCriterion
  std::set<std::string> searchStrings;
  // call method when something in the algorithm changes
  void update(void){wasModified_ = true;}
  // make sure algorithm can run (false: should not run)
  bool check(const MonitorElement * const me);
  // true if algorithm is invalid (e.g. wrong type of reference object)
  virtual bool isInvalid(const MonitorElement * const me) = 0;
  // true if MonitorElement does not have enough entries to run test
  virtual bool notEnoughStats(const MonitorElement * const me) const = 0;
  // true if probability value is valid
  bool validProb(float prob) const{return prob>=0 && prob<=1;}
  // set status & message for disabled tests
  void setDisabled(void);
  // set status & message for invalid tests
  void setInvalid(void);
  // set status & message for tests w/o enough statistics
  void setNotEnoughStats(void);
  // set status & message for succesfull tests
  void setOk(void);
  // set status & message for tests w/ warnings
  void setWarning(void);
  // set status & message for tests w/ errors
  void setError(void);
  // set message after test has run
  virtual void setMessage(void) = 0;

  // probability limits for warnings, errors
  float warningProb_, errorProb_;
  // test result [0, 1]; 
  // (a) for comparison to reference: 
  // probability that histogram is consistent w/ reference
  // (b) for "contents within range":
  // fraction of entries that fall within allowed range
  float prob_;

 private:

  // default "probability" values for setting warnings & errors when running tests
  static const float WARNING_PROB_THRESHOLD;
  static const float ERROR_PROB_THRESHOLD;

  // for creating and deleting class instances
  friend class DaqMonitorBEInterface;
  // for running the test
  friend class QReport;
};

namespace dqm
{
  namespace qtests
  {
    // key: quality test name, value: address of QCriterion
    typedef std::map<std::string, QCriterion *> QC_map;
    typedef QC_map::iterator qc_it;
    typedef QC_map::const_iterator cqc_it;

    typedef std::set<QCriterion *> QC_set;
    typedef QC_set::iterator qcs_it;

    typedef std::vector<QCriterion *>::iterator vqc_it;
  }
}


#endif
