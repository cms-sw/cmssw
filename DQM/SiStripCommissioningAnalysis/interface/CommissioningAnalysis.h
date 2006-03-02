#ifndef DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H
#define DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/HistoSet.h"
#include <string>

class CommissioningHistograms;
class CommissioningMonitorables;
class TH1F;

using namespace std;

/**
   @file DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h
   @class CommissioningAnalysis
   
   @brief Abstract base for derived classes that provide a
   histogram-based analysis for each of the commissioning tasks. 
   
   - The analysis() method takes two "wrapper" objects as args:
   a CommissioningHistograms object as an input; and a
   CommissioningMonitorables object as output.
   
   - The CommissioningHistograms and CommissioningMonitorables classes
   are abstract bases that require a derived implementation specific
   to each commissioning task.
   
   - The derived classes of CommissioningHistograms wrap all the
   histograms required by the appropriate analysis (that is task
   dependent), and the analysis extracts the appropriate parameter(s)
   and makes them available to the user via the corresponding derived
   class of CommissioningMonitorables.

   - A user of this analysis package is required to create 
   instances of the derived classes of CommissioningHistograms and
   CommissioningMonitorables. In order to create an instance of the
   CommissioningHistograms class, a user may use the ExtractRootHisto
   class to extract root histos from MonitorElements of the DQM fwk.
*/
class CommissioningAnalysis {

 public:
  
  CommissioningAnalysis() {;}
  virtual ~CommissioningAnalysis() {;}
  
  /** Performs histogram-based analysis. */
  virtual void analysis( const CommissioningHistograms& histos, 
			 CommissioningMonitorables& monitorables ) = 0;
  
};

// -----------------------------------------------------------------------------
/** 
    @file DQM/SiStripCommissioningAnalysis/CommissioningAnalysis.h
    @class CommissioningHistograms 

    @brief Abstract base class for which there is one concrete
    implementation per commissioning task. The derived classes are
    necesssary to "wrap" all root histograms required by the
    appropriate "histogram analysis" class, as well as providing
    "setter" and "getter" methods.
*/
class CommissioningHistograms {

 public:
  
  inline const string& myName() const { return name_; }

 protected:
  /** Protected constructor enforces derived implementation. */
  CommissioningHistograms( string name ) : name_(name) {;}
 public:
  /** Public virtual destructor */
  virtual ~CommissioningHistograms() {;}
 private:
  /** Private default constructor prevents using default constructor
      in derived class. */
  CommissioningHistograms() {;}

  string name_;

};

// -----------------------------------------------------------------------------
/** 
    @file DQM/SiStripCommissioningAnalysis/CommissioningAnalysis.h
    @class CommissioningMonitorables
    
    @brief Abstract base class for which there is one concrete
    implementation per commissioning task. The derived classes are
    necesssary to "wrap" all HW configuration parameters provided by
    the appropriate "histogram analysis" class, as well as providing
    "setter" and "getter" methods.
*/
class CommissioningMonitorables {

 public:

  inline const string& myName() { return name_; }
  
 protected:
  /** Protected constructor enforces derived implementation. */
  CommissioningMonitorables( string name ) : name_(name) {;}
 public:
  /** Public virtual destructor */
  virtual ~CommissioningMonitorables() {;}
 private:
  /** Private default constructor prevents using default constructor
      in derived class. */
  CommissioningMonitorables() {;}
  
  string name_;

};

#endif // DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

