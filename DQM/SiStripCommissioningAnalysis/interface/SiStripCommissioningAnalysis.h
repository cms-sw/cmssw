#ifndef DQM_SiStripCommissioningAnalysis_SiStripCommissioningAnalysis_H
#define DQM_SiStripCommissioningAnalysis_SiStripCommissioningAnalysis_H

#include <string>

class Histograms;
class Monitorables;

/**
   @file DQM/SiStripCommissioningAnalysis/SiStripCommissioningAnalysis.h
   @class SiStripCommissioningAnalysis

   @brief Abstract base for derived classes that provide a
   histogram-based analysis for each of the commissioning tasks. 
   
   - The histoAnalysis() method takes two arguments: a Histograms
   object as an "input" and a Monitorables object as "output".
   
   - The Histograms object "wraps" all histograms required by the
   analysis, and the analysis extracts the appropriate parameter(s)
   and makes them available to the user via the "Monitorables" object.

   - A user of this analysis package is required to create
   instances of the "Histograms" (containing the appropriate histos)
   and "Monitorables" wrappers.
*/
class SiStripCommissioningAnalysis {

 public:
  
  SiStripCommissioningAnalysis() {;}
  virtual ~SiStripCommissioningAnalysis() {;}
  
  /** Takes histograms as input (in the form of a Histograms object),
      extracts the appropriate HW configuration parameter(s) and
      returns by reference within the arg list the parameters within
      the Monitorables objects. */
  virtual void histoAnalysis( const Histograms&, 
			      Monitorables& ) = 0;
  
};


/** 
    @class Histograms 
    @brief Abstract base class for which there is one concrete
    implementation per commissioning task. The derived classes are
    necesssary to "wrap" all root histograms required by the
    appropriate "histogram analysis" class, as well as providing
    "setter" and "getter" methods.
*/
class Histograms {
 public:
  Histograms() {;}
  virtual ~Histograms() {;}
  /** Simple pure virtual method that forces a user to implement
      concrete implementation of this histogram container. */
  virtual std::string myName() = 0;
};


/** 
    @class Monitorables
    @brief Abstract base class for which there is one concrete
    implementation per commissioning task. The derived classes are
    necesssary to "wrap" all HW configuration parameters provided by
    the appropriate "histogram analysis" class, as well as providing
    "setter" and "getter" methods.
*/
class Monitorables {
 public:
  Monitorables() {;}
  virtual ~Monitorables() {;}
  /** Simple pure virtual method that forces a user to implement
      concrete implementation of this histogram container. */
  virtual std::string myName() = 0;
};


#endif // DQM_SiStripCommissioningAnalysis_SiStripCommissioningAnalysis_H

