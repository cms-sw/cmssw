#ifndef DQM_SiStripCommissioningAnalysis_SiStripCommissioningAnalysis_H
#define DQM_SiStripCommissioningAnalysis_SiStripCommissioningAnalysis_H

class Histograms;
class Monitorables;

/**
   \class SiStripCommissioningAnalysis

   \brief Abstract base for classes that provide histogram-based
   analyses for commissioning tasks. An input "Histograms" object is
   required and the analysis extracts the appropriate parameter(s) and
   makes them available to the user via the "Monitorables" object.
*/
class SiStripCommissioningAnalysis {

 public:
  
  SiStripCommissioningAnalysis() {;}
  virtual ~SiStripCommissioningAnalysis() {;}
  
  virtual void histoAnalysis( const Histograms&, 
			      Monitorables& ) = 0;
  
};


/** 
    \class Histograms
    \brief Abstract base class for which there is one concrete
    implementation per commissioning task. Each derived class should
    contain all root histos required by the appropriate "histogram
    analysis", as well as "setter" / "getter" methods.
*/
class Histograms {
 public:
  Histograms() {;}
  virtual ~Histograms() {;}
  
};

/** 
    \class Monitorables
    \brief Abstract base class for which there is one concrete
    implementation per commissioning task. Each derived class should
    contain all "monitorables" provided by the appropriate "histogram
    analysis", as well as "setter" / "getter" methods.
*/
class Monitorables {
 public:
  Monitorables() {;}
  virtual ~Monitorables() {;}
};


#endif // DQM_SiStripCommissioningAnalysis_SiStripCommissioningAnalysis_H

