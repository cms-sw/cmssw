#ifndef DQM_SiStripCommissioningAnalysis_SiStripPedestalsAnalysis_H
#define DQM_SiStripCommissioningAnalysis_SiStripPedestalsAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/SiStripCommissioningAnalysis.h"
#include <vector>

class PedestalHistograms;
class PedestalMonitorables;
class TH1F;

/** 
   \class SiStripPedestalsAnalysis
   \brief Concrete implementation of the histogram-based analysis for
   pedestals and noise "monitorables". 
*/
class SiStripPedestalsAnalysis : public SiStripCommissioningAnalysis {
  
 public:

  SiStripPedestalsAnalysis() {;}
  virtual ~SiStripPedestalsAnalysis() {;}
  
  virtual void histoAnalysis( const PedestalHistograms&, 
			      PedestalMonitorables& );
  
};


/** 
   \class PedestalHistograms
   \brief Concrete implementation of Histograms that contains the
   necessary histograms required to calculate pedestals and noise.
*/
class PedestalHistograms : public Histograms {
  
 public:
  
  PedestalHistograms() {;}
  virtual ~PedestalHistograms() {;}

  inline TH1F* raw() const { return raw_; }
  inline TH1F* entries() const { return entries_; }
  
 private:

  TH1F* raw_;
  TH1F* entries_;

};


/** 
   \class PedestalMonitorables
   \brief Concrete implementation of Monitorables that contains the
   parameters (pedetals and noise) provided by the histogram-based
   analysis.
*/
class PedestalMonitorables : public Monitorables {
  
 public:

  PedestalMonitorables() : rawPedestals_(), rawNoise_() {;}
  virtual ~PedestalMonitorables() {;}

  inline const std::vector<float>& rawPedestals() const { return rawPedestals_; }
  inline const std::vector<float>& rawNoise() const { return rawNoise_; }

  inline void rawPedestals( std::vector<float>& peds ) { rawPedestals_ = peds; }
  inline void rawNoise( std::vector<float>& noise ) { rawNoise_ = noise; }
  
 private:

  std::vector<float> rawPedestals_;
  std::vector<float> rawNoise_;

};


#endif // DQM_SiStripCommissioningAnalysis_SiStripPedestalsAnalysis_H

