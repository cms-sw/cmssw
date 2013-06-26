#ifndef DPGAnalysis_SiStripTools_MultiplicityCorrelatorHistogramMaker_H
#define DPGAnalysis_SiStripTools_MultiplicityCorrelatorHistogramMaker_H

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"


namespace edm {
  class ParameterSet;
  class Event;
  class Run;
}
class TH1F;
class TH2F;

class MultiplicityCorrelatorHistogramMaker {

 public:
  MultiplicityCorrelatorHistogramMaker();
  MultiplicityCorrelatorHistogramMaker(const edm::ParameterSet& iConfig);
 
  ~MultiplicityCorrelatorHistogramMaker();

  void beginRun(const edm::Run& iRun);
  void fill(const edm::Event& iEvent, const int xmult, const int ymult);
  
 private:
  
  RunHistogramManager m_rhm;
  RunHistogramManager m_fhm;
  bool m_runHisto;
  bool m_runHistoBXProfile;
  bool m_runHistoBX;
  bool m_runHisto2D;
  bool m_runHistoProfileBX;
  double m_scfact;
  TH2F* m_yvsxmult;
  TH1F* m_atanyoverx;
  TH1F** m_atanyoverxrun;
  TProfile** m_atanyoverxvsbxrun;
  TH2F** m_atanyoverxvsbxrun2D;
  TH2F** m_yvsxmultrun;
  TProfile2D** m_yvsxmultprofvsbxrun;
  TProfile2D** m_xvsymultprofvsbxrun;
};


#endif //  DPGAnalysis_SiStripTools_MultiplicityCorrelatorHistogramMaker_H
