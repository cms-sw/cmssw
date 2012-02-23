#ifndef DPGAnalysis_SiStripTools_MultiplicityCorrelatorHistogramMaker_H
#define DPGAnalysis_SiStripTools_MultiplicityCorrelatorHistogramMaker_H

namespace edm {
  class ParameterSet;
}
class TH1F;
class TH2F;

class MultiplicityCorrelatorHistogramMaker {

 public:
  MultiplicityCorrelatorHistogramMaker();
  MultiplicityCorrelatorHistogramMaker(const edm::ParameterSet& iConfig);
 
  ~MultiplicityCorrelatorHistogramMaker();

  void fill(const int xmult, const int ymult);

 private:

  double _scfact;
  TH2F* _yvsxmult;
  TH1F* _atanyoverx;
};


#endif //  DPGAnalysis_SiStripTools_MultiplicityCorrelatorHistogramMaker_H
