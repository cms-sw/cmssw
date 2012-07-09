#ifndef DPGAnalysis_SiStripTools_DigiInvestigatorHistogramMaker_H
#define DPGAnalysis_SiStripTools_DigiInvestigatorHistogramMaker_H

#include <string>
#include <map>
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

namespace edm {
  class ParameterSet;
}
class TH1F;
class TProfile;
class TFileDirectory;

class DigiInvestigatorHistogramMaker {

 public:
  DigiInvestigatorHistogramMaker();
  DigiInvestigatorHistogramMaker(const edm::ParameterSet& iConfig);
 
  ~DigiInvestigatorHistogramMaker();

  void book(const std::string dirname, const std::map<unsigned int, std::string>& labels);
  void book(const std::string dirname);
  void beginRun(const unsigned int nrun);
  void fill(const unsigned int orbit, const std::map<unsigned int,int>& ndigi);

 private:

  std::string _hitname;
  const int _nbins;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;
  int _scalefact;
  const bool _runHisto;
  std::map<unsigned int,int> _binmax;
  std::map<unsigned int, std::string> _labels;


  RunHistogramManager _rhm;
  std::map<unsigned int,TProfile**> _nmultvsorbrun;
  std::map<unsigned int,TH1F*> _nmult;
  std::map<unsigned int,TFileDirectory*> _subdirs;

};


#endif //  DPGAnalysis_SiStripTools_DigiInvestigatorHistogramMaker_H
