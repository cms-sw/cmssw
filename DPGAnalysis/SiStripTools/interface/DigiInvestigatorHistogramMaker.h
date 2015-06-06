#ifndef DPGAnalysis_SiStripTools_DigiInvestigatorHistogramMaker_H
#define DPGAnalysis_SiStripTools_DigiInvestigatorHistogramMaker_H

#include <string>
#include <map>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

namespace edm {
  class ParameterSet;
  class Event;
  class Run;
}
class TH1F;
class TProfile;
class TFileDirectory;

class DigiInvestigatorHistogramMaker {

 public:
  DigiInvestigatorHistogramMaker(edm::ConsumesCollector&& iC);
  DigiInvestigatorHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);

  ~DigiInvestigatorHistogramMaker();

  void book(const std::string dirname, const std::map<unsigned int, std::string>& labels);
  void book(const std::string dirname);
  void beginRun(const edm::Run& iRun);
  void fill(const edm::Event& iEvent, const std::map<unsigned int,int>& ndigi);

 private:

  std::string _hitname;
  const int _nbins;
  const unsigned int m_maxLS;
  const unsigned int m_LSfrac;
  int _scalefact;
  const bool _runHisto;
  const bool _fillHisto;
  std::map<unsigned int,int> _binmax;
  std::map<unsigned int, std::string> _labels;


  RunHistogramManager _rhm;
  RunHistogramManager _fhm;
  std::map<unsigned int,TProfile**> _nmultvsorbrun;
  std::map<unsigned int,TProfile**> _nmultvsbxrun;
  std::map<unsigned int,TProfile**> _nmultvsbxfill;
  std::map<unsigned int,TH1F*> _nmult;
  std::map<unsigned int,TFileDirectory*> _subdirs;

};


#endif //  DPGAnalysis_SiStripTools_DigiInvestigatorHistogramMaker_H
