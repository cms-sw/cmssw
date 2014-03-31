#ifndef DPGAnalysis_SiStripTools_DigiVertexCorrHistogramMaker_H
#define DPGAnalysis_SiStripTools_DigiVertexCorrHistogramMaker_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <string>
#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class Run;
}
class TH2F;
class TProfile;
class TProfile2D;
class TFileDirectory;
class RunHistogramManager;

class DigiVertexCorrHistogramMaker {

 public:
  DigiVertexCorrHistogramMaker();
  DigiVertexCorrHistogramMaker(const edm::ParameterSet& iConfig);

  ~DigiVertexCorrHistogramMaker();

  void book(const std::string dirname, const std::map<unsigned int, std::string>& labels, edm::ConsumesCollector&& iC);
  void book(const std::string dirname, edm::ConsumesCollector&& iC) {book(dirname, iC);}
  void book(const std::string dirname, edm::ConsumesCollector& iC);
  void beginRun(const edm::Run& iRun);
  void fill(const edm::Event& iEvent, const unsigned int nvtx, const std::map<unsigned int,int>& ndigi);

 private:

  std::map<unsigned int,RunHistogramManager*> m_fhm;
  bool m_runHisto;
  std::string m_hitname;
  const int m_nbins;
  const int m_scalefact;
  const int m_maxnvtx;
  std::map<unsigned int,int> m_binmax;
  std::map<unsigned int, std::string> m_labels;

  std::map<unsigned int,TH2F*> m_nmultvsnvtx;
  std::map<unsigned int,TProfile*> m_nmultvsnvtxprof;
  std::map<unsigned int,TProfile2D**> m_nmultvsnvtxvsbxprofrun;
  std::map<unsigned int,TFileDirectory*> m_subdirs;

};


#endif //  DPGAnalysis_SiStripTools_DigiVertexCorrHistogramMaker_H
