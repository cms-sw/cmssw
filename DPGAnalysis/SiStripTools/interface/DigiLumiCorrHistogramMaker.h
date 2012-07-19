#ifndef DPGAnalysis_SiStripTools_DigiLumiCorrHistogramMaker_H
#define DPGAnalysis_SiStripTools_DigiLumiCorrHistogramMaker_H

#include <string>
#include <map>
#include "FWCore/Utilities/interface/InputTag.h"


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

class DigiLumiCorrHistogramMaker {

 public:
  DigiLumiCorrHistogramMaker();
  DigiLumiCorrHistogramMaker(const edm::ParameterSet& iConfig);
 
  ~DigiLumiCorrHistogramMaker();

  void book(const std::string dirname, const std::map<unsigned int, std::string>& labels);
  void book(const std::string dirname);
  void beginRun(const edm::Run& iRun);
  void fill(const edm::Event& iEvent, const std::map<unsigned int,int>& ndigi);

 private:

  const edm::InputTag m_lumiProducer;
  std::map<unsigned int,RunHistogramManager*> m_fhm;
  bool m_runHisto;
  std::string m_hitname;
  const int m_nbins;
  const int m_scalefact; 
  const double m_maxlumi;
  std::map<unsigned int,int> m_binmax;
  std::map<unsigned int, std::string> m_labels;

  std::map<unsigned int,TH2F*> m_nmultvslumi;
  std::map<unsigned int,TProfile*> m_nmultvslumiprof;
  std::map<unsigned int,TProfile2D**> m_nmultvslumivsbxprofrun;
  std::map<unsigned int,TFileDirectory*> m_subdirs;

};


#endif //  DPGAnalysis_SiStripTools_DigiLumiCorrHistogramMaker_H
