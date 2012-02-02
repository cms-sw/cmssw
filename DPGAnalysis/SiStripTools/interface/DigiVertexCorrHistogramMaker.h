#ifndef DPGAnalysis_SiStripTools_DigiVertexCorrHistogramMaker_H
#define DPGAnalysis_SiStripTools_DigiVertexCorrHistogramMaker_H

#include <string>
#include <map>

namespace edm {
  class ParameterSet;
}
class TH2F;
class TFileDirectory;

class DigiVertexCorrHistogramMaker {

 public:
  DigiVertexCorrHistogramMaker();
  DigiVertexCorrHistogramMaker(const edm::ParameterSet& iConfig);
 
  ~DigiVertexCorrHistogramMaker();

  void book(const std::string dirname, const std::map<unsigned int, std::string>& labels);
  void book(const std::string dirname);
  void beginRun(const unsigned int nrun);
  void fill(const unsigned int nvtx, const std::map<unsigned int,int>& ndigi);

 private:

  std::string m_hitname;
  const int m_nbins;
  const int m_scalefact; 
  std::map<unsigned int,int> m_binmax;
  std::map<unsigned int, std::string> m_labels;

  std::map<unsigned int,TH2F*> m_nmultvsnvtx;
  std::map<unsigned int,TFileDirectory*> m_subdirs;

};


#endif //  DPGAnalysis_SiStripTools_DigiVertexCorrHistogramMaker_H
