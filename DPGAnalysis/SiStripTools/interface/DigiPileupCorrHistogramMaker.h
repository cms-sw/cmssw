#ifndef DPGAnalysis_SiStripTools_DigiPileupCorrHistogramMaker_H
#define DPGAnalysis_SiStripTools_DigiPileupCorrHistogramMaker_H

#include <string>
#include <map>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"


namespace edm {
  class ParameterSet;
  class Event;
}
class TH2F;
class TProfile;
class TFileDirectory;

class DigiPileupCorrHistogramMaker {

 public:
  DigiPileupCorrHistogramMaker(edm::ConsumesCollector&& iC);
  DigiPileupCorrHistogramMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);

  ~DigiPileupCorrHistogramMaker();

  void book(const std::string dirname, const std::map<unsigned int, std::string>& labels);
  void book(const std::string dirname);
  void beginRun(const unsigned int nrun);
  void fill(const edm::Event& iEvent, const std::map<unsigned int,int>& ndigi);

 private:

  edm::EDGetTokenT<std::vector<PileupSummaryInfo> > m_pileupcollectionToken;
  bool m_useVisibleVertices;
  std::string m_hitname;
  const int m_nbins;
  const int m_scalefact;
  std::map<unsigned int,int> m_binmax;
  std::map<unsigned int, std::string> m_labels;

  std::map<unsigned int,TH2F*> m_nmultvsmclumi;
  std::map<unsigned int,TProfile*> m_nmultvsmclumiprof;
  std::map<unsigned int,TH2F*> m_nmultvsmcnvtx;
  std::map<unsigned int,TProfile*> m_nmultvsmcnvtxprof;
  std::map<unsigned int,TFileDirectory*> m_subdirs;

};


#endif //  DPGAnalysis_SiStripTools_DigiPileupCorrHistogramMaker_H
