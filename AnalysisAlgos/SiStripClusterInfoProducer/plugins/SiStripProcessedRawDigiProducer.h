#ifndef AnalysisAlgos_SiStripClusterInfoProducer_SiStripProcessedRawDigiProducer_H
#define AnalysisAlgos_SiStripClusterInfoProducer_SiStripProcessedRawDigiProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripMedianCommonModeNoiseSubtraction.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripFastLinearCommonModeNoiseSubtraction.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripTT6CommonModeNoiseSubtraction.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripProcessedRawDigi.h"

#include <map>
#include <string>
#include <functional>
#include <memory>

class SiStripProcessedRawDigiProducer : public edm::EDProducer {
 public:
  explicit SiStripProcessedRawDigiProducer(edm::ParameterSet const&);
  ~SiStripProcessedRawDigiProducer();
  
  enum FEDOUTPUT {NOTFOUND=0, ZS, VR, PR, SM};
  enum CMNTYPE   {UNKNOWN=0, MEDIAN, TT6, FASTLINEAR};  

 private:
  void produce(edm::Event& e, const edm::EventSetup& es);
  
  void vr_process(const edm::DetSetVector<SiStripRawDigi>&,  edm::DetSetVector<SiStripProcessedRawDigi>&);
  void pr_process(const edm::DetSetVector<SiStripRawDigi>&,  edm::DetSetVector<SiStripProcessedRawDigi>&);
  void zs_process(const edm::DetSetVector<SiStripDigi>&,     edm::DetSetVector<SiStripProcessedRawDigi>&);
  void common_process(std::vector<float>&, const uint32_t&,  edm::DetSetVector<SiStripProcessedRawDigi>&);

  typedef std::vector<edm::ParameterSet> Parameters;
  
  std::map<std::string, FEDOUTPUT> inputmap_;
  edm::ParameterSet                    conf_;
  edm::ESHandle<SiStripGain>     gainHandle_;
  std::string            CMNSubtractionMode_;
  bool                  validCMNSubtraction_;

  SiStripPedestalsSubtractor*             SiStripPedestalsSubtractor_;
  SiStripCommonModeNoiseSubtractor* SiStripCommonModeNoiseSubtractor_;
};
#endif //AnalysisAlgos_SiStripClusterInfoProducer_SiStripProcessedRawDigiProducer_H

