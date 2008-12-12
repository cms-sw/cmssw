#ifndef EventFilter_SiStripRawToDigi_SiStripClusterValidator_H
#define EventFilter_SiStripRawToDigi_SiStripClusterValidator_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class SiStripClusterValidator : public edm::EDAnalyzer {
 public:
  SiStripClusterValidator(const edm::ParameterSet& config);
  ~SiStripClusterValidator();
  virtual void beginJob(const edm::EventSetup& setup);
  virtual void endJob();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  void validate(const edm::DetSetVector<SiStripCluster>&, const edm::DetSetVector<SiStripCluster>&);
  void validate(const edmNew::DetSetVector<SiStripCluster>&, const edmNew::DetSetVector<SiStripCluster>&);

 private:

  /// Input collections
  edm::InputTag collection1Tag_;
  edm::InputTag collection2Tag_;
  bool dsvnew_;
  /// used to remember if there have been errors for message in endJob
  bool errors_;
};

std::ostream& operator<<(std::ostream&, const edmNew::DetSetVector<SiStripCluster>&);
std::ostream& operator<<(std::ostream&, const edm::DetSetVector<SiStripCluster>&);

#endif /// RecoLocalTracker_SiStripClusterizer_SiStripClusterValidator_H
