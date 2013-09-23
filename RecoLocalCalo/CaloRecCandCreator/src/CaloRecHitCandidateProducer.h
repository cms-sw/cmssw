#ifndef RecoCandAlgos_CaloRecHitCandidateProducer_h
#define RecoCandAlgos_CaloRecHitCandidateProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class CaloGeometry;
class CaloRecHit;
class HcalTopology;

class CaloRecHitCandidateProducer : public edm::EDProducer {
public:
  CaloRecHitCandidateProducer( const edm::ParameterSet&); 
  ~CaloRecHitCandidateProducer() { }
  void produce( edm::Event&, const edm::EventSetup& );
  double cellTresholdAndWeight (const CaloRecHit&, const HcalTopology&) const;
  
private:
  /// source collection tag
  std::vector<edm::InputTag> mEcalLabels;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
  std::vector<edm::EDGetTokenT<EcalRecHitCollection> > toks_ecal_;
  bool mAllowMissingInputs;
  bool mUseHO;
  double mEBthreshold, mEEthreshold;
  double mHBthreshold, mHESthreshold,  mHEDthreshold; 
  double mHOthreshold, mHF1threshold, mHF2threshold;
  double mEBweight, mEEweight; 
  double mHBweight, mHESweight, mHEDweight, mHOweight, mHF1weight, mHF2weight;

};

#endif
