
#ifndef L2TAUJETSMERGER_H
#define L2TAUJETSMERGER_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


#include <map>
#include <vector>

class L2TauJetsMerger: public edm::EDProducer {
 public:
  explicit L2TauJetsMerger(const edm::ParameterSet&);
  ~L2TauJetsMerger();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
    
  typedef std::vector<edm::InputTag> vtag;
  vtag jetSrc;
  double mEt_Min;
  std::map<int, const reco::CaloJet> myL2L1JetsMap; //first is # L1Tau , second is L2 jets


      class SorterByPt {
      public:
	SorterByPt() {}
	~SorterByPt() {}
	bool operator()(reco::CaloJet jet1 , reco::CaloJet jet2)
	{
	  return jet1.pt()>jet2.pt();
	}
      };


};
#endif
