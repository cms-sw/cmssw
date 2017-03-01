
#ifndef L2TAUJETSMERGER_H
#define L2TAUJETSMERGER_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


#include <map>
#include <vector>

class L2TauJetsMerger: public edm::global::EDProducer<> {
 public:
  explicit L2TauJetsMerger(const edm::ParameterSet&);
  ~L2TauJetsMerger();
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
    
  typedef std::vector<edm::InputTag> vtag;
  typedef std::vector<edm::EDGetTokenT<reco::CaloJetCollection> > vtoken_cjets;
  const vtag jetSrc;
  vtoken_cjets jetSrc_token;
  const double mEt_Min;


      class SorterByPt {
      public:
	SorterByPt() {}
	~SorterByPt() {}
	bool operator()(const reco::CaloJet& jet1 , const reco::CaloJet& jet2)
	{
	  return jet1.pt()>jet2.pt();
	}
      };


};
#endif
