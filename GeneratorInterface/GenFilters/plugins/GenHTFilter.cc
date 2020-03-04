/*
Package:    GeneralInterface/GenFilters/GenHTFilter
Class:      GenHTFilter

class GenHTFilter GenHTFilter.cc GeneratorInterface/GenFilters/src/GenHTFilter.cc

Description: EDFilter which firstly calculates the generated HT (genHT) from GenJets (given a HT definition) and then filters the events on the generator level (given a certain genHT cut).  

Implementation: 

The following input parameters are used (found in the genHTFilter_cfi configuration file fragment used for module initalisation): 
  src = cms.InputTag("X") : GenJet collection label as input
  jetPtCut = cms.double(#) : GenJet pT cut for HT
  jetEtaCut = cms.double(#) : GenJet eta cut for HT
  genHTcut = cms.double(#) : GenHT cut

Original Author:  Mateusz Zarucki
         Created:  Oct 2015
*/

//System include files
#include <memory>
#include <vector>

//User include files
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

//Class declaration
class GenHTFilter : public edm::global::EDFilter<> {
public:
  explicit GenHTFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  //Member data
  const edm::EDGetTokenT<reco::GenJetCollection> token_;
  const double jetPtCut_, jetEtaCut_, genHTcut_;
};

//Constructor
GenHTFilter::GenHTFilter(const edm::ParameterSet& params)
    : token_(consumes<reco::GenJetCollection>(params.getParameter<edm::InputTag>("src"))),
      jetPtCut_(params.getParameter<double>("jetPtCut")),
      jetEtaCut_(params.getParameter<double>("jetEtaCut")),
      genHTcut_(params.getParameter<double>("genHTcut")) {}

bool GenHTFilter::filter(edm::StreamID, edm::Event& evt, const edm::EventSetup& params) const {
  using namespace std;
  using namespace edm;
  using namespace reco;

  //Read GenJets Collection from Event
  edm::Handle<reco::GenJetCollection> generatedJets;
  evt.getByToken(token_, generatedJets);

  //Loop over all jets in Event and calculate genHT
  double genHT = 0.0;
  for (std::vector<reco::GenJet>::const_iterator it = generatedJets->begin(); it != generatedJets->end(); ++it) {
    const reco::GenJet& gjet = *it;

    //Add GenJet pt to genHT if GenJet complies with given HT definition
    if (gjet.pt() > jetPtCut_ && abs(gjet.eta()) < jetEtaCut_) {
      genHT += gjet.pt();
    }
  }
  return (genHT > genHTcut_);  //Return boolean whether genHT passes cut value
}

// Define module as a plug-in
DEFINE_FWK_MODULE(GenHTFilter);
