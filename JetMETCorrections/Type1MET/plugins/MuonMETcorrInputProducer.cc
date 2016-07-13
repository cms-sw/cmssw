#include "JetMETCorrections/Type1MET/plugins/MuonMETcorrInputProducer.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

MuonMETcorrInputProducer::MuonMETcorrInputProducer(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  token_ = consumes<reco::MuonCollection>(cfg.getParameter<edm::InputTag>("src"));
  muonCorrectionMapToken_ = consumes<edm::ValueMap<reco::MuonMETCorrectionData> >(cfg.getParameter<edm::InputTag>("srcMuonCorrections"));

  produces<CorrMETData>();
}

MuonMETcorrInputProducer::~MuonMETcorrInputProducer()
{
// nothing to be done yet...
}

void MuonMETcorrInputProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::unique_ptr<CorrMETData> unclEnergySum(new CorrMETData());

  edm::Handle<reco::MuonCollection> muons;
  evt.getByToken(token_, muons);

  typedef edm::ValueMap<reco::MuonMETCorrectionData> MuonMETCorrectionMap;
  edm::Handle<MuonMETCorrectionMap> muonCorrections;
  evt.getByToken(muonCorrectionMapToken_, muonCorrections);

//--- sum muon corrections. 
//
//    NOTE: MET = -(jets + muon corrections + "unclustered energy"),
//          so "unclustered energy" = -(MET + jets + muons),
//          i.e. muon corrections enter the sum of "unclustered energy" with negative sign.
//
  int numMuons = muons->size();
  for ( int muonIndex = 0; muonIndex < numMuons; ++muonIndex ) {
    const reco::Muon& muon = muons->at(muonIndex);
	
    reco::MuonRef muonRef(muons, muonIndex);

    reco::MuonMETCorrectionData muonCorrection = (*muonCorrections)[muonRef];
    if ( muonCorrection.type() != reco::MuonMETCorrectionData::NotUsed ) {
      unclEnergySum->mex   -= muon.px();
      unclEnergySum->mey   -= muon.py();
      unclEnergySum->sumet -= muon.pt();
    }
  }

//--- add sum of muon corrections to the event
  evt.put(std::move(unclEnergySum));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonMETcorrInputProducer);
