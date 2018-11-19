/*
 * RecoTauPlotDiscriminator
 *
 * Plot the output of a PFTauDiscriminator using TFileService.
 *
 * Author: Evan K. Friis (UC Davis)
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>

class RecoTauPlotDiscriminator : public edm::EDAnalyzer {
  public:
    RecoTauPlotDiscriminator(const edm::ParameterSet& pset);
    ~RecoTauPlotDiscriminator() override {}
    void analyze(const edm::Event &evt, const edm::EventSetup &es) override;
  private:
    edm::InputTag src_;
    typedef std::map<std::string, TH1*> HistoMap;
    typedef std::map<std::string, HistoMap> DiscMap;
    typedef std::vector<edm::InputTag> VInputTag;
    VInputTag discs_;
    bool plotPU_;
    double pileupPtCut_;
    edm::InputTag pileupInfoSrc_;
    edm::InputTag pileupVerticesSrc_;
    DiscMap histos_;
};

RecoTauPlotDiscriminator::RecoTauPlotDiscriminator(const edm::ParameterSet &pset)
  :src_(pset.getParameter<edm::InputTag>("src")) {
  uint32_t nbins = pset.getParameter<uint32_t>("nbins");
  double min = pset.getParameter<double>("min");
  double max = pset.getParameter<double>("max");
  edm::Service<TFileService> fs;
  // Get the discriminators
  discs_ = pset.getParameter<std::vector<edm::InputTag> >("discriminators");

  plotPU_ = pset.getParameter<bool>("plotPU");
  if (plotPU_) {
    pileupPtCut_ = pset.getParameter<double>("pileupTauPtCut");
    pileupInfoSrc_ = pset.getParameter<edm::InputTag>("pileupInfo");
    pileupVerticesSrc_ = pset.getParameter<edm::InputTag>("pileupVertices");
  }

  for(auto const& tag : discs_) {
    HistoMap discMap;
    discMap["plain"] =
        fs->make<TH1F>(tag.label().c_str(), tag.label().c_str(),
                       nbins, min, max);

    // Make correlation plots w.r.t tau pt
    std::string vs_pt_name = tag.label()+"_pt";
    discMap["vs_pt"] =
        fs->make<TH2F>(vs_pt_name.c_str(), vs_pt_name.c_str(),
                       nbins, min, max, 100, 0, 200);

    // W.r.t. jet pt
    std::string vs_jetpt_name = tag.label()+"_jetPt";
    discMap["vs_jetPt"] =
        fs->make<TH2F>(vs_jetpt_name.c_str(), vs_jetpt_name.c_str(),
                       nbins, min, max, 100, 0, 200);

    // W.r.t. embedded pt in alternat lorentz vector (used to hold gen tau pt)
    std::string vs_embedpt_name = tag.label()+"_embedPt";
    discMap["vs_embedPt"] =
        fs->make<TH2F>(vs_embedpt_name.c_str(), vs_embedpt_name.c_str(),
                       nbins, min, max, 100, 0, 200);

    // 3D histogram with tau pt & jet pt
    std::string vs_pt_jetPt_name = tag.label()+"_pt_jetPt";
    discMap["vs_pt_jetPt"] =
        fs->make<TH3F>(vs_pt_jetPt_name.c_str(), vs_pt_jetPt_name.c_str(),
                       nbins, min, max, 100, 0, 200, 100, 0, 200);

    std::string vs_pt_embedPt_name = tag.label()+"_pt_embedPt";
    discMap["vs_pt_embedPt"] =
        fs->make<TH3F>(vs_pt_embedPt_name.c_str(), vs_pt_embedPt_name.c_str(),
                       nbins, min, max, 100, 0, 200, 100, 0, 200);


    std::string vs_eta_name = tag.label()+"_eta";
    discMap["vs_eta"] =
        fs->make<TH2F>(vs_eta_name.c_str(), vs_eta_name.c_str(),
                       nbins, min, max, 100, -2.5, 2.5);

    std::string vs_dm_name = tag.label()+"_dm";
    discMap["vs_dm"] =
        fs->make<TH2F>(vs_dm_name.c_str(), vs_dm_name.c_str(),
                       nbins, min, max, 15, -0.5, 14.5);

    if (plotPU_) {
      std::string vs_truePU_name = tag.label()+"_truePU";
      discMap["vs_truePU"] = fs->make<TH2F>(vs_truePU_name.c_str(),
          vs_truePU_name.c_str(), nbins, min, max, 15, -0.5, 14.5);
      std::string vs_recoPU_name = tag.label()+"_recoPU";
      discMap["vs_recoPU"] = fs->make<TH2F>(vs_recoPU_name.c_str(),
          vs_recoPU_name.c_str(), nbins, min, max, 15, -0.5, 14.5);
    }

    histos_[tag.label()] = discMap;
  }
}

void
RecoTauPlotDiscriminator::analyze(const edm::Event &evt,
                                  const edm::EventSetup &es) {
  // Get the input collection to clean
  edm::Handle<reco::CandidateView> input;
  evt.getByLabel(src_, input);

  // Cast the input candidates to Refs to real taus
  reco::PFTauRefVector inputRefs =
      reco::tau::castView<reco::PFTauRefVector>(input);

  edm::Handle<PileupSummaryInfo> puInfo;
  edm::Handle<reco::VertexCollection> puVertices;
  if (plotPU_) {
    evt.getByLabel(pileupInfoSrc_, puInfo);
    evt.getByLabel(pileupVerticesSrc_, puVertices);
  }

  // Plot the discriminator output for each of our taus
  for(auto const& tau : inputRefs) {
    // Plot each discriminator
    for(auto const& tag : discs_) {
      edm::Handle<reco::PFTauDiscriminator> discHandle;
      evt.getByLabel(tag, discHandle);
      //const HistoMap &discHistos = disc.second;
      double result = (*discHandle)[tau];
      HistoMap& mymap = histos_[tag.label()];
      mymap["plain"]->Fill(result);
      mymap["vs_pt"]->Fill(result, tau->pt());
      mymap["vs_jetPt"]->Fill(result, tau->jetRef()->pt());
      mymap["vs_embedPt"]->Fill(result, tau->alternatLorentzVect().pt());
      dynamic_cast<TH3F*>(mymap["vs_pt_jetPt"])->Fill(
          result, tau->pt(), tau->jetRef()->pt());
      dynamic_cast<TH3F*>(mymap["vs_pt_embedPt"])->Fill(
          result, tau->pt(), tau->alternatLorentzVect().pt());
      mymap["vs_eta"]->Fill(result, tau->eta());
      mymap["vs_dm"]->Fill(result, tau->decayMode());
      if (plotPU_ && tau->pt() > pileupPtCut_) {
        if (puInfo.isValid())
          mymap["vs_truePU"]->Fill(result, puInfo->getPU_NumInteractions());
        mymap["vs_recoPU"]->Fill(result, puVertices->size());
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPlotDiscriminator);
