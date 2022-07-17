
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"

/** \class MiniAODSVAnalyzer
 *
 *  Secondary Vertex Analyzer to run on MiniAOD
 *
 */

class MiniAODSVAnalyzer : public DQMEDAnalyzer {
public:
  explicit MiniAODSVAnalyzer(const edm::ParameterSet& pSet);
  ~MiniAODSVAnalyzer() override = default;

  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  const edm::EDGetTokenT<std::vector<pat::Jet>> jetToken_;
  const std::string svTagInfo_;
  const double ptMin_;
  const double etaMax_;

  MonitorElement* n_sv_;

  MonitorElement* sv_mass_;
  MonitorElement* sv_pt_;
  MonitorElement* sv_ntracks_;
  MonitorElement* sv_chi2norm_;
  MonitorElement* sv_chi2prob_;

  // relation to jet
  MonitorElement* sv_ptrel_;
  MonitorElement* sv_energyratio_;
  MonitorElement* sv_deltaR_;

  MonitorElement* sv_dxy_;
  MonitorElement* sv_dxysig_;
  MonitorElement* sv_d3d_;
  MonitorElement* sv_d3dsig_;
};

MiniAODSVAnalyzer::MiniAODSVAnalyzer(const edm::ParameterSet& pSet)
    : jetToken_(consumes<std::vector<pat::Jet>>(pSet.getParameter<edm::InputTag>("JetTag"))),
      svTagInfo_(pSet.getParameter<std::string>("svTagInfo")),
      ptMin_(pSet.getParameter<double>("ptMin")),
      etaMax_(pSet.getParameter<double>("etaMax")) {}

void MiniAODSVAnalyzer::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& es) {
  ibook.setCurrentFolder("Btag/SV");

  n_sv_ = ibook.book1D("n_sv", "number of SV in jet", 5, 0, 5);
  n_sv_->setAxisTitle("number of SV in jet");

  sv_mass_ = ibook.book1D("sv_mass", "SV mass", 30, 0., 6.);
  sv_mass_->setAxisTitle("SV mass");

  sv_pt_ = ibook.book1D("sv_pt", "SV transverse momentum", 40, 0., 120.);
  sv_pt_->setAxisTitle("SV pt");

  sv_ntracks_ = ibook.book1D("sv_ntracks", "SV number of daugthers", 10, 0, 10);
  sv_ntracks_->setAxisTitle("number of tracks at SV");

  sv_chi2norm_ = ibook.book1D("sv_chi2norm", "normalized Chi2 of vertex", 30, 0, 15);
  sv_chi2norm_->setAxisTitle("normalized Chi2 of SV");

  sv_chi2prob_ = ibook.book1D("sv_chi2prob", "Chi2 probability of vertex", 20, 0., 1.);
  sv_chi2prob_->setAxisTitle("Chi2 probability of SV");

  sv_ptrel_ = ibook.book1D("sv_ptrel", "SV jet transverse momentum ratio", 25, 0., 1.);
  sv_ptrel_->setAxisTitle("pt(SV)/pt(jet)");

  sv_energyratio_ = ibook.book1D("sv_energyratio", "SV jet energy ratio", 25, 0., 1.);
  sv_energyratio_->setAxisTitle("E(SV)/E(jet)");

  sv_deltaR_ = ibook.book1D("sv_deltaR", "SV jet deltaR", 40, 0., 0.4);
  sv_deltaR_->setAxisTitle("deltaR(jet, SV)");

  sv_dxy_ = ibook.book1D("sv_dxy", "2D flight distance", 40, 0., 8.);
  sv_dxy_->setAxisTitle("dxy");

  sv_dxysig_ = ibook.book1D("sv_dxysig", "2D flight distance significance", 25, 0., 250.);
  sv_dxysig_->setAxisTitle("dxy significance");

  sv_d3d_ = ibook.book1D("sv_d3d", "3D flight distance", 40, 0., 8.);
  sv_d3d_->setAxisTitle("d3d");

  sv_d3dsig_ = ibook.book1D("sv_d3dsig", "3D flight distance significance", 25, 0., 250.);
  sv_d3dsig_->setAxisTitle("d3d significance");
}

void MiniAODSVAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<pat::Jet>> jetCollection;
  iEvent.getByToken(jetToken_, jetCollection);

  // Loop over the pat::Jets
  for (std::vector<pat::Jet>::const_iterator jet = jetCollection->begin(); jet != jetCollection->end(); ++jet) {
    // jet selection
    if (jet->hasTagInfo(svTagInfo_) && jet->pt() > ptMin_ && std::abs(jet->eta()) < etaMax_) {
      const reco::CandSecondaryVertexTagInfo* taginfo =
          static_cast<const reco::CandSecondaryVertexTagInfo*>(jet->tagInfo(svTagInfo_));
      n_sv_->Fill(taginfo->nVertices());

      // loop secondary vertices
      for (unsigned int i = 0; i < taginfo->nVertices(); i++) {
        const reco::VertexCompositePtrCandidate& sv = taginfo->secondaryVertex(i);

        sv_mass_->Fill(sv.mass());
        sv_pt_->Fill(sv.pt());
        sv_ntracks_->Fill(sv.numberOfDaughters());
        sv_chi2norm_->Fill(sv.vertexNormalizedChi2());
        sv_chi2prob_->Fill(ChiSquaredProbability(sv.vertexChi2(), sv.vertexNdof()));

        sv_ptrel_->Fill(sv.pt() / jet->pt());
        sv_energyratio_->Fill(sv.energy() / jet->energy());
        sv_deltaR_->Fill(reco::deltaR(sv, jet->momentum()));

        sv_dxy_->Fill(taginfo->flightDistance(i, 2).value());
        sv_dxysig_->Fill(taginfo->flightDistance(i, 2).significance());
        sv_d3d_->Fill(taginfo->flightDistance(i, 3).value());
        sv_d3dsig_->Fill(taginfo->flightDistance(i, 3).significance());
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MiniAODSVAnalyzer);
