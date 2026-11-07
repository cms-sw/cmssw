#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <cstdlib>
#include <iostream>

using namespace std;
using namespace RecoBTag;

JetTagPlotter::JetTagPlotter(const std::string& tagName,
                             const EtaPtBin& etaPtBin,
                             const edm::ParameterSet& pSet,
                             unsigned int mc,
                             bool wf,
                             DQMStore::IBooker& ibook,
                             bool doCTagPlots /*=false*/,
                             bool doDifferentialPlots /*=false*/,
                             double discrCut /*=-999.*/,
                             const std::string& kinematicsTag /*=""*/,
                             bool bookKinematics /*=true*/)
    : BaseBTagPlotter(tagName, etaPtBin),
      discrStart_(pSet.getParameter<double>("discriminatorStart")),
      discrEnd_(pSet.getParameter<double>("discriminatorEnd")),
      nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
      startEffPur_(pSet.getParameter<double>("startEffPur")),
      endEffPur_(pSet.getParameter<double>("endEffPur")),
      mcPlots_(mc),
      willFinalize_(wf),
      doCTagPlots_(doCTagPlots),
      doDifferentialPlots_(doDifferentialPlots),
      cutValue_(discrCut),
      kinematicsExtension_(kinematicsTag.empty() ? theExtensionString
                                                 : "_" + kinematicsTag + etaPtBin.getDescriptionString()),
      bookKinematics_(bookKinematics) {
  // to have a shorter name .....
  const std::string& es = theExtensionString;
  const std::string jetTagDir(es.substr(1));
  const std::string& kes = kinematicsExtension_;
  const std::string kinematicsDir(kes.substr(1));

  if (willFinalize_)
    return;

  // Discriminator: again with reasonable binning
  dDiscriminator_ = std::make_unique<FlavourHistograms<double>>(
      "discr" + es, "Discriminator", 102, discrStart_, discrEnd_, false, true, true, "b", jetTagDir, mcPlots_, ibook);
  dDiscriminator_->settitle("Discriminant");

  if (bookKinematics_) {
    if (mcPlots_) {
      // jet flavour: always with ALL (odd MClevel -> next even one). The DQM (MClevel 0) module
      // never books jetFlavour, so its ALL cannot be filled twice
      dJetFlav_ = std::make_unique<FlavourHistograms<int>>("jetFlavour" + kes,
                                                           "Jet Flavour",
                                                           22,
                                                           -0.5,
                                                           21.5,
                                                           false,
                                                           false,
                                                           false,
                                                           "b",
                                                           kinematicsDir,
                                                           mcPlots_ % 2 ? mcPlots_ + 1 : mcPlots_,
                                                           ibook);
    }

    // reconstructed jet transverse momentum: fine binning at low pT, and the full range up to 7 TeV
    dJetRecPt_ = std::make_unique<FlavourHistograms<double>>(
        "jetPt" + kes, "jet pt", 350, 0.0, 350.0, false, false, true, "b", kinematicsDir, mcPlots_, ibook);
    dJetRecPtExtended_ = std::make_unique<FlavourHistograms<double>>(
        "jetPt_extended" + kes, "jet pt", 700, 0.0, 7000.0, false, false, true, "b", kinematicsDir, mcPlots_, ibook);

    // reconstructed jet eta
    dJetRecPseudoRapidity_ = std::make_unique<FlavourHistograms<double>>("jetEta" + kes,
                                                                         "jet eta",
                                                                         20,
                                                                         -etaPtBin.getEtaMax(),
                                                                         etaPtBin.getEtaMax(),
                                                                         false,
                                                                         false,
                                                                         true,
                                                                         "b",
                                                                         kinematicsDir,
                                                                         mcPlots_,
                                                                         ibook);

    // reconstructed jet phi
    dJetRecPhi_ = std::make_unique<FlavourHistograms<double>>(
        "jetPhi" + kes, "jet phi", 20, -M_PI, M_PI, false, false, true, "b", kinematicsDir, mcPlots_, ibook);
  }

  if (doDifferentialPlots_) {
    // jet Phi larger than requested discrimnator cut
    dJetPhiDiscrCut_ = std::make_unique<FlavourHistograms<double>>("jetPhi_diffEff" + es,
                                                                   "Efficiency vs. jet Phi for discriminator above cut",
                                                                   20,
                                                                   -M_PI,
                                                                   M_PI,
                                                                   false,
                                                                   false,
                                                                   true,
                                                                   "b",
                                                                   jetTagDir,
                                                                   mcPlots_,
                                                                   ibook);

    // jet Eta larger than requested discrimnator cut
    dJetPseudoRapidityDiscrCut_ =
        std::make_unique<FlavourHistograms<double>>("jetEta_diffEff" + es,
                                                    "Efficiency vs. jet eta for discriminator above cut",
                                                    20,
                                                    -etaPtBin.getEtaMax(),
                                                    etaPtBin.getEtaMax(),
                                                    false,
                                                    false,
                                                    true,
                                                    "b",
                                                    jetTagDir,
                                                    mcPlots_,
                                                    ibook);

    // jet pT larger than requested discrimnator cut; 50 GeV bins, bin edges aligned with jetPt_extended (10 GeV)
    dJetPtDiscrCut_ = std::make_unique<FlavourHistograms<double>>("jetPt_diffEff" + es,
                                                                  "Efficiency vs. jet pt for discriminator above cut",
                                                                  140,
                                                                  0.0,
                                                                  7000.0,
                                                                  false,
                                                                  false,
                                                                  true,
                                                                  "b",
                                                                  jetTagDir,
                                                                  mcPlots_,
                                                                  ibook);
  }
}

JetTagPlotter::~JetTagPlotter() {}

void JetTagPlotter::epsPlot(const std::string& name) {
  if (!willFinalize_) {
    dJetFlav_->epsPlot(name);
    dDiscriminator_->epsPlot(name);
    dJetRecPt_->epsPlot(name);
    dJetRecPtExtended_->epsPlot(name);
    dJetRecPseudoRapidity_->epsPlot(name);
    dJetRecPhi_->epsPlot(name);
  } else {
    effPurFromHistos_->epsPlot(name);
  }
}

void JetTagPlotter::psPlot(const std::string& name) {
  std::string cName = "JetTagPlots" + theExtensionString;
  setTDRStyle()->cd();
  TCanvas canvas(cName.c_str(), cName.c_str(), 600, 900);
  canvas.UseCurrentStyle();

  canvas.Divide(2, 3);
  canvas.Print((name + cName + ".ps[").c_str());
  if (!willFinalize_) {
    canvas.cd(1);
    dJetFlav_->plot();
    canvas.cd(2);
    canvas.cd(3);
    dDiscriminator_->plot();
    canvas.cd(5);
    dJetRecPt_->plot();
    canvas.cd(6);
    dJetRecPseudoRapidity_->plot();
    canvas.Print((name + cName + ".ps").c_str());
    canvas.Clear();
    canvas.Divide(2, 3);

    canvas.cd(1);
    dJetRecPhi_->plot();
    canvas.cd(2);
    canvas.cd(3);
    canvas.cd(4);
  } else {
    canvas.cd(5);
    effPurFromHistos_->discriminatorNoCutEffic().plot();
    canvas.cd(6);
    effPurFromHistos_->discriminatorCutEfficScan().plot();
    canvas.Print((name + cName + ".ps").c_str());
    canvas.Clear();
    canvas.Divide(2, 3);
    canvas.cd(1);
    effPurFromHistos_->plot();
  }
  canvas.Print((name + cName + ".ps").c_str());
  canvas.Print((name + cName + ".ps]").c_str());
}

void JetTagPlotter::analyzeTag(const reco::Jet& jet, double jec, float discriminator, int jetFlavour, float w /*=1*/) {
  if (edm::isNotFinite(discriminator))
    dDiscriminator_->fill(jetFlavour, -999.0, w);
  else
    dDiscriminator_->fill(jetFlavour, discriminator, w);
  // kinematics are filled only by the plotter that books them
  if (bookKinematics_) {
    if (mcPlots_)
      dJetFlav_->fill(jetFlavour, std::abs(jetFlavour), w);
    dJetRecPt_->fill(jetFlavour, jet.pt() * jec, w);
    dJetRecPtExtended_->fill(jetFlavour, jet.pt() * jec, w);
    dJetRecPseudoRapidity_->fill(jetFlavour, jet.eta(), w);
    dJetRecPhi_->fill(jetFlavour, jet.phi(), w);
  }
  if (doDifferentialPlots_) {
    if (edm::isFinite(discriminator) && discriminator > cutValue_) {
      dJetPhiDiscrCut_->fill(jetFlavour, jet.phi(), w);
      dJetPseudoRapidityDiscrCut_->fill(jetFlavour, jet.eta(), w);
      dJetPtDiscrCut_->fill(jetFlavour, jet.pt() * jec, w);
    }
  }
}

void JetTagPlotter::analyzeTag(const reco::JetTag& jetTag, double jec, int jetFlavour, float w /*=1*/) {
  const auto& discriminator = jetTag.second;
  if (edm::isNotFinite(discriminator))
    dDiscriminator_->fill(jetFlavour, -999.0, w);
  else
    dDiscriminator_->fill(jetFlavour, discriminator, w);
  // kinematics are filled only by the plotter that books them
  if (bookKinematics_) {
    if (mcPlots_)
      dJetFlav_->fill(jetFlavour, std::abs(jetFlavour), w);
    dJetRecPt_->fill(jetFlavour, jetTag.first->pt() * jec, w);
    dJetRecPtExtended_->fill(jetFlavour, jetTag.first->pt() * jec, w);
    dJetRecPseudoRapidity_->fill(jetFlavour, jetTag.first->eta(), w);
    dJetRecPhi_->fill(jetFlavour, jetTag.first->phi(), w);
  }
  if (doDifferentialPlots_) {
    if (edm::isFinite(discriminator) && discriminator > cutValue_) {
      dJetPhiDiscrCut_->fill(jetFlavour, jetTag.first->phi(), w);
      dJetPseudoRapidityDiscrCut_->fill(jetFlavour, jetTag.first->eta(), w);
      dJetPtDiscrCut_->fill(jetFlavour, jetTag.first->pt() * jec, w);
    }
  }
}

void JetTagPlotter::finalize(DQMStore::IBooker& ibook_, DQMStore::IGetter& igetter_) {
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  const std::string& es = theExtensionString;
  const std::string jetTagDir(es.substr(1));
  const std::string& kes = kinematicsExtension_;
  const std::string kinematicsDir(kes.substr(1));
  dDiscriminator_ = std::make_unique<FlavourHistograms<double>>(
      "discr" + es, "Discriminator", 102, discrStart_, discrEnd_, "b", jetTagDir, mcPlots_, igetter_);

  effPurFromHistos_ = std::make_unique<EffPurFromHistos>(
      *dDiscriminator_, jetTagDir, mcPlots_, ibook_, nBinEffPur_, startEffPur_, endEffPur_);
  effPurFromHistos_->doCTagPlots(doCTagPlots_);
  effPurFromHistos_->compute(ibook_);

  // Produce the differentiel efficiency vs. kinematical variables
  if (doDifferentialPlots_) {
    dJetRecPhi_ = std::make_unique<FlavourHistograms<double>>(
        "jetPhi" + kes, "jet phi", 20, -M_PI, M_PI, "b", kinematicsDir, mcPlots_, igetter_);
    dJetPhiDiscrCut_ = std::make_unique<FlavourHistograms<double>>("jetPhi_diffEff" + es,
                                                                   "Efficiency vs. jet Phi for discriminator above cut",
                                                                   20,
                                                                   -M_PI,
                                                                   M_PI,
                                                                   "b",
                                                                   jetTagDir,
                                                                   mcPlots_,
                                                                   igetter_);
    dJetPhiDiscrCut_->divide(*dJetRecPhi_);
    dJetPhiDiscrCut_->setEfficiencyFlag();

    dJetRecPseudoRapidity_ = std::make_unique<FlavourHistograms<double>>("jetEta" + kes,
                                                                         "jet eta",
                                                                         20,
                                                                         -etaPtBin_.getEtaMax(),
                                                                         etaPtBin_.getEtaMax(),
                                                                         "b",
                                                                         kinematicsDir,
                                                                         mcPlots_,
                                                                         igetter_);
    dJetPseudoRapidityDiscrCut_ =
        std::make_unique<FlavourHistograms<double>>("jetEta_diffEff" + es,
                                                    "Efficiency vs. jet eta for discriminator above cut",
                                                    20,
                                                    -etaPtBin_.getEtaMax(),
                                                    etaPtBin_.getEtaMax(),
                                                    "b",
                                                    jetTagDir,
                                                    mcPlots_,
                                                    igetter_);
    dJetPseudoRapidityDiscrCut_->divide(*dJetRecPseudoRapidity_);
    dJetPseudoRapidityDiscrCut_->setEfficiencyFlag();

    dJetRecPtExtended_ = std::make_unique<FlavourHistograms<double>>(
        "jetPt_extended" + kes, "jet pt", 700, 0.0, 7000.0, "b", kinematicsDir, mcPlots_, igetter_);
    dJetPtDiscrCut_ = std::make_unique<FlavourHistograms<double>>("jetPt_diffEff" + es,
                                                                  "Efficiency vs. jet pt for discriminator above cut",
                                                                  140,
                                                                  0.0,
                                                                  7000.0,
                                                                  "b",
                                                                  jetTagDir,
                                                                  mcPlots_,
                                                                  igetter_);
    dJetPtDiscrCut_->divideRebinned(*dJetRecPtExtended_);
    dJetPtDiscrCut_->setEfficiencyFlag();
  }
}
