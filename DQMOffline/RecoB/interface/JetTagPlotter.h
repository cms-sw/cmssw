#ifndef JetTagPlotter_H
#define JetTagPlotter_H

#include "TH1F.h"

#include <string>
#include <vector>
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DQMServices/Core/interface/DQMStore.h"

class JetTagPlotter : public BaseBTagPlotter {
public:
  JetTagPlotter(const std::string& tagName,
                const EtaPtBin& etaPtBin,
                const edm::ParameterSet& pSet,
                unsigned int mc,
                bool willFinalize,
                DQMStore::IBooker& ibook,
                bool doCTagPlots = false,
                bool doDifferentialPlots = false,
                double discrCut = -999.,
                const std::string& kinematicsTag = "",
                bool bookKinematics = true);

  ~JetTagPlotter() override;

  //void analyzeTag (const reco::JetTag & jetTag, const double & jec, const int & jetFlavour);
  void analyzeTag(const reco::JetTag& jetTag, double jec, int jetFlavour, float w = 1);
  //void analyzeTag (const reco::Jet & jet, const double & jec, const float& discriminator, const int& jetFlavour);
  void analyzeTag(const reco::Jet& jet, double jec, float discriminator, int jetFlavour, float w = 1);

  // final computation, plotting, printing .......
  void finalize(DQMStore::IBooker& ibook_, DQMStore::IGetter& igetter_) override;

  // get "2d" histograms for misid. vs. b-eff
  EffPurFromHistos& getEffPurFromHistos() { return *effPurFromHistos_; }

  void epsPlot(const std::string& name) override;
  void psPlot(const std::string& name) override;

  int nBinEffPur() const { return nBinEffPur_; }
  double startEffPur() const { return startEffPur_; }
  double endEffPur() const { return endEffPur_; }

protected:
  // binning and bounds
  // 1) for 'efficiency' versus discriminator cut histos
  double discrStart_;
  double discrEnd_;
  int nBinEffPur_;
  double startEffPur_;
  double endEffPur_;

  unsigned int mcPlots_;
  bool willFinalize_;

  bool doCTagPlots_;

  // Differential plots: efficiency vs. variable for cut on discrimator > cutValue_
  bool doDifferentialPlots_;
  double cutValue_;

  // jet kinematics (jetPt/Eta/Phi/Flavour) are booked in the folder of kinematicsTag
  // (own folder if empty) and only by the plotter with bookKinematics; the diffEff plots divide by them
  std::string kinematicsExtension_;
  bool bookKinematics_;

  // for the misid vs. eff plots
  std::unique_ptr<EffPurFromHistos> effPurFromHistos_;

  std::unique_ptr<FlavourHistograms<int>> dJetFlav_;

  // Discriminator: again with reasonable binning
  std::unique_ptr<FlavourHistograms<double>> dDiscriminator_;

  // reconstructed jet transverse momentum: low pT with fine binning (jetPt) and full range (jetPt_extended)
  std::unique_ptr<FlavourHistograms<double>> dJetRecPt_;
  std::unique_ptr<FlavourHistograms<double>> dJetRecPtExtended_;

  // reconstructed jet eta
  std::unique_ptr<FlavourHistograms<double>> dJetRecPseudoRapidity_;

  // reconstructed jet phi
  std::unique_ptr<FlavourHistograms<double>> dJetRecPhi_;

  // jet Phi larger than requested discrimnator cut
  std::unique_ptr<FlavourHistograms<double>> dJetPhiDiscrCut_;

  // jet Eta larger than requested discrimnator cut
  std::unique_ptr<FlavourHistograms<double>> dJetPseudoRapidityDiscrCut_;

  // jet pT larger than requested discrimnator cut (coarser than jetPt_extended, which it is divided by)
  std::unique_ptr<FlavourHistograms<double>> dJetPtDiscrCut_;
};

#endif
