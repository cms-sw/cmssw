#ifndef RecoBTag_Analysis_SoftLeptonTagPlotter_h
#define RecoBTag_Analysis_SoftLeptonTagPlotter_h

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SoftLeptonTagPlotter : public BaseTagInfoPlotter {
public:

  SoftLeptonTagPlotter(const std::string & tagName, const EtaPtBin & etaPtBin,
		       const edm::ParameterSet& pSet, unsigned int mc, 
		       bool willFinalize, DQMStore::IBooker & ibook);
  
  ~SoftLeptonTagPlotter(void) ;

  void analyzeTag(const reco::BaseTagInfo * baseTagInfo, double jec, int jetFlavour, float w/*=1*/);

  virtual void finalize(DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_) {}

  void psPlot(const std::string & name);
  void epsPlot(const std::string & name);

private:

  unsigned int mcPlots_;
  bool willFinalize_;

  // keep plots for up to 2 leptons per jet
  static const int s_leptons = 2;
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_leptonId;   // lepton identification discriminant
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_leptonPt;   // lepton transverse momentum
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_sip2dsig;      // 2D signed inpact parameter significance
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_sip3dsig;      // 3D signed inpact parameter significance
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_sip2d;      // 2D signed inpact parameter
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_sip3d;      // 3D signed inpact parameter
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_ptRel;      // transverse momentum wrt. jet axis
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_p0Par;      // parallel momentum wrt. jet axis in the B rest frame
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_etaRel;     // (pseudo)rapidity along jet axis
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_deltaR;     // (pseudo)angular distance to jet axis
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_ratio;      // momentum over jet energy
  std::vector< std::unique_ptr<FlavourHistograms<double>> > m_ratioRel;   // momentum parallel to jet axis over jet energy
  
};

#endif // RecoBTag_Analysis_SoftLeptonTagPlotter_h
