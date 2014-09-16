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
		       const edm::ParameterSet& pSet, const unsigned int& mc, 
		       const bool& update, DQMStore::IBooker & ibook);
  
  ~SoftLeptonTagPlotter( void ) ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, const int & jetFlavour);
  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, const int & jetFlavour, const float & w);

  virtual void finalize() {}


  void psPlot( const std::string & name );
  void epsPlot( const std::string & name );

private:

  unsigned int mcPlots_;

  // keep plots for up to 3 leptons per jet
  static const int s_leptons = 2;
  FlavourHistograms<double> * m_leptonId[s_leptons];   // lepton identification discriminant
  FlavourHistograms<double> * m_leptonPt[s_leptons];   // lepton transverse momentum
  
  FlavourHistograms<double> * m_sip2d[s_leptons];      // 2D signed inpact parameter
  FlavourHistograms<double> * m_sip3d[s_leptons];      // 3D signed inpact parameter
  FlavourHistograms<double> * m_ptRel[s_leptons];      // transverse momentum wrt. jet axis
  FlavourHistograms<double> * m_p0Par[s_leptons];      // parallel momentum wrt. jet axis in the B rest frame
  FlavourHistograms<double> * m_etaRel[s_leptons];     // (pseudo)rapidity along jet axis
  FlavourHistograms<double> * m_deltaR[s_leptons];     // (pseudo)angular distance to jet axis
  FlavourHistograms<double> * m_ratio[s_leptons];      // momentum over jet energy
  FlavourHistograms<double> * m_ratioRel[s_leptons];   // momentum parallel to jet axis over jet energy
  
};

#endif // RecoBTag_Analysis_SoftLeptonTagPlotter_h
