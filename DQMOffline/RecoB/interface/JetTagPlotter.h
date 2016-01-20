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

  JetTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
		 const edm::ParameterSet& pSet, const unsigned int& mc , 
		 const bool& willFinalize, DQMStore::IBooker & ibook, const bool & doCTagPlots = false);

  virtual ~JetTagPlotter () ;

  void analyzeTag (); //added to fill the jet multiplicity on data 
  void analyzeTag (const float & w); //added to fill the jet multiplicity on mc 
  void analyzeTag (const reco::JetTag & jetTag, const double & jec, const int & jetFlavour);
  void analyzeTag (const reco::JetTag & jetTag, const double & jec, const int & jetFlavour, const float & w);
  void analyzeTag (const reco::Jet & jet, const double & jec, const float& discriminator, const int& jetFlavour);
  void analyzeTag (const reco::Jet & jet, const double & jec, const float& discriminator, const int& jetFlavour, const float & w);

  // final computation, plotting, printing .......
  void finalize (DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_) ;

  // get "2d" histograms for misid. vs. b-eff
  EffPurFromHistos * getEffPurFromHistos () { return effPurFromHistos ; }

  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

  int nBinEffPur() const {return nBinEffPur_;}
  double startEffPur() const {return startEffPur_;}
  double endEffPur() const {return endEffPur_;}

 protected:

  // binning and bounds
  // 1) for 'efficiency' versus discriminator cut histos
  int    discrBins  ;
  double discrStart_ ;
  double discrEnd_   ;
  int	nBinEffPur_ ;
  double startEffPur_ ; 
  double endEffPur_ ; 

  unsigned int mcPlots_;
  bool willFinalize_;

  bool doCTagPlots_;

  int *nJets;
  // jet multiplicity
  FlavourHistograms<int> * JetMultiplicity;

  // for the misid vs. eff plots
  EffPurFromHistos * effPurFromHistos ;

  FlavourHistograms<int> * dJetFlav;
  
  // Discriminator: again with reasonable binning
  FlavourHistograms<double> * dDiscriminator;
  
  // reconstructed jet momentum
  FlavourHistograms<double> * dJetRecMomentum;

  // reconstructed jet transverse momentum
  FlavourHistograms<double> * dJetRecPt;

  // reconstructed jet eta
  FlavourHistograms<double> * dJetRecPseudoRapidity;

  // reconstructed jet phi
  FlavourHistograms<double> * dJetRecPhi;

} ;

#endif
