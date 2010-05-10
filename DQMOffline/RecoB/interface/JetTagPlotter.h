#ifndef JetTagPlotter_H
#define JetTagPlotter_H

#include "TH1F.h"

#include <string>
#include <vector>
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
#include "DQMOffline/RecoB/interface/BaseBTagPlotter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"


class JetTagPlotter : public BaseBTagPlotter {


 public:

  JetTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
		 const edm::ParameterSet& pSet, const bool& mc , const bool& update, const bool& willFinalize);

  virtual ~JetTagPlotter () ;

  void analyzeTag (const reco::JetTag & jetTag, const int & jetFlavour);
  void analyzeTag (const reco::Jet & jet, const float& discriminator, const int& jetFlavour);

  // final computation, plotting, printing .......
  void createPlotsForFinalize();
  void finalize () ;

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

  bool finalized;
  bool mcPlots_;
  bool willFinalize_;

  // for the misid vs. eff plots
  EffPurFromHistos * effPurFromHistos ;

  FlavourHistograms<int> * dJetFlav;

  // track multiplicity in jet
  FlavourHistograms<int> * dJetMultiplicity;
  
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

  // associated parton momentum
  FlavourHistograms<double> * dJetPartonMomentum;

  // associated parton pt
  FlavourHistograms<double> * dJetPartonPt;

  // associated parton eta
  FlavourHistograms<double> * dJetPartonPseudoRapidity;
} ;

#endif
