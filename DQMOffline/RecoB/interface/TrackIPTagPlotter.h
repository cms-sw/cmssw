#ifndef TrackIPTagPlotter_H
#define TrackIPTagPlotter_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackIPTagPlotter : public BaseTagInfoPlotter {


 public:

  TrackIPTagPlotter (const TString & tagName, const EtaPtBin & etaPtBin,
	const edm::ParameterSet& pSet, bool update, bool mc);

  ~TrackIPTagPlotter () ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const int & jetFlavour);

  virtual void finalize ();

  void epsPlot(const TString & name);

  void psPlot(const TString & name);

 private:

  int	nBinEffPur_ ;
  double startEffPur_ ; 
  double endEffPur_ ; 
  bool mcPlots_;

  FlavourHistograms<double> * tkcntHistosSig3D[5];
  FlavourHistograms<double> * tkcntHistosSig2D[5];
  FlavourHistograms<double> * tkcntHistosErr3D[5];
  FlavourHistograms<double> * tkcntHistosErr2D[5];
  FlavourHistograms<double> * tkcntHistosVal3D[5];
  FlavourHistograms<double> * tkcntHistosVal2D[5];
  FlavourHistograms<int> * trkNbr3D, * trkNbr2D;
  double lowerIPSBound, upperIPSBound,lowerIPBound, upperIPBound,lowerIPEBound, upperIPEBound ;

  EffPurFromHistos * effPurFromHistos[4] ;

  FlavourHistograms<float> * tkcntHistosProb3D[5];
  FlavourHistograms<float> * tkcntHistosProb2D[5];
  FlavourHistograms<double> *decayLengthValuHisto, *jetDistanceValuHisto;
  FlavourHistograms<double> *decayLengthSignHisto, *jetDistanceSignHisto;

  bool finalized;
} ;

#endif
