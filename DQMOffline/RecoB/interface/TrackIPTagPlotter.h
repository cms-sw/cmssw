#ifndef TrackIPTagPlotter_H
#define TrackIPTagPlotter_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams2D.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackIPTagPlotter : public BaseTagInfoPlotter {


 public:

  TrackIPTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
	const edm::ParameterSet& pSet, const bool& update, const bool& mc, const bool& wf);

  ~TrackIPTagPlotter () ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const int & jetFlavour);

  virtual void createPlotsForFinalize ();
  virtual void finalize ();

  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

  int highestTrackQual(const reco::Track* track);

 private:

  int	nBinEffPur_ ;
  double startEffPur_ ; 
  double endEffPur_ ; 
  bool mcPlots_;
  bool willFinalize_;

  FlavourHistograms<double> * tkcntHistosSig3D[5];
  FlavourHistograms<double> * tkcntHistosSig2D[5];
  FlavourHistograms<double> * tkcntHistosErr3D[5];
  FlavourHistograms<double> * tkcntHistosErr2D[5];
  FlavourHistograms<double> * tkcntHistosVal3D[5];
  FlavourHistograms<double> * tkcntHistosVal2D[5];
  FlavourHistograms<double> * tkcntHistosDecayLengthVal2D[5];
  FlavourHistograms<double> * tkcntHistosDecayLengthVal3D[5];
  FlavourHistograms<double> * tkcntHistosJetDistVal2D[5];
  FlavourHistograms<double> * tkcntHistosJetDistVal3D[5];
  FlavourHistograms<double> * tkcntHistosJetDistSign2D[5];
  FlavourHistograms<double> * tkcntHistosJetDistSign3D[5];
  FlavourHistograms<double> * tkcntHistosTkNChiSqr2D[5];
  FlavourHistograms<double> * tkcntHistosTkNChiSqr3D[5];
  FlavourHistograms<double> * tkcntHistosTkPt2D[5];
  FlavourHistograms<double> * tkcntHistosTkPt3D[5];
  FlavourHistograms<int> * tkcntHistosTkNHits2D[5];
  FlavourHistograms<int> * tkcntHistosTkNHits3D[5];
  FlavourHistograms<int> * trkNbr3D, * trkNbr2D;
  double lowerIPSBound, upperIPSBound,lowerIPBound, upperIPBound,lowerIPEBound, upperIPEBound ;

  EffPurFromHistos * effPurFromHistos[4] ;

  FlavourHistograms<float> * tkcntHistosProb3D[5];
  FlavourHistograms<float> * tkcntHistosProb2D[5];
  FlavourHistograms<double> *ghostTrackWeightHisto;
  FlavourHistograms<double> *ghostTrackDistanceValuHisto, *ghostTrackDistanceSignHisto;

  FlavourHistograms<int> * trackQualHisto;
  FlavourHistograms<int> * selectedTrackQualHisto;
  FlavourHistograms2D<double, int> * trackMultVsJetPtHisto;
  FlavourHistograms2D<double, int> * selectedTrackMultVsJetPtHisto;

  bool finalized;
} ;

#endif
