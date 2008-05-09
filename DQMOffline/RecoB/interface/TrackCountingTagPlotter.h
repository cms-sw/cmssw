#ifndef TrackCountingTagPlotter_H
#define TrackCountingTagPlotter_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackCountingTagPlotter : public BaseTagInfoPlotter {


 public:

  TrackCountingTagPlotter (const TString & tagName, const EtaPtBin & etaPtBin,
	const edm::ParameterSet& pSet, bool update = false);

  ~TrackCountingTagPlotter () ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const int & jetFlavour);

  virtual void finalize ();

  void epsPlot(const TString & name);

  void psPlot(const TString & name);

 private:

  int	nBinEffPur_ ;
  double startEffPur_ ; 
  double endEffPur_ ; 

  FlavourHistograms<double> * tkcntHistosSig3D[5];
  FlavourHistograms<double> * tkcntHistosSig2D[5];
  FlavourHistograms<int> * trkNbr3D, * trkNbr2D;
  double lowerIPSBound, upperIPSBound;

  EffPurFromHistos * effPurFromHistos[4] ;
  bool finalized;
} ;

#endif
