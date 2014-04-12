#ifndef TrackProbabilityTagPlotter_H
#define TrackProbabilityTagPlotter_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackProbabilityTagPlotter : public BaseTagInfoPlotter {


 public:

  TrackProbabilityTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
	const edm::ParameterSet& pSet, const bool& update, const unsigned int& mc, const bool& wf);

  ~TrackProbabilityTagPlotter () ;

  void analyzeTag (const reco::BaseTagInfo * tagInfo, const int & jetFlavour);

  void analyzeTag (const reco::BaseTagInfo * tagInfo, const int & jetFlavour, const float & w);

  virtual void createPlotsForFinalize ();
  virtual void finalize ();

  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

 private:

  int	nBinEffPur_ ;
  double startEffPur_ ; 
  double endEffPur_ ; 
  FlavourHistograms<double> * tkcntHistosSig3D[5];
  FlavourHistograms<double> * tkcntHistosSig2D[5];
  EffPurFromHistos * effPurFromHistos[4] ;
  bool finalized;
  unsigned int mcPlots_;  
  bool willFinalize_;

} ;

#endif
