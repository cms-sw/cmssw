#ifndef TrackCountingTagPlotter_H
#define TrackCountingTagPlotter_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams.h"
// #include "RecoBTag/MCTools/interface/JetFlavour.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TrackCountingTagPlotter : public BaseTagInfoPlotter {


 public:

  TrackCountingTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
			   const edm::ParameterSet& pSet, const bool& update, 
			   const unsigned int& mc, const bool& willfinalize, DQMStore::IBooker & ibook);

  ~TrackCountingTagPlotter () ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, const int & jetFlavour);
  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, const int & jetFlavour, const float & w);

  virtual void finalize ();
  virtual void createPlotsForFinalize (DQMStore::IBooker & ibook);

  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

 private:
  unsigned int mcPlots_;
  int	nBinEffPur_ ;
  double startEffPur_ ; 
  double endEffPur_ ; 

  bool willFinalize_;

  FlavourHistograms<double> * tkcntHistosSig3D[5];
  FlavourHistograms<double> * tkcntHistosSig2D[5];
  FlavourHistograms<int> * trkNbr3D, * trkNbr2D;
  double lowerIPSBound, upperIPSBound;

  EffPurFromHistos * effPurFromHistos[4] ;
  bool finalized;
  DQMStore::IBooker & ibook_;
} ;

#endif
