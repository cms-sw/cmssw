#ifndef IPTagPlotter_H
#define IPTagPlotter_H

#include "DQMOffline/RecoB/interface/BaseTagInfoPlotter.h"
#include "DQMOffline/RecoB/interface/TrackIPHistograms.h"
#include "DQMOffline/RecoB/interface/FlavourHistorgrams2D.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/IPTagInfo.h"

template <class Container, class Base>
class IPTagPlotter : public BaseTagInfoPlotter {


 public:
  IPTagPlotter (const std::string & tagName, const EtaPtBin & etaPtBin,
		     const edm::ParameterSet& pSet, const unsigned int& mc, 
		     const bool& wf, DQMStore::IBooker & ibook);

  ~IPTagPlotter () ;

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, const int & jetFlavour);
  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, const double & jec, const int & jetFlavour, const float & w);

  virtual void finalize (DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_);

  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

  reco::TrackBase::TrackQuality highestTrackQual(const reco::Track * track) const;

 private:

  int	nBinEffPur_ ;
  double startEffPur_ ; 
  double endEffPur_ ; 
  unsigned int mcPlots_;
  bool willFinalize_;
  bool makeQualityPlots_;

  TrackIPHistograms<double> * tkcntHistosSig3D[5];
  TrackIPHistograms<double> * tkcntHistosSig2D[5];
  TrackIPHistograms<double> * tkcntHistosErr3D[5];
  TrackIPHistograms<double> * tkcntHistosErr2D[5];
  TrackIPHistograms<double> * tkcntHistosVal3D[5];
  TrackIPHistograms<double> * tkcntHistosVal2D[5];
  TrackIPHistograms<double> * tkcntHistosDecayLengthVal2D[5];
  TrackIPHistograms<double> * tkcntHistosDecayLengthVal3D[5];
  TrackIPHistograms<double> * tkcntHistosJetDistVal2D[5];
  TrackIPHistograms<double> * tkcntHistosJetDistVal3D[5];
  TrackIPHistograms<double> * tkcntHistosJetDistSign2D[5];
  TrackIPHistograms<double> * tkcntHistosJetDistSign3D[5];
  TrackIPHistograms<double> * tkcntHistosTkNChiSqr2D[5];
  TrackIPHistograms<double> * tkcntHistosTkNChiSqr3D[5];
  TrackIPHistograms<double> * tkcntHistosTkPt2D[5];
  TrackIPHistograms<double> * tkcntHistosTkPt3D[5];
  TrackIPHistograms<int> * tkcntHistosTkNHits2D[5];
  TrackIPHistograms<int> * tkcntHistosTkNHits3D[5];
  TrackIPHistograms<int> * tkcntHistosTkNPixelHits2D[5];
  TrackIPHistograms<int> * tkcntHistosTkNPixelHits3D[5];
  FlavourHistograms<int> * trkNbr3D, * trkNbr2D;
  double lowerIPSBound, upperIPSBound,lowerIPBound, upperIPBound,lowerIPEBound, upperIPEBound ;
  int nBinsIPS, nBinsIP, nBinsIPE;
  double minDecayLength, maxDecayLength, minJetDistance, maxJetDistance;

  EffPurFromHistos * effPurFromHistos[4] ;

  TrackIPHistograms<float> * tkcntHistosProb3D[5];
  TrackIPHistograms<float> * tkcntHistosProb2D[5];
  TrackIPHistograms<float> * tkcntHistosTkProbIPneg2D, * tkcntHistosTkProbIPpos2D;
  TrackIPHistograms<float> * tkcntHistosTkProbIPneg3D, * tkcntHistosTkProbIPpos3D;
  TrackIPHistograms<double> * ghostTrackWeightHisto;
  TrackIPHistograms<double> * ghostTrackDistanceValuHisto, * ghostTrackDistanceSignHisto;

  FlavourHistograms<int> * trackQualHisto;
  FlavourHistograms<int> * selectedTrackQualHisto;
  FlavourHistograms2D<double, int> * trackMultVsJetPtHisto;
  FlavourHistograms2D<double, int> * selectedTrackMultVsJetPtHisto;
} ;

#include "DQMOffline/RecoB/interface/IPTagPlotter_cc.h"

#endif
