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
		     const edm::ParameterSet& pSet, unsigned int mc, 
		     bool wf, DQMStore::IBooker & ibook);

  ~IPTagPlotter ();

  void analyzeTag (const reco::BaseTagInfo * baseTagInfo, double jec, int jetFlavour, float w=1);

  virtual void finalize (DQMStore::IBooker & ibook_, DQMStore::IGetter & igetter_);

  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

  reco::TrackBase::TrackQuality highestTrackQual(const reco::Track * track) const;

 private:

  int nBinEffPur_;
  double startEffPur_; 
  double endEffPur_; 
  unsigned int mcPlots_;
  bool willFinalize_;
  bool makeQualityPlots_;

  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosSig3D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosSig2D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosErr3D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosErr2D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosVal3D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosVal2D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosDecayLengthVal2D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosDecayLengthVal3D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosJetDistVal2D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosJetDistVal3D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosJetDistSign2D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosJetDistSign3D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosTkNChiSqr2D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosTkNChiSqr3D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosTkPt2D;
  std::vector< std::unique_ptr<TrackIPHistograms<double>> > tkcntHistosTkPt3D;
  std::vector< std::unique_ptr<TrackIPHistograms<int>> >    tkcntHistosTkNHits2D;
  std::vector< std::unique_ptr<TrackIPHistograms<int>> >    tkcntHistosTkNHits3D;
  std::vector< std::unique_ptr<TrackIPHistograms<int>> >    tkcntHistosTkNPixelHits2D;
  std::vector< std::unique_ptr<TrackIPHistograms<int>> >    tkcntHistosTkNPixelHits3D;
  
  std::unique_ptr<FlavourHistograms<int>> trkNbr3D, trkNbr2D;
  
  double lowerIPSBound, upperIPSBound,lowerIPBound, upperIPBound,lowerIPEBound, upperIPEBound;
  int nBinsIPS, nBinsIP, nBinsIPE;
  double minDecayLength, maxDecayLength, minJetDistance, maxJetDistance;

  std::vector< std::unique_ptr<EffPurFromHistos> > effPurFromHistos;

  std::vector< std::unique_ptr<TrackIPHistograms<float>> > tkcntHistosProb3D;
  std::vector< std::unique_ptr<TrackIPHistograms<float>> > tkcntHistosProb2D;
  
  std::unique_ptr<TrackIPHistograms<float>> tkcntHistosTkProbIPneg2D, tkcntHistosTkProbIPpos2D;
  std::unique_ptr<TrackIPHistograms<float>> tkcntHistosTkProbIPneg3D, tkcntHistosTkProbIPpos3D;
  std::unique_ptr<TrackIPHistograms<double>> ghostTrackWeightHisto;
  std::unique_ptr<TrackIPHistograms<double>> ghostTrackDistanceValuHisto, ghostTrackDistanceSignHisto;

  std::unique_ptr<FlavourHistograms<int>> trackQualHisto;
  std::unique_ptr<FlavourHistograms<int>> selectedTrackQualHisto;
  std::unique_ptr<FlavourHistograms2D<double, int>> trackMultVsJetPtHisto;
  std::unique_ptr<FlavourHistograms2D<double, int>> selectedTrackMultVsJetPtHisto;
};

#include "DQMOffline/RecoB/interface/IPTagPlotter_cc.h"

#endif
