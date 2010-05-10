#include <cstddef>
#include <string>

#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQMOffline/RecoB/interface/TrackIPTagPlotter.h"

TrackIPTagPlotter::TrackIPTagPlotter(const std::string & tagName,
	const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, const bool& update, const bool& mc, const bool& wf) :
  BaseTagInfoPlotter(tagName, etaPtBin),
  nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
  startEffPur_(pSet.getParameter<double>("startEffPur")),
  endEffPur_(pSet.getParameter<double>("endEffPur")),
  mcPlots_(mc), willFinalize_(wf),lowerIPSBound(-35.0), upperIPSBound(35.0),
  lowerIPBound(-0.1), upperIPBound(0.1), lowerIPEBound(0.0), upperIPEBound(0.4), finalized(false)
{
  const std::string trackIPDir("TrackIPPlots" + theExtensionString);

  trkNbr3D = new FlavourHistograms<int>
	("selTrksNbr_3D" + theExtensionString, "Number of selected tracks for 3D IPS", 31, -0.5, 30.5,
	false, true, true, "b", update,trackIPDir ,mc);

  trkNbr2D = new FlavourHistograms<int>
	("selTrksNbr_2D" + theExtensionString, "Number of selected tracks for 2D IPS", 31, -0.5, 30.5,
	false, true, true, "b", update,trackIPDir ,mc );

  lowerIPSBound = -35.0;
  upperIPSBound = +35.0;

  lowerIPBound = -0.1;
  upperIPBound = 0.1;

  lowerIPEBound = 0;
  upperIPEBound = 0.04;

  // IP significance
  // 3D
  tkcntHistosSig3D[4] = new FlavourHistograms<double>
       ("ips_3D" + theExtensionString, "3D IP significance",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosSig3D[0] = new FlavourHistograms<double>
       ("ips1_3D" + theExtensionString, "3D IP significance 1.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir,mc) ;

  tkcntHistosSig3D[1] = new FlavourHistograms<double>
       ("ips2_3D" + theExtensionString, "3D IP significance 2.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosSig3D[2] = new FlavourHistograms<double>
       ("ips3_3D" + theExtensionString, "3D IP significance 3.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosSig3D[3] = new FlavourHistograms<double>
       ("ips4_3D" + theExtensionString, "3D IP significance 4.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  //2D
  tkcntHistosSig2D[4] = new FlavourHistograms<double>
       ("ips_2D" + theExtensionString, "2D IP significance",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosSig2D[0] = new FlavourHistograms<double>
       ("ips1_2D" + theExtensionString, "2D IP significance 1.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosSig2D[1] = new FlavourHistograms<double>
       ("ips2_2D" + theExtensionString, "2D IP significance 2.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosSig2D[2] = new FlavourHistograms<double>
       ("ips3_2D" + theExtensionString, "2D IP significance 3.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosSig2D[3] = new FlavourHistograms<double>
       ("ips4" + theExtensionString, "2D IP significance 4.trk",
	100, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc) ;

  // IP value
  //3D
  tkcntHistosVal3D[4] = new FlavourHistograms<double>
       ("ip_3D" + theExtensionString, "3D IP value",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosVal3D[0] = new FlavourHistograms<double>
       ("ip1_3D" + theExtensionString, "3D IP value 1.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosVal3D[1] = new FlavourHistograms<double>
       ("ip2_3D" + theExtensionString, "3D IP value 2.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosVal3D[2] = new FlavourHistograms<double>
       ("ip3_3D" + theExtensionString, "3D IP value 3.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosVal3D[3] = new FlavourHistograms<double>
       ("ip4_3D" + theExtensionString, "3D IP value 4.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  //2D
  tkcntHistosVal2D[4] = new FlavourHistograms<double>
       ("ip_2D" + theExtensionString, "2D IP value",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosVal2D[0] = new FlavourHistograms<double>
       ("ip1_2D" + theExtensionString, "2D IP value 1.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosVal2D[1] = new FlavourHistograms<double>
       ("ip2_2D" + theExtensionString, "2D IP value 2.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosVal2D[2] = new FlavourHistograms<double>
       ("ip3_2D" + theExtensionString, "2D IP value 3.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosVal2D[3] = new FlavourHistograms<double>
       ("ip4" + theExtensionString, "2D IP value 4.trk",
	100, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc) ;


  // IP error
  // 3D
  tkcntHistosErr3D[4] = new FlavourHistograms<double>
       ("ipe_3D" + theExtensionString, "3D IP error",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosErr3D[0] = new FlavourHistograms<double>
       ("ipe1_3D" + theExtensionString, "3D IP error 1.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosErr3D[1] = new FlavourHistograms<double>
       ("ipe2_3D" + theExtensionString, "3D IP error 2.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosErr3D[2] = new FlavourHistograms<double>
       ("ipe3_3D" + theExtensionString, "3D IP error 3.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosErr3D[3] = new FlavourHistograms<double>
       ("ipe4_3D" + theExtensionString, "3D IP error 4.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  //2D
  tkcntHistosErr2D[4] = new FlavourHistograms<double>
       ("ipe_2D" + theExtensionString, "2D IP error",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosErr2D[0] = new FlavourHistograms<double>
       ("ipe1_2D" + theExtensionString, "2D IP error 1.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosErr2D[1] = new FlavourHistograms<double>
       ("ipe2_2D" + theExtensionString, "2D IP error 2.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosErr2D[2] = new FlavourHistograms<double>
       ("ipe3_2D" + theExtensionString, "2D IP error 3.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosErr2D[3] = new FlavourHistograms<double>
       ("ipe4_2D" + theExtensionString, "2D IP error 4.trk",
	100, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc) ;

  // decay length
  tkcntHistosDecayLengthVal2D[4] = new FlavourHistograms<double>
       ("decLen_2D" + theExtensionString, "Decay Length 2D",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal2D[0] = new FlavourHistograms<double>
       ("decLen1_2D" + theExtensionString, "2D Decay Length 1.trk",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal2D[1] = new FlavourHistograms<double>
       ("decLen2_2D" + theExtensionString, "2D Decay Length 2.trk",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal2D[2] = new FlavourHistograms<double>
       ("decLen3_2D" + theExtensionString, "2D Decay Length 3.trk",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal2D[3] = new FlavourHistograms<double>
       ("decLen4_2D" + theExtensionString, "2D Decay Length 4.trk",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal3D[4] = new FlavourHistograms<double>
       ("decLen_3D" + theExtensionString, "3D Decay Length",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal3D[0] = new FlavourHistograms<double>
       ("decLen1_3D" + theExtensionString, "3D Decay Length 1.trk",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal3D[1] = new FlavourHistograms<double>
       ("decLen2_3D" + theExtensionString, "3D Decay Length 2.trk",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal3D[2] = new FlavourHistograms<double>
       ("decLen3_3D" + theExtensionString, "3D Decay Length 3.trk",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosDecayLengthVal3D[3] = new FlavourHistograms<double>
       ("decLen4_3D" + theExtensionString, "3D Decay Length 4.trk",
	50, -5.0, 5.0, false, true, true, "b", update,trackIPDir, mc);

  // jet distance
  tkcntHistosJetDistVal2D[4] = new FlavourHistograms<double>
       ("jetDist_2D" + theExtensionString, "JetDistance 2D",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal2D[0] = new FlavourHistograms<double>
       ("jetDist1_2D" + theExtensionString, "JetDistance 2D 1.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal2D[1] = new FlavourHistograms<double>
       ("jetDist2_2D" + theExtensionString, "JetDistance 2D 2.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal2D[2] = new FlavourHistograms<double>
       ("jetDist3_2D" + theExtensionString, "JetDistance 2D 3.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal2D[3] = new FlavourHistograms<double>
       ("jetDist4_2D" + theExtensionString, "JetDistance 2D 4.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal3D[4] = new FlavourHistograms<double>
       ("jetDist_3D" + theExtensionString, "JetDistance 3D",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal3D[0] = new FlavourHistograms<double>
       ("jetDist1_3D" + theExtensionString, "JetDistance 3D 1.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal3D[1] = new FlavourHistograms<double>
       ("jetDist2_3D" + theExtensionString, "JetDistance 3D 2.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal3D[2] = new FlavourHistograms<double>
       ("jetDist3_3D" + theExtensionString, "JetDistance 3D 3.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistVal3D[3] = new FlavourHistograms<double>
       ("jetDist4_3D" + theExtensionString, "JetDistance 3D 4.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign2D[4] = new FlavourHistograms<double>
       ("jetDist_2D" + theExtensionString, "JetDistance Sign 2D",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign2D[0] = new FlavourHistograms<double>
       ("jetDist1_2D" + theExtensionString, "JetDistance Sign 2D 1.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign2D[1] = new FlavourHistograms<double>
       ("jetDist2_2D" + theExtensionString, "JetDistance Sign 2D 2.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign2D[2] = new FlavourHistograms<double>
       ("jetDist3_2D" + theExtensionString, "JetDistance Sign 2D 3.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign2D[3] = new FlavourHistograms<double>
       ("jetDist4_2D" + theExtensionString, "JetDistance Sign 2D 4.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign3D[4] = new FlavourHistograms<double>
       ("jetDist_3D" + theExtensionString, "JetDistance Sign 3D",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign3D[0] = new FlavourHistograms<double>
       ("jetDist1_3D" + theExtensionString, "JetDistance Sign 3D 1.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign3D[1] = new FlavourHistograms<double>
       ("jetDist2_3D" + theExtensionString, "JetDistance Sign 3D 2.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign3D[2] = new FlavourHistograms<double>
       ("jetDist3_3D" + theExtensionString, "JetDistance Sign 3D 3.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  tkcntHistosJetDistSign3D[3] = new FlavourHistograms<double>
       ("jetDist4_3D" + theExtensionString, "JetDistance Sign 3D 4.trk",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);

  // track chi-squared
  tkcntHistosTkNChiSqr2D[4] = new FlavourHistograms<double>
       ("tkNChiSqr_2D" + theExtensionString, "Normalized Chi Squared 2D",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr2D[0] = new FlavourHistograms<double>
       ("tkNChiSqr1_2D" + theExtensionString, "Normalized Chi Squared 2D 1.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr2D[1] = new FlavourHistograms<double>
       ("tkNChiSqr2_2D" + theExtensionString, "Normalized Chi Squared 2D 2.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr2D[2] = new FlavourHistograms<double>
       ("tkNChiSqr3_2D" + theExtensionString, "Normalized Chi Squared 2D 3.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr2D[3] = new FlavourHistograms<double>
       ("tkNChiSqr4_2D" + theExtensionString, "Normalized Chi Squared 2D 4.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr3D[4] = new FlavourHistograms<double>
       ("tkNChiSqr_3D" + theExtensionString, "Normalized Chi Squared 3D",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr3D[0] = new FlavourHistograms<double>
       ("tkNChiSqr1_3D" + theExtensionString, "Normalized Chi Squared 3D 1.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr3D[1] = new FlavourHistograms<double>
       ("tkNChiSqr2_3D" + theExtensionString, "Normalized Chi Squared 3D 2.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr3D[2] = new FlavourHistograms<double>
       ("tkNChiSqr3_3D" + theExtensionString, "Normalized Chi Squared 3D 3.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNChiSqr3D[3] = new FlavourHistograms<double>
       ("tkNChiSqr4_3D" + theExtensionString, "Normalized Chi Squared 3D 4.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc);

  // track pT
  tkcntHistosTkPt2D[4] = new FlavourHistograms<double>
       ("tkPt_2D" + theExtensionString, "Track Pt 2D",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt2D[0] = new FlavourHistograms<double>
       ("tkPt1_2D" + theExtensionString, "Track Pt 2D 1.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt2D[1] = new FlavourHistograms<double>
       ("tkPt2_2D" + theExtensionString, "Track Pt 2D 2.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt2D[2] = new FlavourHistograms<double>
       ("tkPt3_2D" + theExtensionString, "Track Pt 2D 3.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt2D[3] = new FlavourHistograms<double>
       ("tkPt4_2D" + theExtensionString, "Track Pt 2D 4.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt3D[4] = new FlavourHistograms<double>
       ("tkPt_3D" + theExtensionString, "Track Pt 3D",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt3D[0] = new FlavourHistograms<double>
       ("tkPt1_3D" + theExtensionString, "Track Pt 3D 1.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt3D[1] = new FlavourHistograms<double>
       ("tkPt2_3D" + theExtensionString, "Track Pt 3D 2.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt3D[2] = new FlavourHistograms<double>
       ("tkPt3_3D" + theExtensionString, "Track Pt 3D 3.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkPt3D[3] = new FlavourHistograms<double>
       ("tkPt4_3D" + theExtensionString, "Track Pt 3D 4.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc);

  // track nHits
  tkcntHistosTkNHits2D[4] = new FlavourHistograms<int>
       ("tkNHits_2D" + theExtensionString, "Track NHits 2D",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits2D[0] = new FlavourHistograms<int>
       ("tkNHits1_2D" + theExtensionString, "Track NHits 2D 1.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits2D[1] = new FlavourHistograms<int>
       ("tkNHits2_2D" + theExtensionString, "Track NHits 2D 2.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits2D[2] = new FlavourHistograms<int>
       ("tkNHits3_2D" + theExtensionString, "Track NHits 2D 3.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits2D[3] = new FlavourHistograms<int>
       ("tkNHits4_2D" + theExtensionString, "Track NHits 2D 4.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits3D[4] = new FlavourHistograms<int>
       ("tkNHits_3D" + theExtensionString, "Track NHits 3D",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits3D[0] = new FlavourHistograms<int>
       ("tkNHits1_3D" + theExtensionString, "Track NHits 3D 1.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits3D[1] = new FlavourHistograms<int>
       ("tkNHits2_3D" + theExtensionString, "Track NHits 3D 2.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits3D[2] = new FlavourHistograms<int>
       ("tkNHits3_3D" + theExtensionString, "Track NHits 3D 3.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  tkcntHistosTkNHits3D[3] = new FlavourHistograms<int>
       ("tkNHits4_3D" + theExtensionString, "Track NHits 3D 4.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc);

  // probability
  tkcntHistosProb3D[4] = new FlavourHistograms<float>
       ("prob_3D" + theExtensionString, "3D IP probability",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb3D[0] = new FlavourHistograms<float>
       ("prob1_3D" + theExtensionString, "3D IP probability 1.trk",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb3D[1] = new FlavourHistograms<float>
       ("prob2_3D" + theExtensionString, "3D IP probability 2.trk",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb3D[2] = new FlavourHistograms<float>
       ("prob3_3D" + theExtensionString, "3D IP probability 3.trk",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb3D[3] = new FlavourHistograms<float>
       ("prob4_3D" + theExtensionString, "3D IP probability 4.trk",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb2D[4] = new FlavourHistograms<float>
       ("prob_2D" + theExtensionString, "2D IP probability",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb2D[0] = new FlavourHistograms<float>
       ("prob1_2D" + theExtensionString, "2D IP probability 1.trk",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb2D[1] = new FlavourHistograms<float>
       ("prob2_2D" + theExtensionString, "2D IP probability 2.trk",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb2D[2] = new FlavourHistograms<float>
       ("prob3_2D" + theExtensionString, "2D IP probability 3.trk",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  tkcntHistosProb2D[3] = new FlavourHistograms<float>
       ("prob4" + theExtensionString, "2D IP probability 4.trk",
	50, -1.1, 1.1, false, true, true, "b", update,trackIPDir, mc) ;

  ghostTrackDistanceValuHisto = new FlavourHistograms<double>
       ("ghostTrackDist" + theExtensionString, "GhostTrackDistance",
	50, -0.1, 0.1, false, true, true, "b", update,trackIPDir, mc);
  ghostTrackDistanceSignHisto = new FlavourHistograms<double>
       ("ghostTrackDistSign" + theExtensionString, "GhostTrackDistance significance",
	50, -100.0, 100.0, false, true, true, "b", update,trackIPDir, mc);
  ghostTrackWeightHisto = new FlavourHistograms<double>
       ("ghostTrackWeight" + theExtensionString, "GhostTrack fit participation weight",
	50, 0.0, 1.0, false, false, true, "b", update,trackIPDir, mc);

  trackQualHisto = new FlavourHistograms<int>
       ("trackQual" + theExtensionString, "Track Quality of Tracks Associated to Jets",
        4, -1.5, 2.5, false, true, true, "b", update, trackIPDir, mc);

  selectedTrackQualHisto = new FlavourHistograms<int>
       ("selectedTrackQual" + theExtensionString, "Track Quality of Selected Tracks Associated to Jets",
        4, -1.5, 2.5, false, true, true, "b", update, trackIPDir, mc);

  trackMultVsJetPtHisto = new FlavourHistograms2D<double, int>
       ("trackMultVsJetPt" + theExtensionString, "Track Multiplicity vs Jet Pt for Tracks Associated to Jets",
        50, 0.0, 250.0, 21, -0.5, 30.5, false, update, trackIPDir, mc);

  selectedTrackMultVsJetPtHisto = new FlavourHistograms2D<double, int>
       ("selectedTrackMultVsJetPt" + theExtensionString, "Track Multiplicity vs Jet Pt for Selected Tracks Associated to Jets",
        50, 0.0, 250.0, 21, -0.5, 20.5, false, update, trackIPDir, mc);

  if (willFinalize_) createPlotsForFinalize();

}


TrackIPTagPlotter::~TrackIPTagPlotter ()
{

  delete trkNbr3D;
  delete trkNbr2D;
  delete ghostTrackDistanceValuHisto;
  delete ghostTrackDistanceSignHisto;
  delete ghostTrackWeightHisto;
  delete trackQualHisto;
  delete selectedTrackQualHisto;
  delete trackMultVsJetPtHisto;
  delete selectedTrackMultVsJetPtHisto;

  for(int n=0; n != 5; ++n) {
    delete tkcntHistosSig2D[n];
    delete tkcntHistosSig3D[n];
    delete tkcntHistosVal2D[n];
    delete tkcntHistosVal3D[n];
    delete tkcntHistosErr2D[n];
    delete tkcntHistosErr3D[n];
    delete tkcntHistosDecayLengthVal2D[n];
    delete tkcntHistosDecayLengthVal3D[n];
    delete tkcntHistosJetDistVal2D[n];
    delete tkcntHistosJetDistVal3D[n];
    delete tkcntHistosJetDistSign2D[n];
    delete tkcntHistosJetDistSign3D[n];
    delete tkcntHistosTkNChiSqr2D[n];
    delete tkcntHistosTkNChiSqr3D[n];
    delete tkcntHistosTkPt2D[n];
    delete tkcntHistosTkPt3D[n];
    delete tkcntHistosTkNHits2D[n];
    delete tkcntHistosTkNHits3D[n];
    delete tkcntHistosProb2D[n];
    delete tkcntHistosProb3D[n];
  }
  if (finalized) {
    for(int n=0; n != 4; ++n) delete effPurFromHistos[n];
  }
}


void TrackIPTagPlotter::analyzeTag (const reco::BaseTagInfo * baseTagInfo,
	const int & jetFlavour)
{

  const reco::TrackIPTagInfo * tagInfo = 
	dynamic_cast<const reco::TrackIPTagInfo *>(baseTagInfo);

  if (!tagInfo) {
    throw cms::Exception("Configuration")
      << "BTagPerformanceAnalyzer: Extended TagInfo not of type TrackIPTagInfo. " << std::endl;
  }

  const GlobalPoint pv(tagInfo->primaryVertex()->position().x(),
                       tagInfo->primaryVertex()->position().y(),
                       tagInfo->primaryVertex()->position().z());

  const std::vector<reco::TrackIPTagInfo::TrackIPData>& ip = tagInfo->impactParameterData();

  std::vector<float> prob2d, prob3d;
  if (tagInfo->hasProbabilities()) {
    prob2d = tagInfo->probabilities(0);	
    prob3d = tagInfo->probabilities(1);	
  }

  trkNbr3D->fill(jetFlavour, ip.size());
  trkNbr2D->fill(jetFlavour, ip.size());

  std::vector<std::size_t> sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::IP2DSig);
  reco::TrackRefVector sortedTracks = tagInfo->sortedTracks(sortedIndices);
  for(unsigned int n=0; n != sortedIndices.size() && n != 4; ++n) {
    tkcntHistosSig2D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip2d.significance());
    tkcntHistosVal2D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip2d.value());
    tkcntHistosErr2D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip2d.error());
    const double& decayLen = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    tkcntHistosDecayLengthVal2D[n]->fill(jetFlavour, decayLen);
    tkcntHistosJetDistVal2D[n]->fill(jetFlavour, ip[sortedIndices[n]].distanceToJetAxis.value());
    tkcntHistosJetDistSign2D[n]->fill(jetFlavour, ip[sortedIndices[n]].distanceToJetAxis.significance());
    const reco::TrackRef& track = sortedTracks[n];
    tkcntHistosTkNChiSqr2D[n]->fill(jetFlavour, track->normalizedChi2());
    tkcntHistosTkPt2D[n]->fill(jetFlavour, track->pt());
    tkcntHistosTkNHits2D[n]->fill(jetFlavour, track->found());
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::Prob2D);
  for(unsigned int n=0; n != sortedIndices.size() && n != 4; ++n) {
    tkcntHistosProb2D[n]->fill(jetFlavour, prob2d[sortedIndices[n]]);
  }
  for(unsigned int n=sortedIndices.size(); n < 4; ++n){
    tkcntHistosSig2D[n]->fill(jetFlavour, lowerIPSBound-1.0);
    tkcntHistosVal2D[n]->fill(jetFlavour, lowerIPBound-1.0);
    tkcntHistosErr2D[n]->fill(jetFlavour, lowerIPEBound-1.0);
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::IP3DSig);
  sortedTracks = tagInfo->sortedTracks(sortedIndices);
  for(unsigned int n=0; n != sortedIndices.size() && n != 4; ++n) {
    tkcntHistosSig3D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip3d.significance());
    tkcntHistosVal3D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip3d.value());
    tkcntHistosErr3D[n]->fill(jetFlavour, ip[sortedIndices[n]].ip3d.error());
    const double& decayLen = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    tkcntHistosDecayLengthVal3D[n]->fill(jetFlavour, decayLen);
    tkcntHistosJetDistVal3D[n]->fill(jetFlavour, ip[sortedIndices[n]].distanceToJetAxis.value());
    tkcntHistosJetDistSign3D[n]->fill(jetFlavour, ip[sortedIndices[n]].distanceToJetAxis.significance());
    const reco::TrackRef& track = sortedTracks[n];
    tkcntHistosTkNChiSqr3D[n]->fill(jetFlavour, track->normalizedChi2());
    tkcntHistosTkPt3D[n]->fill(jetFlavour, track->pt());
    tkcntHistosTkNHits3D[n]->fill(jetFlavour, track->found());
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::Prob3D);
  for(unsigned int n=0; n != sortedIndices.size() && n != 4; ++n) {
    tkcntHistosProb3D[n]->fill(jetFlavour, prob3d[sortedIndices[n]]);
  }
  for(unsigned int n=sortedIndices.size(); n < 4; ++n){
    tkcntHistosSig3D[n]->fill(jetFlavour, lowerIPSBound-1.0);
    tkcntHistosVal3D[n]->fill(jetFlavour, lowerIPBound-1.0);
    tkcntHistosErr3D[n]->fill(jetFlavour, lowerIPEBound-1.0);
  }
  for(unsigned int n=0; n != ip.size(); ++n) {
    tkcntHistosSig2D[4]->fill(jetFlavour, ip[n].ip2d.significance());
    tkcntHistosVal2D[4]->fill(jetFlavour, ip[n].ip2d.value());
    tkcntHistosErr2D[4]->fill(jetFlavour, ip[n].ip2d.error());
    tkcntHistosProb2D[4]->fill(jetFlavour, prob2d[n]);
    const double& decayLen = (ip[n].closestToJetAxis - pv).mag();
    tkcntHistosDecayLengthVal2D[4]->fill(jetFlavour, decayLen);
    tkcntHistosJetDistVal2D[4]->fill(jetFlavour, ip[n].distanceToJetAxis.value());
    tkcntHistosJetDistSign2D[4]->fill(jetFlavour, ip[n].distanceToJetAxis.significance());
    const reco::TrackRef& track = tagInfo->selectedTracks()[n];
    tkcntHistosTkNChiSqr2D[4]->fill(jetFlavour, track->normalizedChi2());
    tkcntHistosTkPt2D[4]->fill(jetFlavour, track->pt());
    tkcntHistosTkNHits2D[4]->fill(jetFlavour, track->found());
  }
  for(unsigned int n=0; n != ip.size(); ++n) {
    tkcntHistosSig3D[4]->fill(jetFlavour, ip[n].ip3d.significance());
    tkcntHistosVal3D[4]->fill(jetFlavour, ip[n].ip3d.value());
    tkcntHistosErr3D[4]->fill(jetFlavour, ip[n].ip3d.error());
    tkcntHistosProb3D[4]->fill(jetFlavour, prob3d[n]);
    const double& decayLen = (ip[n].closestToJetAxis - pv).mag();
    tkcntHistosDecayLengthVal3D[4]->fill(jetFlavour, decayLen);
    tkcntHistosJetDistVal3D[4]->fill(jetFlavour, ip[n].distanceToJetAxis.value());
    tkcntHistosJetDistSign3D[4]->fill(jetFlavour, ip[n].distanceToJetAxis.significance());
    const reco::TrackRef& track = tagInfo->selectedTracks()[n];
    tkcntHistosTkNChiSqr3D[4]->fill(jetFlavour, track->normalizedChi2());
    tkcntHistosTkPt3D[4]->fill(jetFlavour, track->pt());
    tkcntHistosTkNHits3D[4]->fill(jetFlavour, track->found());
  }
  for(unsigned int n=0; n != ip.size(); ++n) {
    ghostTrackDistanceValuHisto->fill(jetFlavour, ip[n].distanceToGhostTrack.value());
    ghostTrackDistanceSignHisto->fill(jetFlavour, ip[n].distanceToGhostTrack.significance());
    ghostTrackWeightHisto->fill(jetFlavour, ip[n].ghostTrackWeight);
    selectedTrackQualHisto->fill(jetFlavour, highestTrackQual(tagInfo->selectedTracks()[n].get()));
  }
  for(unsigned int n = 0; n != tagInfo->tracks().size(); ++n) {
    trackQualHisto->fill(jetFlavour, highestTrackQual(tagInfo->tracks()[n].get()));
  }

  trackMultVsJetPtHisto->fill(jetFlavour, tagInfo->jet()->pt(), tagInfo->tracks().size());
  selectedTrackMultVsJetPtHisto->fill(jetFlavour, tagInfo->jet()->pt(), tagInfo->selectedTracks().size());
}

void TrackIPTagPlotter::createPlotsForFinalize (){
  const std::string trackIPDir("TrackIPPlots" + theExtensionString);
  effPurFromHistos[0] = new EffPurFromHistos (tkcntHistosSig3D[1],trackIPDir, mcPlots_, 
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[1] = new EffPurFromHistos (tkcntHistosSig3D[2],trackIPDir, mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[2] = new EffPurFromHistos (tkcntHistosSig2D[1],trackIPDir, mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
  effPurFromHistos[3] = new EffPurFromHistos (tkcntHistosSig2D[2],trackIPDir, mcPlots_,
		nBinEffPur_, startEffPur_,
		endEffPur_);
}


void TrackIPTagPlotter::finalize ()
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  for(int n=0; n != 4; ++n) effPurFromHistos[n]->compute();
  finalized = true;
}

void TrackIPTagPlotter::psPlot(const std::string & name)
{
  const std::string cName("TrackIPPlots"+ theExtensionString);
  RecoBTag::setTDRStyle()->cd();
  TCanvas canvas(cName.c_str(), cName.c_str(), 600, 900);
  canvas.UseCurrentStyle();
  canvas.Divide(2,3);
  canvas.Print((name + cName + ".ps[").c_str());

  canvas.cd(1);
  trkNbr3D->plot();
  canvas.cd(2);
  tkcntHistosSig3D[4]->plot();
  for(int n=0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosSig3D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);

  canvas.cd(1);
  trkNbr3D->plot();
  canvas.cd(2);
  tkcntHistosProb3D[4]->plot();
  for(int n=0; n < 4; n++) {
    canvas.cd(3+n);
    tkcntHistosProb3D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);
  canvas.cd(1);
  trkNbr2D->plot();
  canvas.cd(2);
  tkcntHistosSig2D[4]->plot();
  for(int n=0; n != 4; ++n) {
    canvas.cd(3+n);
    tkcntHistosSig2D[n]->plot();
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(2,3);
  canvas.cd(1);
  trkNbr2D->plot();
  canvas.cd(2);
  tkcntHistosProb2D[4]->plot();
  for(int n=0; n != 4; ++n) {
    canvas.cd(3+n);
    tkcntHistosProb2D[n]->plot();
  }

  if (finalized) {
    for(int n=0; n != 2; ++n) {
      canvas.Print((name + cName + ".ps").c_str());
      canvas.Clear();
      canvas.Divide(2,3);
      canvas.cd(1);
      effPurFromHistos[0+n]->discriminatorNoCutEffic()->plot();
      canvas.cd(2);
      effPurFromHistos[0+n]->discriminatorCutEfficScan()->plot();
      canvas.cd(3);
      effPurFromHistos[0+n]->plot();
      canvas.cd(4);
      effPurFromHistos[1+n]->discriminatorNoCutEffic()->plot();
      canvas.cd(5);
      effPurFromHistos[1+n]->discriminatorCutEfficScan()->plot();
      canvas.cd(6);
      effPurFromHistos[1+n]->plot();
    }
  }

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Clear();
  canvas.Divide(1,3);
  canvas.cd(1);
  ghostTrackDistanceValuHisto->plot();
  canvas.cd(2);
  ghostTrackDistanceSignHisto->plot();
  canvas.cd(3);
  ghostTrackWeightHisto->plot();

  canvas.Print((name + cName + ".ps").c_str());
  canvas.Print((name + cName + ".ps]").c_str());
}


void TrackIPTagPlotter::epsPlot(const std::string & name)
{
  trkNbr2D->epsPlot(name);
  trkNbr3D->epsPlot(name);
  ghostTrackDistanceValuHisto->epsPlot(name);
  ghostTrackDistanceSignHisto->epsPlot(name);
  ghostTrackWeightHisto->epsPlot(name);
  for(int n=0; n != 5; ++n) {
    tkcntHistosSig2D[n]->epsPlot(name);
    tkcntHistosSig3D[n]->epsPlot(name);
    tkcntHistosVal2D[n]->epsPlot(name);
    tkcntHistosVal3D[n]->epsPlot(name);
    tkcntHistosErr2D[n]->epsPlot(name);
    tkcntHistosErr3D[n]->epsPlot(name);
    tkcntHistosProb2D[n]->epsPlot(name);
    tkcntHistosProb3D[n]->epsPlot(name);
    tkcntHistosDecayLengthVal2D[n]->epsPlot(name);
    tkcntHistosDecayLengthVal3D[n]->epsPlot(name);
    tkcntHistosJetDistVal2D[n]->epsPlot(name);
    tkcntHistosJetDistVal3D[n]->epsPlot(name);
    tkcntHistosJetDistSign2D[n]->epsPlot(name);
    tkcntHistosJetDistSign3D[n]->epsPlot(name);
    tkcntHistosTkNChiSqr2D[n]->epsPlot(name);
    tkcntHistosTkNChiSqr3D[n]->epsPlot(name);
    tkcntHistosTkPt2D[n]->epsPlot(name);
    tkcntHistosTkPt3D[n]->epsPlot(name);
    tkcntHistosTkNHits2D[n]->epsPlot(name);
    tkcntHistosTkNHits3D[n]->epsPlot(name);
  }
  if (finalized) {
    for(int n=0; n != 4; ++n) effPurFromHistos[n]->epsPlot(name);
  }
}

int TrackIPTagPlotter::highestTrackQual(const reco::Track* track) {
  for(int i = 2; i > -1; --i)
  {
    reco::TrackBase::TrackQuality qual = reco::TrackBase::qualityByName(reco::TrackBase::qualityNames[i]);
    if(track->quality(qual))
      return i;
  }

  return -1;
}
