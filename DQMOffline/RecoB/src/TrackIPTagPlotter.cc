#include <cstddef>
#include <string>

#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQMOffline/RecoB/interface/TrackIPTagPlotter.h"

TrackIPTagPlotter::TrackIPTagPlotter(const std::string & tagName,
				     const EtaPtBin & etaPtBin, const edm::ParameterSet& pSet, const bool& update, 
				     const unsigned int& mc, const bool& wf, DQMStore::IBooker & ibook) :
  BaseTagInfoPlotter(tagName, etaPtBin),
  nBinEffPur_(pSet.getParameter<int>("nBinEffPur")),
  startEffPur_(pSet.getParameter<double>("startEffPur")),
  endEffPur_(pSet.getParameter<double>("endEffPur")),
  mcPlots_(mc), willFinalize_(wf),
  makeQualityPlots_(pSet.getParameter<bool>("QualityPlots")),
  lowerIPSBound(pSet.getParameter<double>("LowerIPSBound")),
  upperIPSBound(pSet.getParameter<double>("UpperIPSBound")),
  lowerIPBound(pSet.getParameter<double>("LowerIPBound")),
  upperIPBound(pSet.getParameter<double>("UpperIPBound")),
  lowerIPEBound(pSet.getParameter<double>("LowerIPEBound")),
  upperIPEBound(pSet.getParameter<double>("UpperIPEBound")),
  nBinsIPS(pSet.getParameter<int>("NBinsIPS")),
  nBinsIP(pSet.getParameter<int>("NBinsIP")),
  nBinsIPE(pSet.getParameter<int>("NBinsIPE")),
  minDecayLength(pSet.getParameter<double>("MinDecayLength")),
  maxDecayLength(pSet.getParameter<double>("MaxDecayLength")),
  minJetDistance(pSet.getParameter<double>("MinJetDistance")),
  maxJetDistance(pSet.getParameter<double>("MaxJetDistance")),
  finalized(false),
  ibook_(ibook)
{
  const std::string trackIPDir(theExtensionString.substr(1));

  trkNbr3D = new TrackIPHistograms<int>
	("selTrksNbr_3D" + theExtensionString, "Number of selected tracks for 3D IPS", 31, -0.5, 30.5,
	 false, true, true, "b", update,trackIPDir ,mc, makeQualityPlots_, ibook);

  trkNbr2D = new TrackIPHistograms<int>
	("selTrksNbr_2D" + theExtensionString, "Number of selected tracks for 2D IPS", 31, -0.5, 30.5,
	 false, true, true, "b", update,trackIPDir ,mc, makeQualityPlots_, ibook);

  // IP significance
  // 3D
  tkcntHistosSig3D[4] = new TrackIPHistograms<double>
       ("ips_3D" + theExtensionString, "3D IP significance",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosSig3D[0] = new TrackIPHistograms<double>
       ("ips1_3D" + theExtensionString, "3D IP significance 1.trk",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir,mc, makeQualityPlots_, ibook) ;

  tkcntHistosSig3D[1] = new TrackIPHistograms<double>
       ("ips2_3D" + theExtensionString, "3D IP significance 2.trk",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosSig3D[2] = new TrackIPHistograms<double>
       ("ips3_3D" + theExtensionString, "3D IP significance 3.trk",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosSig3D[3] = new TrackIPHistograms<double>
       ("ips4_3D" + theExtensionString, "3D IP significance 4.trk",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  //2D
  tkcntHistosSig2D[4] = new TrackIPHistograms<double>
       ("ips_2D" + theExtensionString, "2D IP significance",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosSig2D[0] = new TrackIPHistograms<double>
       ("ips1_2D" + theExtensionString, "2D IP significance 1.trk",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosSig2D[1] = new TrackIPHistograms<double>
       ("ips2_2D" + theExtensionString, "2D IP significance 2.trk",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosSig2D[2] = new TrackIPHistograms<double>
       ("ips3_2D" + theExtensionString, "2D IP significance 3.trk",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosSig2D[3] = new TrackIPHistograms<double>
       ("ips4_2D" + theExtensionString, "2D IP significance 4.trk",
	nBinsIPS, lowerIPSBound, upperIPSBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  // IP value
  //3D
  tkcntHistosVal3D[4] = new TrackIPHistograms<double>
       ("ip_3D" + theExtensionString, "3D IP value",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosVal3D[0] = new TrackIPHistograms<double>
       ("ip1_3D" + theExtensionString, "3D IP value 1.trk",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosVal3D[1] = new TrackIPHistograms<double>
       ("ip2_3D" + theExtensionString, "3D IP value 2.trk",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosVal3D[2] = new TrackIPHistograms<double>
       ("ip3_3D" + theExtensionString, "3D IP value 3.trk",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosVal3D[3] = new TrackIPHistograms<double>
       ("ip4_3D" + theExtensionString, "3D IP value 4.trk",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  //2D
  tkcntHistosVal2D[4] = new TrackIPHistograms<double>
       ("ip_2D" + theExtensionString, "2D IP value",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosVal2D[0] = new TrackIPHistograms<double>
       ("ip1_2D" + theExtensionString, "2D IP value 1.trk",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosVal2D[1] = new TrackIPHistograms<double>
       ("ip2_2D" + theExtensionString, "2D IP value 2.trk",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosVal2D[2] = new TrackIPHistograms<double>
       ("ip3_2D" + theExtensionString, "2D IP value 3.trk",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosVal2D[3] = new TrackIPHistograms<double>
       ("ip4_2D" + theExtensionString, "2D IP value 4.trk",
	nBinsIP, lowerIPBound, upperIPBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;


  // IP error
  // 3D
  tkcntHistosErr3D[4] = new TrackIPHistograms<double>
       ("ipe_3D" + theExtensionString, "3D IP error",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosErr3D[0] = new TrackIPHistograms<double>
       ("ipe1_3D" + theExtensionString, "3D IP error 1.trk",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosErr3D[1] = new TrackIPHistograms<double>
       ("ipe2_3D" + theExtensionString, "3D IP error 2.trk",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosErr3D[2] = new TrackIPHistograms<double>
       ("ipe3_3D" + theExtensionString, "3D IP error 3.trk",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosErr3D[3] = new TrackIPHistograms<double>
       ("ipe4_3D" + theExtensionString, "3D IP error 4.trk",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  //2D
  tkcntHistosErr2D[4] = new TrackIPHistograms<double>
       ("ipe_2D" + theExtensionString, "2D IP error",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosErr2D[0] = new TrackIPHistograms<double>
       ("ipe1_2D" + theExtensionString, "2D IP error 1.trk",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosErr2D[1] = new TrackIPHistograms<double>
       ("ipe2_2D" + theExtensionString, "2D IP error 2.trk",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosErr2D[2] = new TrackIPHistograms<double>
       ("ipe3_2D" + theExtensionString, "2D IP error 3.trk",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosErr2D[3] = new TrackIPHistograms<double>
       ("ipe4_2D" + theExtensionString, "2D IP error 4.trk",
	nBinsIPE, lowerIPEBound, upperIPEBound, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  // decay length
  tkcntHistosDecayLengthVal2D[4] = new TrackIPHistograms<double>
       ("decLen_2D" + theExtensionString, "Decay Length 2D",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal2D[0] = new TrackIPHistograms<double>
       ("decLen1_2D" + theExtensionString, "2D Decay Length 1.trk",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal2D[1] = new TrackIPHistograms<double>
       ("decLen2_2D" + theExtensionString, "2D Decay Length 2.trk",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal2D[2] = new TrackIPHistograms<double>
       ("decLen3_2D" + theExtensionString, "2D Decay Length 3.trk",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal2D[3] = new TrackIPHistograms<double>
       ("decLen4_2D" + theExtensionString, "2D Decay Length 4.trk",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal3D[4] = new TrackIPHistograms<double>
       ("decLen_3D" + theExtensionString, "3D Decay Length",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal3D[0] = new TrackIPHistograms<double>
       ("decLen1_3D" + theExtensionString, "3D Decay Length 1.trk",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal3D[1] = new TrackIPHistograms<double>
       ("decLen2_3D" + theExtensionString, "3D Decay Length 2.trk",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal3D[2] = new TrackIPHistograms<double>
       ("decLen3_3D" + theExtensionString, "3D Decay Length 3.trk",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosDecayLengthVal3D[3] = new TrackIPHistograms<double>
       ("decLen4_3D" + theExtensionString, "3D Decay Length 4.trk",
	50, 0.0, 5.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  // jet distance
  tkcntHistosJetDistVal2D[4] = new TrackIPHistograms<double>
       ("jetDist_2D" + theExtensionString, "JetDistance 2D",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal2D[0] = new TrackIPHistograms<double>
       ("jetDist1_2D" + theExtensionString, "JetDistance 2D 1.trk",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal2D[1] = new TrackIPHistograms<double>
       ("jetDist2_2D" + theExtensionString, "JetDistance 2D 2.trk",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal2D[2] = new TrackIPHistograms<double>
       ("jetDist3_2D" + theExtensionString, "JetDistance 2D 3.trk",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal2D[3] = new TrackIPHistograms<double>
       ("jetDist4_2D" + theExtensionString, "JetDistance 2D 4.trk",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal3D[4] = new TrackIPHistograms<double>
       ("jetDist_3D" + theExtensionString, "JetDistance 3D",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal3D[0] = new TrackIPHistograms<double>
       ("jetDist1_3D" + theExtensionString, "JetDistance 3D 1.trk",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal3D[1] = new TrackIPHistograms<double>
       ("jetDist2_3D" + theExtensionString, "JetDistance 3D 2.trk",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal3D[2] = new TrackIPHistograms<double>
       ("jetDist3_3D" + theExtensionString, "JetDistance 3D 3.trk",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistVal3D[3] = new TrackIPHistograms<double>
       ("jetDist4_3D" + theExtensionString, "JetDistance 3D 4.trk",
	50, -0.1, 0.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;
  /* // the jet distance significance return always 0 => disabled
  tkcntHistosJetDistSign2D[4] = new TrackIPHistograms<double>
       ("jetDistSig_2D" + theExtensionString, "JetDistance Sign 2D",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign2D[0] = new TrackIPHistograms<double>
       ("jetDistSig1_2D" + theExtensionString, "JetDistance Sign 2D 1.trk",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign2D[1] = new TrackIPHistograms<double>
       ("jetDistSig2_2D" + theExtensionString, "JetDistance Sign 2D 2.trk",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign2D[2] = new TrackIPHistograms<double>
       ("jetDistSig3_2D" + theExtensionString, "JetDistance Sign 2D 3.trk",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign2D[3] = new TrackIPHistograms<double>
       ("jetDistSig4_2D" + theExtensionString, "JetDistance Sign 2D 4.trk",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign3D[4] = new TrackIPHistograms<double>
       ("jetDistSig_3D" + theExtensionString, "JetDistance Sign 3D",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign3D[0] = new TrackIPHistograms<double>
       ("jetDistSig1_3D" + theExtensionString, "JetDistance Sign 3D 1.trk",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign3D[1] = new TrackIPHistograms<double>
       ("jetDistSig2_3D" + theExtensionString, "JetDistance Sign 3D 2.trk",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign3D[2] = new TrackIPHistograms<double>
       ("jetDistSig3_3D" + theExtensionString, "JetDistance Sign 3D 3.trk",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosJetDistSign3D[3] = new TrackIPHistograms<double>
       ("jetDistSig4_3D" + theExtensionString, "JetDistance Sign 3D 4.trk",
	100, -0.001, 0.001, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;
  */
  // track chi-squared
  tkcntHistosTkNChiSqr2D[4] = new TrackIPHistograms<double>
       ("tkNChiSqr_2D" + theExtensionString, "Normalized Chi Squared 2D",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr2D[0] = new TrackIPHistograms<double>
       ("tkNChiSqr1_2D" + theExtensionString, "Normalized Chi Squared 2D 1.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr2D[1] = new TrackIPHistograms<double>
       ("tkNChiSqr2_2D" + theExtensionString, "Normalized Chi Squared 2D 2.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr2D[2] = new TrackIPHistograms<double>
       ("tkNChiSqr3_2D" + theExtensionString, "Normalized Chi Squared 2D 3.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr2D[3] = new TrackIPHistograms<double>
       ("tkNChiSqr4_2D" + theExtensionString, "Normalized Chi Squared 2D 4.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr3D[4] = new TrackIPHistograms<double>
       ("tkNChiSqr_3D" + theExtensionString, "Normalized Chi Squared 3D",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr3D[0] = new TrackIPHistograms<double>
       ("tkNChiSqr1_3D" + theExtensionString, "Normalized Chi Squared 3D 1.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr3D[1] = new TrackIPHistograms<double>
       ("tkNChiSqr2_3D" + theExtensionString, "Normalized Chi Squared 3D 2.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr3D[2] = new TrackIPHistograms<double>
       ("tkNChiSqr3_3D" + theExtensionString, "Normalized Chi Squared 3D 3.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNChiSqr3D[3] = new TrackIPHistograms<double>
       ("tkNChiSqr4_3D" + theExtensionString, "Normalized Chi Squared 3D 4.trk",
        50, -0.1, 10.0, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  // track pT
  tkcntHistosTkPt2D[4] = new TrackIPHistograms<double>
       ("tkPt_2D" + theExtensionString, "Track Pt 2D",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt2D[0] = new TrackIPHistograms<double>
       ("tkPt1_2D" + theExtensionString, "Track Pt 2D 1.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt2D[1] = new TrackIPHistograms<double>
       ("tkPt2_2D" + theExtensionString, "Track Pt 2D 2.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt2D[2] = new TrackIPHistograms<double>
       ("tkPt3_2D" + theExtensionString, "Track Pt 2D 3.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt2D[3] = new TrackIPHistograms<double>
       ("tkPt4_2D" + theExtensionString, "Track Pt 2D 4.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt3D[4] = new TrackIPHistograms<double>
       ("tkPt_3D" + theExtensionString, "Track Pt 3D",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt3D[0] = new TrackIPHistograms<double>
       ("tkPt1_3D" + theExtensionString, "Track Pt 3D 1.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt3D[1] = new TrackIPHistograms<double>
       ("tkPt2_3D" + theExtensionString, "Track Pt 3D 2.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt3D[2] = new TrackIPHistograms<double>
       ("tkPt3_3D" + theExtensionString, "Track Pt 3D 3.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkPt3D[3] = new TrackIPHistograms<double>
       ("tkPt4_3D" + theExtensionString, "Track Pt 3D 4.trk",
        50, -0.1, 50.1, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  // track nHits
  tkcntHistosTkNHits2D[4] = new TrackIPHistograms<int>
       ("tkNHits_2D" + theExtensionString, "Track NHits 2D",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits2D[0] = new TrackIPHistograms<int>
       ("tkNHits1_2D" + theExtensionString, "Track NHits 2D 1.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits2D[1] = new TrackIPHistograms<int>
       ("tkNHits2_2D" + theExtensionString, "Track NHits 2D 2.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits2D[2] = new TrackIPHistograms<int>
       ("tkNHits3_2D" + theExtensionString, "Track NHits 2D 3.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits2D[3] = new TrackIPHistograms<int>
       ("tkNHits4_2D" + theExtensionString, "Track NHits 2D 4.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits3D[4] = new TrackIPHistograms<int>
       ("tkNHits_3D" + theExtensionString, "Track NHits 3D",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits3D[0] = new TrackIPHistograms<int>
       ("tkNHits1_3D" + theExtensionString, "Track NHits 3D 1.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits3D[1] = new TrackIPHistograms<int>
       ("tkNHits2_3D" + theExtensionString, "Track NHits 3D 2.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits3D[2] = new TrackIPHistograms<int>
       ("tkNHits3_3D" + theExtensionString, "Track NHits 3D 3.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNHits3D[3] = new TrackIPHistograms<int>
       ("tkNHits4_3D" + theExtensionString, "Track NHits 3D 4.trk",
        31, -0.5, 30.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  //Pixel hits
  tkcntHistosTkNPixelHits2D[4] = new TrackIPHistograms<int>
       ("tkNPixelHits_2D" + theExtensionString, "Track NPixelHits 2D",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits2D[0] = new TrackIPHistograms<int>
       ("tkNPixelHits1_2D" + theExtensionString, "Track NPixelHits 2D 1.trk",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits2D[1] = new TrackIPHistograms<int>
       ("tkNPixelHits2_2D" + theExtensionString, "Track NPixelHits 2D 2.trk",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits2D[2] = new TrackIPHistograms<int>
       ("tkNPixelHits3_2D" + theExtensionString, "Track NPixelHits 2D 3.trk",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits2D[3] = new TrackIPHistograms<int>
       ("tkNPixelHits4_2D" + theExtensionString, "Track NPixelHits 2D 4.trk",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits3D[4] = new TrackIPHistograms<int>
       ("tkNPixelHits_3D" + theExtensionString, "Track NPixelHits 3D",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits3D[0] = new TrackIPHistograms<int>
       ("tkNPixelHits1_3D" + theExtensionString, "Track NPixelHits 3D 1.trk",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits3D[1] = new TrackIPHistograms<int>
       ("tkNPixelHits2_3D" + theExtensionString, "Track NPixelHits 3D 2.trk",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits3D[2] = new TrackIPHistograms<int>
       ("tkNPixelHits3_3D" + theExtensionString, "Track NPixelHits 3D 3.trk",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosTkNPixelHits3D[3] = new TrackIPHistograms<int>
       ("tkNPixelHits4_3D" + theExtensionString, "Track NPixelHits 3D 4.trk",
        11, -0.5, 10.5, false, true, true, "b", update, trackIPDir, mc, makeQualityPlots_, ibook) ;

  // probability
  tkcntHistosProb3D[4] = new TrackIPHistograms<float>
       ("prob_3D" + theExtensionString, "3D IP probability",
	50, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb3D[0] = new TrackIPHistograms<float>
       ("prob1_3D" + theExtensionString, "3D IP probability 1.trk",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb3D[1] = new TrackIPHistograms<float>
       ("prob2_3D" + theExtensionString, "3D IP probability 2.trk",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb3D[2] = new TrackIPHistograms<float>
       ("prob3_3D" + theExtensionString, "3D IP probability 3.trk",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb3D[3] = new TrackIPHistograms<float>
       ("prob4_3D" + theExtensionString, "3D IP probability 4.trk",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb2D[4] = new TrackIPHistograms<float>
       ("prob_2D" + theExtensionString, "2D IP probability",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb2D[0] = new TrackIPHistograms<float>
       ("prob1_2D" + theExtensionString, "2D IP probability 1.trk",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb2D[1] = new TrackIPHistograms<float>
       ("prob2_2D" + theExtensionString, "2D IP probability 2.trk",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb2D[2] = new TrackIPHistograms<float>
       ("prob3_2D" + theExtensionString, "2D IP probability 3.trk",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  tkcntHistosProb2D[3] = new TrackIPHistograms<float>
       ("prob4_2D" + theExtensionString, "2D IP probability 4.trk",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  //probability for tracks with IP value < 0 or IP value > 0
  tkcntHistosTkProbIPneg2D = new TrackIPHistograms<float>
       ("probIPneg_2D" + theExtensionString, "2D negative IP probability",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;
  tkcntHistosTkProbIPpos2D = new TrackIPHistograms<float>
       ("probIPpos_2D" + theExtensionString, "2D positive IP probability",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;
  tkcntHistosTkProbIPneg3D = new TrackIPHistograms<float>
       ("probIPneg_3D" + theExtensionString, "3D negative IP probability",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;
  tkcntHistosTkProbIPpos3D = new TrackIPHistograms<float>
       ("probIPpos_3D" + theExtensionString, "3D positive IP probability",
	52, -1.04, 1.04, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  //ghost Tracks and others
  ghostTrackDistanceValuHisto = new TrackIPHistograms<double>
       ("ghostTrackDist" + theExtensionString, "GhostTrackDistance",
	50, 0.0, 0.1, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;
  ghostTrackDistanceSignHisto = new TrackIPHistograms<double>
       ("ghostTrackDistSign" + theExtensionString, "GhostTrackDistance significance",
	50, -5.0, 15.0, false, true, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;
  ghostTrackWeightHisto = new TrackIPHistograms<double>
       ("ghostTrackWeight" + theExtensionString, "GhostTrack fit participation weight",
	50, 0.0, 1.0, false, false, true, "b", update,trackIPDir, mc, makeQualityPlots_, ibook) ;

  trackQualHisto = new FlavourHistograms<int>
       ("trackQual" + theExtensionString, "Track Quality of Tracks Associated to Jets",
        4, -1.5, 2.5, false, true, true, "b", update, trackIPDir, mc, ibook);

  selectedTrackQualHisto = new FlavourHistograms<int>
       ("selectedTrackQual" + theExtensionString, "Track Quality of Selected Tracks Associated to Jets",
        4, -1.5, 2.5, false, true, true, "b", update, trackIPDir, mc, ibook);

  trackMultVsJetPtHisto = new FlavourHistograms2D<double, int>
       ("trackMultVsJetPt" + theExtensionString, "Track Multiplicity vs Jet Pt for Tracks Associated to Jets",
        50, 0.0, 250.0, 21, -0.5, 30.5, false, update, trackIPDir, mc, true, ibook);

  selectedTrackMultVsJetPtHisto = new FlavourHistograms2D<double, int>
       ("selectedTrackMultVsJetPt" + theExtensionString, "Track Multiplicity vs Jet Pt for Selected Tracks Associated to Jets",
        50, 0.0, 250.0, 21, -0.5, 20.5, false, update, trackIPDir, mc, true, ibook);

  if (willFinalize_) createPlotsForFinalize(ibook);

}


TrackIPTagPlotter::~TrackIPTagPlotter ()
{

  delete trkNbr3D;
  delete trkNbr2D;
  delete tkcntHistosTkProbIPneg2D;
  delete tkcntHistosTkProbIPpos2D;
  delete tkcntHistosTkProbIPneg3D;
  delete tkcntHistosTkProbIPpos3D;
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
    //delete tkcntHistosJetDistSign2D[n];
    //delete tkcntHistosJetDistSign3D[n];
    delete tkcntHistosTkNChiSqr2D[n];
    delete tkcntHistosTkNChiSqr3D[n];
    delete tkcntHistosTkPt2D[n];
    delete tkcntHistosTkPt3D[n];
    delete tkcntHistosTkNHits2D[n];
    delete tkcntHistosTkNHits3D[n];  
    delete tkcntHistosTkNPixelHits2D[n];
    delete tkcntHistosTkNPixelHits3D[n];
    delete tkcntHistosProb2D[n];
    delete tkcntHistosProb3D[n];
  }
  if (finalized) {
    for(int n=0; n != 4; ++n) delete effPurFromHistos[n];
  }
}

void TrackIPTagPlotter::analyzeTag (const reco::BaseTagInfo * baseTagInfo,
				    const double & jec,
				    const int & jetFlavour)
{
  analyzeTag(baseTagInfo, jec, jetFlavour, 1.);
}
void TrackIPTagPlotter::analyzeTag (const reco::BaseTagInfo * baseTagInfo,
				    const double & jec,
				    const int & jetFlavour, const float & w)
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

  std::vector<std::size_t> sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::IP2DSig);
  std::vector<std::size_t> selectedIndices;
  reco::TrackRefVector sortedTracks = tagInfo->sortedTracks(sortedIndices);
  reco::TrackRefVector selectedTracks;
  for(unsigned int n = 0; n != sortedIndices.size(); ++n) {
    double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
    if(decayLength > minDecayLength && decayLength < maxDecayLength &&
       fabs(jetDistance) >= minJetDistance && fabs(jetDistance) < maxJetDistance ) {
      selectedIndices.push_back(sortedIndices[n]);
      selectedTracks.push_back(sortedTracks[n]);
    }
  }

  trkNbr2D->fill(jetFlavour, selectedIndices.size(),w);

  for(unsigned int n=0; n != selectedIndices.size(); ++n) {
    const reco::TrackRef& track = selectedTracks[n];
    const reco::TrackBase::TrackQuality& trackQual = highestTrackQual(track);
    tkcntHistosSig2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.significance(), true,w);
    tkcntHistosVal2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.value(), true,w);
    tkcntHistosErr2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.error(), true,w);
    const double& decayLen = (ip[selectedIndices[n]].closestToJetAxis - pv).mag();
    tkcntHistosDecayLengthVal2D[4]->fill(jetFlavour, trackQual, decayLen, true,w);
    tkcntHistosJetDistVal2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.value(), true,w);
    //tkcntHistosJetDistSign2D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.significance(), true,w);
    tkcntHistosTkNChiSqr2D[4]->fill(jetFlavour, trackQual, track->normalizedChi2(), true,w);
    tkcntHistosTkPt2D[4]->fill(jetFlavour, trackQual, track->pt(), true,w);
    tkcntHistosTkNHits2D[4]->fill(jetFlavour, trackQual, track->found(), true,w);
    tkcntHistosTkNPixelHits2D[4]->fill(jetFlavour, trackQual, track->hitPattern().numberOfValidPixelHits(), true,w);
    if(n >= 4) continue;
    tkcntHistosSig2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.significance(), true,w);
    tkcntHistosVal2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.value(), true,w);
    tkcntHistosErr2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip2d.error(), true,w);
    tkcntHistosDecayLengthVal2D[n]->fill(jetFlavour, trackQual, decayLen, true,w);
    tkcntHistosJetDistVal2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.value(), true,w);
    //tkcntHistosJetDistSign2D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.significance(), true,w);
    tkcntHistosTkNChiSqr2D[n]->fill(jetFlavour, trackQual, track->normalizedChi2(), true,w);
    tkcntHistosTkPt2D[n]->fill(jetFlavour, trackQual, track->pt(), true,w);
    tkcntHistosTkNHits2D[n]->fill(jetFlavour, trackQual, track->found(), true,w);
    tkcntHistosTkNPixelHits2D[n]->fill(jetFlavour, trackQual, track->hitPattern().numberOfValidPixelHits(), true,w);
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::Prob2D);
  selectedIndices.clear();
  sortedTracks = tagInfo->sortedTracks(sortedIndices);
  selectedTracks.clear();
  for(unsigned int n = 0; n != sortedIndices.size(); ++n) {
    double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
    if(decayLength > minDecayLength && decayLength < maxDecayLength &&
       fabs(jetDistance) >= minJetDistance && fabs(jetDistance) < maxJetDistance ) {
      selectedIndices.push_back(sortedIndices[n]);
      selectedTracks.push_back(sortedTracks[n]);
    }
  }
  for(unsigned int n=0; n != selectedIndices.size(); ++n) {
    const reco::TrackRef& track = selectedTracks[n];
    const reco::TrackBase::TrackQuality& trackQual = highestTrackQual(track);
    tkcntHistosProb2D[4]->fill(jetFlavour, trackQual, prob2d[selectedIndices[n]], true,w);
    if(ip[selectedIndices[n]].ip2d.value() < 0) tkcntHistosTkProbIPneg2D->fill(jetFlavour, trackQual, prob2d[selectedIndices[n]], true,w);
    else tkcntHistosTkProbIPpos2D->fill(jetFlavour, trackQual, prob2d[selectedIndices[n]], true,w);
    if(n >= 4) continue;
    tkcntHistosProb2D[n]->fill(jetFlavour, trackQual, prob2d[selectedIndices[n]], true,w);
  }
  for(unsigned int n=selectedIndices.size(); n < 4; ++n){
    const reco::TrackBase::TrackQuality trackQual = reco::TrackBase::undefQuality;
    tkcntHistosSig2D[n]->fill(jetFlavour, trackQual, lowerIPSBound-1.0, false,w);
    tkcntHistosVal2D[n]->fill(jetFlavour, trackQual, lowerIPBound-1.0, false,w);
    tkcntHistosErr2D[n]->fill(jetFlavour, trackQual, lowerIPEBound-1.0, false,w);
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::IP3DSig);
  selectedIndices.clear();
  sortedTracks = tagInfo->sortedTracks(sortedIndices);
  selectedTracks.clear();
  for(unsigned int n = 0; n != sortedIndices.size(); ++n) {
    double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
    if(decayLength > minDecayLength && decayLength < maxDecayLength &&
       fabs(jetDistance) >= minJetDistance && fabs(jetDistance) < maxJetDistance ) {
      selectedIndices.push_back(sortedIndices[n]);
      selectedTracks.push_back(sortedTracks[n]);
    }
  }

  trkNbr3D->fill(jetFlavour, selectedIndices.size(),w);
  int nSelectedTracks = selectedIndices.size();

  for(unsigned int n=0; n != selectedIndices.size(); ++n) {
    const reco::TrackRef& track = selectedTracks[n];
    const reco::TrackBase::TrackQuality& trackQual = highestTrackQual(track);
    tkcntHistosSig3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.significance(), true,w);
    tkcntHistosVal3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.value(), true,w);
    tkcntHistosErr3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.error(), true,w);
    const double& decayLen = (ip[selectedIndices[n]].closestToJetAxis - pv).mag();
    tkcntHistosDecayLengthVal3D[4]->fill(jetFlavour, trackQual, decayLen, true,w);
    tkcntHistosJetDistVal3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.value(), true,w);
    //tkcntHistosJetDistSign3D[4]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.significance(), true,w);
    tkcntHistosTkNChiSqr3D[4]->fill(jetFlavour, trackQual, track->normalizedChi2(), true,w);
    tkcntHistosTkPt3D[4]->fill(jetFlavour, trackQual, track->pt(), true,w);
    tkcntHistosTkNHits3D[4]->fill(jetFlavour, trackQual, track->found(), true,w);
    tkcntHistosTkNPixelHits3D[4]->fill(jetFlavour, trackQual, track->hitPattern().numberOfValidPixelHits(), true,w);
    //ghostTrack infos  
    ghostTrackDistanceValuHisto->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToGhostTrack.value(), true,w);
    ghostTrackDistanceSignHisto->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToGhostTrack.significance(), true,w);
    ghostTrackWeightHisto->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ghostTrackWeight, true,w);
    selectedTrackQualHisto->fill(jetFlavour, trackQual,w);
    if(n >= 4) continue;
    tkcntHistosSig3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.significance(), true,w);
    tkcntHistosVal3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.value(), true,w);
    tkcntHistosErr3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].ip3d.error(), true,w);
    tkcntHistosDecayLengthVal3D[n]->fill(jetFlavour, trackQual, decayLen, true,w);
    tkcntHistosJetDistVal3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.value(), true,w);
    //tkcntHistosJetDistSign3D[n]->fill(jetFlavour, trackQual, ip[selectedIndices[n]].distanceToJetAxis.significance(), true,w);
    tkcntHistosTkNChiSqr3D[n]->fill(jetFlavour, trackQual, track->normalizedChi2(), true,w);
    tkcntHistosTkPt3D[n]->fill(jetFlavour, trackQual, track->pt(), true,w);
    tkcntHistosTkNHits3D[n]->fill(jetFlavour, trackQual, track->found(), true,w);
    tkcntHistosTkNPixelHits3D[n]->fill(jetFlavour, trackQual, track->hitPattern().numberOfValidPixelHits(), true,w);
  }
  sortedIndices = tagInfo->sortedIndexes(reco::TrackIPTagInfo::Prob3D);
  selectedIndices.clear();
  sortedTracks = tagInfo->sortedTracks(sortedIndices);
  selectedTracks.clear();
  for(unsigned int n = 0; n != sortedIndices.size(); ++n) {
    double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
    double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
    if(decayLength > minDecayLength && decayLength < maxDecayLength &&
       fabs(jetDistance) >= minJetDistance && fabs(jetDistance) < maxJetDistance ) {
      selectedIndices.push_back(sortedIndices[n]);
      selectedTracks.push_back(sortedTracks[n]);
    }
  }
  for(unsigned int n=0; n != selectedIndices.size(); ++n) {
    const reco::TrackRef& track = selectedTracks[n];
    const reco::TrackBase::TrackQuality& trackQual = highestTrackQual(track);
    tkcntHistosProb3D[4]->fill(jetFlavour, trackQual, prob3d[selectedIndices[n]], true,w);
    if(ip[selectedIndices[n]].ip3d.value() < 0) tkcntHistosTkProbIPneg3D->fill(jetFlavour, trackQual, prob3d[selectedIndices[n]], true,w);
    else tkcntHistosTkProbIPpos3D->fill(jetFlavour, trackQual, prob3d[selectedIndices[n]], true,w);
    if(n >= 4) continue;
    tkcntHistosProb3D[n]->fill(jetFlavour, trackQual, prob3d[selectedIndices[n]], true,w);
  }
  for(unsigned int n=selectedIndices.size(); n < 4; ++n){
    const reco::TrackBase::TrackQuality trackQual = reco::TrackBase::undefQuality;
    tkcntHistosSig3D[n]->fill(jetFlavour, trackQual, lowerIPSBound-1.0, false,w);
    tkcntHistosVal3D[n]->fill(jetFlavour, trackQual, lowerIPBound-1.0, false,w);
    tkcntHistosErr3D[n]->fill(jetFlavour, trackQual, lowerIPEBound-1.0, false,w);
  }
  for(unsigned int n = 0; n != tagInfo->tracks().size(); ++n) {
    trackQualHisto->fill(jetFlavour, highestTrackQual(tagInfo->tracks()[n]),w);
  }

  //still need to implement weights in FlavourHistograms2D
  trackMultVsJetPtHisto->fill(jetFlavour, tagInfo->jet()->pt()*jec, tagInfo->tracks().size());
  selectedTrackMultVsJetPtHisto->fill(jetFlavour, tagInfo->jet()->pt()*jec, nSelectedTracks); //tagInfo->selectedTracks().size());
}

void TrackIPTagPlotter::createPlotsForFinalize (DQMStore::IBooker & ibook){
  const std::string trackIPDir(theExtensionString.substr(1));
  effPurFromHistos[0] = new EffPurFromHistos (tkcntHistosSig3D[1],trackIPDir, mcPlots_, ibook, 
					      nBinEffPur_, startEffPur_,
					      endEffPur_);
  effPurFromHistos[1] = new EffPurFromHistos (tkcntHistosSig3D[2],trackIPDir, mcPlots_, ibook,
					      nBinEffPur_, startEffPur_,
					      endEffPur_);
  effPurFromHistos[2] = new EffPurFromHistos (tkcntHistosSig2D[1],trackIPDir, mcPlots_, ibook,
					      nBinEffPur_, startEffPur_,
					      endEffPur_);
  effPurFromHistos[3] = new EffPurFromHistos (tkcntHistosSig2D[2],trackIPDir, mcPlots_, ibook,
					      nBinEffPur_, startEffPur_,
					      endEffPur_);
}


void TrackIPTagPlotter::finalize ()
{
  //
  // final processing:
  // produce the misid. vs. eff histograms
  //
  const std::string trackIPDir(theExtensionString.substr(1));
  effPurFromHistos[0] = new EffPurFromHistos (tkcntHistosSig3D[1],trackIPDir, mcPlots_, ibook_,
					      nBinEffPur_, startEffPur_,
					      endEffPur_);
  effPurFromHistos[1] = new EffPurFromHistos (tkcntHistosSig3D[2],trackIPDir, mcPlots_, ibook_,
					      nBinEffPur_, startEffPur_,
					      endEffPur_);
  effPurFromHistos[2] = new EffPurFromHistos (tkcntHistosSig2D[1],trackIPDir, mcPlots_, ibook_,
					      nBinEffPur_, startEffPur_,
					      endEffPur_);
  effPurFromHistos[3] = new EffPurFromHistos (tkcntHistosSig2D[2],trackIPDir, mcPlots_, ibook_,
					      nBinEffPur_, startEffPur_,
					      endEffPur_);
  for(int n=0; n != 4; ++n) effPurFromHistos[n]->compute(ibook_);
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
  tkcntHistosTkProbIPneg2D->epsPlot(name);
  tkcntHistosTkProbIPpos2D->epsPlot(name);
  tkcntHistosTkProbIPneg3D->epsPlot(name);
  tkcntHistosTkProbIPpos3D->epsPlot(name);
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
    //tkcntHistosJetDistSign2D[n]->epsPlot(name);
    //tkcntHistosJetDistSign3D[n]->epsPlot(name);
    tkcntHistosTkNChiSqr2D[n]->epsPlot(name);
    tkcntHistosTkNChiSqr3D[n]->epsPlot(name);
    tkcntHistosTkPt2D[n]->epsPlot(name);
    tkcntHistosTkPt3D[n]->epsPlot(name);
    tkcntHistosTkNHits2D[n]->epsPlot(name);
    tkcntHistosTkNHits3D[n]->epsPlot(name);    
    tkcntHistosTkNPixelHits2D[n]->epsPlot(name);
    tkcntHistosTkNPixelHits3D[n]->epsPlot(name);

  }
  if (finalized) {
    for(int n=0; n != 4; ++n) effPurFromHistos[n]->epsPlot(name);
  }
}

reco::TrackBase::TrackQuality TrackIPTagPlotter::highestTrackQual(const reco::TrackRef& track) const {
  for(reco::TrackBase::TrackQuality i = reco::TrackBase::highPurity; i != reco::TrackBase::undefQuality; i = reco::TrackBase::TrackQuality(i - 1))
  {
    if(track->quality(i))
      return i;
  }

  return reco::TrackBase::undefQuality;
}
