{

gROOT->SetStyle("Plain");
gROOT->ForceStyle();
gStyle->SetOptStat(0);

gStyle->SetPalette(1); 
//gStyle->SetNumberContours(20);  // Default: 20

double width = 600.;

gStyle->SetCanvasDefW(width);
gStyle->SetCanvasDefH(width);

gStyle->SetPadLeftMargin(0.13);
gStyle->SetPadRightMargin(0.11);

gStyle->SetPadTopMargin(0.10);
gStyle->SetPadBottomMargin(0.13);

gStyle->SetTitleOffset(1.25,"Y");
gStyle->SetTitleOffset(1.13,"X");


TGaxis::SetMaxDigits(3);


gStyle->SetTitleX(0.25);        // move upper left corner of title box to specified value
//gStyle->SetTitleY(0.99);        // move upper left corner of title box to specified value


gStyle->SetTitleXSize(0.05);
gStyle->SetTitleYSize(0.05);
gStyle->SetTitleSize(0.05,"XY");
gStyle->SetLabelSize(0.05,"XY");




//++++++++++++++++++++++++++++++++++=====================================+++++++++++++++++++++++++++++++


/*

gROOT->ProcessLine(".L tdrstyle.C");
setTDRStyle();

*/


//++++++++++++++++++++++++++++++++++=====================================+++++++++++++++++++++++++++++++





gROOT->ProcessLine(".L DrawPlot.C+");



gROOT->ProcessLine("DrawPlot drawPlot1(0, true)");
drawPlot1.thesisMode();

drawPlot1.setLegendEntry("data (final #sigma_{align})","data (#sigma_{align}=0)","design");
gStyle->SetOptStat(111110);
drawPlot1.setLegendCoordinate(0.69, 0.70, 0.99, 0.85);
//drawPlot1.setLegendCoordinate(0.69, 0.20, 0.99, 0.35);
drawPlot1.drawPlot("ApeEstimatorSummary","h_weightX", false);
drawPlot1.drawPlot("ApeEstimatorSummary","h_weightY", false);
drawPlot1.setLegendCoordinate(0.69, 0.20, 0.99, 0.35);
drawPlot1.drawPlot("ApeEstimatorSummary","h_residualWidthX1", false);
drawPlot1.drawPlot("ApeEstimatorSummary","h_residualWidthY1", false);
drawPlot1.drawPlot("ApeEstimatorSummary","h_rmsX", false);
drawPlot1.drawPlot("ApeEstimatorSummary","h_rmsY", false);


gROOT->ProcessLine("DrawPlot drawPlot2(0, false)");
drawPlot2.thesisMode();

drawPlot2.setLegendEntry("data (final #sigma_{align})","data (#sigma_{align}=0)","design");
drawPlot2.setLegendCoordinate(0.69, 0.70, 0.99, 0.85);
//drawPlot2.setLegendCoordinate(0.69, 0.20, 0.99, 0.35);
gStyle->SetOptStat(111100);
drawPlot2.drawPlot("ApeEstimator1","h_entriesX", false);
drawPlot2.drawPlot("ApeEstimator1","h_entriesY", false);
drawPlot2.setLegendCoordinate(0.395, 0.845, 0.995, 0.995);
gStyle->SetOptStat(111110);
drawPlot2.drawPlot("ApeEstimator1","h_NorResX", false);
drawPlot2.drawPlot("ApeEstimator1","h_NorResY", false);
drawPlot2.drawPlot("ApeEstimator1","h_ResX", false);
drawPlot2.drawPlot("ApeEstimator1","h_ResY", false);

// Additional plots for strip (mainly; some are common with pixel)
drawPlot2.drawPlot("ApeEstimator2","h_ChargeStrip", false);
drawPlot2.drawPlot("ApeEstimator2","h_SOverN", false);
drawPlot2.drawPlot("ApeEstimator2","h_ChargeOnEdges", false);
drawPlot2.drawPlot("ApeEstimator2","h_BaryStripX", false);
drawPlot2.drawPlot("ApeEstimator2","h_WidthX", false);
drawPlot2.drawPlot("ApeEstimator2","h_PhiSensX", false);
drawPlot2.drawPlot("ApeEstimator2","h_sigmaXHit_1", false);
drawPlot2.drawPlot("ApeEstimator2","h_sigmaXTrk_1", false);

drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsChargeStrip", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsSOverN", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsChargeOnEdges", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsBaryStripX", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsPhiSensX", false);

drawPlot2.drawPlot("ApeEstimator2","p_sigmaXTrkVsD0Beamspot", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXTrkVsDz", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXTrkVsHitsPixel", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXTrkVsHitsStrip", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXTrkVsP", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXTrkVsPhiSensX", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXTrkVsPt", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXTrkVsTheta", false);

drawPlot2.drawPlot("ApeEstimator2","p_widthVsPhiSensX", false);

// Additional plots for pixel
drawPlot2.drawPlot("ApeEstimator2","h_ChargePixel", false);
drawPlot2.drawPlot("ApeEstimator2","h_ChargePixel_y", false);
drawPlot2.drawPlot("ApeEstimator2","h_BaryStripY_y", false);
drawPlot2.drawPlot("ApeEstimator2","h_ClusterProbXY", false);
drawPlot2.drawPlot("ApeEstimator2","h_ClusterProbXY_y", false);
drawPlot2.drawPlot("ApeEstimator2","h_LogClusterProb", false);
drawPlot2.drawPlot("ApeEstimator2","h_LogClusterProb_y", false);
drawPlot2.drawPlot("ApeEstimator2","h_PhiSensY_y", false);
drawPlot2.drawPlot("ApeEstimator2","h_WidthY_y", false);
drawPlot2.drawPlot("ApeEstimator2","h_sigmaYHit_1", false);
drawPlot2.drawPlot("ApeEstimator2","h_sigmaYTrk_1", false);

drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsChargePixel", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsClusterProbXY", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsLogClusterProb", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaXHitVsWidthX", false);

drawPlot2.drawPlot("ApeEstimator2","p_sigmaYHitVsChargePixel", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYHitVsBaryStripY", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYHitVsClusterProbXY", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYHitVsLogClusterProb", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYHitVsWidthY", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYHitVsPhiSensY", false);

drawPlot2.drawPlot("ApeEstimator2","p_sigmaYTrkVsD0Beamspot", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYTrkVsDz", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYTrkVsHitsPixel", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYTrkVsHitsStrip", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYTrkVsP", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYTrkVsPhiSensX", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYTrkVsPt", false);
drawPlot2.drawPlot("ApeEstimator2","p_sigmaYTrkVsTheta", false);

drawPlot2.drawPlot("ApeEstimator2","p_phiSensXVsBarycentreX", false);
drawPlot2.drawPlot("ApeEstimator2","p_phiSensYVsBarycentreY", false);



drawPlot2.setLegendCoordinate(0.395, 0.845, 0.995, 0.995);
gStyle->SetOptStat(111110);
drawPlot2.drawTrackPlot("ApeEstimator2","h_eta", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_etaErr", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_etaSig", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_theta", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_phi", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_phiErr", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_phiSig", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_norChi2", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_p", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_pt", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_ptErr", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_ptSig", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_charge", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_d0Beamspot", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_d0BeamspotErr", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_d0BeamspotSig", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_dz", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_dzErr", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_dzSig", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_prob", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_hitsPixel", false);
drawPlot2.drawTrackPlot("ApeEstimator2","h_hitsStrip", false);

drawPlot2.drawEventPlot("ApeEstimator2","h_trackSizeGood", false);




gROOT->ProcessLine("DrawPlot drawPlot3(15, true)");
drawPlot3.thesisMode();

drawPlot3.setLegendEntry("data (final #sigma_{align})","data (#sigma_{align}=0)","design");
drawPlot3.setLegendCoordinate(0.41, 0.27, 0.71, 0.42);
gStyle->SetOptStat(111110);
// drawPlot3.drawPlot("ApeEstimatorSummary","h_correctionX1", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","h_correctionX2", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","h_correctionY1", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","h_correctionY2", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","h_fitMeanX1", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","h_fitMeanX2", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","h_fitMeanY1", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","h_fitMeanY2", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","meanX", false);
// drawPlot3.drawPlot("ApeEstimatorSummary","meanY", false);
drawPlot3.setLegendCoordinate(0.69, 0.70, 0.99, 0.85);
drawPlot3.drawPlot("ApeEstimatorSummary","h_weightX", false);
drawPlot3.drawPlot("ApeEstimatorSummary","h_weightY", false);
drawPlot3.setLegendCoordinate(0.69, 0.20, 0.99, 0.35);
drawPlot3.drawPlot("ApeEstimatorSummary","h_residualWidthX1", false);
drawPlot3.drawPlot("ApeEstimatorSummary","h_residualWidthY1", false);
//drawPlot3.drawPlot("ApeEstimatorSummary","h_residualWidthX2", false);
//drawPlot3.drawPlot("ApeEstimatorSummary","h_residualWidthY2", false);
drawPlot3.drawPlot("ApeEstimatorSummary","h_rmsX", false);
drawPlot3.drawPlot("ApeEstimatorSummary","h_rmsY", false);



gROOT->ProcessLine("DrawPlot drawPlot4(15, false)");
drawPlot4.thesisMode();

drawPlot4.setLegendEntry("data (final #sigma_{align})","data (#sigma_{align}=0)","design");
drawPlot4.setLegendCoordinate(0.69, 0.70, 0.99, 0.85);
gStyle->SetOptStat(111100);
drawPlot4.drawPlot("ApeEstimator1","h_entriesX", false);
drawPlot4.drawPlot("ApeEstimator1","h_entriesY", false);
drawPlot4.setLegendCoordinate(0.395, 0.845, 0.995, 0.995);
gStyle->SetOptStat(111110);
drawPlot4.drawPlot("ApeEstimator1","h_NorResX");
drawPlot4.drawPlot("ApeEstimator1","h_NorResY");
drawPlot4.drawPlot("ApeEstimator1","h_ResX");
drawPlot4.drawPlot("ApeEstimator1","h_ResY");

drawPlot4.setLegendCoordinate(0.395, 0.845, 0.995, 0.995);
gStyle->SetOptStat(111110);
drawPlot4.drawTrackPlot("ApeEstimator2","h_eta");
drawPlot4.drawTrackPlot("ApeEstimator2","h_etaErr");
drawPlot4.drawTrackPlot("ApeEstimator2","h_etaSig");
drawPlot4.drawTrackPlot("ApeEstimator2","h_theta");
drawPlot4.drawTrackPlot("ApeEstimator2","h_phi");
drawPlot4.drawTrackPlot("ApeEstimator2","h_phiErr");
drawPlot4.drawTrackPlot("ApeEstimator2","h_phiSig");
drawPlot4.drawTrackPlot("ApeEstimator2","h_norChi2");
drawPlot4.drawTrackPlot("ApeEstimator2","h_p");
drawPlot4.drawTrackPlot("ApeEstimator2","h_pt");
drawPlot4.drawTrackPlot("ApeEstimator2","h_ptErr");
drawPlot4.drawTrackPlot("ApeEstimator2","h_ptSig");
drawPlot4.drawTrackPlot("ApeEstimator2","h_charge");
drawPlot4.drawTrackPlot("ApeEstimator2","h_d0Beamspot");
drawPlot4.drawTrackPlot("ApeEstimator2","h_d0BeamspotErr");
drawPlot4.drawTrackPlot("ApeEstimator2","h_d0BeamspotSig");
drawPlot4.drawTrackPlot("ApeEstimator2","h_dz");
drawPlot4.drawTrackPlot("ApeEstimator2","h_dzErr");
drawPlot4.drawTrackPlot("ApeEstimator2","h_dzSig");
drawPlot4.drawTrackPlot("ApeEstimator2","h_prob");
drawPlot4.drawTrackPlot("ApeEstimator2","h_hitsPixel");
drawPlot4.drawTrackPlot("ApeEstimator2","h_hitsStrip");

drawPlot4.drawEventPlot("ApeEstimator2","h_trackSizeGood");



gROOT->ProcessLine(".q");



}






