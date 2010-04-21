{
gROOT->SetStyle("Plain");
int n = 5;

double isoCut[n] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
double isoCutErr[n] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
double chi2[n] = { 1.0813, 0.9948, 0.964385, 0.992457, 0.989026 };
double zYield[n] = { 8818.69, 8832.02, 8827.16,  8804.38, 8802.99 };
double zYieldErr[n] = { 102.324, 102.079, 98.0602, 97.502, 95.9208 };
double effIso[n] = { 0.935329, 0.960251, 0.980071, 0.987804, 0.992889 };
double effIsoErr[n] = { 0.00235985, 0.00187779, 0.00133067, 0.00113768, 0.000920907 };
double effSa[n] = { 0.932455 , 0.932456, 0.933515, 0.934257, 0.93411 };
double effSaErr[n] = { 0.0025364, 0.0024718, 0.00233289, 0.00219377, 0.00228775 }; 
double effTk[n] = { 0.996802, 0.996852, 0.996798, 0.99686, 0.996899 };
double effTkErr[n] = { 0.000584946, 0.000550869, 0.000529331, 0.000509021, 0.000526273 };
double relErr[n];
for(unsigned int i = 0; i < n; ++i) relErr[i] = zYieldErr[i] / zYield[i];

gStyle->SetOptStat(kFALSE);
TCanvas canvas ("canvas","Isolation Study", 200, 10, 700, 500);

TH2D zYieldFrame("frame", "", 1, 0.5, 5.5, 1, 8500, 9200);
TGraphErrors zYieldGraph(n, isoCut, zYield, isoCutErr, zYieldErr);
double mcX[2] = { isoCut[0], isoCut[n-1] };
double mcY[2] = { 8958, 8958 }; 
TGraph mc(2, mcX, mcY);
mc.SetLineWidth(2);
mc.SetLineColor(kRed);
zYieldFrame.GetXaxis()->SetTitle("isolation cut (GeV/c)");
zYieldFrame.GetYaxis()->SetTitle("Z yield from fit");
zYieldGraph.SetMarkerStyle(21);
zYieldGraph.SetLineWidth(2);
zYieldFrame.Draw();
zYieldGraph.Draw("LP");
mc.Draw("L");
canvas.Update();
canvas.SaveAs("zYieldVsIso.eps");

TH2D effIsoFrame("frame", "", 1, 0.5, 5.5, 1, 0.92, 1);
TGraphErrors effIsoGraph(n, isoCut, effIso, isoCutErr, effIsoErr);
effIsoFrame.GetXaxis()->SetTitle("isolation cut (GeV/c)");
effIsoFrame.GetYaxis()->SetTitle("isolation efficiency from fit");
effIsoGraph.SetMarkerStyle(21);
effIsoGraph.SetLineWidth(2);
effIsoFrame.Draw();
effIsoGraph.Draw("LP");
canvas.Update();
canvas.SaveAs("effIsoVsIso.eps");
{
TH2D effSaFrame("frame", "", 1, 0.5, 5.5, 1, 0.92, 0.95);
double mcX[2] = { isoCut[0], isoCut[n-1] };
double mcY[2] = { .938, .938 }; 
TGraph mc(2, mcX, mcY);
mc.SetLineWidth(2);
mc.SetLineColor(kRed);
TGraphErrors effSaGraph(n, isoCut, effSa, isoCutErr, effSaErr);
effSaFrame.GetXaxis()->SetTitle("isolation cut (GeV/c)");
effSaFrame.GetYaxis()->SetTitle("stand-alone efficiency from fit");
effSaGraph.SetMarkerStyle(21);
effSaGraph.SetLineWidth(2);
effSaFrame.Draw();
effSaGraph.Draw("LP");
mc.Draw("L");
canvas.Update();
canvas.SaveAs("effSaVsIso.eps");
}
{
TH2D effTkFrame("frame", "", 1, 0.5, 5.5, 1, 0.994, 1);
double mcX[2] = { isoCut[0], isoCut[n-1] };
double mcY[2] = { .9956, .9956 }; 
TGraph mc(2, mcX, mcY);
mc.SetLineWidth(2);
mc.SetLineColor(kRed);
TGraphErrors effTkGraph(n, isoCut, effTk, isoCutErr, effTkErr);
effTkFrame.GetXaxis()->SetTitle("isolation cut (GeV/c)");
effTkFrame.GetYaxis()->SetTitle("tracker efficiency from fit");
effTkGraph.SetMarkerStyle(21);
effTkGraph.SetLineWidth(2);
effTkFrame.Draw();
effTkGraph.Draw("LP");
mc.Draw("L");
canvas.Update();
canvas.SaveAs("effTkVsIso.eps");
}
TH2D chi2Frame("frame", "", 1, 0.5, 5.5, 1, 0, 2.0);
TGraph chi2Graph(n, isoCut, chi2);
chi2Frame.GetXaxis()->SetTitle("isolation cut (GeV/c)");
chi2Frame.GetYaxis()->SetTitle("fit #chi^{2}");
chi2Graph.SetMarkerStyle(21);
chi2Graph.SetLineWidth(2);
chi2Frame.Draw();
chi2Graph.Draw("LP");
canvas.Update();
canvas.SaveAs("chi2VsIso.eps");

TH2D relErrFrame("frame", "", 1, 0.5, 5.5, 1, 0.0105, 0.012);
TGraph relErrGraph(n, isoCut, relErr);
relErrFrame.GetXaxis()->SetTitle("isolation cut (GeV/c)");
relErrFrame.GetXaxis()->SetTitleOffset(-0.2);
relErrFrame.GetYaxis()->SetTitle("zYield relative error");
relErrGraph.SetMarkerStyle(21);
relErrGraph.SetLineWidth(2);
relErrFrame.Draw();
relErrGraph.Draw("LP");
canvas.Update();
canvas.SaveAs("relErrVsIso.eps");



}
