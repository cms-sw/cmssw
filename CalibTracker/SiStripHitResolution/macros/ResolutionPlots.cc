#include <TH2F.h>

void ResolutionPlots_HistoMaker(const std::string& unit) {
  int n = 20;

  float numbers[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
                     11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};

  std::string YAxisTitle;
  std::string HistoTitle;
  std::string OutputFile;
  std::string RootFile;

  float UL[20] = {0};
  float NonUL[20] = {0};

  if (unit == "centimetres") {
    float UL_Array[] = {0.00137706, 0.00142804, 0.00223064, 0.00313546, 0.00528442, 0.00328381, 0.00365521,
                        0.00362465, 0.00404834, 0.00304523, 0.00154502, 0.00103244, 0.0108947,  0,
                        0.0030021,  0.00229157, 0.00244635, 0.00576464, 0.00461518, 0.00464276};

    float NonUL_Array[] = {0.00211815, 0.00243601, 0.002224,   0.00223777, 0.00284721, 0.00306343, 0.00310523,
                           0.00334504, 0.00333095, 0.00326225, 0.00331065, 0.00237821, 0.00233033, 0,
                           0.00285122, 0.00287095, 0.00226949, 0.0030951,  0.00284769, 0.00285031};

    for (int i = 0; i < 20; i++) {
      UL[i] = UL_Array[i];
      NonUL[i] = NonUL_Array[i];
    }

    YAxisTitle = "Resolution [cm]";
    HistoTitle = "Resolution values for UL and non-UL samples, in centimetres";
    OutputFile = "ResolutionComparison_ULAndNonUL_Centimetres.pdf";
    RootFile = "ResolutionComparison_ULAndNonUL_Centimetres.root";

  } else if (unit == "pitch units") {
    float UL_Array[] = {0.172131, 0.143091, 0.161787, 0.126722,  0.17909,   0.197285,  0.180396,
                        0.170818, 0.182258, 0.166405, 0.0844271, 0.0846207, 0.0400775, 0,
                        0.120119, 0.171899, 0.160656, 0.18299,   0.177929,  0.178037};

    float NonUL_Array[] = {0.153758, 0.151801, 0.14859,  0.148245, 0.147986, 0.146962, 0.147919,
                           0.147431, 0.146219, 0.145619, 0.14549,  0.147042, 0.147267, 0,
                           0.146873, 0.153169, 0.151639, 0.146694, 0.148681, 0.148683};

    for (int i = 0; i < 20; i++) {
      UL[i] = UL_Array[i];
      NonUL[i] = NonUL_Array[i];
    }

    YAxisTitle = "Resolution [pitch units]";
    HistoTitle = "Resolution values for the UL and non-UL samples in pitch units";
    OutputFile = "ResolutionComparison_ULAndNonUL_PitchUnits.pdf";
    RootFile = "ResolutionComparison_ULAndNonUL_PitchUnits.root";

  } else {
    std::cout << "ERROR: Unit must be centimetres or pitch units" << std::endl;
  }

  auto c1 = new TCanvas("c1", "c1", 800, 600);

  TGraph* gr1 = new TGraph(n, numbers, UL);
  gr1->SetName("UL samples");
  gr1->SetTitle("UL samples");
  gr1->SetMarkerStyle(21);
  gr1->SetDrawOption("AP");
  gr1->SetLineColor(0);
  gr1->SetLineWidth(4);
  gr1->SetFillStyle(0);

  TGraph* gr2 = new TGraph(n, numbers, NonUL);
  gr2->SetName("Non-UL samples");
  gr2->SetTitle("Non-UL samples");
  gr2->SetMarkerStyle(22);
  gr2->SetMarkerColor(2);
  gr2->SetDrawOption("P");
  gr2->SetLineColor(0);
  gr2->SetLineWidth(0);
  gr2->SetFillStyle(0);

  TMultiGraph* mg = new TMultiGraph();
  mg->Add(gr1);
  mg->Add(gr2);
  mg->GetHistogram()->SetTitle(HistoTitle.c_str());
  mg->GetHistogram()->SetTitleOffset(0.05);
  mg->GetHistogram()->GetXaxis()->SetTitle("Region");
  mg->GetHistogram()->GetXaxis()->SetTitleOffset(2.5);
  c1->SetBottomMargin(0.2);

  mg->GetHistogram()->GetXaxis()->SetBinLabel(1.0, "TIB L1");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(2.0), "TIB L2");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(3.0), "TIB L3");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(4.0), "TIB L4");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(5.0), "Side TID");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(6.0), "Wheel TID");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(7.0), "Ring TID");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(8.0), "TOB L1");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(9.0), "TOB L2");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(10.0), "TOB L3");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(11.0), "TOB L4");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(12.0), "TOB L5");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(13.0), "TOB L6");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(14.0), "Side TEC");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(15.0), "Wheel TEC");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(16.0), "Ring TEC");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(17.0), "TIB (All)");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(18.0), "TOB (All)");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(mg->GetHistogram()->GetXaxis()->FindBin(19.0), "TID (All)");
  mg->GetHistogram()->GetXaxis()->SetBinLabel(100, "TEC (All)");

  mg->GetHistogram()->GetYaxis()->SetTitle(YAxisTitle.c_str());
  mg->Draw("ALP");
  c1->BuildLegend();
  mg->SaveAs(OutputFile.c_str());

  TFile* output = new TFile(RootFile.c_str(), "RECREATE");
  output->cd();
  mg->Write();
  output->Close();
}

void ResolutionPlots() {
  ResolutionPlots_HistoMaker("centimetres");
  ResolutionPlots_HistoMaker("pitch units");
}
