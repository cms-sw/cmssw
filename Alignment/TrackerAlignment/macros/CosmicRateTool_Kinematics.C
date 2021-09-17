#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <cmath>
#include <TString.h>

void Get_Plot(TH1D, TString);

void CosmicRateTool_Kinematics(const char *fileName) {
  TString InputFile = Form("%s", fileName);
  TFile *file = new TFile(InputFile);

  bool IsFileExist;
  IsFileExist = file->IsZombie();
  if (IsFileExist) {
    cout << endl
         << "====================================================================================================="
         << endl;
    cout << fileName << " is not found. Check the file!" << endl;
    cout << "====================================================================================================="
         << endl
         << endl;
    exit(EXIT_FAILURE);
  }

  TTree *tree = (TTree *)file->Get("cosmicRateAnalyzer/Event");

  vector<double> *pt;
  vector<double> *charge;
  vector<double> *chi2;
  vector<double> *chi2_ndof;
  vector<double> *eta;
  vector<double> *theta;
  vector<double> *phi;
  vector<double> *p;
  vector<double> *d0;
  vector<double> *dz;

  pt = 0;
  charge = 0;
  chi2 = 0;
  chi2_ndof = 0;
  eta = 0;
  theta = 0;
  phi = 0;
  p = 0;
  d0 = 0;
  dz = 0;

  tree->SetBranchAddress("pt", &pt);
  tree->SetBranchAddress("charge", &charge);
  tree->SetBranchAddress("chi2", &chi2);
  tree->SetBranchAddress("chi2_ndof", &chi2_ndof);
  tree->SetBranchAddress("eta", &eta);
  tree->SetBranchAddress("theta", &theta);
  tree->SetBranchAddress("phi", &phi);
  tree->SetBranchAddress("p", &p);
  tree->SetBranchAddress("d0", &d0);
  tree->SetBranchAddress("dz", &dz);

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  //					Various Kinematical Histograms Declerations
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  TH1D h_pt("h_pt", "h_pt", 100, 0, 100);
  TH1D h_charge("h_charge", "h_charge", 10, -5, 5);
  TH1D h_chi2("h_chi2", "h_chi2", 200, 0, 100);
  TH1D h_chi2_ndof("h_chi2_ndof", "h_chi2_ndof", 100, 0, 10);
  TH1D h_eta("h_eta", "h_eta", 500, -3, 3);
  TH1D h_theta("h_theta", "h_theta", 500, -3, 3);
  TH1D h_phi("h_phi", "h_phi", 400, -3.5, 3.5);
  TH1D h_d0("h_d0", "h_d0", 1000, -85, 85);
  TH1D h_dz("h_dz", "h_dz", 1500, -350, 350);

  //----------------------------------------------------------------------------------------------------------------

  int nTotalEvents = 0, nTotalTracks = 0;
  Long64_t n = tree->GetEntriesFast();
  for (Long64_t jentry = 0; jentry < n; jentry++)  // Loop over events
  {
    tree->GetEntry(jentry);

    for (int k = 0; k < pt->size(); k++)  // Loop over tracks
    {
      h_pt.Fill(pt->at(k));
      h_charge.Fill(charge->at(k));
      h_chi2.Fill(chi2->at(k));
      h_chi2_ndof.Fill(chi2_ndof->at(k));
      h_eta.Fill(eta->at(k));
      h_theta.Fill(theta->at(k));
      h_phi.Fill(phi->at(k));
      h_d0.Fill(d0->at(k));
      h_dz.Fill(dz->at(k));

      nTotalTracks++;
    }  // Tracks Loop

    nTotalEvents++;
  }  // Event Loop

  std::cout << "Total Events: " << nTotalEvents << std::endl;
  std::cout << "Total Tracks: " << nTotalTracks << std::endl;

  //++++++++++++++++++++++++++++++++++       Make Directory     ++++++++++++++++++++++++++++++++++++++
  gSystem->Exec("mkdir -p Kinematical_Plots");

  //++++++++++++++++++++++++++++++++++         Plotting         ++++++++++++++++++++++++++++++++++++++
  Get_Plot(h_pt, "pt");
  Get_Plot(h_eta, "eta");
  Get_Plot(h_phi, "phi");
  Get_Plot(h_theta, "theta");
  Get_Plot(h_d0, "d0");
  Get_Plot(h_dz, "dz");
  Get_Plot(h_chi2, "chi2");
  Get_Plot(h_chi2_ndof, "chi2_ndof");
  Get_Plot(h_charge, "charge");
}

void Get_Plot(TH1D h1, TString variable) {
  TCanvas c("c", "c", 556, 214, 661, 641);
  gStyle->SetOptStat(0);   // Dont show statistics
  gStyle->SetOptTitle(0);  // Dont show Title
  c.Range(-7.156863, -810349, 5.764706, 4951034);
  c.SetFillColor(0);
  c.SetBorderMode(0);
  c.SetBorderSize(3);
  c.SetGridx();
  c.SetGridy();
  c.SetTickx(1);
  c.SetTicky(1);
  c.SetLeftMargin(0.1669196);
  c.SetRightMargin(0.05918058);
  c.SetTopMargin(0.08233276);
  c.SetBottomMargin(0.1406518);
  c.SetFrameLineWidth(3);
  c.SetFrameBorderMode(0);
  c.SetFrameLineWidth(3);
  c.SetFrameBorderMode(0);

  TGaxis::SetMaxDigits(3);

  h1.SetLineColor(kRed);
  h1.SetLineWidth(3);

  //---- X-axis Titles -----//
  TString TempEta = "eta", TempChi2 = "chi2";
  if (variable.Contains("pt")) {
    h1.SetXTitle("Track p_{T} (GeV)");
  } else if (variable.Contains("charge")) {
    h1.SetXTitle("Track charge (e)");
  } else if (variable.Data() == TempEta) {
    h1.SetXTitle("Track #eta");
  } else if (variable.Contains("phi")) {
    h1.SetXTitle("Track #phi (rad)");
  } else if (variable.Contains("theta")) {
    h1.SetXTitle("Track #theta (rad)");
  } else if (variable.Contains("d0")) {
    h1.SetXTitle("Track d_{0} (cm)");
  } else if (variable.Contains("dz")) {
    h1.SetXTitle("Track d_{z} (cm)");
  } else if (variable.Data() == TempChi2) {
    h1.SetXTitle("Track #chi^{2}");
  } else if (variable.Contains("ndof")) {
    h1.SetXTitle("Track #chi^{2} per NDF");
  } else {
    std::cout << "Title does not match anything in the categories defined!" << std::endl;
  }

  h1.SetYTitle("Tracks (#)");
  h1.SetLabelSize(0.05);
  h1.GetXaxis()->SetLabelSize(0.05);
  h1.GetXaxis()->SetTitleSize(0.05);
  h1.GetYaxis()->SetLabelSize(0.05);
  h1.GetYaxis()->SetTitleSize(0.06);
  h1.GetXaxis()->SetTitleOffset(1.12);
  h1.Draw();

  // Text on upper right corner of plot //
  TLatex Title = TLatex();
  Title.SetTextFont(42);
  Title.SetTextSize(0.039);
  Title.DrawLatexNDC(0.76, 0.94, "cosmic rays");  //Bv1

  //============== Saving as PDF, png and C ============= //
  TString PlotFormat[] = {"png", "pdf", "C"};
  for (int k = 0; k < 3; k++) {
    TString Format = variable + "." + PlotFormat[k];
    c.SaveAs(Format.Data());
    TString mv_folder_string = "mv " + Format + " Kinematical_Plots";
    gSystem->Exec(mv_folder_string.Data());
  }
  c.Close();
}
