
void makePlots() {
	TFile file("Exercises.root", "update");
	TTree* tv__tree = (TTree *) gROOT->FindObject("CalibratedParticles");

	TDirectory* plots = file.mkdir("plots");
	file.cd("/plots");

	plots->mkdir("linearity");
	file.cd("/plots/linearity");

	tv__tree->Draw(
			"calibrations_.particleEnergy_:sim_energyEvent_>>uncalibrated",
			"calibrations_.provenance_ == 0", "box", 9906, 0);
	TH2D* uncalibrated = (TH2D*) gDirectory->Get("uncalibrated");
	uncalibrated->FitSlicesY();
	TH1D* uncalibrated_1 = (TH1D*) gDirectory->Get("uncalibrated_1");
	uncalibrated_1->SetTitle("Uncalibrated fitted mean");
	uncalibrated_1->SetXTitle("sim_energyEvent_");
	uncalibrated_1->SetYTitle("calibrations_.particleEnergy_");
	uncalibrated_1->SetMarkerStyle(21);
	uncalibrated_1->SetMarkerColor(kViolet+7);
	uncalibrated_1->Write();
	uncalibrated->Write();

	tv__tree->Draw(
			"calibrations_.particleEnergy_:sim_energyEvent_>>calibrated",
			"calibrations_.provenance_ == 1", "box", 9906, 0);
	TH2D* calibrated = (TH2D*) gDirectory->Get("calibrated");
	calibrated->FitSlicesY();
	TH1D* calibrated_1 = (TH1D*) gDirectory->Get("calibrated_1");
	calibrated_1->SetTitle("Calibrated fitted mean");
	calibrated_1->SetXTitle("sim_energyEvent_");
	calibrated_1->SetYTitle("calibrations_.particleEnergy_");
	calibrated_1->SetMarkerStyle(22);
	calibrated_1->SetMarkerColor(kRed);
	calibrated_1->Write();
	calibrated->Write();

	file.cd("/");
	plots->mkdir("bias");
	file.cd("/plots/bias");

	tv__tree->Draw("calibrations_.bias_:sim_energyEvent_>>calibbias",
			"calibrations_.provenance_ == 1", "box", 9906, 0);
	TH2D* calibbias = (TH2D*) gDirectory->Get("calibbias");
	calibbias->FitSlicesY();
	TH1D* calibbias_1 = (TH1D*) gDirectory->Get("calibbias_1");
	calibbias_1->SetTitle("Calibrated bias");
	calibbias_1->SetXTitle("sim_energyEvent_");
	calibbias_1->SetYTitle("calibrations_.bias_");
	calibbias_1->SetMarkerStyle(21);
	calibbias_1->SetMarkerColor(kViolet+7);
	calibbias_1->Write();
	calibbias->Write();

	tv__tree->Draw("calibrations_.bias_:sim_energyEvent_>>uncalibbias",
			"calibrations_.provenance_ == 0", "box", 9906, 0);
	TH2D* uncalibbias = (TH2D*) gDirectory->Get("uncalibbias");
	uncalibbias->FitSlicesY();
	TH1D* uncalibbias_1 = (TH1D*) gDirectory->Get("uncalibbias_1");
	uncalibbias_1->SetTitle("Uncalibrated bias");
	uncalibbias_1->SetXTitle("sim_energyEvent_");
	uncalibbias_1->SetYTitle("calibrations_.bias_");
	uncalibbias_1->SetMarkerStyle(22);
	uncalibbias_1->SetMarkerColor(kRed);
	uncalibbias_1->Write();
	uncalibbias->Write();

}