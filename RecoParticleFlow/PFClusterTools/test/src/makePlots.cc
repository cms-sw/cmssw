void makePlots() {
	TFile file("Exercises.root", "update");
	TTree* tv__tree = (TTree *) gROOT->FindObject("CalibratedParticles");

	TDirectory* plots = file.mkdir("plots");
	file.cd("/plots");

	plots->mkdir("linearity");
	file.cd("/plots/linearity");

	tv__tree->Draw(
			"calibrations_.particleEnergy_:sim_energyEvent_>>uncalibrated",
			"calibrations_.provenance_ == 0", "box");
	TH2D* uncalibrated = (TH2D*) gDirectory->Get("uncalibrated");
	uncalibrated->FitSlicesY();
	uncalibrated->ProfileX();
	TProfile
			* uncalibrated_pfx = (TProfile*) gDirectory->Get("uncalibrated_pfx");
	uncalibrated_pfx->Write();
	TH1D* uncalibrated_1 = (TH1D*) gDirectory->Get("uncalibrated_1");
	uncalibrated_1->SetTitle("Uncalibrated fitted mean");
	uncalibrated_1->SetXTitle("sim_energyEvent_");
	uncalibrated_1->SetYTitle("calibrations_.particleEnergy_");
	uncalibrated_1->SetMarkerStyle(21);
	uncalibrated_1->SetMarkerColor(kRed);
	uncalibrated_1->Write();
	uncalibrated->Write();

	tv__tree->Draw(
			"calibrations_.particleEnergy_:sim_energyEvent_>>calibrated",
			"calibrations_.provenance_ != 0", "box");
	TH2D* calibrated = (TH2D*) gDirectory->Get("calibrated");
	calibrated->FitSlicesY();
	calibrated->ProfileX();
	TProfile* calibrated_pfx = (TProfile*) gDirectory->Get("calibrated_pfx");
	calibrated_pfx->Write();
	TH1D* calibrated_1 = (TH1D*) gDirectory->Get("calibrated_1");
	calibrated_1->SetTitle("Calibrated fitted mean");
	calibrated_1->SetXTitle("sim_energyEvent_");
	calibrated_1->SetYTitle("calibrations_.particleEnergy_");
	calibrated_1->SetMarkerStyle(22);
	calibrated_1->SetMarkerColor(kViolet + 7);
	calibrated_1->Write();
	calibrated->Write();

	file.cd("/");
	plots->mkdir("bias");
	file.cd("/plots/bias");

	tv__tree->Draw("calibrations_.bias_:sim_energyEvent_>>calibbias",
			"calibrations_.provenance_ != 0", "box");
	TH2D* calibbias = (TH2D*) gDirectory->Get("calibbias");
	calibbias->FitSlicesY();
	calibbias->ProfileX();
	TProfile* calibbias_pfx = (TProfile*) gDirectory->Get("calibbias_pfx");
	calibbias_pfx->Write();
	TH1D* calibbias_1 = (TH1D*) gDirectory->Get("calibbias_1");
	calibbias_1->SetTitle("Calibrated bias");
	calibbias_1->SetXTitle("sim_energyEvent_");
	calibbias_1->SetYTitle("calibrations_.bias_");
	calibbias_1->SetMarkerStyle(21);
	calibbias_1->SetMarkerColor(kViolet+7);
	calibbias_1->Write();
	calibbias->Write();

	tv__tree->Draw("calibrations_.bias_:sim_energyEvent_>>uncalibbias",
			"calibrations_.provenance_ == 0", "box");
	TH2D* uncalibbias = (TH2D*) gDirectory->Get("uncalibbias");
	uncalibbias->FitSlicesY();
	uncalibbias->ProfileX();
	TProfile* uncalibbias_pfx = (TProfile*) gDirectory->Get("uncalibbias_pfx");
	uncalibbias_pfx->Write();
	TH1D* uncalibbias_1 = (TH1D*) gDirectory->Get("uncalibbias_1");
	uncalibbias_1->SetTitle("Uncalibrated bias");
	uncalibbias_1->SetXTitle("sim_energyEvent_");
	uncalibbias_1->SetYTitle("calibrations_.bias_");
	uncalibbias_1->SetMarkerStyle(22);
	uncalibbias_1->SetMarkerColor(kRed);
	uncalibbias_1->Write();
	uncalibbias->Write();

	//Now deal with target functions
	file.cd("/");
	plots->mkdir("targetfn");
	file.cd("/plots/targetfn");

	tv__tree->Draw(
			"calibrations_.targetFuncContrib_:calibrations_.truthEnergy_>>calibtarg",
			"calibrations_.provenance_ !=0 ", "box");
	TH2D* calibtarg = (TH2D*) gDirectory->Get("calibtarg");
	calibtarg->Rebin2D(3, 1);
	double start = calibtarg->GetBinLowEdge(1);
	double width = calibtarg->GetBinWidth(1);
	double end = calibtarg->GetBinLowEdge(calibtarg->GetNbinsX() + 1);
	TH1F
			* calibtargfn = new TH1F("calibtargfn", "Calibrated target function;true energy;target", calibtarg->GetNbinsX(), start, end);
	TProfile* calibtarg_pfx = calibtarg->ProfileX();
	for (unsigned k(1); k < calibtarg_pfx->GetNbinsX() + 1; ++k) {
		double targfn = sqrt(calibtarg_pfx->GetBinContent(k)
				/(calibtarg_pfx->GetBinCenter(k)));
		calibtargfn->Fill(calibtarg_pfx->GetBinCenter(k), targfn);
		std::cout << "k: "<< k << ":\t"<< k * width << " = \t"<< targfn<< "\n";

	}
	calibtargfn->SetFillColor(kViolet+7);
	calibtargfn->Write();

	tv__tree->Draw(
			"calibrations_.targetFuncContrib_:calibrations_.truthEnergy_>>uncalibtarg",
			"calibrations_.provenance_==0", "box");
	TH2D* uncalibtarg = (TH2D*) gDirectory->Get("uncalibtarg");
	uncalibtarg->Rebin2D(3, 1);
	start = uncalibtarg->GetBinLowEdge(1);
	width = uncalibtarg->GetBinWidth(1);
	end = uncalibtarg->GetBinLowEdge(uncalibtarg->GetNbinsX() + 1);
	TH1F
			* uncalibtargfn = new TH1F("uncalibtargfn", "Uncalibrated target function;true energy;target", uncalibtarg->GetNbinsX(), start, end);
	TProfile* uncalibtarg_pfx = uncalibtarg->ProfileX();
	for (unsigned k(1); k < uncalibtarg_pfx->GetNbinsX() + 1; ++k) {
		double targfn = sqrt(uncalibtarg_pfx->GetBinContent(k)
				/(uncalibtarg_pfx->GetBinCenter(k)));
		uncalibtargfn->Fill(uncalibtarg_pfx->GetBinCenter(k), targfn);
		std::cout << "k: "<< k << ":\t"<< k * width << " = \t"<< targfn<< "\n";

	}
	uncalibtargfn->SetFillColor(kRed);
	uncalibtargfn->Write();

}
