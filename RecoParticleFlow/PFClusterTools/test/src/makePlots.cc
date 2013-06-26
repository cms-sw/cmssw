void makePlots(TFile& file) {
	//TFile file("Exercises.root", "update");
	TTree* tv__tree = (TTree *) gROOT->FindObject("CalibratedParticles");
	//tv__tree->AddFriend("extraction/Extraction", "Exercises300.root");
	TDirectory* plots = file.mkdir("plots");
	file.cd("/plots");

	plots->mkdir("linearity");
	plots->mkdir("targetFunction");
	plots->mkdir("bias");
	plots->mkdir("ratios");
	plots->mkdir("resolutions");
	plots->mkdir("resolutions30");

	file.cd("/plots/linearity");

	tv__tree->Draw(
			"calibrations_.particleEnergy_:sim_energyEvent_>>uncalibrated",
			"calibrations_.provenance_ == 0", "box");
	doLinearityPlots(file, "uncalibrated", "Uncalibrated", kRed);
	doRatioPlots(file, "uncalibrated", "Uncalibrated", kRed);
	doTargetFunctions(file, "uncalibrated", "Uncalibrated", kRed);

	file.cd("/plots/linearity");
	tv__tree->Draw(
			"calibrations_.particleEnergy_:sim_energyEvent_>>calibrated",
			"calibrations_.provenance_ > 0", "box");
	doLinearityPlots(file, "calibrated", "Calibrated", kBlue);
	doRatioPlots(file, "calibrated", "Calibrated", kBlue);
	doTargetFunctions(file, "calibrated", "Calibrated", kBlue);

	file.cd("/plots/linearity");
	tv__tree->Draw(
			"calibrations_.particleEnergy_:sim_energyEvent_>>calibratedCorr",
			"calibrations_.provenance_ < 0", "box");
	doLinearityPlots(file, "calibratedCorr", "Calibrated and corrected",
			kViolet);
	doRatioPlots(file, "calibratedCorr", "Calibrated and corrected", kViolet);
	doTargetFunctions(file, "calibratedCorr", "Calibrated and corrected",
			kViolet);

	file.cd("/plots/ratios");
	tv__tree->Draw(
			"calibrations_.ratio_:sim_energyEvent_>>calibratedCorr_fullRatio",
			"calibrations_.provenance_ < 0", "col");
	doFullRatioPlots(file, "calibratedCorr", "Corrected", kViolet);
	tv__tree->Draw(
			"calibrations_.ratio_:sim_energyEvent_>>calibrated_fullRatio",
			"calibrations_.provenance_ > 0", "col");
	doFullRatioPlots(file, "calibrated", "Calibrated", kBlue);

	tv__tree->Draw(
			"calibrations_.ratio_:sim_energyEvent_>>uncalibrated_fullRatio",
			"calibrations_.provenance_ == 0", "col");
	doFullRatioPlots(file, "uncalibrated", "Uncalibrated", kRed);

	file.cd("/plots/bias");
	tv__tree->Draw("calibrations_.bias_:sim_energyEvent_>>calibrated_bias",
			"calibrations_.provenance_ > 0", "box");
	doBiasPlots(file, "calibrated_bias", "Calibrated", kBlue);

	tv__tree->Draw("calibrations_.bias_:sim_energyEvent_>>calibratedCorr_bias",
			"calibrations_.provenance_ < 0", "box");
	doBiasPlots(file, "calibratedCorr_bias", "Calibrated and corrected",
			kViolet);

	tv__tree->Draw("calibrations_.bias_:sim_energyEvent_>>uncalibrated_bias",
			"calibrations_.provenance_ == 0", "box");
	doBiasPlots(file, "uncalibrated_bias", "Uncalibrated", kRed);
	
	file.cd("/plots/resolutions30");
	doResolution(tv__tree,"(20,1.0,6.0,100,0,50)","calibrations_.provenance_ < 0", "Corrected", kViolet);
	doResolution(tv__tree,"(20,1.0,6.0,100,0,50)","calibrations_.provenance_ > 0", "Calibrated", kBlue);
	doResolution(tv__tree,"(20,1.0,6.0,100,0,50)","calibrations_.provenance_ == 0", "Uncalibrated", kRed); 
	
	/*
	file.cd("/plots/resolutions");
	 
	doResolution(tv__tree,"(20,1.0,18.0,100,0,500)","calibrations_.provenance_ < 0", "Corrected", kViolet);
	doResolution(tv__tree,"(20,1.0,18.0,100,0,500)","calibrations_.provenance_ > 0", "Calibrated", kBlue);
	doResolution(tv__tree,"(20,1.0,18.0,100,0,500)","calibrations_.provenance_ == 0", "Uncalibrated", kRed); 
	*/
}

void doFullRatioPlots(TFile& f, std::string leadingName, std::string title,
		Color_t color = kBlack) {
	std::string leadingName_1(leadingName);
	leadingName_1.append("_fullRatio");
	TH2D* source = (TH2D*) gDirectory->Get(leadingName_1.c_str());
	source->SetTitle(title.c_str());
	std::string leadingName_profile(leadingName_1);
	leadingName_profile.append("_pfx");
	source->ProfileX();
	TProfile * profile =
			(TProfile*) gDirectory->Get(leadingName_profile.c_str());
	profile->SetTitle(title.c_str());
	profile->SetMarkerStyle(22);
	profile->SetMarkerColor(color);
	profile->SetXTitle("E_{true}");
	profile->SetYTitle("E_{reco}/E_{true}");
	source->SetXTitle("E_{true}");
	source->SetYTitle("E_{reco}/E_{true}");
	profile->Write();
	source->Write();

}

void doLinearityPlots(TFile& f, std::string leadingName, std::string title,
		Color_t color = kBlack) {

	std::string leadingName_1(leadingName);
	leadingName_1.append("_1");
	std::string leadingName_2(leadingName);
	leadingName_2.append("_2");
	std::string leadingName_profile(leadingName);
	leadingName_profile.append("_pfx");
	std::string titleCpy(title);

	TH2D* source = (TH2D*) gDirectory->Get(leadingName.c_str());
	source->FitSlicesY();
	source->ProfileX();
	TProfile * profile =
			(TProfile*) gDirectory->Get(leadingName_profile.c_str());
	title.append(" sample mean");
	profile->SetTitle(title.c_str());
	profile->SetMarkerStyle(22);
	profile->SetMarkerColor(color);
	profile->Write();
	TH1D* source_1 = (TH1D*) gDirectory->Get(leadingName_1.c_str());
	titleCpy.append(" fitted mean");
	source_1->SetTitle(titleCpy.c_str());
	source_1->SetXTitle("E_{true} (GeV)");
	source_1->SetYTitle("calibrations_.particleEnergy_");
	source_1->SetMarkerStyle(22);
	source_1->SetMarkerColor(color);
	source_1->Write();
	source->Write();
	TH1D* source_2 = (TH1D*) gDirectory->Get(leadingName_2.c_str());
	source_2->SetMarkerStyle(22);
	source_2->SetMarkerColor(color);
	source_2->Write();
}

void doBiasPlots(TFile& f, std::string leadingName, std::string title,
		Color_t color = kBlack) {
	f.cd("/plots/bias");
	std::string leadingName_1(leadingName);
	leadingName_1.append("_1");
	std::string leadingName_profile(leadingName);
	leadingName_profile.append("_pfx");
	std::string titleCpy(title);

	TH2D* bias = (TH2D*) gDirectory->Get(leadingName.c_str());
	bias->FitSlicesY();
	bias->ProfileX();
	TProfile * profile =
			(TProfile*) gDirectory->Get(leadingName_profile.c_str());
	title.append(" sample bias");
	profile->SetTitle(title.c_str());
	profile->SetXTitle("E_{true} (GeV)");
	profile->SetYTitle("calibrations_.bias_");
	profile->SetMarkerStyle(22);
	profile->SetMarkerColor(color);
	profile->Write();
	TH1D* bias_1 = (TH1D*) gDirectory->Get(leadingName_1.c_str());
	titleCpy.append(" fitted bias");
	bias_1->SetTitle(titleCpy.c_str());
	bias_1->SetXTitle("E_{true} (GeV)");
	bias_1->SetYTitle("calibrations_.bias_");
	bias_1->SetMarkerStyle(22);
	bias_1->SetMarkerColor(color);
	bias_1->Write();
	bias->Write();
}

void doRatioPlots(TFile& f, std::string leadingName, std::string title,
		Color_t color = kBlack) {
	std::string leadingName_ratio(leadingName);
	leadingName_ratio.append("_ratio");
	std::string leadingName_1(leadingName);
	leadingName_1.append("_pfx");

	TH1D* source = (TH1D*) gDirectory->Get(leadingName_1.c_str());

	double start = source->GetBinLowEdge(1);
	double width = source->GetBinWidth(1);
	double end = source->GetBinLowEdge(source->GetNbinsX() + 1);

	TH1F* ratio = new TH1F(leadingName_ratio.c_str(), title.c_str(), source->GetNbinsX(), start, end);
	ratio->SetXTitle("E_{true} (GeV)");
	ratio->SetYTitle("E_{reco}/E_{true}");
	for (unsigned k(1); k < source->GetNbinsX() + 1; ++k) {
		double ratioVal = source->GetBinContent(k)/(width * k + start);
		ratio->Fill(width * k + start, ratioVal);
	}
	ratio->SetMarkerStyle(22);
	ratio->SetMarkerColor(color);
	ratio->Write();
}

void doTargetFunctions(TFile& f, std::string leadingName, std::string title,
		Color_t color = kBlack) {

	//for each in [un]calibrated(Corr)_1, _2, divide bin entry in _2 by _1 bin entry. Multiply by sqrt of bin center.

	std::string leadingName_1(leadingName);
	std::string leadingName_2(leadingName);
	std::string leadingName_targ(leadingName);
	leadingName_1.append("_1");
	leadingName_2.append("_2");
	leadingName_targ.append("_targ");

	TH1D* source = (TH1D*) gDirectory->Get(leadingName_2.c_str());
	TH1D* means = (TH1D*) gDirectory->Get(leadingName_1.c_str());
	f.cd("/plots/targetFunction");
	double start = source->GetBinLowEdge(1);
	double width = source->GetBinWidth(1);
	double end = source->GetBinLowEdge(source->GetNbinsX() + 1);

	TH1F* targ = new TH1F(leadingName_targ.c_str(), title.c_str(), source->GetNbinsX(), start, end);
	targ->SetXTitle("E_{true} (GeV)");
	targ->SetYTitle("Target function");
	for (unsigned k(1); k < source->GetNbinsX() + 1; ++k) {
		if (means->GetBinContent(k) != 0) {
			double targVal = source->GetBinContent(k)/ means->GetBinContent(k)
					* sqrt(width * k + start);
			targ->Fill(width * k + start, targVal);
		}
	}
	targ->SetMarkerStyle(22);
	targ->SetMarkerColor(color);
	targ->Write();

}

void doResolution(TTree* t, std::string queryBins, std::string cut, 
		std::string name, Color_t color = kBlack) {

	std::string	qryName("calibrations_.particleEnergy_:sqrt(calibrations_.truthEnergy_)>>");
	qryName.append(name);
	qryName.append(queryBins);
	
	std::string name1(name);
	std::string name2(name);
	name1.append("_1");
	name2.append("_2");
	std::string leadingName_res(name);
	leadingName_res.append("_res");

	
	//do tree business here
	t->Draw(qryName.c_str(), cut.c_str(),"box");
	TH2* temp = (TH2*) gDirectory->FindObject(name.c_str());
	temp->FitSlicesY();
	TH1* reso = ((TH1*) gDirectory->FindObject(name2.c_str()))->Clone();
	TH1* mean = ((TH1*) gDirectory->FindObject(name1.c_str()))->Clone();
	reso->Divide(mean);
	reso->SetTitle(name.c_str());
	reso->SetName(leadingName_res.c_str());
	reso->SetMarkerStyle(22);
	reso->SetMarkerColor(color);
	TF1* f = new TF1(name.c_str(), "[0]/x + [1]/(x*x)");
	reso->Fit(f);
	f->Write();
	reso->Write();
	leadingName_res.append("_graph");

}
