{
	setTDRStyle();
	TH1F * h_lorentzAngle_layer1 = new TH1F("lorentzAnglelayer1","lorentzAngle", 8 , 0.5, 8.5);
	TH1F * h_lorentzAngle_layer2 = new TH1F("lorentzAnglelayer2","lorentzAngle", 8 , 0.5, 8.5);
	TH1F * h_lorentzAngle_layer3 = new TH1F("lorentzAnglelayer3","lorentzAngle", 8 , 0.5, 8.5);
	ifstream fin;
	fin.open("lorentzFit.txt");
	int module,  layer;
	double offset,  error,   slope,   error,   rel, pull,    chi2,    prob;
	fin.ignore(100,'\n');
// 	fin >> module >>  layer >> offset >> error >> slope >>   error >>   rel >> pull >>    chi2 >>    prob;
// 	for(int i_layer = 1; i_layer <= 3; i_layer++){
	for(int i_module = 1; i_module <= 8; i_module++){
		fin >> module >>  layer >> offset >> error >> slope >>   error >>   rel >> pull >>    chi2 >>    prob;
		cout << slope << endl;
		h_lorentzAngle_layer1->SetBinContent(i_module, slope);
		h_lorentzAngle_layer1->SetBinError(i_module, error);
	}
	for(int i_module = 1; i_module <= 8; i_module++){
		fin >> module >>  layer >> offset >> error >> slope >>   error >>   rel >> pull >>    chi2 >>    prob;
		cout << slope << endl;
		h_lorentzAngle_layer2->SetBinContent(i_module, slope);
		h_lorentzAngle_layer2->SetBinError(i_module, error);
	}
	for(int i_module = 1; i_module <= 8; i_module++){
		fin >> module >>  layer >> offset >> error >> slope >>   error >>   rel >> pull >>    chi2 >>    prob;
		cout << slope << endl;
		h_lorentzAngle_layer3->SetBinContent(i_module, slope);
		h_lorentzAngle_layer3->SetBinError(i_module, error);
	}

	h_lorentzAngle_layer3->SetMarkerColor(7);
	h_lorentzAngle_layer3->SetLineColor(7);
	h_lorentzAngle_layer3->SetMarkerStyle(20);
	h_lorentzAngle_layer3->SetMarkerSize(0.9);
	h_lorentzAngle_layer3->SetMaximum(0.55);
	h_lorentzAngle_layer3->SetMinimum(0.35);
	h_lorentzAngle_layer3->GetXaxis()->SetTitle("module");
	h_lorentzAngle_layer3->GetYaxis()->SetTitle("tan(#Theta_{L})   ");
	
	h_lorentzAngle_layer3->Draw("");

	TLine* l = new TLine(0.5, 0.424, 8.5, 0.424); 
	l->SetLineColor(2);
	l->SetLineWidth(2);
	l->Draw();
	
	h_lorentzAngle_layer2->SetMarkerColor(4);
	h_lorentzAngle_layer2->SetLineColor(4);
	h_lorentzAngle_layer2->SetMarkerStyle(21);
	h_lorentzAngle_layer2->SetMarkerSize(0.7);
	h_lorentzAngle_layer2->Draw("same");
				
	h_lorentzAngle_layer1->SetMarkerColor(6);
	h_lorentzAngle_layer1->SetLineColor(6);
	h_lorentzAngle_layer1->SetMarkerStyle(22);
	h_lorentzAngle_layer1->SetMarkerSize(1.1);
	h_lorentzAngle_layer1->Draw("same");
	
	TLegend *legend1=new TLegend(0.7,0.75,0.94,0.94);
// 	legend1->SetTextSize(0.045); 
	legend1->AddEntry(h_lorentzAngle_layer1,"layer 1","lp");
	legend1->AddEntry(h_lorentzAngle_layer2,"layer 2","lp");
	legend1->AddEntry(h_lorentzAngle_layer3,"layer 3","lp");
	legend1->Draw("");

} 
