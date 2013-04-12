#include "Style.C"

void InputVariables() {
// Example of Root macro based on $ROOTSYS/tutorials/tree/copytree3.C

 setTDRStyle();   
   gSystem->Load("$ROOTSYS/test/libEvent");

//Get old file, old tree and set top branch address
   TFile *oldfile = new TFile("train_save_all.root");
   TTree *oldtree = (TTree*)oldfile->Get("combinedMVA_save");
   Int_t nentries = (Int_t)oldtree->GetEntries();
   
   //Access jet trackSip3dSig to split depending on ntracks
   vector<double>  *algoDiscriminator;
   oldtree->SetBranchAddress("algoDiscriminator",&algoDiscriminator);
//   Int_t flavour;
//   oldtree->SetBranchAddress("flavour",&flavour);

   //Create a new file + a clone of old tree in new file
   TFile *newfile = new TFile("InputVariables.root","recreate");
   //TTree *newtree = oldtree->CloneTree(0);
 TH1F* DiscriminatorJP = new TH1F("JP","JP",60,0,3);
 TH1F* DiscriminatorCSV = new TH1F("CSV","CSV",60,0,1);
 TH1F* DiscriminatorSM = new TH1F("SM","SM",60,0,1);
 TH1F* DiscriminatorSE = new TH1F("SE","SE",60,0,1);
 TH1F* DiscriminatorJP_B = new TH1F("JP_B","JP_B",60,0,3);
 TH1F* DiscriminatorCSV_B = new TH1F("CSV_B","CSV_B",60,0,1);
 TH1F* DiscriminatorSM_B = new TH1F("SM_B","SM_B",60,0,1);
 TH1F* DiscriminatorSE_B = new TH1F("SE_B","SE_B",60,0,1);
 TH1F* DiscriminatorJP_C = new TH1F("JP_C","JP_C",60,0,3);
 TH1F* DiscriminatorCSV_C = new TH1F("CSV_C","CSV_C",60,0,1);
 TH1F* DiscriminatorSM_C = new TH1F("SM_C","SM_C",60,0,1);
 TH1F* DiscriminatorSE_C = new TH1F("SE_C","SE_C",60,0,1);
 TH1F* DiscriminatorJP_DUSG = new TH1F("JP_DUSG","JP_DUSG",60,0,3);
 TH1F* DiscriminatorCSV_DUSG = new TH1F("CSV_DUSG","CSV_DUSG",60,0,1);
 TH1F* DiscriminatorSM_DUSG = new TH1F("SM_DUSG","SM_DUSG",60,0,1);
 TH1F* DiscriminatorSE_DUSG = new TH1F("SE_DUSG","SE_DUSG",60,0,1);
 TH2F* DiscriminatorJPCSV = new TH2F("DiscriminatorJPCSV","DiscriminatorJPCSV",60,0,3,60,0,1);
 TH2F* DiscriminatorJPSM = new TH2F("DiscriminatorJPSM","DiscriminatorJPSM",60,0,3,60,0,1);
 TH2F* DiscriminatorJPSE = new TH2F("DiscriminatorJPSE","DiscriminatorJPSE",60,0,3,60,0,1);
 TH2F* DiscriminatorCSVSM = new TH2F("DiscriminatorCSVSM","DiscriminatorCSVSM",60,0,1,60,0,1);
 TH2F* DiscriminatorCSVSE = new TH2F("DiscriminatorCSVSE","DiscriminatorCSVSE",60,0,1,60,0,1);
 TH2F* DiscriminatorSMSE = new TH2F("DiscriminatorSMSE","DiscriminatorSMSE",60,0,1,60,0,1);

    
   for (Int_t i=0;i<nentries; i++) {
//   for (Int_t i=0;i<100; i++) {
        oldtree->GetEntry(i);

//separate in flavour of jet???

//if(flavour){	
	 for(int j = 0; j < algoDiscriminator->size(); j++){	    
		if(j==0) DiscriminatorJP->Fill((*algoDiscriminator)[j]);
		if(j==1) DiscriminatorCSV->Fill((*algoDiscriminator)[j]);
		if(j==2) DiscriminatorSM->Fill((*algoDiscriminator)[j]);
		if(j==3) DiscriminatorSE->Fill((*algoDiscriminator)[j]);
        
 	if( algoDiscriminator->size()==4){
	//	cout << "JP: " << (*algoDiscriminator)[0] << endl;
	//	cout << "CSV: " << (*algoDiscriminator)[1] << endl;
	//	cout << "SM: " << (*algoDiscriminator)[2] << endl;
	//	cout << "SE: " << (*algoDiscriminator)[3] << endl;
//		if((*algoDiscriminator)[0] < 9999. && (*algoDiscriminator)[1] < 9999. && (*algoDiscriminator)[0] > -0.5 && (*algoDiscriminator)[1] > -0.5)	
DiscriminatorJPCSV->Fill((*algoDiscriminator)[0],(*algoDiscriminator)[1]);
//		if((*algoDiscriminator)[0] < 9999. && (*algoDiscriminator)[2] < 9999. && (*algoDiscriminator)[0] > -0.5 && (*algoDiscriminator)[2] > -0.5)				
DiscriminatorJPSM->Fill((*algoDiscriminator)[0],(*algoDiscriminator)[2]);
//		if((*algoDiscriminator)[0] < 9999. && (*algoDiscriminator)[3] < 9999. && (*algoDiscriminator)[0] > -0.5 && (*algoDiscriminator)[3] > -0.5)				
DiscriminatorJPSE->Fill((*algoDiscriminator)[0],(*algoDiscriminator)[3]);
//		if((*algoDiscriminator)[1] < 9999. && (*algoDiscriminator)[2] < 9999. && (*algoDiscriminator)[1] > -0.5 && (*algoDiscriminator)[2] > -0.5)				
DiscriminatorCSVSM->Fill((*algoDiscriminator)[1],(*algoDiscriminator)[2]);
//		if((*algoDiscriminator)[1] < 9999. && (*algoDiscriminator)[3] < 9999. && (*algoDiscriminator)[1] > -0.5 && (*algoDiscriminator)[3] > -0.5)				
DiscriminatorCSVSE->Fill((*algoDiscriminator)[1],(*algoDiscriminator)[3]);
//		if((*algoDiscriminator)[2] < 9999. && (*algoDiscriminator)[3] < 9999. && (*algoDiscriminator)[2] > -0.5 && (*algoDiscriminator)[3] > -0.5)					
		DiscriminatorSMSE->Fill((*algoDiscriminator)[2],(*algoDiscriminator)[3]);
	}
  }    
   }
		Double_t correlationJPCSV = DiscriminatorJPCSV->GetCorrelationFactor();
		Double_t correlationJPSM = DiscriminatorJPSM->GetCorrelationFactor();
		Double_t correlationJPSE = DiscriminatorJPSE->GetCorrelationFactor();
		Double_t correlationCSVSM = DiscriminatorCSVSM->GetCorrelationFactor();
		Double_t correlationCSVSE = DiscriminatorCSVSE->GetCorrelationFactor();
		Double_t correlationSMSE = DiscriminatorSMSE->GetCorrelationFactor();

  		//Double_t correlationJPCSV = 0.72;
		//Double_t correlationJPSM = 0.46;
		//Double_t correlationJPSE = 0.33;
		//Double_t correlationCSVSM = 0.44;
		//Double_t correlationCSVSE = 0.29;
		//Double_t correlationSMSE = 0.25;

	cout <<correlationJPCSV << endl;
	cout <<correlationJPSM << endl;
	cout <<correlationJPSE << endl;
	cout <<correlationCSVSM << endl;
	cout <<correlationCSVSE << endl;
	cout <<correlationSMSE << endl;

/*	TGraph2D * correlation = new TGraph2D(16);
	correlation->SetNpx(4);
	correlation->SetNpy(4);
	correlation->SetPoint(0,1,4,1); //JP vs JP
	correlation->SetPoint(1,2,4,correlationJPCSV); //JP vs CSV
	correlation->SetPoint(2,3,4,correlationJPSM); //JP vs SM
	correlation->SetPoint(3,4,4,correlationJPSE); //JP vs SE
	correlation->SetPoint(4,1,3,correlationJPCSV); //CSV vs JP
	correlation->SetPoint(5,2,3,1); //CSV vs CSV
	correlation->SetPoint(6,2,2,correlationCSVSM); //CSV vs SM
	correlation->SetPoint(7,2,1,correlationCSVSE); //CSV vs SE
	correlation->SetPoint(8,1,2,correlationJPSM); //SM vs JP
	correlation->SetPoint(9,3,3,correlationCSVSM); //SM vs CSV
	correlation->SetPoint(10,3,2,1); //SM vs SM
	correlation->SetPoint(11,3,1,correlationSMSE); //SM vs SE
	correlation->SetPoint(12,1,1,correlationJPSE); //SE vs JP
	correlation->SetPoint(13,4,3,correlationCSVSE); //SE vs CSV
	correlation->SetPoint(14,4,2,correlationSMSE); //SE vs SM
	correlation->SetPoint(15,4,1,1); //SE vs SE

  TCanvas *c = new TCanvas("c","Graph2D",0,0,600,600);
	correlation->SetTitle("");
	correlation->GetXaxis()->SetNdivisions(100,0);
	correlation->GetYaxis()->SetNdivisions(100,0);
	correlation->GetXaxis()->SetLabelSize(0.08);
	correlation->GetYaxis()->SetLabelSize(0.08);
	correlation->GetXaxis()->SetBinLabel(1, "JP");
	correlation->GetXaxis()->SetBinLabel(2, "CSV");
	correlation->GetXaxis()->SetBinLabel(3, "SE");
	correlation->GetXaxis()->SetBinLabel(4, "SM");
	correlation->GetYaxis()->SetBinLabel(1, "SM");
	correlation->GetYaxis()->SetBinLabel(2, "SE");
	correlation->GetYaxis()->SetBinLabel(3, "CSV");
	correlation->GetYaxis()->SetBinLabel(4, "JP");
  gStyle->SetPalette(1);
	gPad->SetTicks(2,1);
//	correlation->Draw("text,colz");
	correlation->Draw("colz");
*/
	
   TFile *oldfile_b = new TFile("train_B_save.root");
   TFile *oldfile_c = new TFile("train_C_save.root");
//   TFile *oldfile_dusg = new TFile("train_DUSG_save.root");
   TTree *treeb = (TTree*)oldfile_b->Get("combinedMVA_save");
   TTree *treec = (TTree*)oldfile_c->Get("combinedMVA_save");
//   TTree *treedusg = (TTree*)oldfile_dusg->Get("combinedMVA_save");
   Int_t nentriesb = (Int_t)treeb->GetEntries();
   Int_t nentriesc = (Int_t)treec->GetEntries();
//   Int_t nentriesdusg = (Int_t)treedusg->GetEntries();
   
   //Access jet trackSip3dSig to split depending on ntracks
   vector<double>  *algoDiscriminatorb;
   treeb->SetBranchAddress("algoDiscriminator",&algoDiscriminatorb);
   for (Int_t i=0;i<nentriesb; i++) {
//   for (Int_t i=0;i<100; i++) {
     	treeb->GetEntry(i);
	 		for(int j = 0; j < algoDiscriminatorb->size(); j++){	    
				if(j==0) DiscriminatorJP_B->Fill((*algoDiscriminatorb)[j]);
				if(j==1) DiscriminatorCSV_B->Fill((*algoDiscriminatorb)[j]);
				if(j==2) DiscriminatorSM_B->Fill((*algoDiscriminatorb)[j]);
				if(j==3) DiscriminatorSE_B->Fill((*algoDiscriminatorb)[j]);
	 		}
	 }
   vector<double>  *algoDiscriminatorc;
   treec->SetBranchAddress("algoDiscriminator",&algoDiscriminatorc);
   for (Int_t i=0;i<nentriesc; i++) {
//   for (Int_t i=0;i<100; i++) {
     	treec->GetEntry(i);
	 		for(int j = 0; j < algoDiscriminatorc->size(); j++){	    
				if(j==0) DiscriminatorJP_C->Fill((*algoDiscriminatorc)[j]);
				if(j==1) DiscriminatorCSV_C->Fill((*algoDiscriminatorc)[j]);
				if(j==2) DiscriminatorSM_C->Fill((*algoDiscriminatorc)[j]);
				if(j==3) DiscriminatorSE_C->Fill((*algoDiscriminatorc)[j]);
	 		}
	 }

/*    vector<double>  *algoDiscriminatordusg;
   treedusg->SetBranchAddress("algoDiscriminator",&algoDiscriminatordusg);
   for (Int_t i=0;i<nentriesdusg; i++) {
//   for (Int_t i=0;i<100; i++) {
     	treedusg->GetEntry(i);
	 		for(int j = 0; j < algoDiscriminatordusg->size(); j++){	    
				if(j==0) DiscriminatorJP_DUSG->Fill((*algoDiscriminatordusg)[j]);
				if(j==1) DiscriminatorCSV_DUSG->Fill((*algoDiscriminatordusg)[j]);
				if(j==2) DiscriminatorSM_DUSG->Fill((*algoDiscriminatordusg)[j]);
				if(j==3) DiscriminatorSE_DUSG->Fill((*algoDiscriminatordusg)[j]);
	 		}
	 }
 */
/* //plot the histograms!
//b = red, c = green, light = blue
TCanvas *c1 = new TCanvas("c1");
DiscriminatorJP_DUSG->SetTitle("");
DiscriminatorJP_DUSG->GetXaxis()->SetTitle("discriminator JP");
DiscriminatorJP_DUSG->GetXaxis()->SetRangeUser(0,3);
DiscriminatorJP_B->SetLineWidth(2);
DiscriminatorJP_C->SetLineWidth(2);
DiscriminatorJP_DUSG->SetLineWidth(2);
DiscriminatorJP_B->SetLineColor(2);
DiscriminatorJP_C->SetLineColor(8);
DiscriminatorJP_DUSG->SetLineColor(4);
DiscriminatorJP_DUSG->DrawNormalized();		
DiscriminatorJP_C->DrawNormalized("same");		
DiscriminatorJP_B->DrawNormalized("same");		
TLegend* leg = new TLegend(0.65,0.65,0.95,0.9);
leg->SetFillColor(0);
leg->AddEntry(DiscriminatorJP_B,"b","L");
leg->AddEntry(DiscriminatorJP_C,"c","L");
leg->AddEntry(DiscriminatorJP_DUSG,"dusg","L");
leg->Draw("SAME");
c1->SaveAs("DiscriminatorJP.png");
 */
//b = red, c = green, light = blue
TCanvas *c2 = new TCanvas("c2");
DiscriminatorCSV_B->SetTitle("");
DiscriminatorCSV_B->GetXaxis()->SetTitle("discriminator CSV");
DiscriminatorCSV_B->GetXaxis()->SetRangeUser(0,1);
DiscriminatorCSV_B->SetLineWidth(2);
DiscriminatorCSV_C->SetLineWidth(2);
//DiscriminatorCSV_DUSG->SetLineWidth(2);
DiscriminatorCSV_B->SetLineColor(2);
DiscriminatorCSV_C->SetLineColor(8);
//DiscriminatorCSV_DUSG->SetLineColor(4);
DiscriminatorCSV_B->DrawNormalized();		
DiscriminatorCSV_C->DrawNormalized("same");		
//DiscriminatorCSV_DUSG->DrawNormalized("same");		
TLegend* leg = new TLegend(0.7,0.75,0.95,0.9);
leg->SetFillColor(0);
leg->AddEntry(DiscriminatorCSV_B,"b","L");
leg->AddEntry(DiscriminatorCSV_C,"c","L");
//leg->AddEntry(DiscriminatorCSV_DUSG,"dusg","L");
//leg->Draw("SAME");
c2->SaveAs("DiscriminatorCSV.png");

//b = red, c = green, light = blue
TCanvas *c3 = new TCanvas("c3");
DiscriminatorSM_B->SetTitle("");
DiscriminatorSM_B->GetXaxis()->SetTitle("discriminator SM");
DiscriminatorSM_B->GetXaxis()->SetRangeUser(0,1);
DiscriminatorSM_B->SetLineWidth(2);
DiscriminatorSM_C->SetLineWidth(2);
//DiscriminatorSM_DUSG->SetLineWidth(2);
DiscriminatorSM_B->SetLineColor(2);
DiscriminatorSM_C->SetLineColor(8);
//DiscriminatorSM_DUSG->SetLineColor(4);
DiscriminatorSM_B->DrawNormalized();		
DiscriminatorSM_C->DrawNormalized("same");		
//DiscriminatorSM_DUSG->DrawNormalized("same");		
TLegend* leg = new TLegend(0.6,0.4,0.95,0.9);
leg->SetFillColor(0);
leg->AddEntry(DiscriminatorSM_B,"b","L");
leg->AddEntry(DiscriminatorSM_C,"c","L");
//leg->AddEntry(DiscriminatorSM_DUSG,"dusg","L");
//leg->Draw("SAME");
c3->SaveAs("DiscriminatorSM.png");

//b = red, c = green, light = blue
TCanvas *c4 = new TCanvas("c4");
DiscriminatorSE_B->SetTitle("");
DiscriminatorSE_B->GetXaxis()->SetTitle("discriminator SE");
DiscriminatorSE_B->GetXaxis()->SetRangeUser(0,1);
DiscriminatorSE_B->SetLineWidth(2);
DiscriminatorSE_C->SetLineWidth(2);
//DiscriminatorSE_DUSG->SetLineWidth(2);
DiscriminatorSE_B->SetLineColor(2);
DiscriminatorSE_C->SetLineColor(8);
//DiscriminatorSE_DUSG->SetLineColor(4);
DiscriminatorSE_B->DrawNormalized();		
DiscriminatorSE_C->DrawNormalized("same");		
DiscriminatorSE_DUSG->DrawNormalized("same");		
TLegend* leg = new TLegend(0.6,0.4,0.95,0.9);
leg->SetFillColor(0);
leg->AddEntry(DiscriminatorSE_B,"b","L");
leg->AddEntry(DiscriminatorSE_C,"c","L");
//leg->AddEntry(DiscriminatorSE_DUSG,"dusg","L");
//leg->Draw("SAME");
c4->SaveAs("DiscriminatorSE.png");



newfile->cd();
DiscriminatorJP->Write();
DiscriminatorCSV->Write();
DiscriminatorSM->Write();
DiscriminatorSE->Write();

DiscriminatorJP_B->Write();
DiscriminatorCSV_B->Write();
DiscriminatorSM_B->Write();
DiscriminatorSE_B->Write();
DiscriminatorJP_C->Write();
DiscriminatorCSV_C->Write();
DiscriminatorSM_C->Write();
DiscriminatorSE_C->Write();
DiscriminatorJP_DUSG->Write();
DiscriminatorCSV_DUSG->Write();
DiscriminatorSM_DUSG->Write();
DiscriminatorSE_DUSG->Write();

DiscriminatorJPCSV->Write();
DiscriminatorJPSM->Write();
DiscriminatorJPSE->Write();
DiscriminatorCSVSM->Write();
DiscriminatorCSVSE->Write();
DiscriminatorSMSE->Write();
//correlation->Write();
newfile->Close();


	
		

   //newtree->Print();
  /// newtree->AutoSave();
   delete oldfile;
   //delete newfile;

	  }
