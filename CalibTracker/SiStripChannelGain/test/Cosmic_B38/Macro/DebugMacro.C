void AddContributionFromOtherAPV(TH1D* Proj, unsigned int CurrentAPV, TH1F* APV_DetId, TH1F* APV_Id, TH2F* APV_Charge);

void DebugMacro()
{
   gROOT->Reset();
   gStyle->SetCanvasColor(0);
   gStyle->SetOptStat("neMRuoi");
   gStyle->SetOptFit(111);
   gStyle->SetPalette(1);
   bool save = true;

   TFile* file  = new TFile("file:../SiStripCalib.root");

   TH1F* APV_MPV    = file->FindObjectAny("APV_MPV");
   TH1F* APV_DetId  = file->FindObjectAny("APV_DetId");
   TH1F* APV_Id     = file->FindObjectAny("APV_Id");
   TH1F* APV_Eta    = file->FindObjectAny("APV_Eta");
   TH2F* APV_Charge = file->FindObjectAny("APV_Charge");
  if(!(APV_MPV && APV_DetId && APV_Id && APV_Eta && APV_Charge))printf("Loading of Histos failed\n");

   TCanvas* c1 = new TCanvas();

   unsigned int K=0;
   for(int i=0;i<APV_MPV->GetXaxis()->GetNbins();i++){
        int MPV = APV_MPV->GetBinContent(i);
        if(APV_Id->GetBinContent(i)!=0)continue;
	if(MPV>0 && MPV<170){
	   printf("Low MPV (%3i) for APV_%i_%i\n",MPV, APV_DetId->GetBinContent(i), APV_Id->GetBinContent(i) );

           TH1D* Proj = APV_Charge->ProjectionY(" ",i,i,"e");
           Proj = (TH1D*)Proj->Clone();
           if(Proj==NULL)continue;
           AddContributionFromOtherAPV(Proj,i,APV_DetId,APV_Id,APV_Charge);


 	   if(Proj->GetEntries()<40)continue;
           printf("#Entries = %i\n", Proj->GetEntries());

           Proj = Proj->Rebin(4);


           TF1* MyLandau = new TF1("MyLandau","landau",0, 5400.0);
           MyLandau->SetParameter("MPV",300);
           Proj->Fit("MyLandau","QR WW 0");
           TF1 * fitfunction = (TF1*) Proj->GetListOfFunctions()->First();


   	   Proj->SetTitle();
//	   Proj->SetStats(kFALSE);
	   Proj->GetXaxis()->SetTitle("Charged (ADC Normalized)");
	   Proj->GetYaxis()->SetTitle("#Entries");
	   Proj->GetYaxis()->SetTitleOffset(1.05);
           Proj->SetAxisRange(0,900 ,"X");
	   Proj->SetLineColor(4);
	   Proj->SetLineWidth(2);         
	   Proj->Draw("HIST");
	   fitfunction->Draw("same");

	   char filepath[1024];sprintf(filepath,"DebugPictures/DetId%i_MPV%03i_Entries%05i.png",APV_DetId->GetBinContent(i), MPV, Proj->GetEntries());
	   if(save)c1->SaveAs(filepath);
           delete Proj;

	   K++;
	   //if(K==1)return;
	}
   }

}

void AddContributionFromOtherAPV(TH1D* Proj, unsigned int CurrentAPV, TH1F* APV_DetId, TH1F* APV_Id, TH2F* APV_Charge){
   int DetId = APV_DetId->GetBinContent(CurrentAPV);
   int Id    = APV_Id->GetBinContent(CurrentAPV);

   for(int i=CurrentAPV;i<APV_DetId->GetXaxis()->GetNbins()&&i<CurrentAPV+6;i++){
      if(APV_DetId->GetBinContent(i)!=DetId)continue;
      if(APV_Id->GetBinContent(i)   ==Id   )continue;

       TH1D* Proj2 = APV_Charge->ProjectionY(" ",i,i,"e");
       Proj2 = (TH1D*)Proj2->Clone();
       if(Proj2==NULL)continue;

       printf("Add histos From %i_%i   --> +%iEntries\n",APV_DetId->GetBinContent(i),APV_Id->GetBinContent(i),Proj->GetEntries());

      
       Proj->Add(Proj2,1.0);
   }
}






