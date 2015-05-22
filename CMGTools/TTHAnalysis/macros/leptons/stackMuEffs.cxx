void stackMuEffs(TString what="pt") {
    gStyle->SetOptStat(0);
    gStyle->SetErrorX(0.5);
    TFile *effs  = (TFile*) gROOT->GetListOfFiles()->At(0);
    TProfile *mcEffMP = (TProfile*) effs->Get("mediumIdEff_"+what+"_TT_true");
    TProfile *mcEffMB = (TProfile*) effs->Get("mediumIdEff_"+what+"_TT_bjets");
    TProfile *mcEffTP = (TProfile*) effs->Get("tightIdEff_"+what+"_TT_true");
    TProfile *mcEffTB = (TProfile*) effs->Get("tightIdEff_"+what+"_TT_bjets");

    mcEffMP->SetLineWidth(2);
    mcEffMP->SetLineColor(kOrange+7);
    mcEffMP->SetMarkerColor(kOrange+7);
    mcEffMP->SetMarkerSize(0.9);
    mcEffMB->SetMarkerSize(1.4);
    mcEffMB->SetMarkerStyle(21);
    mcEffMB->SetLineWidth(4);
    mcEffMB->SetLineColor(kGray+1);
    mcEffMB->SetMarkerColor(kGray+1);
    
    mcEffTP->SetLineWidth(2);
    mcEffTP->SetLineColor(kViolet+1);
    mcEffTP->SetMarkerColor(kViolet+1);
    mcEffTP->SetMarkerSize(0.9);
    mcEffTB->SetMarkerSize(1.4);
    mcEffTB->SetLineWidth(4);
    mcEffTB->SetMarkerStyle(21);
    mcEffTB->SetLineColor(kAzure-9);
    mcEffTB->SetMarkerColor(kAzure-9);

    TLine xline(mcEffTB->GetXaxis()->GetXmin(),1.0,mcEffTB->GetXaxis()->GetXmax(),1.0);
    xline.SetLineWidth(3);
    xline.SetLineStyle(2);

    mcEffTB->Draw("AXIS"); 
    xline.DrawClone();
    mcEffTB->Draw("SAME"); 
    mcEffMB->Draw("SAME"); 
    mcEffTP->Draw("SAME"); 
    mcEffMP->Draw("SAME"); 

    mcEffTB->GetYaxis()->SetTitle("Efficiency"); 
    mcEffTB->GetYaxis()->SetDecimals(1); 
    mcEffTB->GetYaxis()->SetRangeUser(0.891, 1.019); 
    mcEffTB->GetXaxis()->SetTitleOffset(1.0); 

    TString odir = gSystem->DirName(gFile->GetName());
    c1->Print(odir+"/effstack.pdf");
    c1->Print(odir+"/effstack.eps");
}
