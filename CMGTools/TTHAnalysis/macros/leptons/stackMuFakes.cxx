void stackMuFakes(TString what="pt") {
    gStyle->SetOptStat(0);
    gStyle->SetErrorX(0.5);
    TFile *fakes  = (TFile*) gROOT->GetListOfFiles()->At(0);
    fakes->ls();
    TProfile *mcFakeMP = (TProfile*) fakes->Get("mediumIdFake_"+what+"_TT_fake");
    TProfile *mcFakeTP = (TProfile*) fakes->Get("tightIdFake_"+what+"_TT_fake");

    mcFakeMP->SetLineWidth(2);
    mcFakeMP->SetLineColor(kBlack);
    mcFakeMP->SetMarkerColor(kBlack);
    mcFakeMP->SetMarkerSize(0.9);

    mcFakeTP->SetMarkerSize(1.4);
    mcFakeTP->SetLineWidth(4);
    mcFakeTP->SetMarkerStyle(21);
    mcFakeTP->SetLineColor(kCyan-8);
    mcFakeTP->SetMarkerColor(kCyan-8);

    mcFakeTP->Draw(); 
    mcFakeMP->Draw("SAME"); 

    mcFakeTP->GetYaxis()->SetTitle("Efficiency"); 
    mcFakeTP->GetYaxis()->SetDecimals(1); 
    mcFakeTP->GetYaxis()->SetRangeUser(0.0, 1.039); 
    mcFakeTP->GetXaxis()->SetTitleOffset(1.0); 

    TString odir = gSystem->DirName(gFile->GetName());
    c1->Print(odir+"/fakestack.pdf");
    c1->Print(odir+"/fakestack.eps");
}
