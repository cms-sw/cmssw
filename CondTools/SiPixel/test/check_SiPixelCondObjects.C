void check_SiPixelCondObjects(){

 
  do_check("histos_HLT.root","histos_HLT_startup.root","HLT");
  do_check("histos_Offline.root","histos_Offline_startup.root","Offline");
}


void do_check(TString filename_official, TString filename_test,TString sampletype){
  
  TFile *file_official= new TFile(filename_official,"read");
  TFile *file_test= new TFile(filename_test,"read");

  TH1F *gain1_off = (TH1F*)file_official->Get("GainAll"); 

  if(!gain1_off){
    std::cout << "Histogram GainAll does not exist in file " << filename_official << std::endl;
    return;
  }
  TString newname1 = gain1_off->GetName();
  newname1+="official";
  gain1_off->SetName(newname1);
  gain1_off->SetTitle("Comparison all gain "+sampletype);
  gain1_off->GetXaxis()->SetTitle("gain");
  TH1F *gain2_off = (TH1F*)file_official->Get("Summary_Gain");
  if(!gain2_off){
    std::cout << "Histogram Summary_Gain does not exist in file " << filename_official << std::endl;
    return;
  }
  TString newname2 = gain2_off->GetName();
  newname2+="official";
  gain2_off->SetName(newname2);
  gain2_off->SetTitle("Comparison gain per module "+ sampletype);
  gain2_off->GetXaxis()->SetTitle("arbitrary module number");
  gain2_off->GetYaxis()->SetTitle("gain");
  TH1F *gain1_test = (TH1F*)file_test->Get("GainAll");
  if(!gain1_test){
    std::cout << "Histogram GainAll does not exist in file " << filename_test << std::endl;
    return;
  }
  TH1F *gain2_test = (TH1F*)file_test->Get("Summary_Gain");
  if(!gain2_test){
    std::cout << "Histogram Summary_Gain does not exist in file " << filename_test << std::endl;
    return;
  }
    
  TH1F *pedestal1_off = (TH1F*)file_official->Get("PedestalsAll"); 
  if(!pedestal1_off){
    std::cout << "Histogram PedestalsAll does not exist in file " << filename_official << std::endl;
    return;
  }
  TString newname3 = pedestal1_off->GetName();
  newname3+="official";
  pedestal1_off->SetName(newname3);
  pedestal1_off->SetTitle("Comparison all pedestal "+sampletype);
  pedestal1_off->GetXaxis()->SetTitle("pedestal");
  TH1F *pedestal2_off = (TH1F*)file_official->Get("Summary_Pedestal");
  if(!pedestal2_off){
    std::cout << "Histogram Summary_Pedestal does not exist in file " << filename_official << std::endl;
    return;
  }
  TString newname4 = pedestal2_off->GetName();
  newname4+="official";
  pedestal2_off->SetName(newname4);
  pedestal2_off->SetTitle("Comparison pedestal per module "+sampletype);
  pedestal2_off->GetXaxis()->SetTitle("arbitrary module number");
  pedestal2_off->GetYaxis()->SetTitle("pedestal");
  TH1F *pedestal1_test = (TH1F*)file_test->Get("PedestalsAll");  
  if(!pedestal1_test){
    std::cout << "Histogram PedestalsAll does not exist in file " << filename_test << std::endl;
    return;
  }
  TH1F *pedestal2_test = (TH1F*)file_test->Get("Summary_Pedestal");
  if(!pedestal2_test){
    std::cout << "Histogram Summary_Pedestal does not exist in file " << filename_test << std::endl;
    return;
  }
  // check for dead pixel histos:
  bool deadhists_exist=true;
  bool deadhists_ref_exist=true;
  
  TH1F *dead2_off = (TH1F*)file_official->Get("Summary_dead");
  TH1F *dead1_off = (TH1F*)file_official->Get("DeadAll");
  if(!dead1_off || !dead2_off){
    std::cout << "Histogram Summary_Dead/DeadAll does not exist in file " << filename_official << std::endl;
    deadhists_exist=false;
    deadhists_ref_exist=false;
  }
  else{
    TString newname5 = dead1_off->GetName();
    newname5+="official";
    TString newname6 = dead2_off->GetName();
    newname6+="official";
    
    dead1_off->SetName(newname5);
    dead1_off->SetTitle("Comparison dead fraction all modules "+sampletype);
    dead2_off->SetName(newname6);
    dead2_off->SetTitle("Comparison dead fraction per module "+sampletype);
    dead2_off->GetXaxis()->SetTitle("arbitrary module number");
    dead2_off->GetYaxis()->SetTitle("pedestal");
  }
  
  TH1F *dead2_test = (TH1F*)file_test->Get("Summary_dead");
  TH1F *dead1_test = (TH1F*)file_test->Get("DeadAll");
  if(!dead1_test || !dead2_test){
    std::cout << "Histogram Summary_Dead/DeadAll does not exist in file " << filename_test << std::endl;
    deadhists_exist=false;
  }
  else{
    TString newname5 = dead1_off->GetName();
    newname5+="official";
    TString newname6 = dead2_off->GetName();
    newname6+="official";
    
    dead1_off->SetName(newname5);
    dead1_off->SetTitle("Comparison dead fraction all modules "+sampletype);
    dead2_off->SetName(newname6);
    dead2_off->SetTitle("Comparison dead fraction per module "+sampletype);
    dead2_off->GetXaxis()->SetTitle("arbitrary module number");
    dead2_off->GetYaxis()->SetTitle("pedestal");
  }
  
  
   
  std::cout << "\n\n******\n" << sampletype << " comparison of the following files:" << std::endl;
  std::cout << "* reference:   " << filename_official << std::endl;
  std::cout << "* test:        " << filename_test << std::endl;
  TCanvas *cv= new TCanvas("comparison"+sampletype,"comparison"+sampletype);
  if(deadhists_exist)
    cv->Divide(2,3);
  else
    cv->Divide(2,2);
  cv->cd(1);
  double res1 = make_plot(gain1_off,gain1_test);
  std::cout << "match of " << gain1_off->GetTitle()<<" histograms: " << res1 << std::endl;
  cv->cd(2);
  double res2 = make_plot(gain2_off,gain2_test);
  std::cout << "match of " << gain2_off->GetTitle() << " histograms: " << res2 << std::endl;
  cv->cd(3);
  double res3 = make_plot(pedestal1_off,pedestal1_test);
  std::cout << "match of "<< pedestal1_off->GetTitle() << " histograms: " << res3 << std::endl;
  cv->cd(4);
  double res4 = make_plot(pedestal2_off,pedestal2_test);
  double res5=1;
  double res6=1;
  if(deadhists_ref_exist){
    cv->cd(5);
    if(deadhists_exist)
      res5=make_plot(dead1_off,dead1_test);
    else
      dead1_off->Draw();
    cv->cd(6);
    if(deadhists_exist)
      res6=make_plot(dead2_off,dead2_test);
    else
      dead2_off->Draw();
  }
 
  std::cout << "match of "<< pedestal1_off->GetTitle() << " histograms: " << res4 << std::endl;
  double compatbg = res1*res2*res3*res4;
  std::cout << "TOTAL compatibility: " << compatbg << " (1=completely identical, 0=not compatible at all)" << std::endl; 
  TString summaryplotname = filename_official+"_asRefvs_"+filename_test;
  summaryplotname.ReplaceAll(".root","");
  summaryplotname.ReplaceAll(".","");
  summaryplotname.ReplaceAll("/","");
  summaryplotname.ReplaceAll("~","");
  summaryplotname+=".pdf";
  cv->cd();
  cv->Update();
  cv->Print(summaryplotname);
  
}

double make_plot(TH1F* off, TH1F *test){
  if(off->GetSum()>0)
    off->Scale((float)test->GetSum()/(float)off->GetSum());
  else if (off->GetEntries()>0)
    off->Scale((float)test->GetEntries()/(float)off->GetEntries());
  if(off->GetMaximum()<test->GetMaximum())
    off->SetMaximum(1.3*test->GetMaximum());
  else
    off->SetMaximum(1.3*off->GetMaximum());
  off->SetLineColor(1);
  off->SetLineStyle(2);
  test->SetLineColor(2);
  test->SetMarkerColor(2);
  test->SetMarkerStyle(23);
  off->Draw("hist");
  test->Draw("psame");
  off->Draw("histsame");
  double kval = off->KolmogorovTest(test);
  TString matchstr = "K = ";
  matchstr+=kval;
  matchstr.ReplaceAll("  "," ");
  matchstr.ReplaceAll("  "," ");
  matchstr.ReplaceAll("  "," ");

  TLegend *leg = new TLegend(0.4,0.75,0.9,0.9);
  leg->SetFillStyle(0);
  leg->SetBorderSize(0);
  leg->AddEntry(test,"data","lp");
  leg->AddEntry(off,"reference","l");
  leg->AddEntry("",matchstr,"");
  leg->Draw("same");
  
  return kval;
}




