{
  char *file_name = "cscHists.root";
  string  nameOfTheHisto= "InefficientALCT_dydz";
  bool excludeChamberList[36];
  for(int iCh =0;iCh<36;++iCh){
    excludeChamberList[iCh] = false;
  }
  //  write here which chanmbers to exclude (numbers from 1) 
  //excludeChamberList[14-1] = true;  

  int endcap = 1;// + is "1", - is "2"
  int station = 2;
  int ring = 2;
  std::cout<<" Adding all "<<nameOfTheHisto<<" histograms from a E/S/R = "<<endcap<<"/"<<station<<"/"<<ring<<" and plotting the result..."<<std::endl;
  TFile *f1=
    (TFile*)gROOT->GetListOfFiles()->FindObject(file_name);
  if (!f1){
    TFile *f1 = new TFile(file_name);
  }
  
  //gStyle->SetOptStat(1110);
  //gStyle->SetErrorX(0);
  //gPad->SetFillColor(0);
  //gStyle->SetPalette(1);
  
  TH1F *data_p1;
  TH1F *sum_histo;

  char * charName;
  string histo;
  int iterations = 0;
  for(int iE=1;iE<3;++iE){
    if(iE!=endcap) continue; 
    for(int iS=1;iS<5;++iS){
      if(iS!=station) continue; 
      for(int iR=1;iR<4;++iR){
        if(iR != ring) continue;
	if(1!=iS && iR>2){
	  continue;
	}
	else if(2==iR && 4==iS){
	  continue;
	}
	for(int iC=1;iC<37;++iC){
          if(excludeChamberList[iC-1]) continue;
	  if(1!=iS && 1==iR && iC >18){
	    continue;
	  }
	  histo = Form("Chambers__E%d_S%d_R%d_Chamber_%d/%s_Ch%d",iE,iS,iR,iC,nameOfTheHisto,iC);
          //std::cout<<" histo name = "<<histo<<std::endl; 
	  charName = histo.c_str();
	  data_p1 =(TH1F*)(f1->Get(charName));
	  if(!iterations){
	    sum_histo=(TH1F*)data_p1->Clone();
	  }
	  else{
	    sum_histo->Add(data_p1);
	  }
	  ++iterations;
	}
      }
    }
  }
  sum_histo->Draw();

}
