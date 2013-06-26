void makeAtable(string model, bool twiki, bool HLT) {
  
  //  TFile* file = new TFile(string("html/SUSYVal/histo_"+model+"/outputfile.root").c_str());
  TFile* file = new TFile(string("html/SUSYVal_168_vs_177/histo_"+model+"/outputfile.root").c_str());
  TH1D* histo1;
  TH1D* histo2;
  vector<string> tablelines;
  
  string code1 = "1_7_7";
  string code2 = "1_8_4";

  if(twiki == true) 
    tablelines.push_back(model+" | "+code1+" | "+code2+"|");
  else {
    tablelines.push_back("\\begin{table}");
    tablelines.push_back("\\begin{center}");
    tablelines.push_back("\\begin{tabular}{|c|c|c|}");
    tablelines.push_back("\\hline\\hline");
    tablelines.push_back(model+" & "+code1+" & "+code2+" \\\\");
    tablelines.push_back("\\hline");
  }    

  vector<int> e_index;
  vector<string> e_line;

  vector<int> jet_index;
  vector<string> jet_line;

  vector<int> mu_index;
  vector<string> mu_line;

  for(int iplot=0; iplot<8;iplot++) {
    char name[256];
    if(HLT == true) 
      sprintf(name,"HltPaths_eff_%i",iplot);
    else
      sprintf(name,"L1Paths_eff_%i",iplot);
    histo1 = (TH1D*) file->Get(name);
    if(HLT == true) 
      sprintf(name,"HltPaths_eff_%i",iplot+8);
    else
      sprintf(name,"L1Paths_eff_%i",iplot+8);
    histo2 = (TH1D*) file->Get(name);

    for(int iline=0; iline< histo1->GetXaxis()->GetNbins(); iline++) {
      if(string(histo1->GetXaxis()->GetBinLabel(iline+1)).find("Total") 
	 != string::npos) 
	break;
      
      double eff1 = histo1->GetBinContent(iline+1);
      double err1 = histo1->GetBinError(iline+1);
      double eff2 = histo2->GetBinContent(iline+1);
      double err2 = histo2->GetBinError(iline+1);
      
      if(twiki == true) {
	sprintf(name,"%s | %f +/- %f | %f +/- %f|",
		histo1->GetXaxis()->GetBinLabel(iline+1),
		eff1, err1, eff2, err2);
      } else {
	sprintf(name,"%s & %f \pm %f & %f \pm %f \\\\",
		histo1->GetXaxis()->GetBinLabel(iline+1),
		eff1, err1, eff2, err2);
      }

      string pathname(histo1->GetXaxis()->GetBinLabel(iline+1));
      if(HLT == true) {
      } else {
	if(pathname.find("EG") != string::npos) {
	  
	}
	  }

      tablelines.push_back(string(name));
    }
  }
  if(twiki == false) {
    tablelines.push_back("\\hline");
    tablelines.push_back("\\end{tabular}");
    tablelines.push_back("\\caption{\label{}}");
    tablelines.push_back("\\end{center}");
    tablelines.push_back("\\end{table}");
  }
  
  string options;
  if(twiki == true)
    if(HLT == true)
      options = "HLT_twiki";
    else
      options = "L1_twiki";
  else
    if(HLT == true)
      options = "HLT_latex";
    else
      options = "L1_latex";
    

  FILE*  f=fopen(string(model+"_table_"+options+".txt").c_str(),"w");
  for(int j = 0; j< int(tablelines.size()); j++) {
    fprintf(f,"%s\n",tablelines[j].c_str());
  }
  fclose(f);

}

void maketable(string model) {
  makeAtable(model,true,true);
  makeAtable(model,true,false);
  makeAtable(model,false,true);
  makeAtable(model,false,false);
}
