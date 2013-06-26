void maketable_HLT(string model, string oldversion, string newversion, bool oldrelease) {
  
  //  TFile* file = new TFile(string("html/SUSYVal/histo_"+model+"/outputfile.root").c_str());
  TFile* file = new TFile(string("html/SUSYVal_"+oldversion+"_vs_"+newversion+"/"+model+"/outputfile.root").c_str());
  TH1D* histo1;
  TH1D* histo2;
  vector<string> tablelines;
  
  bool oldrelease = true;

  vector<double> e_eff;
  vector<string> e_line;

  vector<double> p_eff;
  vector<string> p_line;

  vector<double> jet_eff;
  vector<string> jet_line;

  vector<double > m_eff;
  vector<string> m_line;

  for(int iplot=0; iplot<8;iplot++) {
    char name[256];
    if(oldrelease == true)
      sprintf(name,"HltPaths_eff_%i",iplot);
    else
      sprintf(name,"HltPaths_eff_%i",iplot+8);
    histo = (TH1D*) file->Get(name);

    for(int iline=0; iline< histo->GetXaxis()->GetNbins(); iline++) {
      if(string(histo->GetXaxis()->GetBinLabel(iline+1)).find("Total") 
	 != string::npos) 
	break;
      
      double eff = histo->GetBinContent(iline+1);
      double err = histo->GetBinError(iline+1);
      
      sprintf(name,"| %s | %f +/- %f |",histo->GetXaxis()->GetBinLabel(iline+1),eff, err);

      if(string(name).find("Electron") != string::npos) {
	e_eff.push_back(eff);
	e_line.push_back(string(name));
      }

      if(string(name).find("Photon") != string::npos) {
	p_eff.push_back(eff);
	p_line.push_back(string(name));
      }

      if(string(name).find("Jet") != string::npos ||
	 string(name).find("jet") != string::npos) {
	jet_eff.push_back(eff);
	jet_line.push_back(string(name));
      }

      if(string(name).find("Mu") != string::npos) {
	m_eff.push_back(eff);
	m_line.push_back(string(name));
      }
    }
  }
  
  sort(e_eff.begin(),   e_eff.end());
  sort(p_eff.begin(),   p_eff.end());
  sort(jet_eff.begin(), jet_eff.end());
  sort(m_eff.begin(),   m_eff.end());


  vector<string> firstfour;
  for(int i=0; i<4; i++) {
    char eff_ch[256];
    sprintf(eff_ch,"%d",e_eff[i]);
    for(int j=0; j<int(e_line.size()); j++) {
      if(e_line[j].find(eff_ch) != string::npos) {
	firstfour.push_back(e_line[j]);
	e_line[j] = string("none");
	break;
      }
    }
  }
  
  for(int i=0; i<firstfour.size(); i++) 
    cout << firstfour[i] << endl;

  firstfour.clear();

  for(int i=0; i<4; i++) {
    char eff_ch[256];
    sprintf(eff_ch,"%d",p_eff[i]);
    for(int j=0; j<int(p_line.size()); j++) {
      if(p_line[j].find(eff_ch) != string::npos) {
	firstfour.push_back(p_line[j]);
	p_line[j] = "none";
	break;
      }
    }
  }

  for(int i=0; i<firstfour.size(); i++) 
    cout << firstfour[i] << endl;

  firstfour.clear();

  for(int i=0; i<4; i++) {
    char eff_ch[256];
    sprintf(eff_ch,"%d",jet_eff[i]);
    for(int j=0; j<int(jet_line.size()); j++) {
      if(jet_line[j].find(eff_ch) != string::npos) {
	firstfour.push_back(jet_line[j]);
	jet_line[j] = "none";
	break;
      }
    }
  }
  
  for(int i=0; i<firstfour.size(); i++) 
    cout << firstfour[i] << endl;

  firstfour.clear();

  for(int i=0; i<4; i++) {
    char eff_ch[256];
    sprintf(eff_ch,"%d",m_eff[i]);
    for(int j=0; j<int(m_line.size()); j++) {
      if(m_line[j].find(eff_ch) != string::npos) {
	firstfour.push_back(m_line[j]);
	m_line[j] = "none";
	break;
      }
    }
  }
  
  for(int i=0; i<firstfour.size(); i++) 
    cout << firstfour[i] << endl;
}


void maketable_HLT() {
  string oldversion = "183";
  string newversion = "200";
  cout << "---++ RELEASE: " << oldversion << endl;
  cout << "---++ LM1 " << endl;
  maketable_HLT("LM1",oldversion,newversion,true);
  cout << "---++ LM5 " << endl;
  maketable_HLT("LM5",oldversion,newversion,true);
  cout << "--- ++ LM9p " << endl;
  maketable_HLT("LM9p",oldversion,newversion,true);
  cout << "---++ GM1b " << endl;
  maketable_HLT("GM1b",oldversion,newversion,true);
  cout << "---++ RSgrav " << endl;
  maketable_HLT("RSgrav",oldversion,newversion,true);
  cout << "---++ Zprime " << endl;
  maketable_HLT("Zprime",oldversion,newversion,true);
  cout << endl;
  cout << endl;
  cout << endl;
  cout << endl;
  cout << endl;
  cout << "---++ " << newversion << endl;
  cout << "---++ LM1 " << endl;
  maketable_HLT("LM1",oldversion,newversion,false);
  cout << "---++ LM5 " << endl;
  maketable_HLT("LM5",oldversion,newversion,false);
  cout << "---++ LM9p " << endl;
  maketable_HLT("LM9p",oldversion,newversion,false);
  cout << "---++ GM1b " << endl;
  maketable_HLT("GM1b",oldversion,newversion,false);
  cout << "---++ RSgrav " << endl;
  maketable_HLT("RSgrav",oldversion,newversion,false);
  cout << "---++ Zprime " << endl;
  maketable_HLT("Zprime",oldversion,newversion,false);
}


