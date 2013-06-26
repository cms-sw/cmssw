{

 TFile* f1_out = new TFile("./hcalCorrsFile.root","update");


ifstream respcorrs1("../data/response_corrections.txt");
ifstream respcorrs2("../data/calibConst_IsoTrk_testCone_26.3cm.txt");

TProfile* corrs1 = new TProfile("corrs1", "resresponse_corrections", 84,-42,42);
TProfile* corrs2 = new TProfile("corrs2", "calibConst_IsoTrk_testCone_26.3cm", 84,-42,42);


  Int_t   iEta;
  UInt_t  iPhi;
  Int_t  depth;
  //TString sdName;
  string sdName;
  UInt_t  detId;
  Float_t value;

 std::string line;
 
 while (getline(respcorrs1, line)) 
   {
     if(!line.size() || line[0]=='#') 	 continue;
     std::istringstream linestream(line);
     linestream >> iEta >> iPhi >> depth >> sdName >> value >> hex >> detId;
     if (sdName!="HO" && depth==1)    corrs1 -> Fill(iEta, value);
   }

 while (getline(respcorrs2, line)) 
   {
     if(!line.size() || line[0]=='#') 	 continue;
     std::istringstream linestream(line);
     linestream >> iEta >> iPhi >> depth >> sdName >> value >> hex >> detId;
     if (sdName!="HO" && depth==1)  corrs2-> Fill(iEta, value);
   }

  corrs1 -> Write("", TObject::kOverwrite);
  corrs2 -> Write("", TObject::kOverwrite);
  
  f1_out->Close();

}
