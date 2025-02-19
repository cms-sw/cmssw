{
  string filename;
  cout<<"filename: "<<endl;
  cin >> filename;

  TFile f(filename.c_str());
  f.cd("ecalMipGraphs");

  TTree* namesListTree = (TTree*) gDirectory->Get("canvasNames");
  
  std::vector<std::string>* canvasNames = new std::vector<std::string>();
  namesListTree->SetBranchAddress("names",&canvasNames);
  namesListTree->GetEntry(0);
  
  TCanvas* currentCanvas;
  TCanvas canB("navigation","navigation",10,50,180,200);
  TButton *but = new TButton ("Next",".x $CMSSW_BASE/src/CaloOnlineTools/EcalTools/data/macro/DrawCanvasNext.C",0,0,1,.5);
  but->Draw();
  TButton *butPrev = new TButton ("Prev",".x $CMSSW_BASE/src/CaloOnlineTools/EcalTools/data/macro/DrawCanvasPrev.C",0,.5,1,1);
  butPrev->Draw();
  int canvasNum=0;
  std::string name = canvasNames->at(canvasNum);
  currentCanvas = (TCanvas*) gDirectory->Get(name.c_str());
  currentCanvas->Draw();
  currentCanvas->SetWindowPosition(200,50);
  currentCanvas->SetWindowSize(900,900);
}
