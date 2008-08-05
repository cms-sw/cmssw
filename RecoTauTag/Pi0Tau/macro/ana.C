

void ana(TString ds = "stau"){

  gROOT->LoadMacro("IsoAna.cc+");

  TChain* chain = new TChain("tree");

  chain->Add("hist_"+ds+".root");

  IsoAna* ana = new IsoAna(chain);

  ana->Loop(ds);

}
