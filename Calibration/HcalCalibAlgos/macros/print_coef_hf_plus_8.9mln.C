{
gROOT->Reset();
gROOT->SetStyle("Plain");

gStyle->SetOptStat(1111);
gStyle->SetOptFit(111);
     
TH1F  *h1etacoefmin30a = new TH1F("h1etacoefmin30a", "h1etacoefmin30a", 100, 0., 2.);
TH1F  *h1etacoefmin31a = new TH1F("h1etacoefmin31a", "h1etacoefmin31a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin32a = new TH1F("h1etacoefmin32a", "h1etacoefmin32a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin30 = new TH1F("h1etacoefmin30", "h1etacoefmin30", 100, 0., 2.);
TH1F  *h1etacoefmin31 = new TH1F("h1etacoefmin31", "h1etacoefmin31", 100, 0.6, 1.4);
TH1F  *h1etacoefmin32 = new TH1F("h1etacoefmin32", "h1etacoefmin32", 100, 0.6, 1.4);

TH1F  *h1etacoefmin33 = new TH1F("h1etacoefmin33", "h1etacoefmin33", 100, 0.6, 1.4);
TH1F  *h1etacoefmin34 = new TH1F("h1etacoefmin34", "h1etacoefmin34", 100, 0.6, 1.4);
TH1F  *h1etacoefmin35 = new TH1F("h1etacoefmin35", "h1etacoefmin35", 100, 0.6, 1.4);
TH1F  *h1etacoefmin36 = new TH1F("h1etacoefmin36", "h1etacoefmin36", 100, 0.6, 1.4);
TH1F  *h1etacoefmin37 = new TH1F("h1etacoefmin37", "h1etacoefmin37", 100, 0.6, 1.4);
TH1F  *h1etacoefmin38 = new TH1F("h1etacoefmin38", "h1etacoefmin38", 100, 0.6, 1.4);
TH1F  *h1etacoefmin39 = new TH1F("h1etacoefmin39", "h1etacoefmin39", 100, 0.7, 1.3);
TH1F  *h1etacoefmin40 = new TH1F("h1etacoefmin40", "h1etacoefmin40", 100, 0.7, 1.3);
TH1F  *h1etacoefmin41 = new TH1F("h1etacoefmin41", "h1etacoefmin41", 100, 0.7, 1.3);
// Two-dim

TH2F  *h2etacoefmin30 = new TH2F("h2etacoefmin30", "h2etacoefmin30",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin31 = new TH2F("h2etacoefmin31", "h2etacoefmin31",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin32 = new TH2F("h2etacoefmin32", "h2etacoefmin32",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin33 = new TH2F("h2etacoefmin33", "h2etacoefmin33",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin34 = new TH2F("h2etacoefmin34", "h2etacoefmin34",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin35 = new TH2F("h2etacoefmin35", "h2etacoefmin35",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin36 = new TH2F("h2etacoefmin36", "h2etacoefmin36",72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin37 = new TH2F("h2etacoefmin37", "h2etacoefmin37",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin38 = new TH2F("h2etacoefmin38", "h2etacoefmin38",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin39 = new TH2F("h2etacoefmin39", "h2etacoefmin39",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin40 = new TH2F("h2etacoefmin40", "h2etacoefmin40",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin41 = new TH2F("h2etacoefmin41", "h2etacoefmin41",72, 0.5, 72.5, 100, 0.7, 1.3);


cout<<" Book histos "<<endl;

std::string line;
std::ifstream in20( "coefficients_8.9mln.txt" );

Int_t i11 = 0;

Int_t maxc[36] = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71};
Int_t maxc1[18] = {1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69};

while( std::getline( in20, line)){
int subd,eta,phi,dep;
Float_t var,err;
istringstream linestream(line);
linestream>>subd>>dep>>eta>>phi>>var>>err;
  if( subd == 4 && eta > 0 ) {

    if(dep == 1 && eta == 30) cout<<var<<endl;
    if(dep == 1 && eta == 30) {h2etacoefmin30->Fill(phi,var);}
    if(dep == 1 && eta == 31) {h2etacoefmin31->Fill(phi,var);}
    if(dep == 1 && eta == 32) {h2etacoefmin32->Fill(phi,var);} 
    if(dep == 1 && eta == 33) {h2etacoefmin33->Fill(phi,var);}
    if(dep == 1 && eta == 34) {h2etacoefmin34->Fill(phi,var);}
    if(dep == 1 && eta == 35) {h2etacoefmin35->Fill(phi,var);}
    if(dep == 1 && eta == 36) {h2etacoefmin36->Fill(phi,var);} 
    if(dep == 1 && eta == 37) {h2etacoefmin37->Fill(phi,var);} 
    if(dep == 1 && eta == 38) {h2etacoefmin38->Fill(phi,var);}       
    if(dep == 1 && eta == 39) {h2etacoefmin39->Fill(phi,var);}
    if(dep == 1 && eta == 40) {h2etacoefmin40->Fill(phi,var);} 
    if(dep == 1 && eta == 41) {h2etacoefmin41->Fill(phi,var);} 
    
/*
    if( phi == 70 || var < 0.95 ) continue;
*/    
    
    if(dep == 1 && eta == 30) {h1etacoefmin30->Fill(var);if(var<1.02) h1etacoefmin30a->Fill(var);}
    if(dep == 1 && eta == 31) {h1etacoefmin31->Fill(var);if(var<1.02) h1etacoefmin31a->Fill(var);}
    if(dep == 1 && eta == 32) {h1etacoefmin32->Fill(var);if(var<1.02) h1etacoefmin32a->Fill(var);}
    if(dep == 1 && eta == 33) {h1etacoefmin33->Fill(var);} 
    if(dep == 1 && eta == 34) {h1etacoefmin34->Fill(var);} 
    if(dep == 1 && eta == 35) {h1etacoefmin35->Fill(var);}       
    if(dep == 1 && eta == 36) {h1etacoefmin36->Fill(var);}
    if(dep == 1 && eta == 37) {h1etacoefmin37->Fill(var);} 
    if(dep == 1 && eta == 38) {h1etacoefmin38->Fill(var);} 
    if(dep == 1 && eta == 39) {h1etacoefmin39->Fill(var);}
    if(dep == 1 && eta == 40) {h1etacoefmin40->Fill(var);}
    if(dep == 1 && eta == 41) {h1etacoefmin41->Fill(var);}


  } // subd = 2
}


TFile efile("coefficients_219_val_hf_plus_8.9mln.root","recreate");

h1etacoefmin30->Write();h1etacoefmin30a->Write();
h1etacoefmin31->Write();h1etacoefmin31a->Write();
h1etacoefmin32->Write();h1etacoefmin32a->Write();
h1etacoefmin33->Write();
h1etacoefmin34->Write();
h1etacoefmin35->Write();
h1etacoefmin36->Write();
h1etacoefmin37->Write();
h1etacoefmin38->Write();
h1etacoefmin39->Write();
h1etacoefmin40->Write();
h1etacoefmin41->Write();

h2etacoefmin30->Write();
h2etacoefmin31->Write();
h2etacoefmin32->Write();
h2etacoefmin33->Write();
h2etacoefmin34->Write();
h2etacoefmin35->Write();
h2etacoefmin36->Write();
h2etacoefmin37->Write();
h2etacoefmin38->Write();
h2etacoefmin39->Write();
h2etacoefmin40->Write();
h2etacoefmin41->Write();

}
