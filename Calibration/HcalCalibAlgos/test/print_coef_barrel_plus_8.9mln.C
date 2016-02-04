{
gROOT->Reset();
gROOT->SetStyle("Plain");

gStyle->SetOptStat(1111);
gStyle->SetOptFit(111);
     

TH1F  *h1etacoefmin1a = new TH1F("h1etacoefmin1a", "h1etacoefmin1a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin2a = new TH1F("h1etacoefmin2a", "h1etacoefmin2a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin3a = new TH1F("h1etacoefmin3a", "h1etacoefmin3a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin4a = new TH1F("h1etacoefmin4a", "h1etacoefmin4a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin5a = new TH1F("h1etacoefmin5a", "h1etacoefmin5a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin6a = new TH1F("h1etacoefmin6a", "h1etacoefmin6a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin7a = new TH1F("h1etacoefmin7a", "h1etacoefmin7a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin8a = new TH1F("h1etacoefmin8a", "h1etacoefmin8a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin9a = new TH1F("h1etacoefmin9a", "h1etacoefmin9a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin10a = new TH1F("h1etacoefmin10a", "h1etacoefmin10a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin11a = new TH1F("h1etacoefmin11a", "h1etacoefmin11a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin12a = new TH1F("h1etacoefmin12a", "h1etacoefmin12a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin13a = new TH1F("h1etacoefmin13a", "h1etacoefmin13a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin14a = new TH1F("h1etacoefmin14a", "h1etacoefmin14a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin15a = new TH1F("h1etacoefmin15a", "h1etacoefmin15a", 100, 0., 2.);
TH1F  *h1etacoefmin16_1a = new TH1F("h1etacoefmin16_1a", "h1etacoefmin16_1a", 100, 0., 2.);
TH1F  *h1etacoefmin16_2a = new TH1F("h1etacoefmin16_2a", "h1etacoefmin16_2a", 100, 0., 2.);



TH1F  *h1etacoefmin1b = new TH1F("h1etacoefmin1b", "h1etacoefmin1b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin2b = new TH1F("h1etacoefmin2b", "h1etacoefmin2b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin2bn = new TH1F("h1etacoefmin2bn", "h1etacoefmin2bn", 100, 0.6, 1.4);
TH1F  *h1etacoefmin3b = new TH1F("h1etacoefmin3b", "h1etacoefmin3b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin3bn = new TH1F("h1etacoefmin3bn", "h1etacoefmin3bn", 100, 0.6, 1.4);
TH1F  *h1etacoefmin4b = new TH1F("h1etacoefmin4b", "h1etacoefmin4b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin4bn = new TH1F("h1etacoefmin4bn", "h1etacoefmin4bn", 100, 0.6, 1.4);
TH1F  *h1etacoefmin5b = new TH1F("h1etacoefmin5b", "h1etacoefmin5b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin6b = new TH1F("h1etacoefmin6b", "h1etacoefmin6b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin7b = new TH1F("h1etacoefmin7b", "h1etacoefmin7b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin8b = new TH1F("h1etacoefmin8b", "h1etacoefmin8b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin9b = new TH1F("h1etacoefmin9b", "h1etacoefmin9b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin10b = new TH1F("h1etacoefmin10b", "h1etacoefmin10b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin11b = new TH1F("h1etacoefmin11b", "h1etacoefmin11b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin12b = new TH1F("h1etacoefmin12b", "h1etacoefmin12b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin13b = new TH1F("h1etacoefmin13b", "h1etacoefmin13b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin14b = new TH1F("h1etacoefmin14b", "h1etacoefmin14b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin15b = new TH1F("h1etacoefmin15b", "h1etacoefmin15b", 100, 0., 2.);
TH1F  *h1etacoefmin16_1b = new TH1F("h1etacoefmin16_1b", "h1etacoefmin16_1b", 100, 0., 2.);
TH1F  *h1etacoefmin16_2b = new TH1F("h1etacoefmin16_2b", "h1etacoefmin16_2b", 100, 0., 2.);

//
TH1F  *h1etacoefmin1 = new TH1F("h1etacoefmin1", "h1etacoefmin1", 100, 0.6, 1.4);
TH1F  *h1etacoefmin1l = new TH1F("h1etacoefmin1l", "h1etacoefmin1l", 100, 0.6, 1.4);
TH1F  *h1etacoefmin2 = new TH1F("h1etacoefmin2", "h1etacoefmin2", 100, 0.6, 1.4);
TH1F  *h1etacoefmin2l = new TH1F("h1etacoefmin2l", "h1etacoefmin2l", 100, 0.6, 1.4);
TH1F  *h1etacoefmin3 = new TH1F("h1etacoefmin3", "h1etacoefmin3", 100, 0.6, 1.4);
TH1F  *h1etacoefmin3l = new TH1F("h1etacoefmin3l", "h1etacoefmin3l", 100, 0.6, 1.4);
TH1F  *h1etacoefmin4 = new TH1F("h1etacoefmin4", "h1etacoefmin4", 100, 0.6, 1.4);
TH1F  *h1etacoefmin4l = new TH1F("h1etacoefmin4l", "h1etacoefmin4l", 100, 0.6, 1.4);
TH1F  *h1etacoefmin5 = new TH1F("h1etacoefmin5", "h1etacoefmin5", 100, 0.6, 1.4);
TH1F  *h1etacoefmin5l = new TH1F("h1etacoefmin5l", "h1etacoefmin5l", 100, 0.6, 1.4);
TH1F  *h1etacoefmin6 = new TH1F("h1etacoefmin6", "h1etacoefmin6", 100, 0.6, 1.4);
TH1F  *h1etacoefmin6l = new TH1F("h1etacoefmin6l", "h1etacoefmin6l", 100, 0.6, 1.4);
TH1F  *h1etacoefmin7 = new TH1F("h1etacoefmin7", "h1etacoefmin7", 100, 0.6, 1.4);
TH1F  *h1etacoefmin7l = new TH1F("h1etacoefmin7l", "h1etacoefmin7l", 100, 0.6, 1.4);
TH1F  *h1etacoefmin8 = new TH1F("h1etacoefmin8", "h1etacoefmin8", 100, 0.6, 1.4);
TH1F  *h1etacoefmin8l = new TH1F("h1etacoefmin8l", "h1etacoefmin8l", 100, 0.6, 1.4);
TH1F  *h1etacoefmin9 = new TH1F("h1etacoefmin9", "h1etacoefmin9", 100, 0.7, 1.3);
TH1F  *h1etacoefmin9l = new TH1F("h1etacoefmin9l", "h1etacoefmin9l", 100, 0.7, 1.3);
TH1F  *h1etacoefmin10 = new TH1F("h1etacoefmin10", "h1etacoefmin10", 100, 0.7, 1.3);
TH1F  *h1etacoefmin10l = new TH1F("h1etacoefmin10l", "h1etacoefmin10l", 100, 0.7, 1.3);
TH1F  *h1etacoefmin11 = new TH1F("h1etacoefmin11", "h1etacoefmin11", 100, 0.7, 1.3);
TH1F  *h1etacoefmin11l = new TH1F("h1etacoefmin11l", "h1etacoefmin11l", 100, 0.7, 1.3);
TH1F  *h1etacoefmin12 = new TH1F("h1etacoefmin12", "h1etacoefmin12", 100, 0.7, 1.3);
TH1F  *h1etacoefmin12l = new TH1F("h1etacoefmin12l", "h1etacoefmin12l", 100, 0.7, 1.3);
TH1F  *h1etacoefmin13 = new TH1F("h1etacoefmin13", "h1etacoefmin13", 100, 0.7, 1.3);
TH1F  *h1etacoefmin13l = new TH1F("h1etacoefmin13l", "h1etacoefmin13l", 100, 0.7, 1.3);
TH1F  *h1etacoefmin14 = new TH1F("h1etacoefmin14", "h1etacoefmin14", 100, 0.7, 1.3);
TH1F  *h1etacoefmin14l = new TH1F("h1etacoefmin14l", "h1etacoefmin14l", 100, 0.7, 1.3);
TH1F  *h1etacoefmin15 = new TH1F("h1etacoefmin15", "h1etacoefmin15", 100, 0.7, 1.3);
TH1F  *h1etacoefmin15l = new TH1F("h1etacoefmin15l", "h1etacoefmin15l", 100, 0.7, 1.3);
TH1F  *h1etacoefmin16_1 = new TH1F("h1etacoefmin16_1", "h1etacoefmin16_1", 100, 0.7, 1.3);
TH1F  *h1etacoefmin16_1l = new TH1F("h1etacoefmin16_1l", "h1etacoefmin16_1l", 100, 0.7, 1.3);

TH1F  *h1etacoefmin16_2 = new TH1F("h1etacoefmin16_2", "h1etacoefmin16_2", 100, 0.7, 1.3);
TH1F  *h1etacoefmin16_2l = new TH1F("h1etacoefmin16_2l", "h1etacoefmin16_2l", 100, 0.7, 1.3);
//
TH2F  *h2etacoefmin1a = new TH2F("h2etacoefmin1a", "h2etacoefmin1a", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin2a = new TH2F("h2etacoefmin2a", "h2etacoefmin2a", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin3a = new TH2F("h2etacoefmin3a", "h2etacoefmin3a", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin4a = new TH2F("h2etacoefmin4a", "h2etacoefmin4a", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin5a = new TH2F("h2etacoefmin5a", "h2etacoefmin5a", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin6a = new TH2F("h2etacoefmin6a", "h2etacoefmin6a", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin7a = new TH2F("h2etacoefmin7a", "h2etacoefmin7a", 72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin8a = new TH2F("h2etacoefmin8a", "h2etacoefmin8a", 72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin9a = new TH2F("h2etacoefmin9a", "h2etacoefmin9a", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin10a = new TH2F("h2etacoefmin10a", "h2etacoefmin10a", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin11a = new TH2F("h2etacoefmin11a", "h2etacoefmin11a", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin12a = new TH2F("h2etacoefmin12a", "h2etacoefmin12a", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin13a = new TH2F("h2etacoefmin13a", "h2etacoefmin13a", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin14a = new TH2F("h2etacoefmin14a", "h2etacoefmin14a", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin15a = new TH2F("h2etacoefmin15a", "h2etacoefmin15a", 72, 0.5, 72.5,100, 0., 2.);
TH2F  *h2etacoefmin16_1a = new TH2F("h2etacoefmin16_1a", "h2etacoefmin16_1a", 72, 0.5, 72.5,100, 0., 2.);
TH2F  *h2etacoefmin16_2a = new TH2F("h2etacoefmin16_2a", "h2etacoefmin16_2a", 72, 0.5, 72.5,100, 0., 2.);
//
TH2F  *h2etacoefmin1b = new TH2F("h2etacoefmin1b", "h2etacoefmin1b", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin2b = new TH2F("h2etacoefmin2b", "h2etacoefmin2b", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin3b = new TH2F("h2etacoefmin3b", "h2etacoefmin3b", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin4b = new TH2F("h2etacoefmin4b", "h2etacoefmin4b", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin5b = new TH2F("h2etacoefmin5b", "h2etacoefmin5b", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin6b = new TH2F("h2etacoefmin6b", "h2etacoefmin6b", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin7b = new TH2F("h2etacoefmin7b", "h2etacoefmin7b", 72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin8b = new TH2F("h2etacoefmin8b", "h2etacoefmin8b", 72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin9b = new TH2F("h2etacoefmin9b", "h2etacoefmin9b", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin10b = new TH2F("h2etacoefmin10b", "h2etacoefmin10b", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin11b = new TH2F("h2etacoefmin11b", "h2etacoefmin11b", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin12b = new TH2F("h2etacoefmin12b", "h2etacoefmin12b", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin13b = new TH2F("h2etacoefmin13b", "h2etacoefmin13b", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin14b = new TH2F("h2etacoefmin14b", "h2etacoefmin14b", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin15b = new TH2F("h2etacoefmin15b", "h2etacoefmin15b", 72, 0.5, 72.5,100, 0., 2.);
TH2F  *h2etacoefmin16_1b = new TH2F("h2etacoefmin16_1b", "h2etacoefmin16_1b", 72, 0.5, 72.5,100, 0., 2.);
TH2F  *h2etacoefmin16_2b = new TH2F("h2etacoefmin16_2b", "h2etacoefmin16_2b", 72, 0.5, 72.5,100, 0., 2.);

//
TH2F  *h2etacoefmin1 = new TH2F("h2etacoefmin1", "h2etacoefmin1", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin2 = new TH2F("h2etacoefmin2", "h2etacoefmin2", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin3 = new TH2F("h2etacoefmin3", "h2etacoefmin3", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin4 = new TH2F("h2etacoefmin4", "h2etacoefmin4", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin5 = new TH2F("h2etacoefmin5", "h2etacoefmin5", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin6 = new TH2F("h2etacoefmin6", "h2etacoefmin6", 72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin7 = new TH2F("h2etacoefmin7", "h2etacoefmin7", 72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin8 = new TH2F("h2etacoefmin8", "h2etacoefmin8", 72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin9 = new TH2F("h2etacoefmin9", "h2etacoefmin9", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin10 = new TH2F("h2etacoefmin10", "h2etacoefmin10", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin11 = new TH2F("h2etacoefmin11", "h2etacoefmin11", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin12 = new TH2F("h2etacoefmin12", "h2etacoefmin12", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin13 = new TH2F("h2etacoefmin13", "h2etacoefmin13", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin14 = new TH2F("h2etacoefmin14", "h2etacoefmin14", 72, 0.5, 72.5,100, 0.7, 1.3);
TH2F  *h2etacoefmin15 = new TH2F("h2etacoefmin15", "h2etacoefmin15", 72, 0.5, 72.5,100, 0., 2.);
TH2F  *h2etacoefmin16_1 = new TH2F("h2etacoefmin16_1", "h2etacoefmin16_1", 72, 0.5, 72.5,100, 0., 2.);
TH2F  *h2etacoefmin16_2 = new TH2F("h2etacoefmin16_2", "h2etacoefmin16_2", 72, 0.5, 72.5,100, 0., 2.);

cout<<" Book histos "<<endl;

std::string line;
std::ifstream in20( "coefficients_8.9mln.txt" );

Int_t i11 = 0;

Int_t maxc[36] = {1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37,40,41,44,45,48,49,52,53,56,57,60,61,64,65,68,69,72};

while( std::getline( in20, line)){
int subd,eta,phi,dep;
Float_t var,err;
istringstream linestream(line);
linestream>>subd>>dep>>eta>>phi>>var>>err;
//cout<<"eta "<<eta<<endl;
  if(subd==1) {
    
     if( phi == maxc[i11]) {
       if(dep == 1 && eta == 1) {
   //    cout<<" 1 Phi saved "<< phi<<endl;
       h1etacoefmin1a->Fill(var);
       h2etacoefmin1a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       }
       if(dep == 1 && eta == 2) {
    //   cout<<" 2 Phi saved "<< phi<<endl;
       h1etacoefmin2a->Fill(var);
       h2etacoefmin2a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 3) {
    //   cout<<" 3 Phi saved "<< phi<<endl;
       h1etacoefmin3a->Fill(var);
       h2etacoefmin3a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 4) {
    //   cout<<" 4 Phi saved "<< phi<<endl;
       h1etacoefmin4a->Fill(var);
       h2etacoefmin4a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 5) {
     //  cout<<" 5 Phi saved "<< phi<<endl;
       h1etacoefmin5a->Fill(var);
       h2etacoefmin5a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 6) {
     //  cout<<" 6 Phi saved "<< phi<<endl;
       h1etacoefmin6a->Fill(var);
       h2etacoefmin6a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 7) {
     //  cout<<" 7 Phi saved "<< phi<<endl;
       h1etacoefmin7a->Fill(var);
       h2etacoefmin7a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 8) {
     //  cout<<" 8 Phi saved "<< phi<<endl;
       h1etacoefmin8a->Fill(var);
       h2etacoefmin8a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 9) {
     //  cout<<" 9 Phi saved "<< phi<<endl;
       h1etacoefmin9a->Fill(var);
       h2etacoefmin9a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 10) {
      // cout<<" 10 Phi saved "<< phi<<endl;
       h1etacoefmin10a->Fill(var);
       h2etacoefmin10a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 11) {
      // cout<<" 11 Phi saved "<< phi<<endl;
       h1etacoefmin11a->Fill(var);
       h2etacoefmin11a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 12) {
      // cout<<" 12 Phi saved "<< phi<<endl;
       h1etacoefmin12a->Fill(var);
       h2etacoefmin12a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 13) {
      // cout<<" 13 Phi saved "<< phi<<endl;
       h1etacoefmin13a->Fill(var);
       h2etacoefmin13a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 14) {
      // cout<<" 14 Phi saved "<< phi<<endl;
       h1etacoefmin14a->Fill(var);
       h2etacoefmin14a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 15) {
     //  cout<<" 15 Phi saved "<< phi<<endl;
       h1etacoefmin15a->Fill(var);
       h2etacoefmin15a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == 16) {
      // cout<<" 16 Phi saved "<< phi<<endl;
       h1etacoefmin16_1a->Fill(var);
       h2etacoefmin16_1a->Fill(phi,var);
       i11++;
       }
     } else{
       if(dep == 1 && eta == 1) {
       h1etacoefmin1b->Fill(var);
       h2etacoefmin1b->Fill(phi,var);
       }
       if(dep == 1 && eta == 2) {
       h1etacoefmin2b->Fill(var);
       h2etacoefmin2b->Fill(phi,var);
       if( var > 0.95 ) h1etacoefmin2bn->Fill(var);
       }
       if(dep == 1 && eta == 3) {
       h1etacoefmin3b->Fill(var);
       h2etacoefmin3b->Fill(phi,var);
       if( var > 0.9 ) h1etacoefmin3bn->Fill(var);
       }
       if(dep == 1 && eta == 4) {
       h1etacoefmin4b->Fill(var);
       h2etacoefmin4b->Fill(phi,var);
       if( var > 0.9 ) h1etacoefmin4bn->Fill(var);
       }
       if(dep == 1 && eta == 5) {
       h1etacoefmin5b->Fill(var);
       h2etacoefmin5b->Fill(phi,var);
       }
        if(dep == 1 && eta == 6) {
       h1etacoefmin6b->Fill(var);
       h2etacoefmin6b->Fill(phi,var);
       }
       if(dep == 1 && eta == 7) {
       h1etacoefmin7b->Fill(var);
       h2etacoefmin7b->Fill(phi,var);
       }
       if(dep == 1 && eta == 8) {
       h1etacoefmin8b->Fill(var);
       h2etacoefmin8b->Fill(phi,var);
       }
       if(dep == 1 && eta == 9) {
       h1etacoefmin9b->Fill(var);
       h2etacoefmin9b->Fill(phi,var);
       }
       if(dep == 1 && eta == 10) {
       h1etacoefmin10b->Fill(var);
       h2etacoefmin10b->Fill(phi,var);
       }
       if(dep == 1 && eta == 11) {
       h1etacoefmin11b->Fill(var);
       h2etacoefmin11b->Fill(phi,var);
       }
       if(dep == 1 && eta == 12) {
       h1etacoefmin12b->Fill(var);
       h2etacoefmin12b->Fill(phi,var);
       }
       if(dep == 1 && eta == 13) {
       h1etacoefmin13b->Fill(var);
       h2etacoefmin13b->Fill(phi,var);
       }
       if(dep == 1 && eta == 14) {
       h1etacoefmin14b->Fill(var);
       h2etacoefmin14b->Fill(phi,var);
       }
       if(dep == 1 && eta == 15) {
       h1etacoefmin15b->Fill(var);
       h2etacoefmin15b->Fill(phi,var);
       }
       if(dep == 1 && eta == 16) {
       h1etacoefmin16_1b->Fill(var);
       h2etacoefmin16_1b->Fill(phi,var);
       }
      
     }  

    if(dep == 1 && eta == 1) {h2etacoefmin1->Fill(phi,var);}
    if(dep == 1 && eta == 2) {h2etacoefmin2->Fill(phi,var);}
    if(dep == 1 && eta == 3) {h2etacoefmin3->Fill(phi,var);}
    if(dep == 1 && eta == 4) {h2etacoefmin4->Fill(phi,var);} 
    if(dep == 1 && eta == 5) {h2etacoefmin5->Fill(phi,var);}
    if(dep == 1 && eta == 6) {h2etacoefmin6->Fill(phi,var);}
    if(dep == 1 && eta == 7) {h2etacoefmin7->Fill(phi,var);}
    if(dep == 1 && eta == 8) {h2etacoefmin8->Fill(phi,var);} 
    if(dep == 1 && eta == 9) {h2etacoefmin9->Fill(phi,var);} 
    if(dep == 1 && eta == 10) {h2etacoefmin10->Fill(phi,var);}       
    if(dep == 1 && eta == 11) {h2etacoefmin11->Fill(phi,var);}
    if(dep == 1 && eta == 12) {h2etacoefmin12->Fill(phi,var);} 
    if(dep == 1 && eta == 13) {h2etacoefmin13->Fill(phi,var);} 
    if(dep == 1 && eta == 14) {h2etacoefmin14->Fill(phi,var);}
    if(dep == 1 && eta == 15) {h2etacoefmin15->Fill(phi,var);}
    if(dep == 1 && eta == 16) {h2etacoefmin16_1->Fill(phi,var);}
    if(dep == 2 && eta == 16) {h2etacoefmin16_2->Fill(phi,var);}

        
    if(dep == 1 && eta == 1) {h1etacoefmin1->Fill(var); if( var > 0.95 && var < 1.1) h1etacoefmin1l->Fill(var);}
    if(dep == 1 && eta == 2) {h1etacoefmin2->Fill(var); if( var > 0.95 && var < 1.1) h1etacoefmin2l->Fill(var);}
    if(dep == 1 && eta == 3) {h1etacoefmin3->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin3l->Fill(var);}
    if(dep == 1 && eta == 4) {h1etacoefmin4->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin4l->Fill(var);} 
    if(dep == 1 && eta == 5) {h1etacoefmin5->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin5l->Fill(var);}
    if(dep == 1 && eta == 6) {h1etacoefmin6->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin6l->Fill(var);}
    if(dep == 1 && eta == 7) {h1etacoefmin7->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin7l->Fill(var);}
    if(dep == 1 && eta == 8) {h1etacoefmin8->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin8l->Fill(var);} 
    if(dep == 1 && eta == 9) {h1etacoefmin9->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin9l->Fill(var);} 
    if(dep == 1 && eta == 10) {h1etacoefmin10->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin10l->Fill(var);}       
    if(dep == 1 && eta == 11) {h1etacoefmin11->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin11l->Fill(var); if(var>1.) cout<<"phi= "<<phi<<endl;}
    if(dep == 1 && eta == 12) {h1etacoefmin12->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin12l->Fill(var);} 
    if(dep == 1 && eta == 13) {h1etacoefmin13->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin13l->Fill(var);} 
    if(dep == 1 && eta == 14) {h1etacoefmin14->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin14l->Fill(var);}
    if(dep == 1 && eta == 15) {h1etacoefmin15->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin15l->Fill(var);}
    if(dep == 1 && eta == 16) {h1etacoefmin16_1->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin16_1l->Fill(var);}
    if(dep == 2 && eta == 16) {h1etacoefmin16_2->Fill(var);if( var > 0.95 && var < 1.1) h1etacoefmin16_2l->Fill(var);}
    
  } // subd = 1
  if( subd > 1 ) break;
}


TFile efile("coefficients_219_val_barrel_plus_8.9mln.root","recreate");

h1etacoefmin1a->Write();
h1etacoefmin1a->Write();
h1etacoefmin2a->Write();
h1etacoefmin3a->Write();
h1etacoefmin4a->Write();
h1etacoefmin5a->Write();
h1etacoefmin6a->Write();
h1etacoefmin7a->Write();
h1etacoefmin8a->Write();
h1etacoefmin9a->Write();
h1etacoefmin10a->Write();
h1etacoefmin11a->Write();
h1etacoefmin12a->Write();
h1etacoefmin13a->Write();
h1etacoefmin14a->Write();
h1etacoefmin15a->Write();
h1etacoefmin16_1a->Write();
h1etacoefmin16_2a->Write();

h1etacoefmin1b->Write();
h1etacoefmin2b->Write();h1etacoefmin2bn->Write();
h1etacoefmin3b->Write();h1etacoefmin3bn->Write();
h1etacoefmin4b->Write();h1etacoefmin4bn->Write();
h1etacoefmin5b->Write();
h1etacoefmin6b->Write();
h1etacoefmin7b->Write();
h1etacoefmin8b->Write();
h1etacoefmin9b->Write();
h1etacoefmin10b->Write();
h1etacoefmin11b->Write();
h1etacoefmin12b->Write();
h1etacoefmin13b->Write();
h1etacoefmin14b->Write();
h1etacoefmin15b->Write();
h1etacoefmin16_1b->Write();
h1etacoefmin16_2b->Write();

h1etacoefmin1->Write(); h1etacoefmin1l->Write();
h1etacoefmin2->Write(); h1etacoefmin2l->Write();
h1etacoefmin3->Write();h1etacoefmin3l->Write();
h1etacoefmin4->Write();h1etacoefmin4l->Write();
h1etacoefmin5->Write();h1etacoefmin5l->Write();
h1etacoefmin6->Write();h1etacoefmin6l->Write();
h1etacoefmin7->Write();h1etacoefmin7l->Write();
h1etacoefmin8->Write();h1etacoefmin8l->Write();
h1etacoefmin9->Write();h1etacoefmin9l->Write();
h1etacoefmin10->Write();h1etacoefmin10l->Write();
h1etacoefmin11->Write();h1etacoefmin11l->Write();
h1etacoefmin12->Write();h1etacoefmin12l->Write();
h1etacoefmin13->Write();h1etacoefmin13l->Write();
h1etacoefmin14->Write();h1etacoefmin14l->Write();
h1etacoefmin15->Write();h1etacoefmin15l->Write();
h1etacoefmin16_1->Write();h1etacoefmin16_1l->Write();
h1etacoefmin16_2->Write();h1etacoefmin16_2l->Write();



h2etacoefmin1a->Write();
h2etacoefmin1a->Write();
h2etacoefmin2a->Write();
h2etacoefmin3a->Write();
h2etacoefmin4a->Write();
h2etacoefmin5a->Write();
h2etacoefmin6a->Write();
h2etacoefmin7a->Write();
h2etacoefmin8a->Write();
h2etacoefmin9a->Write();
h2etacoefmin10a->Write();
h2etacoefmin11a->Write();
h2etacoefmin12a->Write();
h2etacoefmin13a->Write();
h2etacoefmin14a->Write();
h2etacoefmin15a->Write();
h2etacoefmin16_1a->Write();
h2etacoefmin16_2a->Write();

h2etacoefmin1b->Write();
h2etacoefmin2b->Write();
h2etacoefmin3b->Write();
h2etacoefmin4b->Write();
h2etacoefmin5b->Write();
h2etacoefmin6b->Write();
h2etacoefmin7b->Write();
h2etacoefmin8b->Write();
h2etacoefmin9b->Write();
h2etacoefmin10b->Write();
h2etacoefmin11b->Write();
h2etacoefmin12b->Write();
h2etacoefmin13b->Write();
h2etacoefmin14b->Write();
h2etacoefmin15b->Write();
h2etacoefmin16_1b->Write();
h2etacoefmin16_2b->Write();

h2etacoefmin1->Write();
h2etacoefmin2->Write();
h2etacoefmin3->Write();
h2etacoefmin4->Write();
h2etacoefmin5->Write();
h2etacoefmin6->Write();
h2etacoefmin7->Write();
h2etacoefmin8->Write();
h2etacoefmin9->Write();
h2etacoefmin10->Write();
h2etacoefmin11->Write();
h2etacoefmin12->Write();
h2etacoefmin13->Write();
h2etacoefmin14->Write();
h2etacoefmin15->Write();
h2etacoefmin16_1->Write();
h2etacoefmin16_2->Write();

}
