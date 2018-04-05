{
gROOT->Reset();
gROOT->SetStyle("Plain");

gStyle->SetOptStat(1111);
gStyle->SetOptFit(111);
     

TH1F  *h1etacoefmin16_3a = new TH1F("h1etacoefmin16_3a", "h1etacoefmin16_3a", 100, 0., 2.);
TH1F  *h1etacoefmin17a = new TH1F("h1etacoefmin17a", "h1etacoefmin17a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin18a = new TH1F("h1etacoefmin18a", "h1etacoefmin18a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin19a = new TH1F("h1etacoefmin19a", "h1etacoefmin19a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin20a = new TH1F("h1etacoefmin20a", "h1etacoefmin20a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin21a = new TH1F("h1etacoefmin21a", "h1etacoefmin21a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin22a = new TH1F("h1etacoefmin22a", "h1etacoefmin22a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin23a = new TH1F("h1etacoefmin23a", "h1etacoefmin23a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin24a = new TH1F("h1etacoefmin24a", "h1etacoefmin24a", 100, 0.6, 1.4);
TH1F  *h1etacoefmin25a = new TH1F("h1etacoefmin25a", "h1etacoefmin25a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin26a = new TH1F("h1etacoefmin26a", "h1etacoefmin26a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin27a = new TH1F("h1etacoefmin27a", "h1etacoefmin27a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin28a = new TH1F("h1etacoefmin28a", "h1etacoefmin28a", 100, 0.7, 1.3);
TH1F  *h1etacoefmin29a = new TH1F("h1etacoefmin29a", "h1etacoefmin29a", 100, 0.7, 1.3);

TH1F  *h1etacoefmin16_3b = new TH1F("h1etacoefmin16_3b", "h1etacoefmin16_3b", 100, 0., 2.);
TH1F  *h1etacoefmin17b = new TH1F("h1etacoefmin17b", "h1etacoefmin17b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin18b = new TH1F("h1etacoefmin18b", "h1etacoefmin18b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin19b = new TH1F("h1etacoefmin19b", "h1etacoefmin19b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin20b = new TH1F("h1etacoefmin20b", "h1etacoefmin20b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin21b = new TH1F("h1etacoefmin21b", "h1etacoefmin21b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin22b = new TH1F("h1etacoefmin22b", "h1etacoefmin22b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin23b = new TH1F("h1etacoefmin23b", "h1etacoefmin23b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin24b = new TH1F("h1etacoefmin24b", "h1etacoefmin24b", 100, 0.6, 1.4);
TH1F  *h1etacoefmin25b = new TH1F("h1etacoefmin25b", "h1etacoefmin25b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin26b = new TH1F("h1etacoefmin26b", "h1etacoefmin26b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin27b = new TH1F("h1etacoefmin27b", "h1etacoefmin27b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin28b = new TH1F("h1etacoefmin28b", "h1etacoefmin28b", 100, 0.7, 1.3);
TH1F  *h1etacoefmin29b = new TH1F("h1etacoefmin29b", "h1etacoefmin29b", 100, 0.7, 1.3);

TH1F  *h1etacoefmin16_3 = new TH1F("h1etacoefmin16_3", "h1etacoefmin16_3", 100, 0., 2.);
TH1F  *h1etacoefmin17 = new TH1F("h1etacoefmin17", "h1etacoefmin17", 100, 0.6, 1.4);
TH1F  *h1etacoefmin18 = new TH1F("h1etacoefmin18", "h1etacoefmin18", 100, 0.6, 1.4);
TH1F  *h1etacoefmin19 = new TH1F("h1etacoefmin19", "h1etacoefmin19", 100, 0.6, 1.4);
TH1F  *h1etacoefmin20 = new TH1F("h1etacoefmin20", "h1etacoefmin20", 100, 0.6, 1.4);
TH1F  *h1etacoefmin21 = new TH1F("h1etacoefmin21", "h1etacoefmin21", 100, 0.6, 1.4);
TH1F  *h1etacoefmin22 = new TH1F("h1etacoefmin22", "h1etacoefmin22", 100, 0.6, 1.4);
TH1F  *h1etacoefmin23 = new TH1F("h1etacoefmin23", "h1etacoefmin23", 100, 0.6, 1.4);
TH1F  *h1etacoefmin24 = new TH1F("h1etacoefmin24", "h1etacoefmin24", 100, 0.6, 1.4);
TH1F  *h1etacoefmin25 = new TH1F("h1etacoefmin25", "h1etacoefmin25", 100, 0.7, 1.3);
TH1F  *h1etacoefmin26 = new TH1F("h1etacoefmin26", "h1etacoefmin26", 100, 0.7, 1.3);
TH1F  *h1etacoefmin27 = new TH1F("h1etacoefmin27", "h1etacoefmin27", 100, 0.7, 1.3);
TH1F  *h1etacoefmin28 = new TH1F("h1etacoefmin28", "h1etacoefmin28", 100, 0.7, 1.3);
TH1F  *h1etacoefmin29 = new TH1F("h1etacoefmin29", "h1etacoefmin29", 100, 0.7, 1.3);
// Two-dim


TH2F  *h2etacoefmin16_3a = new TH2F("h2etacoefmin16_3a", "h2etacoefmin16_3a",72, 0.5, 72.5, 100, 0., 2.);
TH2F  *h2etacoefmin17a = new TH2F("h2etacoefmin17a", "h2etacoefmin17a",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin18a = new TH2F("h2etacoefmin18a", "h2etacoefmin18a",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin19a = new TH2F("h2etacoefmin19a", "h2etacoefmin19a",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin20a = new TH2F("h2etacoefmin20a", "h2etacoefmin20a",72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin21a = new TH2F("h2etacoefmin21a", "h2etacoefmin21a",72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin22a = new TH2F("h2etacoefmin22a", "h2etacoefmin22a",72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin23a = new TH2F("h2etacoefmin23a", "h2etacoefmin23a",72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin24a = new TH2F("h2etacoefmin24a", "h2etacoefmin24a",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin25a = new TH2F("h2etacoefmin25a", "h2etacoefmin25a",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin26a = new TH2F("h2etacoefmin26a", "h2etacoefmin26a",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin27a = new TH2F("h2etacoefmin27a", "h2etacoefmin27a",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin28a = new TH2F("h2etacoefmin28a", "h2etacoefmin28a",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin29a = new TH2F("h2etacoefmin29a", "h2etacoefmin29a", 72, 0.5, 72.5,100, 0.7, 1.3);

TH2F  *h2etacoefmin16_3b = new TH2F("h2etacoefmin16_3b", "h2etacoefmin16_3b",72, 0.5, 72.5, 100, 0., 2.);
TH2F  *h2etacoefmin17b = new TH2F("h2etacoefmin17b", "h2etacoefmin17b",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin18b = new TH2F("h2etacoefmin18b", "h2etacoefmin18b",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin19b = new TH2F("h2etacoefmin19b", "h2etacoefmin19b",72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin20b = new TH2F("h2etacoefmin20b", "h2etacoefmin20b",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin21b = new TH2F("h2etacoefmin21b", "h2etacoefmin21b",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin22b = new TH2F("h2etacoefmin22b", "h2etacoefmin22b",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin23b = new TH2F("h2etacoefmin23b", "h2etacoefmin23b",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin24b = new TH2F("h2etacoefmin24b", "h2etacoefmin24b",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin25b = new TH2F("h2etacoefmin25b", "h2etacoefmin25b",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin26b = new TH2F("h2etacoefmin26b", "h2etacoefmin26b",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin27b = new TH2F("h2etacoefmin27b", "h2etacoefmin27b",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin28b = new TH2F("h2etacoefmin28b", "h2etacoefmin28b",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin29b = new TH2F("h2etacoefmin29b", "h2etacoefmin29b",72, 0.5, 72.5, 100, 0.7, 1.3);

TH2F  *h2etacoefmin16_3 = new TH2F("h2etacoefmin16_3", "h2etacoefmin16_3",72, 0.5, 72.5, 100, 0., 2.);
TH2F  *h2etacoefmin17 = new TH2F("h2etacoefmin17", "h2etacoefmin17",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin18 = new TH2F("h2etacoefmin18", "h2etacoefmin18",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin19 = new TH2F("h2etacoefmin19", "h2etacoefmin19",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin20 = new TH2F("h2etacoefmin20", "h2etacoefmin20",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin21 = new TH2F("h2etacoefmin21", "h2etacoefmin21",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin22 = new TH2F("h2etacoefmin22", "h2etacoefmin22",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin23 = new TH2F("h2etacoefmin23", "h2etacoefmin23",72, 0.5, 72.5,100, 0.6, 1.4);
TH2F  *h2etacoefmin24 = new TH2F("h2etacoefmin24", "h2etacoefmin24",72, 0.5, 72.5, 100, 0.6, 1.4);
TH2F  *h2etacoefmin25 = new TH2F("h2etacoefmin25", "h2etacoefmin25",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin26 = new TH2F("h2etacoefmin26", "h2etacoefmin26",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin27 = new TH2F("h2etacoefmin27", "h2etacoefmin27",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin28 = new TH2F("h2etacoefmin28", "h2etacoefmin28",72, 0.5, 72.5, 100, 0.7, 1.3);
TH2F  *h2etacoefmin29 = new TH2F("h2etacoefmin29", "h2etacoefmin29",72, 0.5, 72.5, 100, 0.7, 1.3);


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
  if( subd == 2 && eta < 0 && abs(eta)<30) {
    
cout<<"eta "<<subd<<" "<<eta<<" phi "<<phi<<endl;
     if( phi == maxc[i11]) {
     
       if(dep == 3 && eta == -16) {
       cout<<" 1 Phi saved "<< phi<<endl;
       h1etacoefmin16_3a->Fill(var);
       h2etacoefmin16_3a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       }
       
       if(dep == 1 && eta == -17) {
       cout<<" 2 Phi saved "<< phi<<endl;
       h1etacoefmin17a->Fill(var);
       h2etacoefmin17a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == -18) {
       cout<<" 3 Phi saved "<< phi<<endl;
       h1etacoefmin18a->Fill(var);
       h2etacoefmin18a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == -19) {
       cout<<" 4 Phi saved "<< phi<<endl;
       h1etacoefmin19a->Fill(var);
       h2etacoefmin19a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       
       }
       if(dep == 1 && eta == -20) {
       cout<<" 5 Phi saved "<< phi<<endl;
       h1etacoefmin20a->Fill(var);
       h2etacoefmin20a->Fill(phi,var);
       i11++;
       if(i11 == 36) i11=0;
       }
       }else{
       
       if(dep == 3 && eta == -16) {
       h1etacoefmin16_3b->Fill(var);
       h2etacoefmin16_3b->Fill(phi,var);
       }
       if(dep == 1 && eta == -17) {
       h1etacoefmin17b->Fill(var);
       h2etacoefmin17b->Fill(phi,var);
       //if( var > 0.95 ) h1etacoefmin2bn->Fill(var);
       }
       if(dep == 1 && eta == -18) {
       h1etacoefmin18b->Fill(var);
       h2etacoefmin18b->Fill(phi,var);
       //if( var > 0.9 ) h1etacoefmin3bn->Fill(var);
       }
       if(dep == 1 && eta == -19) {
       h1etacoefmin19b->Fill(var);
       h2etacoefmin19b->Fill(phi,var);
       //if( var > 0.9 ) h1etacoefmin4bn->Fill(var);
       }
       if(dep == 1 && eta == -20) {
       h1etacoefmin20b->Fill(var);
       h2etacoefmin20b->Fill(phi,var);
       }
       
       }
       
      if(abs(eta)>20 && phi == maxc1[i11]) {
      
       if(dep == 1 && eta == -21) {
       cout<<" 6 Phi saved "<< phi<<endl;
       h1etacoefmin21a->Fill(var);
       h2etacoefmin21a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       
       }
       if(dep == 1 && eta == -22) {
       cout<<" 7 Phi saved "<< phi<<endl;
       h1etacoefmin22a->Fill(var);
       h2etacoefmin22a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       
       }
       if(dep == 1 && eta == -23) {
       cout<<" 8 Phi saved "<< phi<<endl;
       h1etacoefmin23a->Fill(var);
       h2etacoefmin23a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       
       }
       if(dep == 1 && eta == -24) {
       cout<<" 9 Phi saved "<< phi<<endl;
       h1etacoefmin24a->Fill(var);
       h2etacoefmin24a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       
       }
       if(dep == 1 && eta == -25) {
       cout<<" 10 Phi saved "<< phi<<endl;
       h1etacoefmin25a->Fill(var);
       h2etacoefmin25a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       
       }
       if(dep == 1 && eta == -26) {
       cout<<" 11 Phi saved "<< phi<<endl;
       h1etacoefmin26a->Fill(var);
       h2etacoefmin26a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       
       }
       if(dep == 1 && eta == -27) {
       cout<<" 12 Phi saved "<< phi<<endl;
       h1etacoefmin27a->Fill(var);
       h2etacoefmin27a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       
       }
       if(dep == 1 && eta == -28) {
       cout<<" 13 Phi saved "<< phi<<endl;
       h1etacoefmin28a->Fill(var);
       h2etacoefmin28a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       
       }
       if(dep == 1 && eta == -29) {
       cout<<" 14 Phi saved "<< phi<<endl;
       h1etacoefmin29a->Fill(var);
       h2etacoefmin29a->Fill(phi,var);
       i11++;
       if(i11 == 18) i11=0;
       }
     } else{
        if(dep == 1 && eta == -21) {
       h1etacoefmin21b->Fill(var);
       h2etacoefmin21b->Fill(phi,var);
       }
       if(dep == 1 && eta == -22) {
       h1etacoefmin22b->Fill(var);
       h2etacoefmin22b->Fill(phi,var);
       }
       if(dep == 1 && eta == -23) {
       h1etacoefmin23b->Fill(var);
       h2etacoefmin23b->Fill(phi,var);
       }
       if(dep == 1 && eta == -24) {
       h1etacoefmin24b->Fill(var);
       h2etacoefmin24b->Fill(phi,var);
       }
       if(dep == 1 && eta == -25) {
       h1etacoefmin25b->Fill(var);
       h2etacoefmin25b->Fill(phi,var);
       }
       if(dep == 1 && eta == -26) {
       h1etacoefmin26b->Fill(var);
       h2etacoefmin26b->Fill(phi,var);
       }
       if(dep == 1 && eta == -27) {
       h1etacoefmin27b->Fill(var);
       h2etacoefmin27b->Fill(phi,var);
       }
       if(dep == 1 && eta == -28) {
       h1etacoefmin28b->Fill(var);
       h2etacoefmin28b->Fill(phi,var);
       }
       if(dep == 1 && eta == -29) {
       h1etacoefmin29b->Fill(var);
       h2etacoefmin29b->Fill(phi,var);
       }
      
     }  


    if(dep == 3 && eta == -16) {h2etacoefmin16_3->Fill(phi,var);}
    if(dep == 1 && eta == -17) {h2etacoefmin17->Fill(phi,var);}
    if(dep == 1 && eta == -18) {h2etacoefmin18->Fill(phi,var);}
    if(dep == 1 && eta == -19) {h2etacoefmin19->Fill(phi,var);} 
    if(dep == 1 && eta == -20) {h2etacoefmin20->Fill(phi,var);}
    if(dep == 1 && eta == -21) {h2etacoefmin21->Fill(phi,var);}
    if(dep == 1 && eta == -22) {h2etacoefmin22->Fill(phi,var);}
    if(dep == 1 && eta == -23) {h2etacoefmin23->Fill(phi,var);} 
    if(dep == 1 && eta == -24) {h2etacoefmin24->Fill(phi,var);} 
    if(dep == 1 && eta == -25) {h2etacoefmin25->Fill(phi,var);}       
    if(dep == 1 && eta == -26) {h2etacoefmin26->Fill(phi,var);}
    if(dep == 1 && eta == -27) {h2etacoefmin27->Fill(phi,var);} 
    if(dep == 1 && eta == -28) {h2etacoefmin28->Fill(phi,var);} 
    if(dep == 1 && eta == -29) {h2etacoefmin29->Fill(phi,var);}
    
/*
    if( phi == 70 || var < 0.95 ) continue;
*/    
    
    if(dep == 3 && eta == -16) {h1etacoefmin16_3->Fill(var);}
    if(dep == 1 && eta == -17) {h1etacoefmin17->Fill(var);}
    if(dep == 1 && eta == -18) {h1etacoefmin18->Fill(var);}
    if(dep == 1 && eta == -19) {h1etacoefmin19->Fill(var);} 
    if(dep == 1 && eta == -20) {h1etacoefmin20->Fill(var);}
    if(dep == 1 && eta == -21) {h1etacoefmin21->Fill(var);}
    if(dep == 1 && eta == -22) {h1etacoefmin22->Fill(var);}
    if(dep == 1 && eta == -23) {h1etacoefmin23->Fill(var);} 
    if(dep == 1 && eta == -24) {h1etacoefmin24->Fill(var);} 
    if(dep == 1 && eta == -25) {h1etacoefmin25->Fill(var);}       
    if(dep == 1 && eta == -26) {h1etacoefmin26->Fill(var);}
    if(dep == 1 && eta == -27) {h1etacoefmin27->Fill(var);} 
    if(dep == 1 && eta == -28) {h1etacoefmin28->Fill(var);} 
    if(dep == 1 && eta == -29) {h1etacoefmin29->Fill(var);}


  } // subd = 2
  if( subd > 2 ) break;
}


TFile efile("coefficients_219_val_endcap_minus_8.9mln.root","recreate");
/*
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
*/

h2etacoefmin16_3->Write();
h2etacoefmin17->Write();
h2etacoefmin18->Write();
h2etacoefmin19->Write();
h2etacoefmin20->Write();
h2etacoefmin21->Write();
h2etacoefmin22->Write();
h2etacoefmin23->Write();
h2etacoefmin24->Write();
h2etacoefmin25->Write();
h2etacoefmin26->Write();
h2etacoefmin27->Write();
h2etacoefmin28->Write();
h2etacoefmin29->Write();

h2etacoefmin16_3a->Write();
h2etacoefmin17a->Write();
h2etacoefmin18a->Write();
h2etacoefmin19a->Write();
h2etacoefmin20a->Write();
h2etacoefmin21a->Write();
h2etacoefmin22a->Write();
h2etacoefmin23a->Write();
h2etacoefmin24a->Write();
h2etacoefmin25a->Write();
h2etacoefmin26a->Write();
h2etacoefmin27a->Write();
h2etacoefmin28a->Write();
h2etacoefmin29a->Write();

h2etacoefmin16_3b->Write();
h2etacoefmin17b->Write();
h2etacoefmin18b->Write();
h2etacoefmin19b->Write();
h2etacoefmin20b->Write();
h2etacoefmin21b->Write();
h2etacoefmin22b->Write();
h2etacoefmin23b->Write();
h2etacoefmin24b->Write();
h2etacoefmin25b->Write();
h2etacoefmin26b->Write();
h2etacoefmin27b->Write();
h2etacoefmin28b->Write();
h2etacoefmin29b->Write();


h1etacoefmin16_3->Write();
h1etacoefmin17->Write();
h1etacoefmin18->Write();
h1etacoefmin19->Write();
h1etacoefmin20->Write();
h1etacoefmin21->Write();
h1etacoefmin22->Write();
h1etacoefmin23->Write();
h1etacoefmin24->Write();
h1etacoefmin25->Write();
h1etacoefmin26->Write();
h1etacoefmin27->Write();
h1etacoefmin28->Write();
h1etacoefmin29->Write();

h1etacoefmin16_3a->Write();
h1etacoefmin17a->Write();
h1etacoefmin18a->Write();
h1etacoefmin19a->Write();
h1etacoefmin20a->Write();
h1etacoefmin21a->Write();
h1etacoefmin22a->Write();
h1etacoefmin23a->Write();
h1etacoefmin24a->Write();
h1etacoefmin25a->Write();
h1etacoefmin26a->Write();
h1etacoefmin27a->Write();
h1etacoefmin28a->Write();
h1etacoefmin29a->Write();

h1etacoefmin16_3b->Write();
h1etacoefmin17b->Write();
h1etacoefmin18b->Write();
h1etacoefmin19b->Write();
h1etacoefmin20b->Write();
h1etacoefmin21b->Write();
h1etacoefmin22b->Write();
h1etacoefmin23b->Write();
h1etacoefmin24b->Write();
h1etacoefmin25b->Write();
h1etacoefmin26b->Write();
h1etacoefmin27b->Write();
h1etacoefmin28b->Write();
h1etacoefmin29b->Write();

}
