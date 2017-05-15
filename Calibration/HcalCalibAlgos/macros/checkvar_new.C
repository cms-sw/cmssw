{
gROOT->Reset();
gROOT->SetStyle("Plain");

gStyle->SetOptStat(1111);
gStyle->SetOptFit(111);
   Float_t plmeang_nn[120][120][5][5],plnoise[120][120][5][5];
   Float_t minmeang_nn[120][120][5][5],minnoise[120][120][5][5];
   
   FILE *Out1 = fopen("coef_with_noise.txt", "w+");  
   FILE *Out2 = fopen("coef_without_noise_12mln.txt", "w+");

   cout<<" Read "<<endl;
   
   std::ifstream in20( "var_noise_12mln.txt" );
   std::string line;
   while( std::getline( in20, line)){
   istringstream linestream(line);
   Float_t var,err;
   int subd,eta,phi,dep;
   linestream>>subd>>eta>>phi>>dep>>var;
   
   
   if( eta > 0 )
   {
     plnoise[eta][phi][dep][subd] = var;
   } else
   {
      minnoise[abs(eta)][phi][dep][subd] = var;
   }
   
   if( eta == 16) cout<<"subd "<<subd<<" "<<eta<<" "<<phi<<" "<<dep<<" "<<plnoise[eta][phi][dep][subd]<<endl;   
   
   
   }
   cout<<" End of noise read "<<endl;
   
   std::ifstream in21( "var.txt" );
   
   while( std::getline( in21, line)){
   istringstream linestream(line);
   Float_t var,err;
   int subd,eta,phi,dep;
   linestream>>subd>>eta>>phi>>dep>>var>>err;
   if( eta > 0 )
   {
     plmeang_nn[eta][phi][dep][subd] = var;
   } else
   {
      minmeang_nn[abs(eta)][phi][dep][subd] = var;
   }

   if( eta == 16) cout<<"subd "<<subd<<" "<<eta<<" "<<phi<<" "<<dep<<" "<<plmeang_nn[eta][phi][dep][subd]<<endl;   
   
   }
// 
// Choose depth
//   
   Int_t idep = 1;
   
   Float_t plmean1[30][5][5]; 
   Float_t minmean1[30][5][5];
   Float_t plmean1_nn[30][5][5]; 
   Float_t minmean1_nn[30][5][5];
   
   for(Int_t k=1; k<3; k++) 
   {
   
   for(Int_t i=1; i<30; i++) {
   
    Int_t nch1 = 72.;
    if( i > 20 ) nch1 = 36.; 
    
   plmean1[i][idep][k] = 0.; 
   minmean1[i][idep][k] = 0.;
   plmean1_nn[i][idep][k] = 0.; 
   minmean1_nn[i][idep][k] = 0.;
   
   for(Int_t j=1; j<73; j++){

//    if( k == 2 && j == 1 && i == 16 ) cout<<plmeang_nn[i][j][idep][k]<<endl;
      
    plmean1[i][idep][k] = plmean1[i][idep][k] + plmeang_nn[i][j][idep][k];
    minmean1[i][idep][k] = minmean1[i][idep][k] + minmeang_nn[i][j][idep][k];
    plmean1_nn[i][idep][k] = plmean1_nn[i][idep][k] + plmeang_nn[i][j][idep][k] - plnoise[i][j][idep][k];
    minmean1_nn[i][idep][k] = minmean1_nn[i][idep][k] + minmeang_nn[i][j][idep][k] - minnoise[i][j][idep][k];
    
   }
    
    plmean1[i][idep][k] = plmean1[i][idep][k]/nch1;
    minmean1[i][idep][k] = minmean1[i][idep][k]/nch1;
    
    plmean1_nn[i][idep][k] = plmean1_nn[i][idep][k]/nch1;
    minmean1_nn[i][idep][k] = minmean1_nn[i][idep][k]/nch1;
//    cout<<" mean "<<plmean1[i][idep][k]<<" "<<minmean1[i][idep][k]<<endl;
//    cout<<" mean_nn "<<plmean1_nn[i][idep][k]<<" "<<minmean1_nn[i][idep][k]<<endl;
   }
   }    
    
    TH1F  *h12mlnetacoefpl1 = new TH1F("h12mlnetacoefpl1", "h12mlnetacoefpl1", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl1_noise = new TH1F("h12mlnetacoefpl1_noise", "h12mlnetacoefpl1_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl2 = new TH1F("h12mlnetacoefpl2", "h12mlnetacoefpl2", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl2_noise = new TH1F("h12mlnetacoefpl2_noise", "h12mlnetacoefpl2_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl3 = new TH1F("h12mlnetacoefpl3", "h12mlnetacoefpl3", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl3_noise = new TH1F("h12mlnetacoefpl3_noise", "h12mlnetacoefpl3_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl4 = new TH1F("h12mlnetacoefpl4", "h12mlnetacoefpl4", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl4_noise = new TH1F("h12mlnetacoefpl4_noise", "h12mlnetacoefpl4_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl5 = new TH1F("h12mlnetacoefpl5", "h12mlnetacoefpl5", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl5_noise = new TH1F("h12mlnetacoefpl5_noise", "h12mlnetacoefpl5_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl6 = new TH1F("h12mlnetacoefpl6", "h12mlnetacoefpl6", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl6_noise = new TH1F("h12mlnetacoefpl6_noise", "h12mlnetacoefpl6_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl7 = new TH1F("h12mlnetacoefpl7", "h12mlnetacoefpl7", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl7_noise = new TH1F("h12mlnetacoefpl7_noise", "h12mlnetacoefpl7_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl8 = new TH1F("h12mlnetacoefpl8", "h12mlnetacoefpl8", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl8_noise = new TH1F("h12mlnetacoefpl8_noise", "h12mlnetacoefpl8_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl9 = new TH1F("h12mlnetacoefpl9", "h12mlnetacoefpl9", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl9_noise = new TH1F("h12mlnetacoefpl9_noise", "h12mlnetacoefpl9_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl10 = new TH1F("h12mlnetacoefpl10", "h12mlnetacoefpl10", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl10_noise = new TH1F("h12mlnetacoefpl10_noise", "h12mlnetacoefpl10_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl11 = new TH1F("h12mlnetacoefpl11", "h12mlnetacoefpl11", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl11_noise = new TH1F("h12mlnetacoefpl11_noise", "h12mlnetacoefpl11_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl12 = new TH1F("h12mlnetacoefpl12", "h12mlnetacoefpl12", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl12_noise = new TH1F("h12mlnetacoefpl12_noise", "h12mlnetacoefpl12_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl13 = new TH1F("h12mlnetacoefpl13", "h12mlnetacoefpl13", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl13_noise = new TH1F("h12mlnetacoefpl13_noise", "h12mlnetacoefpl13_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl14 = new TH1F("h12mlnetacoefpl14", "h12mlnetacoefpl14", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl14_noise = new TH1F("h12mlnetacoefpl14_noise", "h12mlnetacoefpl14_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl15 = new TH1F("h12mlnetacoefpl15", "h12mlnetacoefpl15", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl15_noise = new TH1F("h12mlnetacoefpl15_noise", "h12mlnetacoefpl15_noise", 100, 0., 2.);
// ieta=16 is in HB and HE
    TH1F  *h12mlnetacoefpl16_HB = new TH1F("h12mlnetacoefpl16_HB", "h12mlnetacoefpl16_HB", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl16_HB_noise = new TH1F("h12mlnetacoefpl16_HB_noise", "h12mlnetacoefpl16_HB_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl16_HE = new TH1F("h12mlnetacoefpl16_HE", "h12mlnetacoefpl16_HE", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl16_HE_noise = new TH1F("h12mlnetacoefpl16_HE_noise", "h12mlnetacoefpl16_HE_noise", 100, 0., 2.);
//    
    TH1F  *h12mlnetacoefpl17 = new TH1F("h12mlnetacoefpl17", "h12mlnetacoefpl17", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl17_noise = new TH1F("h12mlnetacoefpl17_noise", "h12mlnetacoefpl17_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl18 = new TH1F("h12mlnetacoefpl18", "h12mlnetacoefpl18", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl18_noise = new TH1F("h12mlnetacoefpl18_noise", "h12mlnetacoefpl18_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl19 = new TH1F("h12mlnetacoefpl19", "h12mlnetacoefpl19", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl19_noise = new TH1F("h12mlnetacoefpl19_noise", "h12mlnetacoefpl19_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl20 = new TH1F("h12mlnetacoefpl20", "h12mlnetacoefpl20", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl20_noise = new TH1F("h12mlnetacoefpl20_noise", "h12mlnetacoefpl20_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl21 = new TH1F("h12mlnetacoefpl21", "h12mlnetacoefpl21", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl21_noise = new TH1F("h12mlnetacoefpl21_noise", "h12mlnetacoefpl21_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl22 = new TH1F("h12mlnetacoefpl22", "h12mlnetacoefpl22", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl22_noise = new TH1F("h12mlnetacoefpl22_noise", "h12mlnetacoefpl22_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl23 = new TH1F("h12mlnetacoefpl23", "h12mlnetacoefpl23", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl23_noise = new TH1F("h12mlnetacoefpl23_noise", "h12mlnetacoefpl23_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl24 = new TH1F("h12mlnetacoefpl24", "h12mlnetacoefpl24", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl24_noise = new TH1F("h12mlnetacoefpl24_noise", "h12mlnetacoefpl24_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl25 = new TH1F("h12mlnetacoefpl25", "h12mlnetacoefpl25", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl25_noise = new TH1F("h12mlnetacoefpl25_noise", "h12mlnetacoefpl25_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl26 = new TH1F("h12mlnetacoefpl26", "h12mlnetacoefpl26", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl26_noise = new TH1F("h12mlnetacoefpl26_noise", "h12mlnetacoefpl26_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl27 = new TH1F("h12mlnetacoefpl27", "h12mlnetacoefpl27", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl27_noise = new TH1F("h12mlnetacoefpl27_noise", "h12mlnetacoefpl27_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl28 = new TH1F("h12mlnetacoefpl28", "h12mlnetacoefpl28", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl28_noise = new TH1F("h12mlnetacoefpl28_noise", "h12mlnetacoefpl28_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl29 = new TH1F("h12mlnetacoefpl29", "h12mlnetacoefpl29", 100, 0., 2.);
    TH1F  *h12mlnetacoefpl29_noise = new TH1F("h12mlnetacoefpl29_noise", "h12mlnetacoefpl29_noise", 100, 0., 2.);

    TH1F  *h12mlnetacoefmin1 = new TH1F("h12mlnetacoefmin1", "h12mlnetacoefmin1", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin1_noise = new TH1F("h12mlnetacoefmin1_noise", "h12mlnetacoefmin1_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin2 = new TH1F("h12mlnetacoefmin2", "h12mlnetacoefmin2", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin2_noise = new TH1F("h12mlnetacoefmin2_noise", "h12mlnetacoefmin2_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin3 = new TH1F("h12mlnetacoefmin3", "h12mlnetacoefmin3", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin3_noise = new TH1F("h12mlnetacoefmin3_noise", "h12mlnetacoefmin3_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin4 = new TH1F("h12mlnetacoefmin4", "h12mlnetacoefmin4", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin4_noise = new TH1F("h12mlnetacoefmin4_noise", "h12mlnetacoefmin4_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin5 = new TH1F("h12mlnetacoefmin5", "h12mlnetacoefmin5", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin5_noise = new TH1F("h12mlnetacoefmin5_noise", "h12mlnetacoefmin5_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin6 = new TH1F("h12mlnetacoefmin6", "h12mlnetacoefmin6", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin6_noise = new TH1F("h12mlnetacoefmin6_noise", "h12mlnetacoefmin6_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin7 = new TH1F("h12mlnetacoefmin7", "h12mlnetacoefmin7", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin7_noise = new TH1F("h12mlnetacoefmin7_noise", "h12mlnetacoefmin7_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin8 = new TH1F("h12mlnetacoefmin8", "h12mlnetacoefmin8", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin8_noise = new TH1F("h12mlnetacoefmin8_noise", "h12mlnetacoefmin8_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin9 = new TH1F("h12mlnetacoefmin9", "h12mlnetacoefmin9", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin9_noise = new TH1F("h12mlnetacoefmin9_noise", "h12mlnetacoefmin9_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin10 = new TH1F("h12mlnetacoefmin10", "h12mlnetacoefmin10", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin10_noise = new TH1F("h12mlnetacoefmin10_noise", "h12mlnetacoefmin10_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin11 = new TH1F("h12mlnetacoefmin11", "h12mlnetacoefmin11", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin11_noise = new TH1F("h12mlnetacoefmin11_noise", "h12mlnetacoefmin11_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin12 = new TH1F("h12mlnetacoefmin12", "h12mlnetacoefmin12", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin12_noise = new TH1F("h12mlnetacoefmin12_noise", "h12mlnetacoefmin12_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin13 = new TH1F("h12mlnetacoefmin13", "h12mlnetacoefmin13", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin13_noise = new TH1F("h12mlnetacoefmin13_noise", "h12mlnetacoefmin13_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin14 = new TH1F("h12mlnetacoefmin14", "h12mlnetacoefmin14", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin14_noise = new TH1F("h12mlnetacoefmin14_noise", "h12mlnetacoefmin14_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin15 = new TH1F("h12mlnetacoefmin15", "h12mlnetacoefmin15", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin15_noise = new TH1F("h12mlnetacoefmin15_noise", "h12mlnetacoefmin15_noise", 100, 0., 2.);
//   ieta=16 is in HB and HE 
    TH1F  *h12mlnetacoefmin16_HB = new TH1F("h12mlnetacoefmin16_HB", "h12mlnetacoefmin16_HB", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin16_HB_noise = new TH1F("h12mlnetacoefmin16_HB_noise", "h12mlnetacoefmin16_HB_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin16_HE = new TH1F("h12mlnetacoefmin16_HE", "h12mlnetacoefmin16_HE", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin16_HE_noise = new TH1F("h12mlnetacoefmin16_HE_noise", "h12mlnetacoefmin16_HE_noise", 100, 0., 2.);
//
    TH1F  *h12mlnetacoefmin17 = new TH1F("h12mlnetacoefmin17", "h12mlnetacoefmin17", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin17_noise = new TH1F("h12mlnetacoefmin17_noise", "h12mlnetacoefmin17_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin18 = new TH1F("h12mlnetacoefmin18", "h12mlnetacoefmin18", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin18_noise = new TH1F("h12mlnetacoefmin18_noise", "h12mlnetacoefmin18_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin19 = new TH1F("h12mlnetacoefmin19", "h12mlnetacoefmin19", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin19_noise = new TH1F("h12mlnetacoefmin19_noise", "h12mlnetacoefmin19_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin20 = new TH1F("h12mlnetacoefmin20", "h12mlnetacoefmin20", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin20_noise = new TH1F("h12mlnetacoefmin20_noise", "h12mlnetacoefmin20_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin21 = new TH1F("h12mlnetacoefmin21", "h12mlnetacoefmin21", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin21_noise = new TH1F("h12mlnetacoefmin21_noise", "h12mlnetacoefmin21_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin22 = new TH1F("h12mlnetacoefmin22", "h12mlnetacoefmin22", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin22_noise = new TH1F("h12mlnetacoefmin22_noise", "h12mlnetacoefmin22_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin23 = new TH1F("h12mlnetacoefmin23", "h12mlnetacoefmin23", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin23_noise = new TH1F("h12mlnetacoefmin23_noise", "h12mlnetacoefmin23_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin24 = new TH1F("h12mlnetacoefmin24", "h12mlnetacoefmin24", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin24_noise = new TH1F("h12mlnetacoefmin24_noise", "h12mlnetacoefmin24_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin25 = new TH1F("h12mlnetacoefmin25", "h12mlnetacoefmin25", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin25_noise = new TH1F("h12mlnetacoefmin25_noise", "h12mlnetacoefmin25_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin26 = new TH1F("h12mlnetacoefmin26", "h12mlnetacoefmin26", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin26_noise = new TH1F("h12mlnetacoefmin26_noise", "h12mlnetacoefmin26_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin27 = new TH1F("h12mlnetacoefmin27", "h12mlnetacoefmin27", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin27_noise = new TH1F("h12mlnetacoefmin27_noise", "h12mlnetacoefmin27_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin28 = new TH1F("h12mlnetacoefmin28", "h12mlnetacoefmin28", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin28_noise = new TH1F("h12mlnetacoefmin28_noise", "h12mlnetacoefmin28_noise", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin29 = new TH1F("h12mlnetacoefmin29", "h12mlnetacoefmin29", 100, 0., 2.);
    TH1F  *h12mlnetacoefmin29_noise = new TH1F("h12mlnetacoefmin29_noise", "h12mlnetacoefmin29_noise", 100, 0., 2.);

//    
// Two dimensional histogram 
//   
    TH2F  *h12mlnetacoefpl2D1 = new TH2F("h12mlnetacoefpl2D1", "h12mlnetacoefpl2D1", 72, 0.5, 72.5, 100, 0., 2.);
    TH2F  *h12mlnetacoefpl2D1_noise = new TH2F("h12mlnetacoefpl2D1_noise", "h12mlnetacoefpl2D1_noise",72, 0.5, 72.5, 100, 0., 2.);
    TH2F  *h12mlnetacoefmin2D1 = new TH2F("h12mlnetacoefmin2D1", "h12mlnetacoefmin2D1",72, 0.5, 72.5, 100, 0., 2.);
    TH2F  *h12mlnetacoefmin2D1_noise = new TH2F("h12mlnetacoefmin2D1_noise", "h12mlnetacoefmin2D1_noise",72, 0.5, 72.5, 100, 0., 2.);


    TH2F  *h12mlnetacoefpl2D16 = new TH2F("h12mlnetacoefpl2D16", "h12mlnetacoefpl2D16", 72, 0.5, 72.5, 100, 0., 2.);
    TH2F  *h12mlnetacoefpl2D16_noise = new TH2F("h12mlnetacoefpl2D16_noise", "h12mlnetacoefpl2D16_noise",72, 0.5, 72.5, 100, 0., 2.);
    TH2F  *h12mlnetacoefmin2D16 = new TH2F("h12mlnetacoefmin2D16", "h12mlnetacoefmin2D16",72, 0.5, 72.5, 100, 0., 2.);
    TH2F  *h12mlnetacoefmin2D16_noise = new TH2F("h12mlnetacoefmin2D16_noise", "h12mlnetacoefmin2D16_noise",72, 0.5, 72.5, 100, 0., 2.);





    for(Int_t k=1; k<3; k++) 
    {     
    for (Int_t i=1; i<30; i++)
    {
    for (Int_t j=1; j<73; j++)
    {
//      if(i==1) cout<<" First "<<plmeang_nn[i][j][idep][k]<<" "<<plnoise[i][j][idep][k]<<" "<<plmeang_nn[i][j][idep][k] - plnoise[i][j][idep][k]<<" "<<plmean1[1][idep][k]<<
//          " "<<plmean1_nn[1][idep][k]<<endl;
	
	  
//      if( i == 16) cout<<j<<" "<<k<<" "<<plmeang_nn[i][j][idep][k]-plnoise[i][j][idep][k]<<" "<<plmeang_nn[i][j][idep][k]<<" "<<plnoise[i][j][idep][k]<<endl; 	
      if (plmeang_nn[i][j][idep][k]-plnoise[i][j][idep][k] > 0) {
	   Float_t tt0 = sqrt(plmean1_nn[i][idep][k]/(plmeang_nn[i][j][idep][k] - plnoise[i][j][idep][k]));
	   
    //       if( i == 16 && k == 1 ) cout<<" "<<plmeang_nn[i][j][idep][k]-plnoise[i][j][idep][k]<<" "<<plmean1_nn[i][idep][k]<<" "<<tt0<<endl;
	   
	   if( i == 1  ) {
//	      cout<<" 0J= "<<j<<endl;
	      h12mlnetacoefpl1->Fill(tt0);
	      h12mlnetacoefpl2D1->Fill((float)j, tt0);
	   } 
	     
	   if( i == 2  ) h12mlnetacoefpl2->Fill(tt0);
	   if( i == 3  ) h12mlnetacoefpl3->Fill(tt0);
	   if( i == 4  ) h12mlnetacoefpl4->Fill(tt0);
	   if( i == 5  ) h12mlnetacoefpl5->Fill(tt0);
	   if( i == 6  ) h12mlnetacoefpl6->Fill(tt0);
	   if( i == 7  ) h12mlnetacoefpl7->Fill(tt0);
	   if( i == 8  ) h12mlnetacoefpl8->Fill(tt0);
	   if( i == 9  ) h12mlnetacoefpl9->Fill(tt0);
	   if( i == 10  ) h12mlnetacoefpl10->Fill(tt0);
	   if( i == 11  ) {
	     cout<<" J= "<<j<<" Mean "<<plmean1_nn[i][idep][k]<<" Value "<<
	     plmeang_nn[i][j][idep][k] - plnoise[i][j][idep][k]<<" tt0 "<<tt0<<endl; 
	     h12mlnetacoefpl11->Fill(tt0);
	   }
	   if( i == 12  ) h12mlnetacoefpl12->Fill(tt0);
	   if( i == 13  ) h12mlnetacoefpl13->Fill(tt0);
	   if( i == 14  ) h12mlnetacoefpl14->Fill(tt0);
	   if( i == 15  ) h12mlnetacoefpl15->Fill(tt0);
	   if( i == 16  ) { if (k == 1) h12mlnetacoefpl16_HB->Fill(tt0); if( k == 2 ) h12mlnetacoefpl16_HE->Fill(tt0);}
	   if( i == 16 )  h12mlnetacoefpl2D16->Fill((float)j, tt0);
	   if( i == 17  ) h12mlnetacoefpl17->Fill(tt0);
	   if( i == 18  ) h12mlnetacoefpl18->Fill(tt0);
	   if( i == 19  ) h12mlnetacoefpl19->Fill(tt0);
	   if( i == 20  ) h12mlnetacoefpl20->Fill(tt0);
	   if( i == 21  ) h12mlnetacoefpl21->Fill(tt0);
	   if( i == 22  ) h12mlnetacoefpl22->Fill(tt0);
	   if( i == 23  ) h12mlnetacoefpl23->Fill(tt0);
	   if( i == 24  ) h12mlnetacoefpl24->Fill(tt0);
	   if( i == 25  ) h12mlnetacoefpl25->Fill(tt0);
	   if( i == 26  ) h12mlnetacoefpl26->Fill(tt0);
	   if( i == 27  ) h12mlnetacoefpl27->Fill(tt0);
	   if( i == 28  ) h12mlnetacoefpl28->Fill(tt0);
	   if( i == 29  ) h12mlnetacoefpl29->Fill(tt0);
	   	   
	   fprintf(Out2,"%s %d %d %d %.5f\n","HB",i,j,idep,tt0);
	   
      }
      
      if (plmeang_nn[i][j][idep][k] > 0) {
	   if( i == 1  ) {
//	      cout<<" 1J= "<<j<<endl;
	   } 
      
	   Float_t tt0 = sqrt(plmean1[i][idep][k]/plmeang_nn[i][j][idep][k]);
	   if( i == 1  ) h12mlnetacoefpl1_noise->Fill(tt0);
	   if( i == 2  ) h12mlnetacoefpl2_noise->Fill(tt0);
	   if( i == 3  ) h12mlnetacoefpl3_noise->Fill(tt0);
	   if( i == 4  ) h12mlnetacoefpl4_noise->Fill(tt0);
	   if( i == 5  ) h12mlnetacoefpl5_noise->Fill(tt0);
	   if( i == 6  ) h12mlnetacoefpl6_noise->Fill(tt0);
	   if( i == 7  ) h12mlnetacoefpl7_noise->Fill(tt0);
	   if( i == 8  ) h12mlnetacoefpl8_noise->Fill(tt0);
	   if( i == 9  ) h12mlnetacoefpl9_noise->Fill(tt0);
	   if( i == 10  ) h12mlnetacoefpl10_noise->Fill(tt0);
	   if( i == 11  ) h12mlnetacoefpl11_noise->Fill(tt0);
	   if( i == 12  ) h12mlnetacoefpl12_noise->Fill(tt0);
	   if( i == 13  ) h12mlnetacoefpl13_noise->Fill(tt0);
	   if( i == 14  ) h12mlnetacoefpl14_noise->Fill(tt0);
	   if( i == 15  ) h12mlnetacoefpl15_noise->Fill(tt0);
	   if( i == 16  ) {if (k == 1) h12mlnetacoefpl16_HB_noise->Fill(tt0); if( k == 2 ) h12mlnetacoefpl16_HE_noise->Fill(tt0);}
	   if( i == 17  ) h12mlnetacoefpl17_noise->Fill(tt0);
	   if( i == 18  ) h12mlnetacoefpl18_noise->Fill(tt0);
	   if( i == 19  ) h12mlnetacoefpl19_noise->Fill(tt0);
	   if( i == 20  ) h12mlnetacoefpl20_noise->Fill(tt0);
	   if( i == 21  ) h12mlnetacoefpl21_noise->Fill(tt0);
	   if( i == 22  ) h12mlnetacoefpl22_noise->Fill(tt0);
	   if( i == 23  ) h12mlnetacoefpl23_noise->Fill(tt0);
	   if( i == 24  ) h12mlnetacoefpl24_noise->Fill(tt0);
	   if( i == 25  ) h12mlnetacoefpl25_noise->Fill(tt0);
	   if( i == 26  ) h12mlnetacoefpl26_noise->Fill(tt0);
	   if( i == 27  ) h12mlnetacoefpl27_noise->Fill(tt0);
	   if( i == 28  ) h12mlnetacoefpl28_noise->Fill(tt0);
	   if( i == 29  ) h12mlnetacoefpl29_noise->Fill(tt0);
	   
	   if( i == 1  ) h12mlnetacoefpl2D1_noise->Fill((float)j, tt0);
	   if( i == 16  ) h12mlnetacoefpl2D16_noise->Fill((float)j, tt0);
	   	   
	   fprintf(Out1,"%s %d %d %d %.5f\n","HB",i,j,idep,tt0);
	   if( i == 1  ) {
//	      cout<<" 1nJ= "<<j<<endl;
	   } 
 	   
      }
      
      Int_t ii = -1*i;
      
      if (minmeang_nn[i][j][idep][k]-minnoise[i][j][idep][k] > 0) {
 	   if( i == 1  ) {
//	      cout<<" 2J= "<<j<<endl;
	   } 
     
	   Float_t tt0 = sqrt(minmean1_nn[i][idep][k]/(minmeang_nn[i][j][idep][k] - minnoise[i][j][idep][k]));
	   
	   if( i == 1  ) h12mlnetacoefmin1->Fill(tt0);
	   if( i == 2  ) h12mlnetacoefmin2->Fill(tt0);
	   if( i == 3  ) h12mlnetacoefmin3->Fill(tt0);
	   if( i == 4  ) h12mlnetacoefmin4->Fill(tt0);
	   if( i == 5  ) h12mlnetacoefmin5->Fill(tt0);
	   if( i == 6  ) h12mlnetacoefmin6->Fill(tt0);
	   if( i == 7  ) h12mlnetacoefmin7->Fill(tt0);
	   if( i == 8  ) h12mlnetacoefmin8->Fill(tt0);
	   if( i == 9  ) h12mlnetacoefmin9->Fill(tt0);
	   if( i == 10  ) h12mlnetacoefmin10->Fill(tt0);
	   if( i == 11  ) h12mlnetacoefmin11->Fill(tt0);
	   if( i == 12  ) h12mlnetacoefmin12->Fill(tt0);
	   if( i == 13  ) h12mlnetacoefmin13->Fill(tt0);
	   if( i == 14  ) h12mlnetacoefmin14->Fill(tt0);
	   if( i == 15  ) h12mlnetacoefmin15->Fill(tt0);
	   if( i == 16  ) { if (k == 1) h12mlnetacoefmin16_HB->Fill(tt0); if( k == 2 ) h12mlnetacoefmin16_HE->Fill(tt0);}
	   if( i == 17  ) h12mlnetacoefmin17->Fill(tt0);
	   if( i == 18  ) h12mlnetacoefmin18->Fill(tt0);
	   if( i == 19  ) h12mlnetacoefmin19->Fill(tt0);
	   if( i == 20  ) h12mlnetacoefmin20->Fill(tt0);
	   if( i == 21  ) h12mlnetacoefmin21->Fill(tt0);
	   if( i == 22  ) h12mlnetacoefmin22->Fill(tt0);
	   if( i == 23  ) h12mlnetacoefmin23->Fill(tt0);
	   if( i == 24  ) h12mlnetacoefmin24->Fill(tt0);
	   if( i == 25  ) h12mlnetacoefmin25->Fill(tt0);
	   if( i == 26  ) h12mlnetacoefmin26->Fill(tt0);
	   if( i == 27  ) h12mlnetacoefmin27->Fill(tt0);
	   if( i == 28  ) h12mlnetacoefmin28->Fill(tt0);
	   if( i == 29  ) h12mlnetacoefmin29->Fill(tt0);
	   
	   fprintf(Out2,"%s %d %d %d %.5f\n","HB",ii,j,idep,tt0);
	   if( i == 1  ) h12mlnetacoefmin2D1->Fill((float)j, tt0);
	   if( i == 16  ) h12mlnetacoefmin2D16->Fill((float)j, tt0);
	   
      }
      
      if (minmeang_nn[i][j][idep][k] > 0) {
 	   if( i == 1  ) {
//	      cout<<" 3J= "<<j<<endl;
	   } 
      
	   Float_t tt0 = sqrt(minmean1[i][idep][k]/minmeang_nn[i][j][idep][k]);
	   if( i == 1  ) h12mlnetacoefmin1_noise->Fill(tt0);
	   if( i == 2  ) h12mlnetacoefmin2_noise->Fill(tt0);
	   if( i == 3  ) h12mlnetacoefmin3_noise->Fill(tt0);
	   if( i == 4  ) h12mlnetacoefmin4_noise->Fill(tt0);
	   if( i == 5  ) h12mlnetacoefmin5_noise->Fill(tt0);
	   if( i == 6  ) h12mlnetacoefmin6_noise->Fill(tt0);
	   if( i == 7  ) h12mlnetacoefmin7_noise->Fill(tt0);
	   if( i == 8  ) h12mlnetacoefmin8_noise->Fill(tt0);
	   if( i == 9  ) h12mlnetacoefmin9_noise->Fill(tt0);
	   if( i == 10  ) h12mlnetacoefmin10_noise->Fill(tt0);
	   if( i == 11  ) h12mlnetacoefmin11_noise->Fill(tt0);
	   if( i == 12  ) h12mlnetacoefmin12_noise->Fill(tt0);
	   if( i == 13  ) h12mlnetacoefmin13_noise->Fill(tt0);
	   if( i == 14  ) h12mlnetacoefmin14_noise->Fill(tt0);
	   if( i == 15  ) h12mlnetacoefmin15_noise->Fill(tt0);
	   if( i == 16  ) { if (k == 1) h12mlnetacoefmin16_HB_noise->Fill(tt0); if( k == 2 ) h12mlnetacoefmin16_HE_noise->Fill(tt0);}
	   if( i == 17  ) h12mlnetacoefmin17_noise->Fill(tt0);
	   if( i == 18  ) h12mlnetacoefmin18_noise->Fill(tt0);
	   if( i == 19  ) h12mlnetacoefmin19_noise->Fill(tt0);
	   if( i == 20  ) h12mlnetacoefmin20_noise->Fill(tt0);
	   if( i == 21  ) h12mlnetacoefmin21_noise->Fill(tt0);
	   if( i == 22  ) h12mlnetacoefmin22_noise->Fill(tt0);
	   if( i == 23  ) h12mlnetacoefmin23_noise->Fill(tt0);
	   if( i == 24  ) h12mlnetacoefmin24_noise->Fill(tt0);
	   if( i == 25  ) h12mlnetacoefmin25_noise->Fill(tt0);
	   if( i == 26  ) h12mlnetacoefmin26_noise->Fill(tt0);
	   if( i == 27  ) h12mlnetacoefmin27_noise->Fill(tt0);
	   if( i == 28  ) h12mlnetacoefmin28_noise->Fill(tt0);
	   if( i == 29  ) h12mlnetacoefmin29_noise->Fill(tt0);
	   
	   fprintf(Out1,"%s %d %d %d %.5f\n","HB",ii,j,idep,tt0);
	   if( i == 1  ) h12mlnetacoefmin2D1_noise->Fill((float)j, tt0);
	   if( i == 16  ) h12mlnetacoefmin2D16_noise->Fill((float)j, tt0);
	   
      }
      
      
    }
// Print histograms    
    }
    }
//    h12mlnetacoefpl1->Draw();
    fclose(Out1);
    fclose(Out2);
    
//    h12mlnetacoefdist->Draw();
//    h12mlnetacoefdist->Fit("gaus","","",0.8,1.1);


     TFile efile("coefficients_12mln.root","recreate");
     
 	   h12mlnetacoefpl1_noise->Write();
	   h12mlnetacoefpl2_noise->Write();
	   h12mlnetacoefpl3_noise->Write();
	   h12mlnetacoefpl4_noise->Write();
	   h12mlnetacoefpl5_noise->Write();
	   h12mlnetacoefpl6_noise->Write();
	   h12mlnetacoefpl7_noise->Write();
	   h12mlnetacoefpl8_noise->Write();
	   h12mlnetacoefpl9_noise->Write();
	   h12mlnetacoefpl10_noise->Write();
	   h12mlnetacoefpl11_noise->Write();
	   h12mlnetacoefpl12_noise->Write();
	   h12mlnetacoefpl13_noise->Write();
	   h12mlnetacoefpl14_noise->Write();
	   h12mlnetacoefpl15_noise->Write();
	   h12mlnetacoefpl16_HB_noise->Write();
	   h12mlnetacoefpl16_HE_noise->Write();
	   h12mlnetacoefpl17_noise->Write();
	   h12mlnetacoefpl18_noise->Write();
	   h12mlnetacoefpl19_noise->Write();
	   h12mlnetacoefpl20_noise->Write();
	   h12mlnetacoefpl21_noise->Write();
	   h12mlnetacoefpl22_noise->Write();
	   h12mlnetacoefpl23_noise->Write();
	   h12mlnetacoefpl24_noise->Write();
	   h12mlnetacoefpl25_noise->Write();
	   h12mlnetacoefpl26_noise->Write();
	   h12mlnetacoefpl27_noise->Write();
	   h12mlnetacoefpl28_noise->Write();
	   h12mlnetacoefpl29_noise->Write();
	   
 	   h12mlnetacoefpl1->Write();
	   h12mlnetacoefpl2->Write();
	   h12mlnetacoefpl3->Write();
	   h12mlnetacoefpl4->Write();
	   h12mlnetacoefpl5->Write();
	   h12mlnetacoefpl6->Write();
	   h12mlnetacoefpl7->Write();
	   h12mlnetacoefpl8->Write();
	   h12mlnetacoefpl9->Write();
	   h12mlnetacoefpl10->Write();
	   h12mlnetacoefpl11->Write();
	   h12mlnetacoefpl12->Write();
	   h12mlnetacoefpl13->Write();
	   h12mlnetacoefpl14->Write();
	   h12mlnetacoefpl15->Write();
	   h12mlnetacoefpl16_HB->Write();
	   h12mlnetacoefpl16_HE->Write();
	   h12mlnetacoefpl17->Write();
	   h12mlnetacoefpl18->Write();
	   h12mlnetacoefpl19->Write();
	   h12mlnetacoefpl20->Write();
	   h12mlnetacoefpl21->Write();
 	   h12mlnetacoefpl22->Write();
	   h12mlnetacoefpl23->Write();
	   h12mlnetacoefpl24->Write();
	   h12mlnetacoefpl25->Write();
	   h12mlnetacoefpl26->Write();
	   h12mlnetacoefpl27->Write();
	   h12mlnetacoefpl28->Write();
	   h12mlnetacoefpl29->Write();
   
 	   h12mlnetacoefpl1_noise->Clear();
	   h12mlnetacoefpl2_noise->Clear();
	   h12mlnetacoefpl3_noise->Clear();
	   h12mlnetacoefpl4_noise->Clear();
	   h12mlnetacoefpl5_noise->Clear();
	   h12mlnetacoefpl6_noise->Clear();
	   h12mlnetacoefpl7_noise->Clear();
	   h12mlnetacoefpl8_noise->Clear();
	   h12mlnetacoefpl9_noise->Clear();
	   h12mlnetacoefpl10_noise->Clear();
	   h12mlnetacoefpl11_noise->Clear();
	   h12mlnetacoefpl12_noise->Clear();
	   h12mlnetacoefpl13_noise->Clear();
	   h12mlnetacoefpl14_noise->Clear();
	   h12mlnetacoefpl15_noise->Clear();
	   h12mlnetacoefpl16_HB_noise->Clear();
	   h12mlnetacoefpl16_HE_noise->Clear();
	   h12mlnetacoefpl17_noise->Clear();
	   h12mlnetacoefpl18_noise->Clear();
	   h12mlnetacoefpl19_noise->Clear();
	   h12mlnetacoefpl20_noise->Clear();
	   h12mlnetacoefpl21_noise->Clear();	  
	   h12mlnetacoefpl22_noise->Clear();
	   h12mlnetacoefpl23_noise->Clear();
	   h12mlnetacoefpl24_noise->Clear();
	   h12mlnetacoefpl25_noise->Clear();
	   h12mlnetacoefpl26_noise->Clear();
	   h12mlnetacoefpl27_noise->Clear();
	   h12mlnetacoefpl28_noise->Clear();
	   h12mlnetacoefpl29_noise->Clear();

 	   h12mlnetacoefpl1->Clear();
	   h12mlnetacoefpl2->Clear();
	   h12mlnetacoefpl3->Clear();
	   h12mlnetacoefpl4->Clear();
	   h12mlnetacoefpl5->Clear();
	   h12mlnetacoefpl6->Clear();
	   h12mlnetacoefpl7->Clear();
	   h12mlnetacoefpl8->Clear();
	   h12mlnetacoefpl9->Clear();
	   h12mlnetacoefpl10->Clear();
	   h12mlnetacoefpl11->Clear();
	   h12mlnetacoefpl12->Clear();
	   h12mlnetacoefpl13->Clear();
	   h12mlnetacoefpl14->Clear();
	   h12mlnetacoefpl15->Clear();
	   h12mlnetacoefpl16_HB->Clear();
	   h12mlnetacoefpl16_HE->Clear();
	   h12mlnetacoefpl17->Clear();
	   h12mlnetacoefpl18->Clear();
	   h12mlnetacoefpl19->Clear();
	   h12mlnetacoefpl20->Clear();
	   h12mlnetacoefpl21->Clear();
	   h12mlnetacoefpl22->Clear();
	   h12mlnetacoefpl23->Clear();
	   h12mlnetacoefpl24->Clear();
	   h12mlnetacoefpl25->Clear();
	   h12mlnetacoefpl26->Clear();
	   h12mlnetacoefpl27->Clear();
	   h12mlnetacoefpl28->Clear();
	   h12mlnetacoefpl29->Clear();
	   
 	   h12mlnetacoefmin1_noise->Write();
	   h12mlnetacoefmin2_noise->Write();
	   h12mlnetacoefmin3_noise->Write();
	   h12mlnetacoefmin4_noise->Write();
	   h12mlnetacoefmin5_noise->Write();
	   h12mlnetacoefmin6_noise->Write();
	   h12mlnetacoefmin7_noise->Write();
	   h12mlnetacoefmin8_noise->Write();
	   h12mlnetacoefmin9_noise->Write();
	   h12mlnetacoefmin10_noise->Write();
	   h12mlnetacoefmin11_noise->Write();
	   h12mlnetacoefmin12_noise->Write();
	   h12mlnetacoefmin13_noise->Write();
	   h12mlnetacoefmin14_noise->Write();
	   h12mlnetacoefmin15_noise->Write();
	   h12mlnetacoefmin16_HB_noise->Write();
	   h12mlnetacoefmin16_HE_noise->Write();
	   h12mlnetacoefmin17_noise->Write();
	   h12mlnetacoefmin18_noise->Write();
	   h12mlnetacoefmin19_noise->Write();
	   h12mlnetacoefmin20_noise->Write();
	   h12mlnetacoefmin21_noise->Write();
	   h12mlnetacoefmin22_noise->Write();
	   h12mlnetacoefmin23_noise->Write();
	   h12mlnetacoefmin24_noise->Write();
	   h12mlnetacoefmin25_noise->Write();
	   h12mlnetacoefmin26_noise->Write();
	   h12mlnetacoefmin27_noise->Write();
	   h12mlnetacoefmin28_noise->Write();
	   h12mlnetacoefmin29_noise->Write();
	   
 	   h12mlnetacoefmin1->Write();
	   h12mlnetacoefmin2->Write();
	   h12mlnetacoefmin3->Write();
	   h12mlnetacoefmin4->Write();
	   h12mlnetacoefmin5->Write();
	   h12mlnetacoefmin6->Write();
	   h12mlnetacoefmin7->Write();
	   h12mlnetacoefmin8->Write();
	   h12mlnetacoefmin9->Write();
	   h12mlnetacoefmin10->Write();
	   h12mlnetacoefmin11->Write();
	   h12mlnetacoefmin12->Write();
	   h12mlnetacoefmin13->Write();
	   h12mlnetacoefmin14->Write();
	   h12mlnetacoefmin15->Write();
	   h12mlnetacoefmin16_HB->Write();
	   h12mlnetacoefmin16_HE->Write();
	   h12mlnetacoefmin17->Write();
	   h12mlnetacoefmin18->Write();
	   h12mlnetacoefmin19->Write();
	   h12mlnetacoefmin20->Write();
	   h12mlnetacoefmin21->Write();
 	   h12mlnetacoefmin22->Write();
	   h12mlnetacoefmin23->Write();
	   h12mlnetacoefmin24->Write();
	   h12mlnetacoefmin25->Write();
	   h12mlnetacoefmin26->Write();
	   h12mlnetacoefmin27->Write();
	   h12mlnetacoefmin28->Write();
	   h12mlnetacoefmin29->Write();
   
 	   h12mlnetacoefmin1_noise->Clear();
	   h12mlnetacoefmin2_noise->Clear();
	   h12mlnetacoefmin3_noise->Clear();
	   h12mlnetacoefmin4_noise->Clear();
	   h12mlnetacoefmin5_noise->Clear();
	   h12mlnetacoefmin6_noise->Clear();
	   h12mlnetacoefmin7_noise->Clear();
	   h12mlnetacoefmin8_noise->Clear();
	   h12mlnetacoefmin9_noise->Clear();
	   h12mlnetacoefmin10_noise->Clear();
	   h12mlnetacoefmin11_noise->Clear();
	   h12mlnetacoefmin12_noise->Clear();
	   h12mlnetacoefmin13_noise->Clear();
	   h12mlnetacoefmin14_noise->Clear();
	   h12mlnetacoefmin15_noise->Clear();
	   h12mlnetacoefmin16_HB_noise->Clear();
	   h12mlnetacoefmin16_HE_noise->Clear();
	   h12mlnetacoefmin17_noise->Clear();
	   h12mlnetacoefmin18_noise->Clear();
	   h12mlnetacoefmin19_noise->Clear();
	   h12mlnetacoefmin20_noise->Clear();
	   h12mlnetacoefmin21_noise->Clear();	  
	   h12mlnetacoefmin22_noise->Clear();
	   h12mlnetacoefmin23_noise->Clear();
	   h12mlnetacoefmin24_noise->Clear();
	   h12mlnetacoefmin25_noise->Clear();
	   h12mlnetacoefmin26_noise->Clear();
	   h12mlnetacoefmin27_noise->Clear();
	   h12mlnetacoefmin28_noise->Clear();
	   h12mlnetacoefmin29_noise->Clear();

 	   h12mlnetacoefmin1->Clear();
	   h12mlnetacoefmin2->Clear();
	   h12mlnetacoefmin3->Clear();
	   h12mlnetacoefmin4->Clear();
	   h12mlnetacoefmin5->Clear();
	   h12mlnetacoefmin6->Clear();
	   h12mlnetacoefmin7->Clear();
	   h12mlnetacoefmin8->Clear();
	   h12mlnetacoefmin9->Clear();
	   h12mlnetacoefmin10->Clear();
	   h12mlnetacoefmin11->Clear();
	   h12mlnetacoefmin12->Clear();
	   h12mlnetacoefmin13->Clear();
	   h12mlnetacoefmin14->Clear();
	   h12mlnetacoefmin15->Clear();
	   h12mlnetacoefmin16_HB->Clear();
	   h12mlnetacoefmin16_HB->Clear();
	   h12mlnetacoefmin17->Clear();
	   h12mlnetacoefmin18->Clear();
	   h12mlnetacoefmin19->Clear();
	   h12mlnetacoefmin20->Clear();
	   h12mlnetacoefmin21->Clear();
	   h12mlnetacoefmin22->Clear();
	   h12mlnetacoefmin23->Clear();
	   h12mlnetacoefmin24->Clear();
	   h12mlnetacoefmin25->Clear();
	   h12mlnetacoefmin26->Clear();
	   h12mlnetacoefmin27->Clear();
	   h12mlnetacoefmin28->Clear();
	   h12mlnetacoefmin29->Clear();
	   
    h12mlnetacoefpl2D1->Write();
    h12mlnetacoefpl2D1_noise->Write();
    h12mlnetacoefmin2D1->Write();
    h12mlnetacoefmin2D1_noise->Write();
    h12mlnetacoefpl2D1->Clear();
    h12mlnetacoefpl2D1_noise->Clear();
    h12mlnetacoefmin2D1->Clear();
    h12mlnetacoefmin2D1_noise->Clear();
    
    h12mlnetacoefpl2D16->Write();
    h12mlnetacoefpl2D16_noise->Write();
    h12mlnetacoefmin2D16->Write();
    h12mlnetacoefmin2D16_noise->Write();
    h12mlnetacoefpl2D16->Clear();
    h12mlnetacoefpl2D16_noise->Clear();
    h12mlnetacoefmin2D16->Clear();
    h12mlnetacoefmin2D16_noise->Clear();
     
}
