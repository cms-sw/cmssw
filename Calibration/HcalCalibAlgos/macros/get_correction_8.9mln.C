{
gROOT->Reset();
gROOT->SetStyle("Plain");

gStyle->SetOptStat(1111);
gStyle->SetOptFit(111);

   Float_t plsignal[50][73][5][5],plnoise[50][73][5][5];
   Float_t minsignal[50][73][5][5],minnoise[50][73][5][5];
   Float_t excluded_min[50][73][5][5];
   
   FILE *Out1 = fopen("coefficients_8.9mln.txt", "w+");  
   
    for(Int_t k=1; k<50; k++) 
    {
    for(Int_t i=1; i<73; i++) 
    {  
    for(Int_t j=1; j<5; j++) 
    {  
    for(Int_t l=1; l<5; l++) 
    {  
      excluded_min[k][i][j][l] = 0.;
    } // l  
    } // j
    } // i
    } // k 
   
   std::string line;
   
   std::ifstream in21( "HB_exclusion.txt" );

   while( std::getline( in21, line)){
   istringstream linestream(line);

   int eta,phi,dep;
   linestream>>dep>>eta>>phi;
    cout<<" Eta="<<eta<<endl;
   if( eta > 0 )
   {
     cout<<" eta > 0 "<<endl;
   } else
   {
     excluded_min[abs(eta)][phi][dep][1] = 1;
   }
  }


   cout<<" Read "<<endl;
   
   std::ifstream in20( "var_noise_8.9mln.txt" );
   
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
   }
   cout<<" End of noise read "<<endl;
   
   std::ifstream in21( "var_minbias_8.9mln.txt" );
   
   while( std::getline( in21, line)){
   istringstream linestream(line);
   Float_t var,err;
   int subd,eta,phi,dep;
   linestream>>subd>>eta>>phi>>dep>>var>>err;
   if( eta > 0 )
   {
     plsignal[eta][phi][dep][subd] = var;
   } else
   {
      minsignal[abs(eta)][phi][dep][subd] = var;
   }
   }
   
   cout<<" End of signal read "<<endl;
// 
// Choose depth
//   
   
   Float_t plmean[50][5][5]; 
   Float_t minmean[50][5][5];
   Int_t numchan[50][5][5];
   
   for(Int_t k=1; k<5; k++) 
   {
    for(Int_t idep0=1; idep0<5; idep0++) 
   {  
    for(Int_t i=1; i<42; i++) {
    
     plmean[i][idep0][k] = 0.; 
     minmean[i][idep0][k] = 0.;

     Int_t nch1 = 0; 
     Int_t nchp = 0; 
     Int_t nchm = 0;
       
     for(Int_t j=1; j<73; j++){
        if(minsignal[i][j][idep0][k]>0.) nch1++;
        if(plsignal[i][j][idep0][k]>0.) nchp++;
        if(minsignal[i][j][idep0][k]>0.) nchm++;
        plmean[i][idep0][k] = plmean[i][idep0][k] + plsignal[i][j][idep0][k] - plnoise[i][j][idep0][k];
        minmean[i][idep0][k] = minmean[i][idep0][k] + minsignal[i][j][idep0][k] - minnoise[i][j][idep0][k];
      } // j
          
    numchan[i][idep0][k] = nch1;    
    if(nch1 == 0) continue;      
    plmean[i][idep0][k] = plmean[i][idep0][k]/nchp;
    minmean[i][idep0][k] = minmean[i][idep0][k]/nchm;
    cout<<" k, idep0, i, nch1= "<<k<<" "<<idep0<<" "<<i<<" "<<nch1<<endl;
  // Do not calibrate HO
    Float_t err = 0.00001;
    
    if( k != 3 ) {
    for(Int_t j=1;j<73;j++) {
       if( plsignal[i][j][idep0][k] > 0.) {       
       Float_t tt0; 
       if( plsignal[i][j][idep0][k] - plnoise[i][j][idep0][k]>0.0000001 && plmean[i][idep0][k]>0.) {
         tt0 = sqrt(plmean[i][idep0][k]/(plsignal[i][j][idep0][k] - plnoise[i][j][idep0][k]));
       } else { 
	 tt0 = 1.;
       }
         fprintf(Out1,"%d %d %d %d %.5f %.5f\n",k,idep0,i,j,tt0,err);       
       } // plnoise
    } // j
    for(Int_t j=1;j<73;j++) {   
       if( minsignal[i][j][idep0][k] > 0. ) {       
       Float_t tt0;        
       if(minsignal[i][j][idep0][k] - minnoise[i][j][idep0][k]>0.&& minmean[i][idep0][k]>0. ) {
          tt0 = sqrt(minmean[i][idep0][k]/(minsignal[i][j][idep0][k] - minnoise[i][j][idep0][k]));
        } else {  
          tt0 = 1.;
       }      
       Int_t ieta = -1*i;
       fprintf(Out1,"%d %d %d %d %.5f %.5f\n",k,idep0,ieta,j,tt0,err);
       } // minnoise
     } // j
   } // TMP  
    } else { 
        for(Int_t j=1;j<73;j++){   
           Float_t tt0=1.; 
           Int_t ieta = i;
           fprintf(Out1,"%d %d %d %d %.5f %.5f\n",k,idep0,ieta,j,tt0,err);
        }// j
         
        for(Int_t j=1;j<73;j++){   
           Float_t tt0=1.; 
           Int_t ieta = -1*i;
           fprintf(Out1,"%d %d %d %d %.5f %.5f\n",k,idep0,ieta,j,tt0,err);
        }// j
     } // k = 3
      cout<<" End "<<endl;
   } // i
   } // idep0
   } // k   
    
   fclose(Out1);
}
