#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>

#include<vector>
#include<string>
#include <TH1F.h>
#include <TFile.h>


using namespace std;
int main(int argc,  char * argv[]){

  if(argc < 4){cout<<" Usage: executable first_file_name second_file_name out_file(without suffix)"<<endl; 
  return -4;}
  cout<<"Please make sure that the peak time is expected in SAMPLE units"<<endl;
  cout<<"This program will run using the relative timing"<<endl;

  int fromSL1 = 0;
  int fromSL2 = 0;
  int noCompare =0;
  if (argc > 4) { 
     fromSL1 = atoi(argv[4]); 
  }
  if (argc > 5) { 
     fromSL2 = atoi(argv[5]); 
  }  
  if (argc > 6) {
     noCompare = atoi(argv[6]);
  }


  char Buffer[5000];
  //int SMn =0;
  //int HowManyTT =0;

  // reading the peak file
  double Shift1[71], Shift2[71], SMave1[71], SMave2[71];
  for(int i=0;i<71;i++){Shift1[i]=-1000.; Shift2[i]=-1000.; SMave1[i] = -1000.; SMave2[i] = -1000.;}
  int TTnum,SLSMn,SLshift;
  //float rms,shift,rel_shift, rel_rms;
  int HowManyShifts = 0;
  
  ifstream TxtFile1(argv[2]);
  if( !(TxtFile1.is_open()) ){cout<<"Error: file"<<argv[2]<<" not found!!"<<endl;return -2;}
  while( !(TxtFile1.eof()) ){
  
    TxtFile1.getline(Buffer,5000);
    if (!strstr(Buffer,"#") && !(strspn(Buffer," ") == strlen(Buffer)))
       {
             if (fromSL1) sscanf(Buffer,"%d %d %d",&SLSMn,&TTnum,&SLshift);
             else sscanf(Buffer," %d %d",&TTnum,&SLshift);
              if(TTnum < 1 || TTnum >68){cout<<"Wrong TT in txt file: "<<TTnum<<endl;continue;}
	         HowManyShifts++;
              Shift1[TTnum]=-SLshift;//
    }
  }//end of file
  TxtFile1.close();
  cout<<"Found "<<HowManyShifts<<" tt timing while reading the file "<<argv[2]<<" (should be up to 68)"<<endl;
  
  HowManyShifts = 0;
  
  ifstream TxtFile2(argv[1]);
  if( !(TxtFile2.is_open()) ){cout<<"Error: file"<<argv[1]<<" not found!!"<<endl;return -2;}
  while( !(TxtFile2.eof()) ){
  
    TxtFile2.getline(Buffer,5000);
    if (!strstr(Buffer,"#") && !(strspn(Buffer," ") == strlen(Buffer)))
       {
             if (fromSL2) sscanf(Buffer,"%d %d %d",&SLSMn,&TTnum,&SLshift);
             else sscanf(Buffer," %d %d",&TTnum,&SLshift);
              if(TTnum < 1 || TTnum >68){cout<<"Wrong TT in txt file: "<<TTnum<<endl;continue;}
	         HowManyShifts++;
              Shift2[TTnum]=-SLshift;//
	      if (noCompare) Shift2[TTnum]=0;
      }
  }//end of file
  TxtFile2.close();
  cout<<"Found "<<HowManyShifts<<" tt timing while reading the file "<<argv[1]<<" (should be up to 68)"<<endl;
  /*
  if (fromSL){
     HowManyShifts = 0;
     ifstream AveFile1(argv[5]);
     if( !(AveFile1.is_open()) ){cout<<"Error: file"<<argv[5]<<" not found!!"<<endl;return -2;}
     while( !(AveFile1.eof()) ){
  
       AveFile1.getline(Buffer,5000);
       if (!strstr(Buffer,"#") && !(strspn(Buffer," ") == strlen(Buffer)))
          {
	      sscanf(Buffer," %d %f",&SMnum,&rel_shift);
	      //cout<<"TT: "<<TTnum<<endl;
	      if(SMnum < 1 || SMnum >54){cout<<"Wrong SM in txt file: "<<SMnum<<endl;continue;}
	      if(rel_shift <= 4.5){cout<<" Average < 4.5 in SM: "<<SMnum<<" skipped"<<endl;continue;}
	      if(rel_shift > 8. ){ cout<<" Average > 8 in SM: "<<SMnum<<" skipped"<<endl;continue;}
	      //std::cout <<"SM1 "<< SMnum <<  " value " << rel_shift << std::endl;
	      HowManyShifts++;
	      float move = (0. - rel_shift)*25.;
	      SMave1[SMnum] = move;
	      //if( fabs(move)> 10.){cout<<"!! Large shift ( "<<move<<" ns) required for SM: "<<SMnum<<endl; }   
       }
     }//end of file
     AveFile1.close();
     cout<<"Found "<<HowManyShifts<<" SM's while reading the file "<<argv[5]<<" (should be up to 54)"<<endl;
  
     HowManyShifts = 0;
     ifstream AveFile2(argv[6]);
     if( !(AveFile2.is_open()) ){cout<<"Error: file"<<argv[6]<<" not found!!"<<endl;return -2;}
     while( !(AveFile2.eof()) ){
  
       AveFile2.getline(Buffer,5000);
       if (!strstr(Buffer,"#") && !(strspn(Buffer," ") == strlen(Buffer)))
          {
	   sscanf(Buffer,"%d %f",&SMnum,&rel_shift);
	   //cout<<"TT: "<<TTnum<<endl;
	   if(SMnum < 1 || SMnum >54){cout<<"Wrong SM in txt file: "<<SMnum<<endl;continue;}
	   if(rel_shift <= 4.5){cout<<" Average < 4.5 in SM: "<<SMnum<<" skipped"<<endl;continue;}
	   if(rel_shift > 8. ){ cout<<" Average > 8 in SM: "<<SMnum<<" skipped"<<endl;continue;}
	   //std::cout <<"SM1 "<< SMnum <<  " value " << rel_shift << std::endl;
	   HowManyShifts++;
	   float move = (0. - rel_shift)*25.;
	   SMave2[SMnum] = move;
	   //if( fabs(move)> 10.){cout<<"!! Large shift ( "<<move<<" ns) required for TT: "<<SMnum<<endl; }   
       }
     }//end of file
     AveFile2.close();
     cout<<"Found "<<HowManyShifts<<" SM's while reading the file "<<argv[6]<<" (should be up to 54)"<<endl;
   }

  //Perform the average over the SM's
  double average1 = 0.0;
  double number1 = 0.0;
  double average2 = 0.0;
  double number2 = 0.0;
  
  for ( int i = 1; i<55; i++){
     if (SMave1[i] > -1000.0 ) {number1++; average1 += SMave1[i];} 
     if (SMave2[i] > -1000.0 ) {number2++; average2 += SMave2[i];}
  }
  
  if (number1 > 0.0) average1 /= number1;
  if (number2 > 0.0) average2 /= number2;
  std::cout << " Average of 1 " << average1 << std::endl;
  std::cout << " Average of 2 " << average2 << std::endl;
*/
  TH1F *diffHist = new TH1F("DifferenceHist","DifferenceHist",500,-25.,25.);
  
  // calculate the differences between the two files
  for(int i=1;i<69;i++){
    if(Shift1[i]>-1000. && Shift2[i] > -1000.){
            double diff = Shift2[i]-Shift1[i];
            diffHist->Fill(diff);
            if ( diff < -2 || diff > 2 ) std::cout << "NONONO TT " << i << " is more than 5 ns off: " << diff << std::endl;
            }
  }
  
  string rootFileName = argv[3]; 
  rootFileName += ".root";
  
  TFile *myfile = new TFile(rootFileName.c_str(),"RECREATE");
  myfile->cd();
  diffHist->Write();
  myfile->Write();
  myfile->Close();
  
  
  
  
  return 0;
}

