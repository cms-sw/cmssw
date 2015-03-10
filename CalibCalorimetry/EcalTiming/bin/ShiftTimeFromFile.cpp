#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>

#include<vector>
#include<string.h>

using namespace std;
int main(int argc,  char * argv[]){

  if(argc < 4){cout<<" Usage: executable initial_xml_file_name shift_file.txt out_file(without suffix)"<<endl; 
  return -4;}
  cout<<"Please make sure that the peak time is expected in SAMPLE units"<<endl;
  cout<<"This program will use the absolute timing"<<endl;
  ifstream FileXML(argv[1]);
  if( !(FileXML.is_open()) ){cout<<"Error: file"<<argv[1]<<" not found!!"<<endl;return -2;}
  char Buffer[5000];
  float SampleDelay[71],FineDelay[71];
  for(int i=0;i<71;i++){SampleDelay[i]=-100;FineDelay[i]=-100;}
  int SMn =0;
  bool find_SMnum = true;
  int HowManyTT =0;
  while( !(FileXML.eof()) ){
    FileXML.getline(Buffer,5000);
    //    if (!strstr(Buffer,"#") && !(strspn(Buffer," ") == strlen(Buffer)))
    string initial(Buffer);
    if( find_SMnum && initial.find("<SUPERMODULE>") != string::npos){
      char stSM[100];
      sscanf(Buffer,"%s",stSM);
      sscanf(stSM,"<SUPERMODULE>%d</SUPERMODULE>",&SMn);
      find_SMnum = false;
    }

    if( initial.find("<DELAY_OFFSET>") != string::npos ){
      FileXML.getline(Buffer,5000);// get the line with SM id
      FileXML.getline(Buffer,5000);// get the line with TT id
      char st1[200];
      int TT = -1;
      sscanf(Buffer,"%s",st1);
      sscanf(st1,"<TRIGGERTOWER>%d</TRIGGERTOWER>",&TT);
      //cout<<"TT: "<<TT<<endl;
      //cout<<"Buffer: "<<Buffer<<"  []TT: "<<TT<<endl;
      if(TT< 1 || TT >70){cout<<"Wrong TT: "<<TT<<endl;}
      else{
	int samp_del = -10, fine_del = -10;
	char st2[200], st3[200];
	FileXML.getline(Buffer,5000);// line for sample delay
	sscanf(Buffer,"%s",st2);
	FileXML.getline(Buffer,5000);// line for fine delay
	sscanf(Buffer,"%s",st3);

	sscanf(st2,"<SAMPLE_DELAY>%d</SAMPLE_DELAY>",&samp_del);
	sscanf(st3,"<FINE_DELAY>%d</FINE_DELAY>",&fine_del);
	SampleDelay[TT] = samp_del;
	FineDelay[TT] = fine_del;
	//cout<<"sample delay: "<<samp_del<<"  fine delay: "<<fine_del<<endl;
	if(SampleDelay[TT] >0 && FineDelay[TT] >=0 ){HowManyTT++;}
	else{cout<<"Error for delays in TT: "<<TT<<" SampleDelay: "<<SampleDelay[TT] <<" FineDealy: "<< FineDelay[TT]<<endl;}
      }

    }//end of detecting offset of a TT

  }//end of file
  FileXML.close();
  cout<<"Found "<<HowManyTT<<" good TT while reading the file "<<argv[1]<<" (Should be 70)"<<endl;

  // reading the shift file
  ifstream TxtFile(argv[2]);
  if( !(TxtFile.is_open()) ){cout<<"Error: file"<<argv[2]<<" not found!!"<<endl;return -2;}
  float SampleShift[71],FineShift[71];
  for(int i=0;i<71;i++){SampleShift[i]=-100;FineShift[i]=-100;}
  int TTnum;
  float moveBy;
  int HowManyShifts = 0;
  while( !(TxtFile.eof()) ){
    TxtFile.getline(Buffer,5000);
    if (!strstr(Buffer,"#") && !(strspn(Buffer," ") == strlen(Buffer)))
      {
	sscanf(Buffer,"%d %f",&TTnum,&moveBy);
	//cout<<"TT: "<<TTnum<<endl;
	if(TTnum < 1 || TTnum >68){cout<<"Wrong TT in txt file: "<<TTnum<<endl;continue;}
	//if(shift <= 0){cout<<" shift <= 0! in TT: "<<TTnum<<" skipped"<<endl;continue;}
	//if(rms < 0){ cout<<" rms < 0! in TT: "<<TTnum<<" skipped"<<endl;continue;}
	HowManyShifts++;
	float move = moveBy;
	if( move >= 0 ){SampleShift[TTnum] = int(move); FineShift[TTnum] = round((move - int(move))*25);}
	else{SampleShift[TTnum] = int(move) - 1; FineShift[TTnum]= round((move - int(move)+1)*25);}
	if( fabs(move)> 1.0){cout<<"!! Large shift ( "<<move<<" samples) required for TT: "<<TTnum<<endl; }
      }
  }//end of file
  TxtFile.close();
  cout<<"Found "<<HowManyShifts<<" tt timing while reading the file "<<argv[2]<<" (should be 68)"<<endl;
  
  // calculate the new values for the offset file.
  float FinalSampleDelay[71],FinalFineDelay[71];
   for(int i=0;i<71;i++){FinalSampleDelay[i]=41;FinalFineDelay[i]=0;}
  for (int tt=1;tt<69;tt++){
    if(SampleDelay[tt]<0 || FineDelay[tt]<0){cout<<"Problems in reading the TT "<<tt<<" from the xml file! Correct it manually in the new xml file!"<<endl; continue;}
    if(SampleShift[tt] == -100 || FineShift[tt] == -100){
      cout<<"Timing correction not found for TT: "<<tt<<", keeping the previous value!"<<endl;
      SampleShift[tt] = 0; FineShift[tt] = 0;
    }
    FinalSampleDelay[tt]=SampleDelay[tt]+SampleShift[tt];
    FinalFineDelay[tt]=FineDelay[tt]+FineShift[tt];
    if( FinalFineDelay[tt] <0 ){ FinalSampleDelay[tt] -= 1; FinalFineDelay[tt]+= 25; }
    else if( FinalFineDelay[tt] > 24 ){ FinalSampleDelay[tt] += 1; FinalFineDelay[tt]-= 24; }
  }
  
  string txtFileName = argv[3]; txtFileName += ".txt";
  ofstream txt_outfile;
  txt_outfile.open(txtFileName.c_str(),ios::out);
  txt_outfile<< "#  Needed shift in terms of samples and fine tuning (ns) for each TT"<<endl;
  txt_outfile<<"#   TT   sample shift  \t  fine shift"<<std::endl;
  for(int i=0;i<71;i++){
    if(SampleShift[i] == -100){continue;}
    txt_outfile <<"   "<<setw(4)<<i<<"  "<<setw(4)<<SampleShift[i]<<" \t  "<<setw(4)<<FineShift[i]<<endl;  }
  txt_outfile.close();

  string xmlFileName = argv[3]; xmlFileName += ".xml";
  ofstream xml_outfile;
  xml_outfile.open(xmlFileName.c_str(),ios::out);
  
  xml_outfile<<"<delayOffsets>"<<endl;
  xml_outfile<<" <DELAY_OFFSET_RELEASE VERSION_ID = \"SM"<<SMn<<"_VER1\"> \n";
  xml_outfile<<"      <RELEASE_ID>RELEASE_1</RELEASE_ID>\n";
  xml_outfile<<"     <SUPERMODULE>" <<SMn<< "</SUPERMODULE>\n";
  xml_outfile<<"     <TIME_STAMP> 270705 </TIME_STAMP>"<<endl;
  //add the time for the mem at the beginning
  xml_outfile<<"   <DELAY_OFFSET>\n";
  xml_outfile<<"            <SUPERMODULE> "<<SMn <<" </SUPERMODULE>\n";
  xml_outfile<<"             <TRIGGERTOWER>69</TRIGGERTOWER>\n";
  xml_outfile<<"             <SAMPLE_DELAY>40</SAMPLE_DELAY>\n";
  xml_outfile<<"             <FINE_DELAY>0</FINE_DELAY>\n";
  xml_outfile<<"    </DELAY_OFFSET>"<<endl;
  xml_outfile<<"   <DELAY_OFFSET>\n";
  xml_outfile<<"            <SUPERMODULE> "<<SMn <<" </SUPERMODULE>\n";
  xml_outfile<<"             <TRIGGERTOWER>70</TRIGGERTOWER>\n";
  xml_outfile<<"             <SAMPLE_DELAY>40</SAMPLE_DELAY>\n";
  xml_outfile<<"             <FINE_DELAY>0</FINE_DELAY>\n";
  xml_outfile<<"    </DELAY_OFFSET>"<<endl;

  for(int i=1;i<69;i++){
    xml_outfile<<"   <DELAY_OFFSET>\n";
    xml_outfile<<"            <SUPERMODULE> "<<SMn <<" </SUPERMODULE>\n";
    xml_outfile<<"             <TRIGGERTOWER>" << i <<"</TRIGGERTOWER>\n";
    xml_outfile<<"             <SAMPLE_DELAY>" << FinalSampleDelay[i] <<"</SAMPLE_DELAY>\n";
    xml_outfile<<"             <FINE_DELAY>" << FinalFineDelay[i] <<"</FINE_DELAY>\n";
    xml_outfile<<"    </DELAY_OFFSET>"<<endl;
  }
  xml_outfile<<" </DELAY_OFFSET_RELEASE>"<<endl;
  xml_outfile<<"</delayOffsets>"<<endl;
  xml_outfile.close();
  
  //cout<<int(-1.4)<<"  "<<int(-1.7)<<endl;

  return 0;
}

