#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>

#include<vector>
#include<string>

using namespace std;
int main(int argc,  char * argv[]){

  if(argc < 5){cout<<" Usage: executable initial_xml_file_name out_file(without suffix) move_first_half move_second_half"<<endl;
  cout<<" the shitfs are expected in ns"<<endl;
  return -4;}
  

  int move1 = atoi(argv[3]);
  int move2 = atoi(argv[4]);


  ifstream FileXML(argv[1]);
  if( !(FileXML.is_open()) ){cout<<"Error: file"<<argv[1]<<" not found!!"<<endl;return -2;}
  char Buffer[5000];
  int TimeOffset[71];
  for(int i=0;i<71;i++){TimeOffset[i]=-100;}
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
	int time_off = -10;
	char st2[200];
	FileXML.getline(Buffer,5000);// line for the delay
	sscanf(Buffer,"%s",st2);
	sscanf(st2,"<TIME_OFFSET>%d</TIME_OFFSET>",&time_off);
	TimeOffset[TT]=time_off;
	if(TimeOffset[TT] >0 ){HowManyTT++;}
	else{cout<<"Error for delays in TT: "<<TT<<" Offsets: "<<TimeOffset[TT] <<endl;}
      }

    }//end of detecting offset of a TT

  }//end of file
  FileXML.close();
  cout<<"Found "<<HowManyTT<<" good TT while reading the file "<<argv[1]<<" (Should be 70)"<<endl;

  // reading the shift file

  int FinalOffsets[70];
  for(int i=0;i<71;i++){FinalOffsets[i]=-100;}
  int TTnum;
  for(TTnum =1; TTnum<69; TTnum++){
    int move = 0;
    if(TTnum> 4 && (TTnum - 1)%4 <2){move = move2;}
    else {move = move1;}
    FinalOffsets[TTnum]= TimeOffset[TTnum]-move;
    if(FinalOffsets[TTnum] < 1 ){
      cout<<"Offsets in TT: "<<TTnum<<" is <1. It will be set to 0"<<endl;
      FinalOffsets[TTnum]=0;
    }
    //if( fabs(move)> 1.0){cout<<"!! Large shift ( "<<move<<" samples) required for TT: "<<TTnum<<endl; }
  }
  

  string xmlFileName = argv[2]; xmlFileName += ".xml";
  ofstream xml_outfile;
  xml_outfile.open(xmlFileName.c_str(),ios::out);
  
  xml_outfile<<"<delayOffsets>"<<endl;
  xml_outfile<<" <DELAY_OFFSET_RELEASE VERSION_ID = \"SM"<<SMn<<"_VER1\"> \n";
  xml_outfile<<"      <RELEASE_ID>RELEASE_1</RELEASE_ID>\n";
  xml_outfile<<"             <SUPERMODULE>" <<SMn<< "</SUPERMODULE>\n";
  xml_outfile<<"     <TIME_STAMP> 270705 </TIME_STAMP>"<<endl;
  //add the time for the mem at the beginning
  xml_outfile<<"   <DELAY_OFFSET>\n";
  xml_outfile<<"             <SUPERMODULE>"<<SMn <<"</SUPERMODULE>\n";
  xml_outfile<<"             <TRIGGERTOWER>69</TRIGGERTOWER>\n";
  xml_outfile<<"             <TIME_OFFSET>48</TIME_OFFSET>\n";
  xml_outfile<<"    </DELAY_OFFSET>"<<endl;
  xml_outfile<<"   <DELAY_OFFSET>\n";
  xml_outfile<<"             <SUPERMODULE>"<<SMn <<"</SUPERMODULE>\n";
  xml_outfile<<"             <TRIGGERTOWER>70</TRIGGERTOWER>\n";
  xml_outfile<<"             <TIME_OFFSET>48</TIME_OFFSET>\n";
  xml_outfile<<"    </DELAY_OFFSET>"<<endl;

  for(int i=1;i<69;i++){
    xml_outfile<<"   <DELAY_OFFSET>\n";
    xml_outfile<<"             <SUPERMODULE>"<<SMn <<"</SUPERMODULE>\n";
    xml_outfile<<"             <TRIGGERTOWER>"<< i <<"</TRIGGERTOWER>\n";
    xml_outfile<<"             <TIME_OFFSET>"<< FinalOffsets[i] <<"</TIME_OFFSET>\n";
    xml_outfile<<"    </DELAY_OFFSET>"<<endl;
  }
  xml_outfile<<" </DELAY_OFFSET_RELEASE>"<<endl;
  xml_outfile<<"</delayOffsets>"<<endl;
  xml_outfile.close();

  cout<<"The timing has been shifted of "<<move1<<" ns in the first half and of "<<move2<<" ns in the second half"<<endl;
  
  //cout<<int(-1.4)<<"  "<<int(-1.7)<<endl;

  return 0;
}

