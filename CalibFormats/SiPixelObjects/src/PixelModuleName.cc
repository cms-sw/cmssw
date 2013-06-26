//
// This class stores the name and related
// hardware mapings for a module
//


#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cctype>
#include <cstdlib>

using namespace std;
using namespace pos;


PixelModuleName::PixelModuleName():
    id_(0)
{}


PixelModuleName::PixelModuleName(PixelROCName roc)
{

  unsigned int id=roc.id();
  unsigned int idtmp=(id&0x1FFFFFFF)>>4;
  if ((id&0x80000000)==0) idtmp=(idtmp&0xFFFFFFFC);
  
  id_=idtmp|(id&0xE0000000);

}


PixelModuleName::PixelModuleName(string modulename)
{

    parsename(modulename);

}

void PixelModuleName::setIdFPix(char np, char LR,int disk,
			 int blade, int panel){

    std::string mthn = "[PixelModuleName::setIdFPix()]\t\t\t    " ;
    id_=0;

    //cout<< __LINE__ << "]\t" << mthn << "subdet: " << subdet <<endl; 
    //cout<< __LINE__ << "]\t" << mthn << "np    : " << np     <<endl; 
    //cout<< __LINE__ << "]\t" << mthn << "LR    : " << LR     <<endl; 
    //cout<< __LINE__ << "]\t" << mthn << "disk  : " << disk   <<endl; 

    
    if (np=='p') id_=(id_|0x40000000);
    //cout<< __LINE__ << "]\t" << mthn <<"2 id_=" << hex << id_ << dec << endl;
    if (LR=='I') id_=(id_|0x20000000);
    //cout<< __LINE__ << "]\t" << mthn <<"3 id_=" << hex << id_ << dec << endl;
    id_=(id_|(disk<<8));
    //cout<< __LINE__ << "]\t" << mthn <<"4 id_=" << hex << id_ << dec << endl;
    id_=(id_|(blade<<3));
    //cout<< __LINE__ << "]\t" << mthn <<"5 id_=" << hex << id_ << dec << endl;
    id_=(id_|((panel-1)<<2));
    //cout<< __LINE__ << "]\t" << mthn <<"6 id_=" << hex << id_ << dec << endl;

}


void PixelModuleName::setIdBPix(char np, char LR,int sec,
			     int layer, int ladder, char HF,
			     int module){

    std::string mthn = "[PixelModuleName::setIdBPix()]\t\t\t    " ;
    id_=0;

    //cout<< __LINE__ << "]\t" << mthn << "BPix ladder: " << ladder << endl;
    //cout<< __LINE__ << "]\t" << mthn << "np         : " << np     << endl;
    //cout<< __LINE__ << "]\t" << mthn << "LR         : " << LR     << endl;
    //cout<< __LINE__ << "]\t" << mthn << "disk       : " << disk   << endl;

    
    id_=0x80000000;

    if (np=='p') id_=(id_|0x40000000);
    //cout<< __LINE__ << "]\t" << mthn <<"2 id_=" << hex << id_ << dec << endl;
    if (LR=='I') id_=(id_|0x20000000);
    //cout<< __LINE__ << "]\t" << mthn <<"3 id_=" << hex << id_ << dec << endl;
    id_=(id_|((sec-1)<<10));
    //cout<< __LINE__ << "]\t" << mthn <<"4 id_=" << hex << id_ << dec << endl;
    if (HF=='F') id_=(id_|0x00000080);

    id_=(id_|(layer<<8));
    //cout<< __LINE__ << "]\t" << mthn <<"5 id_=" << hex << id_ << dec << endl;
    id_=(id_|(ladder<<2));
    //cout<< __LINE__ << "]\t" << mthn <<"6 id_=" << hex << id_ << dec << endl;
    id_=(id_|((module-1)));
    //cout<< __LINE__ << "]\t" << mthn <<"7 id_=" << hex << id_ << dec << endl;

}

void PixelModuleName::check(bool check, const string& name){

  std::string mthn = "[PixelModuleName::check()]\t\t\t    " ;
  if (check) return;

  cout << __LINE__ << "]\t" << mthn << "ERROR tried to parse string: '" << name ;
  cout << "' as a module name. Will terminate." << endl;

  ::abort();

}

void PixelModuleName::parsename(string name){

   //
    // The name should be on the format
    //
    // FPix_BpR_D1_BLD1_PNL1
    //

    //cout << "ROC name:"<<name<<endl;

    check(name[0]=='F'||name[0]=='B',name);

    if (name[0]=='F'){
	check(name[0]=='F',name);
	check(name[1]=='P',name);
	check(name[2]=='i',name);
	check(name[3]=='x',name);
	check(name[4]=='_',name);
	check(name[5]=='B',name);
	check((name[6]=='m')||(name[6]=='p'),name);
	char np=name[6];
	check((name[7]=='I')||(name[7]=='O'),name);
	char LR=name[7];
	check(name[8]=='_',name);
	check(name[9]=='D',name);
	char digit[2]={0,0};
	digit[0]=name[10];
	int disk=atoi(digit);
	check(name[11]=='_',name);
	check(name[12]=='B',name);
	check(name[13]=='L',name);
	check(name[14]=='D',name);
	check(isdigit(name[15]),name);
	digit[0]=name[15];
	int bld=atoi(digit);
	unsigned int offset=0;
	if (isdigit(name[16])){
	    digit[0]=name[16];
  	    bld=10*bld+atoi(digit);
	    offset++;
	}

	check(name[16+offset]=='_',name);
	check(name[17+offset]=='P',name);
	check(name[18+offset]=='N',name);
	check(name[19+offset]=='L',name);
	check(isdigit(name[20+offset]),name);
	digit[0]=name[20+offset];
	int pnl=atoi(digit);
    
	setIdFPix(np,LR,disk,bld,pnl);
    }
    else{
	check(name[0]=='B',name);
	check(name[1]=='P',name);
	check(name[2]=='i',name);
	check(name[3]=='x',name);
	check(name[4]=='_',name);
	check(name[5]=='B',name);
	check((name[6]=='m')||(name[6]=='p'),name);
	char np=name[6];
	check((name[7]=='I')||(name[7]=='O'),name);
	char LR=name[7];
	check(name[8]=='_',name);
	check(name[9]=='S',name);
	check(name[10]=='E',name);
	check(name[11]=='C',name);
	char digit[2]={0,0};
	digit[0]=name[12];
	int sec=atoi(digit);
	check(name[13]=='_',name);
	check(name[14]=='L',name);
	check(name[15]=='Y',name);
	check(name[16]=='R',name);
	check(isdigit(name[17]),name);
	digit[0]=name[17];
	int layer=atoi(digit);
	check(name[18]=='_',name);
	check(name[19]=='L',name);
	check(name[20]=='D',name);
	check(name[21]=='R',name);
	check(isdigit(name[22]),name);
	digit[0]=name[22];
	int ladder=atoi(digit);
	unsigned int offset=0;
	if (isdigit(name[23])){
	    offset++;
	    digit[0]=name[22+offset];
	    ladder=10*ladder+atoi(digit);
	}
	check(name[23+offset]=='H'||name[23+offset]=='F',name);
        char HF=name[23+offset];
	check(name[24+offset]=='_',name);
	check(name[25+offset]=='M',name);
	check(name[26+offset]=='O',name);
	check(name[27+offset]=='D',name);
	check(isdigit(name[28+offset]),name);
	digit[0]=name[28+offset];
	int module=atoi(digit);
	setIdBPix(np,LR,sec,layer,ladder,HF,module);
    }

}

PixelModuleName::PixelModuleName(ifstream& s){

    string tmp;

    s >> tmp;

    parsename(tmp);

}
    

string PixelModuleName::modulename() const{

    string s;

    std::ostringstream s1;

    if (detsub()=='F') {
	s1<<"FPix"; 
	s1<<"_B";
	s1<<mp();
	s1<<IO();
	s1<<"_D";
	s1<<disk();
	s1<<"_BLD";
	s1<<blade();
	s1<<"_PNL";
	s1<<panel();

    }
    else{
	s1<<"BPix"; 
	s1<<"_B";
	s1<<mp();
	s1<<IO();
	s1<<"_SEC";
	s1<<sec();
	s1<<"_LYR";
	s1<<layer();
	s1<<"_LDR";
	s1<<ladder();
	s1<<HF();
	s1<<"_MOD";
	s1<<module();
	
    }

    s=s1.str();
   
    return s;

} 



ostream& pos::operator<<(ostream& s, const PixelModuleName& pixelroc){


    // FPix_BpR_D1_BLD1_PNL1_PLQ1_ROC1

    s<<pixelroc.modulename();

    return s;
}


const PixelModuleName& PixelModuleName::operator=(const PixelModuleName& aROC){
    
    id_=aROC.id_;

    return *this;

}
