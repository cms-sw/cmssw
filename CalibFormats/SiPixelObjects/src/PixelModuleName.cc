//
// This class stores the name and related
// hardware mapings for a module
//


#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include <string>
#include <iostream>
#include <sstream>
#include <cctype>

#include <assert.h>

using namespace pos;


PixelModuleName::PixelModuleName():
    id_(0)
{}

PixelModuleName::PixelModuleName(std::string modulename)
{

    parsename(modulename);

}

void PixelModuleName::setIdFPix(char np, char LR,int disk,
			 int blade, int panel){

    id_=0;

    //std::cout<<"PixelModuleName::setId subdet:"<<subdet<<std::endl; 
    //std::cout<<"PixelModuleName::setId np:"<<np<<std::endl; 
    //std::cout<<"PixelModuleName::setId LR:"<<LR<<std::endl; 
    //std::cout<<"PixelModuleName::setId disk:"<<disk<<std::endl; 

    
    if (np=='p') id_=(id_|0x40000000);
    //std::cout<<"PixelModuleName::setId 2 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    if (LR=='I') id_=(id_|0x20000000);
    //std::cout<<"PixelModuleName::setId 3 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|(disk<<8));
    //std::cout<<"PixelModuleName::setId 4 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|(blade<<3));
    //std::cout<<"PixelModuleName::setId 5 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|((panel-1)<<2));
    //std::cout<<"PixelModuleName::setId 6 id_="<<std::hex<<id_<<std::dec<<std::endl; 

}


void PixelModuleName::setIdBPix(char np, char LR,int sec,
			     int layer, int ladder, char HF,
			     int module){

    id_=0;

    //std::cout<<"PixelModuleName::setIdBPix ladder:"<<ladder<<std::endl; 
    //std::cout<<"PixelModuleName::setId np:"<<np<<std::endl; 
    //std::cout<<"PixelModuleName::setId LR:"<<LR<<std::endl; 
    //std::cout<<"PixelModuleName::setId disk:"<<disk<<std::endl; 

    
    id_=0x80000000;

    if (np=='p') id_=(id_|0x40000000);
    //std::cout<<"PixelModuleName::setId 2 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    if (LR=='I') id_=(id_|0x20000000);
    //std::cout<<"PixelModuleName::setId 3 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|((sec-1)<<10));
    //std::cout<<"PixelModuleName::setId 4 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    if (HF=='F') id_=(id_|0x00000080);

    id_=(id_|(layer<<8));
    //std::cout<<"PixelModuleName::setId 5 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|(ladder<<2));
    //std::cout<<"PixelModuleName::setId 6 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|((module-1)));
    //std::cout<<"PixelModuleName::setId 7 id_="<<std::hex<<id_<<std::dec<<std::endl; 

}



void PixelModuleName::parsename(std::string name){

   //
    // The name should be on the format
    //
    // FPix_BpR_D1_BLD1_PNL1
    //

    //std::cout << "ROC name:"<<name<<std::endl;

    assert(name[0]=='F'||name[0]=='B');

    if (name[0]=='F'){
	assert(name[0]=='F');
	assert(name[1]=='P');
	assert(name[2]=='i');
	assert(name[3]=='x');
	assert(name[4]=='_');
	assert(name[5]=='B');
	assert((name[6]=='m')||(name[6]=='p'));
	char np=name[6];
	assert((name[7]=='I')||(name[7]=='O'));
	char LR=name[7];
	assert(name[8]=='_');
	assert(name[9]=='D');
	char digit[2]={0,0};
	digit[0]=name[10];
	int disk=atoi(digit);
	assert(name[11]=='_');
	assert(name[12]=='B');
	assert(name[13]=='L');
	assert(name[14]=='D');
	assert(std::isdigit(name[15]));
	digit[0]=name[15];
	int bld=atoi(digit);
	unsigned int offset=0;
	if (std::isdigit(name[16])){
	    digit[0]=name[16];
  	    bld=10*bld+atoi(digit);
	    offset++;
	}

	assert(name[16+offset]=='_');
	assert(name[17+offset]=='P');
	assert(name[18+offset]=='N');
	assert(name[19+offset]=='L');
	assert(std::isdigit(name[20+offset]));
	digit[0]=name[20+offset];
	int pnl=atoi(digit);
    
	setIdFPix(np,LR,disk,bld,pnl);
    }
    else{
	assert(name[0]=='B');
	assert(name[1]=='P');
	assert(name[2]=='i');
	assert(name[3]=='x');
	assert(name[4]=='_');
	assert(name[5]=='B');
	assert((name[6]=='m')||(name[6]=='p'));
	char np=name[6];
	assert((name[7]=='I')||(name[7]=='O'));
	char LR=name[7];
	assert(name[8]=='_');
	assert(name[9]=='S');
	assert(name[10]=='E');
	assert(name[11]=='C');
	char digit[2]={0,0};
	digit[0]=name[12];
	int sec=atoi(digit);
	assert(name[13]=='_');
	assert(name[14]=='L');
	assert(name[15]=='Y');
	assert(name[16]=='R');
	assert(std::isdigit(name[17]));
	digit[0]=name[17];
	int layer=atoi(digit);
	assert(name[18]=='_');
	assert(name[19]=='L');
	assert(name[20]=='D');
	assert(name[21]=='R');
	assert(std::isdigit(name[22]));
	digit[0]=name[22];
	int ladder=atoi(digit);
	unsigned int offset=0;
	if (std::isdigit(name[23])){
	    offset++;
	    digit[0]=name[22+offset];
	    ladder=10*ladder+atoi(digit);
	}
	assert(name[23+offset]=='H'||name[23+offset]=='F');
        char HF=name[23+offset];
	assert(name[24+offset]=='_');
	assert(name[25+offset]=='M');
	assert(name[26+offset]=='O');
	assert(name[27+offset]=='D');
	assert(std::isdigit(name[28+offset]));
	digit[0]=name[28+offset];
	int module=atoi(digit);
	setIdBPix(np,LR,sec,layer,ladder,HF,module);
    }

}

PixelModuleName::PixelModuleName(std::ifstream& s){

    std::string tmp;

    s >> tmp;

    parsename(tmp);

}
    

std::string PixelModuleName::modulename() const{

    std::string s;

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
	s1<<panel()<<(char)(0);

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
	s1<<module()<<(char)(0);
	
    }

    s=s1.str();
   
    return s;

} 



std::ostream& pos::operator<<(std::ostream& s, const PixelModuleName& pixelroc){


    // FPix_BpR_D1_BLD1_PNL1_PLQ1_ROC1

    s<<pixelroc.modulename();

    return s;
}


const PixelModuleName& PixelModuleName::operator=(const PixelModuleName& aROC){
    
    id_=aROC.id_;

    return *this;

}
