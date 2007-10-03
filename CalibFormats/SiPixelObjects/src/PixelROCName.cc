//
// This class stores the name and related
// hardware mapings for a ROC
//


#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include <string>
#include <iostream>
#include <strstream>
#include <cctype>

#include <assert.h>


using namespace pos;

PixelROCName::PixelROCName():
    id_(0)
{}

PixelROCName::PixelROCName(std::string rocname)
{

    parsename(rocname);

}

void PixelROCName::setIdFPix(char np, char LR,int disk,
			 int blade, int panel, int plaquet, int roc){

    id_=0;

    //std::cout<<"PixelROCName::setId subdet:"<<subdet<<std::endl; 
    //std::cout<<"PixelROCName::setId np:"<<np<<std::endl; 
    //std::cout<<"PixelROCName::setId LR:"<<LR<<std::endl; 
    //std::cout<<"PixelROCName::setId disk:"<<disk<<std::endl; 

    
    assert(roc>=0&&roc<10);

    if (np=='p') id_=(id_|0x40000000);
    //std::cout<<"PixelROCName::setId 2 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    if (LR=='I') id_=(id_|0x20000000);
    //std::cout<<"PixelROCName::setId 3 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|(disk<<12));
    //std::cout<<"PixelROCName::setId 4 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|(blade<<7));
    //std::cout<<"PixelROCName::setId 5 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|((panel-1)<<6));
    //std::cout<<"PixelROCName::setId 6 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|((plaquet-1)<<4));
    //std::cout<<"PixelROCName::setId 7 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|roc);

    //std::cout<<"PixelROCName::setIdFPix final id_="<<std::hex<<id_<<std::dec<<std::endl; 

}


void PixelROCName::setIdBPix(char np, char LR,int sec,
			     int layer, int ladder, char HF,
			     int module, int roc){

    id_=0;

    //std::cout<<"PixelROCName::setIdBPix ladder:"<<ladder<<std::endl; 
    //std::cout<<"PixelROCName::setId np:"<<np<<std::endl; 
    //std::cout<<"PixelROCName::setId LR:"<<LR<<std::endl; 
    //std::cout<<"PixelROCName::setId disk:"<<disk<<std::endl; 

    
    assert(roc>=0&&roc<16);

    id_=0x80000000;

    if (np=='p') id_=(id_|0x40000000);
    //std::cout<<"PixelROCName::setId 2 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    if (LR=='I') id_=(id_|0x20000000);
    //std::cout<<"PixelROCName::setId 3 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|((sec-1)<<14));
    //std::cout<<"PixelROCName::setId 4 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    if (HF=='F') id_=(id_|0x00000800);

    id_=(id_|(layer<<12));
    //std::cout<<"PixelROCName::setId 5 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|(ladder<<6));
    //std::cout<<"PixelROCName::setId 6 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|((module-1)<<4));
    //std::cout<<"PixelROCName::setId 7 id_="<<std::hex<<id_<<std::dec<<std::endl; 
    id_=(id_|roc);

    //std::cout<<"PixelROCName::setIdBPix final id_="<<std::hex<<id_<<std::dec<<std::endl; 

}



void PixelROCName::parsename(std::string name){

   //
    // The name should be on the format
    //
    // FPix_BpR_D1_BLD1_PNL1_PLQ1_ROC1
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
	assert(name[21+offset]=='_');
	assert(name[22+offset]=='P');
	assert(name[23+offset]=='L');
	assert(name[24+offset]=='Q');
	assert(std::isdigit(name[25+offset]));
	digit[0]=name[25+offset];
	int plq=atoi(digit);
	assert(name[26+offset]=='_');
	assert(name[27+offset]=='R');
	assert(name[28+offset]=='O');
	assert(name[29+offset]=='C');
	assert(std::isdigit(name[30+offset]));
	digit[0]=name[30+offset];
	int roc=atoi(digit);
	if (name.size()==32+offset){
	    digit[0]=name[31+offset];
	    roc=roc*10+atoi(digit);
	}
    
	setIdFPix(np,LR,disk,bld,pnl,plq,roc);
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
	assert(name[29+offset]=='_');
	assert(name[30+offset]=='R');
	assert(name[31+offset]=='O');
	assert(name[32+offset]=='C');
	assert(std::isdigit(name[33+offset]));
	digit[0]=name[33+offset];
	int roc=atoi(digit);
	if (name.size()==35+offset){
	    digit[0]=name[34+offset];
	    roc=roc*10+atoi(digit);
	}
    
	setIdBPix(np,LR,sec,layer,ladder,HF,module,roc);
    }

}

PixelROCName::PixelROCName(std::ifstream& s){

    std::string tmp;

    s >> tmp;

    parsename(tmp);

}
    

std::string PixelROCName::rocname() const{

    std::string s;

    std::strstream s1;

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
	s1<<"_PLQ";
	s1<<plaquet();
	s1<<"_ROC";
	s1<<roc()<<(char)(0);

	assert(roc()>=0&&roc()<=10);
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
	s1<<"_ROC";
	s1<<roc()<<(char)(0);
	
	assert(roc()>=0&&roc()<=15);
    }

    s=s1.str();
   
    return s;

} 



std::ostream& pos::operator<<(std::ostream& s, const PixelROCName& pixelroc){


    // FPix_BpR_D1_BLD1_PNL1_PLQ1_ROC1

    s<<pixelroc.rocname();

    return s;
}


const PixelROCName& PixelROCName::operator=(const PixelROCName& aROC){
    
    id_=aROC.id_;

    return *this;

}
