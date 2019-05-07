//Base class for processing modules
#ifndef FPGAMEMORYBASE_H
#define FPGAMEMORYBASE_H

using namespace std;

class FPGAMemoryBase{

public:

  FPGAMemoryBase(string name, unsigned int iSector){
    name_=name;
    iSector_=iSector;
    bx_=0;
    event_=0;
  }

  virtual ~FPGAMemoryBase(){}

  string getName() const {return name_;}
  string getLastPartOfName() const {return name_.substr(name_.find_last_of('_')+1);}

  virtual void clean()=0;

  static string hexFormat(string binary){

    string tmp="";
  
    unsigned int value=0;
    unsigned int bits=0;
    
    for(unsigned int i=0;i<binary.size();i++) {
      unsigned int slot=binary.size()-i-1;
      if (!(binary[slot]=='0'||binary[slot]=='1')) continue;
      value+=((binary[slot]-'0')<<bits);
      bits++;
      if (bits==4||i==binary.size()-1) {
	assert(value<16);
	if (value==0) tmp+="0";
	if (value==1) tmp+="1";
	if (value==2) tmp+="2";
	if (value==3) tmp+="3";
	if (value==4) tmp+="4";
	if (value==5) tmp+="5";
	if (value==6) tmp+="6";
	if (value==7) tmp+="7";
	if (value==8) tmp+="8";
	if (value==9) tmp+="9";
	if (value==10) tmp+="A";
	if (value==11) tmp+="B";
	if (value==12) tmp+="C";
	if (value==13) tmp+="D";
	if (value==14) tmp+="E";
	if (value==15) tmp+="F";
	bits=0;
	value=0;
      }
    }

    string hexstring="0x";
    for (unsigned int i=0;i<tmp.size();i++){
      hexstring+=tmp[tmp.size()-i-1];
    }
    
    return hexstring;
     
}

protected:

  string name_;
  unsigned int iSector_;

  ofstream out_;
  int bx_;
  int event_;


};

#endif
