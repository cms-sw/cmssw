//Base class for processing modules
#ifndef MEMORYBASE_H
#define MEMORYBASE_H

using namespace std;

class MemoryBase{

public:

  MemoryBase(string name, unsigned int iSector){
    name_=name;
    iSector_=iSector;
    bx_=0;
    event_=0;
    extended_=hourglassExtended;
    nHelixPar_=nHelixPar;
  }

  virtual ~MemoryBase(){}

  string getName() const {return name_;}
  string getLastPartOfName() const {return name_.substr(name_.find_last_of('_')+1);}

  virtual void clean()=0;

  //Converts string in binary to hex (used in writing out memory content)
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

  //method sets the layer and disk based on the name. pos is the position in the
  //memory name where the layer or disk is specified
  void initLayerDisk(unsigned int pos, int& layer, int& disk){

    string subname=name_.substr(pos,2);
    layer=0;
    disk=0;

    if (subname=="L1") layer=1;
    if (subname=="L2") layer=2;
    if (subname=="L3") layer=3;
    if (subname=="L4") layer=4;
    if (subname=="L5") layer=5;
    if (subname=="L6") layer=6;
    if (subname=="D1") disk=1;
    if (subname=="D2") disk=2;
    if (subname=="D3") disk=3;
    if (subname=="D4") disk=4;
    if (subname=="D5") disk=5;
    if (layer==0&&disk==0) {
      cout << "Memoryname = "<<name_<<" subname = "<<subname
	   <<" layer "<<layer<<" disk "<<disk<<endl;
    }
    assert((layer!=0)||(disk!=0));
  }

  // Based on memory name check if this memory is used for special seeding:
  // overlap is layer-disk seeding
  // extra is the L2L3 seeding
  // extended is the seeding for displaced tracks
  void initSpecialSeeding(unsigned int pos, bool& overlap, bool& extra, bool& extended) {
    
    overlap=false;
    extra=false;
    extended=false;

    string subname=name_.substr(pos,1);

    static const std::set<std::string> overlapset = {"X","Y","W","Q","R","S","T","Z","x","y","w","q","r","s","t","z"};
    overlap=overlapset.find(subname)!=overlapset.end();
    
    static const std::set<std::string> extraset = {"I","J","K","L"};
    extra=extraset.find(subname)!=extraset.end();

    static const std::set<std::string> extendedset = {"a","b","c","d","e","f","g","h","x","y","z","w","q","r","s","t"};
    extended=extendedset.find(subname)!=extendedset.end();
    
  }

  void openFile(bool first, std::string filebase){
    
    std::string fname=filebase;
    fname+=getName();

    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_ = 0;
      event_ = 1;
      out_.open(fname.c_str());
    } else {
      out_.open(fname.c_str(),std::ofstream::app);
    }
      
    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    
    bx_++;
    event_++;
    if (bx_>7) bx_=0;
  }

  
protected:

  string name_;
  unsigned int iSector_;
  bool extended_;
  unsigned int nHelixPar_;
  
  ofstream out_;
  int bx_;
  int event_;


};

#endif
