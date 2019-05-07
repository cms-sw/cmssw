//Base class for processing modules
#ifndef FPGAPROCESSBASE_H
#define FPGAPROCESSBASE_H

using namespace std;

class FPGAProcessBase{

public:

  FPGAProcessBase(string name, unsigned int iSector){
    name_=name;
    iSector_=iSector;
  }

  virtual ~FPGAProcessBase() { } 

  virtual void addOutput(FPGAMemoryBase* memory,string output)=0;

  virtual void addInput(FPGAMemoryBase* memory,string input)=0;

  string getName() const {return name_;}

  unsigned int nbits(unsigned int power) {

    if (power==2) return 1;
    if (power==4) return 2;
    if (power==8) return 3;
    if (power==16) return 4;
    if (power==32) return 5;

    cout << "nbits: power = "<<power<<endl;
    assert(0);

    return -1;
    
  }


protected:

  string name_;
  unsigned int iSector_;


};

#endif
