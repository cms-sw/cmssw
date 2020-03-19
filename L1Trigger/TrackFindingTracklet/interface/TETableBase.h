#ifndef TETABLEBASE_H
#define TETABLEBASE_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class TETableBase{

public:

  TETableBase() {
   
  }

  ~TETableBase() {

  }


  void writeVMTable(std::string name, bool positive=true) {

    ofstream out;
    out.open(name.c_str());
    out << "{"<<endl;
    for(unsigned int i=0;i<table_.size();i++){
      if (i!=0) out<<","<<endl;

      assert(nbits_>0);

      int itable = table_[i];
      if (positive) {
	if (table_[i] < 0) itable = (1<<nbits_)-1; 
      }
      
      //FPGAWord tmp;  
      //tmp.set(itable, nbits_,positive,__LINE__,__FILE__);      
      //out << tmp.str() << endl;

      out << itable;

    }
    out <<endl<<"};"<<endl;
    out.close();
  }


protected:

  vector<int> table_;
  int nbits_;
  
};



#endif



