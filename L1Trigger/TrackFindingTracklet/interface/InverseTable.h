#ifndef INVERSETABLE_H
#define INVERSETABLE_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class InverseTable{

public:

  InverseTable() {
   
  }

  ~InverseTable() {

  }


  void initR(int nbits,
	     int offset,
	     int invbits,
	     bool pos
	    ) {

    nbits_=nbits;
    entries_=(1<<nbits);
    
    for(int i=0;i<entries_;i++) {
      int idrrel=i;
      if(!pos){
	if (i>((1<<(nbits-1))-1)) {
	  idrrel=i-(1<<nbits);
	}
      }
      int idr=offset+idrrel;
      table_.push_back(round_int((1<<invbits)/(1.0*idr)));
    }

  }
  void initT(int nbits,
	     int offset,
	     int invbits,
	     bool pos
	    ) {

    nbits_=nbits;
    entries_=(1<<nbits);
    
    for(int i=0;i<entries_;i++) {
      int itrel=i;
      if(!pos)
	itrel = i-entries_;
      int it=itrel<<offset;
      int invt = round_int((1<<invbits)/(1.0*it));
      table_.push_back(invt);
    }

  }
	    

  void write(std::string fname) {

    ofstream out(fname.c_str());

    for (int i=0;i<entries_;i++){
      //cout << "i "<<i<<endl;
      unsigned int tt = table_[i];
      out <<std::hex<<tt<<endl;
    }
    out.close();
  }


  int lookup(int drrel) const {
    assert(drrel>=0);
    //cout << "nbits = "<<nbits_<<endl;
    //cout << "drrel = "<<drrel<<endl;
    assert(drrel<(1<<nbits_));
    return table_[drrel];
  }

  int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
  }


private:

  int nbits_;
  int entries_;
  vector<int> table_;
  

};



#endif



