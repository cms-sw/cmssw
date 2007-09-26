#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

// that is saved: list of (detid, badstrip|flag)
typedef std::pair<short,short> p_channelflag;
typedef std::vector<p_channelflag> v_channelflag;
typedef std::pair<unsigned int,v_channelflag> p_detidchannelflag;
typedef std::vector<p_detidchannelflag> v_detidallbadstrips;

class DefectsFromConstructionDB{
 
 private:
  const char  *inputfile;
  v_detidallbadstrips v_allbadstrips;
  
  
 public:
  
  

  DefectsFromConstructionDB(const char *infile){inputfile=infile;};
  DefectsFromConstructionDB(){};

  ~DefectsFromConstructionDB(){};

 
  v_detidallbadstrips GetBadStripsFromConstructionDB();
  void print();


};


