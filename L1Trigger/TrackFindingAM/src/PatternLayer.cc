#include <sstream>
#include <iomanip>
#include "../interface/PatternLayer.h"

map<string, vector<short> > PatternLayer::positions_cache;

PatternLayer::PatternLayer(){
  bits=0;
  memset(dc_bits,3,DC_BITS*sizeof(char));
}

char PatternLayer::getDC(int index){
  if(index<0 || index>=DC_BITS)
    index=0;
  return dc_bits[index];
}

void PatternLayer::setDC(int index, char val){
  if(val<0 || val>3)
    val=3;
  if(index<0 || index>=DC_BITS)
    index=0;
  dc_bits[index]=val;
}

string PatternLayer::getCode(){
  int c=bits.to_ulong();
  ostringstream oss;
  oss<<std::setfill('0');
  oss<<setw(5)<<c;
  string res=oss.str();
  return res;
}

int PatternLayer::getIntValue() const{
  return bits.to_ulong();
}

void PatternLayer::setIntValue(int v){
  unsigned int val = v;
  bits=val;
}

int PatternLayer::getDCBitsNumber(){
  for(int i=0;i<DC_BITS;i++){
    if(dc_bits[i]==3){
      return i;
    }
  }
  return 3;
}

void PatternLayer::getPositionsFromDC(vector<char> dc, vector<short>& positions){
  while(dc.size()!=0 && dc[0]!=3){
    char val = dc[0];

    if(positions.size()==0){
      if(val==2){
	positions.push_back(0);
	positions.push_back(1);
      }
      else{
	positions.push_back(val);
      }
    }
    else{
      unsigned int fin=positions.size();
      for(unsigned int i=0;i<fin;i++){
	short nv=positions[0];
	if(val==2){
	  positions.push_back(nv*2);
	  positions.push_back((nv*2)+1);
	}
	else{
	  positions.push_back(nv*2+val);
	}
	positions.erase(positions.begin());
      }
    }
    dc.erase(dc.begin());
  }
}

vector<short> PatternLayer::getPositionsFromDC(){

  vector<char> v;
  for(int i=0;i<DC_BITS;i++){
    v.push_back(dc_bits[i]);
  }

  string ref(v.begin(),v.end());

  //check if we already have the result
  map<string, vector<short> >::iterator it = positions_cache.find(ref);
  if(it!=positions_cache.end()){ // already computed
    return it->second;
  }
  
  //not yet computed
  vector<short> n_vec;
  getPositionsFromDC(v,n_vec); 
  //keep the result for later usage
  positions_cache[ref]=n_vec;
  return n_vec;
}
