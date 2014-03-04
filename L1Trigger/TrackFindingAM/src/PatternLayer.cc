#include <sstream>
#include <iomanip>
#include "../interface/PatternLayer.h"

map<string, int> PatternLayer::GRAY_POSITIONS = PatternLayer::CreateMap();
map<string, vector<string> > PatternLayer::positions_cache;

map<string, int> PatternLayer::CreateMap(){
  map<string, int> p;
  p[""]=0;
  p["0"]=0;
  p["1"]=1;
  p["00"]=0;
  p["01"]=1;
  p["10"]=3;
  p["11"]=2;
  p["000"]=0;
  p["001"]=1;
  p["011"]=2;
  p["010"]=3;
  p["110"]=4;
  p["111"]=5;
  p["101"]=6;
  p["100"]=7;
  return p;
}

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

void PatternLayer::getPositionsFromDC(vector<char> dc, vector<string>& positions){
  int index = -1;
  bool containsX = false;

  for(int i=0;i<DC_BITS;i++){
    if(dc[i]==2){
      containsX=true;
      index=i;
      break;
    }
  }
  
  if(!containsX){ // no don't care bit : nothing to do
    ostringstream oss;
    for(int i=0;i<DC_BITS;i++){
      if(dc[i]!=3)
	oss<<(int)dc[i];
    }
    positions.push_back(oss.str());
    return;
  }
  else{
    vector<string> v1;
    vector<string> v2;
    vector<char> newDC1 = dc;
    vector<char> newDC2 = dc;

    newDC1[index]=0;
    newDC2[index]=1;
    getPositionsFromDC(newDC1, v1);
    getPositionsFromDC(newDC2, v2);
    for(unsigned int i=0;i<v1.size();i++){
      positions.push_back(v1[i]);
    }
    for(unsigned int i=0;i<v2.size();i++){
      positions.push_back(v2[i]);
    }
  }
}

vector<string> PatternLayer::getPositionsFromDC(){

  vector<char> v;
  for(int i=0;i<DC_BITS;i++){
    v.push_back(dc_bits[i]);
  }

  string ref(v.begin(),v.end());

  //check if we already have the result
  map<string, vector<string> >::iterator it = positions_cache.find(ref);
  if(it!=positions_cache.end()){ // already computed
    return it->second;
  }

  //not yet computed
  vector<string> n_vec;
  getPositionsFromDC(v,n_vec);
  //keep the result for later usage
  positions_cache[ref]=n_vec;
  return n_vec;
}
