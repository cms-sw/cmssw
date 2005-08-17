#ifndef __HEADER_H_HCAL_CLASSES_CPP_H_
#define __HEADER_H_HCAL_CLASSES_CPP_H_

#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <iostream>


namespace HcalDb
{

/*
 Eta:   6 bits (1..41)
 Phi:   7 bits (1..72) 
 Depth  3 bits (1..6)
 Z:     2 bits (1..2)
*/

struct CellId
 {
  unsigned id;
  enum {INVALID=0xffffffff};
  
  CellId():id(INVALID){}
  
  CellId(unsigned eta,unsigned phi,unsigned depth,int z);
   
  CellId(const CellId &c):id(c.id){}
  
  const CellId& operator=(const CellId& c)
   {
    id=c.id;
    return *this;
   }
  
  int getZ() const {int tmp=id&0xff; return (tmp-1);} // Z is -1 or 1, but is atored as 0 or 2!
  unsigned getDepth() const  {return ((id>>8)&0xff);}
  unsigned getPhi() const    {return ((id>>16)&0xff);}
  unsigned getEta() const    {return ((id)>>24)&0xff;}
 };
 
 
inline bool operator<(const HcalDb::CellId& k1, const HcalDb::CellId& k2)
 {
  return k1.id<k2.id;
 }
 

std::ostream& operator<<(std::ostream& o,const HcalDb::CellId &c);

 
template<class T> class THcalDbClass
 {
  private:
   std::map<CellId,int> ids;
   
   //Prohibit copying and assignment
   THcalDbClass(const THcalDbClass& rhs){}
   THcalDbClass& operator=(const THcalDbClass& rhs){return *this;}
   
  public:
   std::vector<T> data;
   
   THcalDbClass(){}
   
   void append(const T& v)
    {
     int ind=data.size();
     // XXX - drop duplicate records for now - garbage in DB !!!!!!!!
     if(ids.find(v.id)!=ids.end()) 
      {
       //std::cout<<"ERROR: key already exists: "<<v.id.id.i<<std::endl;
       //throw std::logic_error("Key already exists");
       return;
      }
     ids[v.id]=ind;
     data.push_back(v);
    }
   
    
   const T* getById(const CellId& key) const
    {
     std::map<CellId,int>::const_iterator it=ids.find(key);
     if(it==ids.end()) 
      {
       return static_cast<T*>(0);
      }
     int ind=(*it).second;
     
     return &(data[ind]);
    }
 };

 
struct Gain
 {
  CellId id;
  float gain;
  float error;
 };

struct Ped
 {
  CellId id;
  float ped[4]; // 4 caps
 };

 
struct QieCM
 {
  CellId id;
  int adc_channel;
  float linearized;
  float non_linearized;
 };
 

struct Range
 {
  CellId id;
  std::vector<float> slope;
  std::vector<float> offset;

  /* cap_id 0..3, n 0..3 */
  float getSlope(int cap_id,int n) const {return slope[cap_id*4+n];}
  float getOffset(int cap_id,int n) const {return offset[cap_id*4+n];}
 };


struct QLinearData
 {
  unsigned bin;
  float value;
 };

typedef THcalDbClass<Gain> Gains;
typedef THcalDbClass<Ped> Pedestals;
typedef THcalDbClass<QieCM> QieCMs;
typedef THcalDbClass<Range> Ranges;
typedef std::vector<QLinearData> QLinearization;


const Gains*     getGains(unsigned timestamp);
const Pedestals* getPedestals(unsigned timestamp);
const QieCMs*    getQieCMs(unsigned int timestamp);
const Ranges*    getRanges(unsigned timestamp);
const QLinearization *getQLinearization(unsigned timestamp);


}; // End of HcalDb namespace



#endif //__HEADER_H_HCAL_CLASSES_CPP_H_
