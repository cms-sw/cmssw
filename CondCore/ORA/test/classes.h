#ifndef TESTCLASSES_H
#define TESTCLASSES_H

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <map>
#include "CondCore/ORA/interface/Reference.h"
#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Ptr.h"
#include "CondCore/ORA/interface/PVector.h"
#include "CondCore/ORA/interface/QueryableVector.h"
#include "CondCore/ORA/interface/UniqueRef.h"
#include <boost/shared_ptr.hpp>

class SimpleMember {

 public:
  SimpleMember():m_flag(false),m_id(0){}
  SimpleMember( long id ):m_flag(true),m_id(id){}
  SimpleMember(const SimpleMember& mem):m_flag(mem.m_flag),m_id(mem.m_id){}
  SimpleMember& operator=(const SimpleMember& mem){
    m_flag = mem.m_flag;
    m_id = mem.m_id;
    return *this;
  }
  bool operator==(const SimpleMember& rhs) const {
    if(m_flag!=rhs.m_flag) {
      std::cout << "## SimpleMember: m_flag is different."<<std::endl;
      return false;
    }
    if(m_id!=rhs.m_id) {
      std::cout << "## SimpleMember: exp m_id="<<m_id<<" found="<<rhs.m_id<<std::endl;
      return false;
    }
    return true;
  }
  bool operator!=(const SimpleMember& rhs) const {
    return !operator==(rhs);
  }
  
  bool m_flag;
  long m_id;
};

struct Dummy {
    std::vector<SimpleMember> dm;
};

class BaseClass {
  public:
  BaseClass():m_f(0.),m_s(""),m_objectData(){
  }
  BaseClass(unsigned int id):m_f(0.+id),m_s(""),m_objectData(){
    std::ostringstream os;
    os << "BaseClass_"<<id;    
    m_s = os.str();
    m_objectData.m_flag = true;
    m_objectData.m_id = id+1;
  }
  
  bool operator==(const BaseClass& rhs){
    if(m_f!=rhs.m_f) return false;
    if(m_s!=rhs.m_s) return false;
    if(m_objectData.m_flag!=rhs.m_objectData.m_flag) return false;
    if(m_objectData.m_id!=rhs.m_objectData.m_id) return false;
    return true;
  }
  bool operator!=(const BaseClass& rhs){
    return !operator==(rhs);
  }
  
  float m_f;
  std::string m_s;
  SimpleMember m_objectData;
};

class SimpleClass : public BaseClass {

 public:

  //SimpleClass():m_intData(0),m_stringData(""),m_objectData(),m_arrayData(),m_map(),m_code(ZERO){}
  //SimpleClass(unsigned int id):m_intData(id),m_stringData(""),m_objectData(),m_arrayData(),m_map(),m_code(ZERO){
  SimpleClass():BaseClass(),m_intData(0),m_stringData(""),m_objectData(),m_code(ZERO){}
  SimpleClass(unsigned int id):BaseClass(id),m_intData(id),m_stringData(""),m_objectData(),m_code(ZERO){
    m_intData = id;
    std::ostringstream os;
    os << "SimpleClass_"<<id; 
    m_stringData = os.str();
    m_objectData.m_flag = true;
    m_objectData.m_id = id;
    //for(int i=0;i<(int)id;i++){
    //  m_arrayData.push_back(i);
    //}
    //for(unsigned long long i=0;i<(unsigned long long)id;i++){
    //   m_map.insert(std::make_pair(i,i));
    //}
    m_code = ONE;
    if(id % 2 !=0 ){
      m_code = TWO;
    }
  }

  SimpleClass(const SimpleClass& obj){
    m_intData = obj.m_intData;
    m_stringData = obj.m_stringData;
    m_objectData = obj.m_objectData;
    //m_arrayData = obj.m_arrayData;
    //m_map = obj.m_map;
    m_code = obj.m_code;
  }
  SimpleClass& operator=(const SimpleClass& obj){
    m_intData = obj.m_intData;
    m_stringData = obj.m_stringData;
    m_objectData = obj.m_objectData;
    //m_arrayData = obj.m_arrayData;
    //m_map = obj.m_map;
    m_code = obj.m_code;
    return *this;
  }

  bool operator==(const SimpleClass& rhs){
    if(BaseClass::operator!=(rhs)) return false;
    if(m_intData!=rhs.m_intData) return false;
    if(m_stringData!=rhs.m_stringData) return false;
    if(m_objectData.m_flag!=rhs.m_objectData.m_flag) return false;
    if(m_objectData.m_id!=rhs.m_objectData.m_id) return false;
    //if(m_arrayData!=rhs.m_arrayData) return false;
    //if(m_map!=rhs.m_map) return false;
    if(m_code!=rhs.m_code) return false;
    return true;
  }
  bool operator!=(const SimpleClass& rhs){
    return !operator==(rhs);
  }

  void print(){
    std::cout << "BASE F="<<BaseClass::m_f<<std::endl;
    std::cout << "BASE S="<<BaseClass::m_s<<std::endl;
    std::string bbs("TRUE");
    if(!BaseClass::m_objectData.m_flag) bbs = "FALSE";
    std::cout << "BASE ODFLAG="<<bbs<<std::endl;
    std::cout << "BASE ODID="<<BaseClass::m_objectData.m_id<<std::endl;
    std::cout << "ID="<<m_intData<<std::endl;
    std::cout << "STR="<<m_stringData<<std::endl;
    std::string bs("TRUE");
    if(!m_objectData.m_flag) bs = "FALSE";
    std::cout << "ODFLAG="<<bs<<std::endl;
    std::cout << "ODID="<<m_objectData.m_id<<std::endl;
    std::cout << "CODE="<<m_code<<std::endl;
  }

  enum MySimpleClassCode { ZERO, ONE, TWO };
  unsigned int m_intData;
  std::string m_stringData;
  SimpleMember m_objectData;
  //std::vector<int> m_arrayData;
  //std::map<unsigned long long,unsigned long long> m_map;
  MySimpleClassCode m_code;
};

class ArrayClass {

 public:

  ArrayClass():m_arrayData(),m_map(){}
  ArrayClass(unsigned int id):m_arrayData(),m_map(){
    for(int i=0;i<(int)id;i++){
      m_arrayData.push_back(i);
    }
    for(int i=0;i<(int)id;i++){
      std::ostringstream os0;
      os0 << "SC_"<<i; 
      std::ostringstream os1;
      os1 << "SimpleClass_"<<i; 
      m_map.insert(std::make_pair(os0.str(),os1.str()));
    }
  }
  
  bool operator==(const ArrayClass& rhs){
    if(m_arrayData!=rhs.m_arrayData) return false;
    if(m_map!=rhs.m_map) return false;
    return true;
  }
  bool operator!=(const ArrayClass& rhs){
    return !operator==(rhs);
  }

  void print(){
    for(size_t j=0;j<m_arrayData.size();j++) std::cout << "v["<<j<<"]="<<m_arrayData[j]<<std::endl;
    for( std::map<std::string,std::string>::const_iterator iM = m_map.begin();
         iM != m_map.end(); iM++ ){
      std::cout << "m["<<iM->first<<"]="<<iM->second<<std::endl;
    }
    
  }

  std::vector<int> m_arrayData;
  std::map<std::string,std::string> m_map;
};

class MultiArrayClass {

 public:

  MultiArrayClass():m_a(){}
  MultiArrayClass(unsigned int id):m_a(){
    for(int i=0;i<(int)id;i++){
      std::vector<int> vj;
      for(int j=0;j<i+1;j++){
        vj.push_back(j);
      }
      m_a.push_back(vj);
    }
  }
  
  bool operator==(const MultiArrayClass& rhs){
    if(m_a!=rhs.m_a) return false;
    return true;
  }
  bool operator!=(const MultiArrayClass& rhs){
    return !operator==(rhs);
  }

  void print(){
    for(size_t j=0;j<m_a.size();j++) {
      std::cout <<"*** vec["<<j<<"]*** size="<<m_a[j].size()<<std::endl;
      for( size_t ij=0;ij<m_a[j].size();ij++) std::cout << "m_a["<<j<<"]["<<ij<<"]="<<m_a[j][ij]<<std::endl;
    }
  }

  std::vector<std::vector<int> > m_a;
};

class OtherMember {

 public:
  OtherMember():m_flag(false),m_id(0),m_v(){}
  OtherMember( long id ):m_flag(true),m_id(id),m_v(){
    for(int i=10;i<15;i++) m_v.push_back(i);
  }
  OtherMember(const OtherMember& mem):m_flag(mem.m_flag),m_id(mem.m_id),m_v(mem.m_v){}
  OtherMember& operator=(const OtherMember& mem){
    m_flag = mem.m_flag;
    m_id = mem.m_id;
    m_v = mem.m_v;
    return *this;
  }
  bool operator==(const OtherMember& rhs) const {
    if(m_flag!=rhs.m_flag) return false;
    if(m_id!=rhs.m_id) return false;
    if(m_v!=rhs.m_v) return false;
    return true;
  }
  bool operator!=(const OtherMember& rhs) const {
    return !operator==(rhs);
  }
  
  bool m_flag;
  long m_id;
  std::vector<int> m_v;
};

class MultiArrayClass2 {

  public:
  MultiArrayClass2():m_a(){}
  MultiArrayClass2(unsigned int id):m_a(){
    for(int i=0;i<(int)id;i++){
      std::vector<OtherMember> vj;
      for(int j=0;j<i+1;j++){
        OtherMember sm(j);
        if( i%2 != 0) sm.m_flag=false;
        vj.push_back(sm);
      }
      m_a.push_back(vj);
    }
  }
  
  bool operator==(const MultiArrayClass2& rhs){
    if(m_a!=rhs.m_a) return false;
    return true;
  }
  bool operator!=(const MultiArrayClass2& rhs){
    return !operator==(rhs);
  }

  void print(){
    for(size_t j=0;j<m_a.size();j++) {
      std::cout <<"*** vec["<<j<<"]*** size="<<m_a[j].size()<<std::endl;
      for( size_t ij=0;ij<m_a[j].size();ij++) {
        std::string f("FALSE");
        if(m_a[j][ij].m_flag) f = "TRUE";
        std::cout << "m_a["<<j<<"]["<<ij<<"] id="<<m_a[j][ij].m_id<<" flag="<<f<<std::endl;
        for(size_t ijk=0;ijk<m_a[j][ij].m_v.size();ijk++)
          std::cout << "m_a["<<j<<"]["<<ij<<"] mv="<<m_a[j][ij].m_v[ijk]<<std::endl;
      }
    }
  }

  std::vector<std::vector<OtherMember> > m_a;
};

class SimpleCl {

 public:

  SimpleCl():m_intData(0),m_stringData(""),m_objectData(),m_arrayData(),m_map(),m_code(ZERO){}
  SimpleCl(unsigned int id):m_intData(id),m_stringData(""),m_objectData(),m_arrayData(),m_map(),m_code(ZERO){
    //SimpleCl():m_intData(0),m_stringData(""),m_objectData(),m_arrayData(),m_code(ZERO){}
    //SimpleCl(unsigned int id):m_intData(id),m_stringData(""),m_objectData(),m_arrayData(),m_code(ZERO){
    m_intData = id;
    std::ostringstream os;
    os << "SimpleCl_"<<id; 
    m_stringData = os.str();
    m_objectData.m_flag = true;
    m_objectData.m_id = id;
    for(int i=0;i<(int)id;i++){
      m_arrayData.push_back(i);
    }
    for(unsigned long long i=0;i<(unsigned long long)id;i++){
       m_map.insert(std::make_pair(i,i));
    }
    m_code = ONE;
    if(id % 2 !=0 ){
      m_code = TWO;
    }
  }

  SimpleCl(const SimpleCl& obj){
    m_intData = obj.m_intData;
    m_stringData = obj.m_stringData;
    m_objectData = obj.m_objectData;
    m_arrayData = obj.m_arrayData;
    m_map = obj.m_map;
    m_code = obj.m_code;
  }

  ~SimpleCl(){
   }
  
  
  SimpleCl& operator=(const SimpleCl& obj){
    m_intData = obj.m_intData;
    m_stringData = obj.m_stringData;
    m_objectData = obj.m_objectData;
    m_arrayData = obj.m_arrayData;
    m_map = obj.m_map;
    m_code = obj.m_code;
    return *this;
  }

  bool operator==(const SimpleCl& rhs){
    if(m_intData!=rhs.m_intData) return false;
    if(m_stringData!=rhs.m_stringData) return false;
    if(m_objectData.m_flag!=rhs.m_objectData.m_flag) return false;
    if(m_objectData.m_id!=rhs.m_objectData.m_id) return false;
    if(m_arrayData!=rhs.m_arrayData) return false;
    if(m_map!=rhs.m_map) return false;
    if(m_code!=rhs.m_code) return false;
    return true;
  }
  bool operator!=(const SimpleCl& rhs){
    return !operator==(rhs);
  }

  enum MySimpleClCode { ZERO, ONE, TWO };
  unsigned int m_intData;
  std::string m_stringData;
  SimpleMember m_objectData;
  std::vector<int> m_arrayData;
  std::map<unsigned long long,unsigned long long> m_map;
  MySimpleClCode m_code;
};

class SM {

 public:
  SM():m_flag(false),m_id(0){}
  SM(long id):m_flag(true),m_id(id){
    //for(int i=0;i<3;i++) m_v.push_back(i);
  }
  SM(const SM& mem):m_flag(mem.m_flag),m_id(mem.m_id){}
  SM& operator=(const SM& mem){
    m_flag = mem.m_flag;
    m_id = mem.m_id;
    //m_v = mem.m_v;
    return *this;
  }

  bool operator==(const SM& mem)const{
    if(m_flag!=mem.m_flag) return false;
    if(m_id!=mem.m_id) return false;
    //if(m_v!=mem.m_v) return false;
    return true;
  }
  bool operator!=(const SM& mem)const{
    return !operator==(mem);
  }
  
  bool m_flag;
  long m_id;
  //std::vector<int> m_v;

};

class SimpleStruct {
  public:
  SimpleStruct():m_i(0){
    for(int i=0;i<3;i++) m_arr0[i]=0;    
    for(int i=0;i<15;i++) m_arr1[i]=0;
  }
  SimpleStruct(int id):m_i(id){
    for(int i=0;i<3;i++) m_arr0[i]=i+id;    
    for(int i=0;i<15;i++) m_arr1[i]=i+id;
  }
  SimpleStruct(const SimpleStruct& mem):m_i(mem.m_i){
    for(int i=0;i<3;i++) m_arr0[i]=mem.m_arr0[1];    
    for(int i=0;i<15;i++) m_arr1[i]=mem.m_arr1[i];
  }
  SimpleStruct& operator=(const SimpleStruct& mem){
    m_i = mem.m_i;
    for(int i=0;i<3;i++) m_arr0[i]=mem.m_arr0[1];    
    for(int i=0;i<15;i++) m_arr1[i]=mem.m_arr1[i];
    return *this;
  }
  bool operator==(const SimpleStruct& rhs){
    if(m_i!=rhs.m_i) {
      std::cout << "** ERROR: m_i exp="<<m_i<<" found="<<rhs.m_i<<std::endl;
      return false;
    }
    for(int i=0;i<3;i++) if(m_arr0[i]!=rhs.m_arr0[i]) {
      std::cout << "** ERROR: m_arr0["<<i<<"] exp="<<m_arr0[i]<<" found="<<rhs.m_arr0[i]<<std::endl;
      return false;
    }
    
    for(int i=0;i<15;i++) if(m_arr1[i]!=rhs.m_arr1[i]) {
      std::cout << "** ERROR: m_arr1["<<i<<"] exp="<<m_arr1[i]<<std::endl;
      //for(int j=0;j<110;j++) std::cout << "** m_arr1 ["<<j<<"] exp="<<m_arr1[j]<<" found="<<rhs.m_arr1[j]<<std::endl;
      return false;
    }
    
    return true;
  }
  bool operator!=(const SimpleStruct& rhs){
    return !operator==(rhs);
  }
  int m_i;
  int m_arr0[3];
  int m_arr1[15];
};

class SA {
  public:

  class Inner 
    {
      public:
      Inner():m_v(){}
      Inner(int id):m_v(){
        for(int i=0;i<id;i++) m_v.push_back(i);
      }
      virtual ~Inner(){}
      bool operator==(const Inner& rhs){
        if(m_v!=rhs.m_v) return false;
        return true;
      }
      bool operator!=(const Inner& rhs){
        return !operator==(rhs);
      }
      std::vector<int> m_v;
      
    };

  public:

  //SA():m_intData(0),m_0_i1_i2_i3_m_id(0),m_stringData012345678012345678012345678012345678012345678(""),m_objectData(),m_arrayData(),m_2(),m_innData(){
  SA():m_intData(0),m_0_i1_i2_i3_m_id(0),m_stringData012345678012345678012345678012345678012345678(""){
    for(int i=0;i<5;i++)
      for(int j=0;j<4;j++) m_carra0[i][j]=0;
    for(int i=0;i<5;i++) m_carra[i]=0;
    for(int i=0;i<150;i++) m_carra1[i]=0;    
    for(int i=0;i<110;i++)
      for(int j=0;j<2;j++) m_carra2[i][j]=0;
    for(int i=0;i<30;i++) m_ocarra[i]=SimpleStruct();
    for(int i=0;i<10;i++) {
      m_maps[i]=std::map<std::string,std::string>();
    }
    for(int i=0;i<2;i++) {
      m_vecs[i] = std::vector<int>();
    }
    for(int i=0;i<2;i++)
      for(int j=0;j<3;j++)
        for(int k=0;k<4;k++) m_0[i][j][k] = SM();
    for(int i=0;i<10;i++)
      for(int j=0;j<20;j++)
        for(int k=0;k<30;k++) m_1[i][j][k] = SM();
  }

    //SA(unsigned int id):m_intData(id),m_0_i1_i2_i3_m_id(id),m_stringData012345678012345678012345678012345678012345678(""),m_objectData(),m_arrayData(),m_2(),m_innData(id){
  SA(unsigned int id):m_intData(id),m_0_i1_i2_i3_m_id(id),m_stringData012345678012345678012345678012345678012345678(""){
    m_intData = id;
    m_0_i1_i2_i3_m_id = id;
    std::ostringstream os;
    os << "SimpleClass_"<<id; 
    m_stringData012345678012345678012345678012345678012345678 = os.str();
    /**
    m_objectData.m_flag = true;
    m_objectData.m_id = id;
    for(int i=0;i<(int)id;i++){
      m_arrayData.push_back(i);
      }
    **/  
    for(int i=0;i<5;i++)
      for(int j=0;j<4;j++) m_carra0[i][j]=id*1000+i*j;
    for(int i=0;i<5;i++) m_carra[i]=id*1000+i;
    for(int i=0;i<150;i++) m_carra1[i]=id*100+i;
    for(int i=0;i<110;i++)
      for(int j=0;j<2;j++) m_carra2[i][j]=id*100+i*j;
    for(int i=0;i<30;i++) m_ocarra[i]=SimpleStruct(i);
    for(int i=0;i<10;i++) {
      m_maps[i]=std::map<std::string,std::string>();
      for(unsigned int j=0;j<10;j++) {
        std::ostringstream os0;
        os0 << i*j;
        std::ostringstream os1;
        os1 << os.str() << "_"<< i*j;
        m_maps[i].insert(std::make_pair(os0.str(),os1.str()));
      }
    }
    for(int i=0;i<2;i++) {
      m_vecs[i] = std::vector<int>();
      for(unsigned int j=0;j<10;j++) m_vecs[i].push_back(i*j);
    }
    for(int i=0;i<2;i++)
      for(int j=0;j<3;j++)
        for(int k=0;k<4;k++) m_0[i][j][k] = SM(i*j*k);
    for(int i=0;i<10;i++)
      for(int j=0;j<20;j++)
        for(int k=0;k<30;k++) m_1[i][j][k] = SM(i*j*k);;
    /**
    for(int i=0;i<(int)id;i++){
      std::vector<std::vector<int> > v0;
      for(int j=0;j<2;j++){
        std::vector<int> v1;
        for(int k=0;k<3;k++) v1.push_back(i*j*k);
        v0.push_back(v1);
      }
      m_2.push_back(v0);
    }
    **/  
  }
  

    //SA(const SA& obj):m_intData(obj.m_intData),m_0_i1_i2_i3_m_id(obj.m_0_i1_i2_i3_m_id),m_stringData012345678012345678012345678012345678012345678(obj.m_stringData012345678012345678012345678012345678012345678),m_objectData(obj.m_objectData),m_arrayData(obj.m_arrayData),m_2(obj.m_2),m_innData(){
    SA(const SA& obj):m_intData(obj.m_intData),m_0_i1_i2_i3_m_id(obj.m_0_i1_i2_i3_m_id),m_stringData012345678012345678012345678012345678012345678(obj.m_stringData012345678012345678012345678012345678012345678){
    for(int i=0;i<5;i++)
      for(int j=0;j<4;j++) m_carra0[i][j]=obj.m_carra0[i][j];
    for(int i=0;i<5;i++) m_carra[i]=obj.m_carra[i];
    for(int i=0;i<150;i++) m_carra1[i]=obj.m_carra1[i];
    for(int i=0;i<110;i++)
      for(int j=0;j<2;j++) m_carra2[i][j]=obj.m_carra2[i][j];
    for(int i=0;i<30;i++) m_ocarra[i]=obj.m_ocarra[i];
    for(int i=0;i<10;i++) m_maps[i] = obj.m_maps[i];
    for(int i=0;i<2;i++) m_vecs[i] = obj.m_vecs[i];
    for(int i=0;i<2;i++)
      for(int j=0;j<3;j++)
        for(int k=0;k<4;k++) m_0[i][j][k] = obj.m_0[i][j][k];
    for(int i=0;i<10;i++)
      for(int j=0;j<20;j++)
        for(int k=0;k<30;k++) m_1[i][j][k] = obj.m_1[i][j][k];
    /**
    m_innData.m_v = obj.m_innData.m_v;
    **/    
  }
  
  SA& operator=(const SA& obj){
    m_intData = obj.m_intData;
    m_0_i1_i2_i3_m_id = obj.m_0_i1_i2_i3_m_id;
    m_stringData012345678012345678012345678012345678012345678 = obj.m_stringData012345678012345678012345678012345678012345678;
    /**
    m_objectData = obj.m_objectData;
    m_arrayData = obj.m_arrayData;
    **/
    for(int i=0;i<5;i++)
      for(int j=0;j<4;j++) m_carra0[i][j]=obj.m_carra0[i][j];
    for(int i=0;i<5;i++) m_carra[i]=obj.m_carra[i];
    for(int i=0;i<150;i++) m_carra1[i]=obj.m_carra1[i];    
    for(int i=0;i<110;i++)
      for(int j=0;j<2;j++) m_carra2[i][j]=obj.m_carra2[i][j];
    for(int i=0;i<30;i++) m_ocarra[i]=obj.m_ocarra[i];
    for(int i=0;i<10;i++) m_maps[i] = obj.m_maps[i];
    for(int i=0;i<2;i++) m_vecs[i] = obj.m_vecs[i];
    for(int i=0;i<2;i++)
      for(int j=0;j<3;j++)
        for(int k=0;k<4;k++) m_0[i][j][k] = obj.m_0[i][j][k];
    for(int i=0;i<10;i++)
      for(int j=0;j<20;j++)
        for(int k=0;k<30;k++) m_1[i][j][k] = obj.m_1[i][j][k];
    /**
    m_2 = obj.m_2;
    m_innData.m_v = obj.m_innData.m_v;
    **/
    return *this;
  }

  bool operator==(const SA& rhs){
    if(m_intData!=rhs.m_intData) return false;
    if(m_0_i1_i2_i3_m_id!=rhs.m_0_i1_i2_i3_m_id) return false;
    if(m_stringData012345678012345678012345678012345678012345678!=rhs.m_stringData012345678012345678012345678012345678012345678) return false;
    //if(m_objectData.m_flag!=rhs.m_objectData.m_flag) return false;
    //if(m_objectData.m_id!=rhs.m_objectData.m_id) return false;
    //if(m_arrayData!=rhs.m_arrayData) return false;
    for(int i=0;i<5;i++)
    for(int j=0;j<4;j++) if(m_carra0[i][j]!=rhs.m_carra0[i][j]) return false;
    for(int i=0;i<5;i++) if(m_carra[i]!=rhs.m_carra[i]) return false;
    for(int i=0;i<150;i++) if(m_carra1[i]!=rhs.m_carra1[i]) {
      return false;
    }
    for(int i=0;i<110;i++)
      for(int j=0;j<2;j++) if(m_carra2[i][j]!=rhs.m_carra2[i][j]) return false;
    for(int i=0;i<30;i++) if(m_ocarra[i]!=rhs.m_ocarra[i]) {
      std::cout << "** ERROR for ind i="<<i<<std::endl;
      return false;
    }
    for(int i=0;i<10;i++) if(m_maps[i]!=rhs.m_maps[i]) return false;
    for(int i=0;i<2;i++) if(m_vecs[i]!=rhs.m_vecs[i]) return false;
    bool ok = true;
    for(int i=0;i<2;i++)
      for(int j=0;j<3;j++)
        for(int k=0;k<4;k++) if(m_0[i][j][k]!=rhs.m_0[i][j][k]) {
          std::cout << "## Error: found="<<m_0[i][j][k].m_id<<" exp="<<rhs.m_0[i][j][k].m_id<<std::endl;
          ok = false;
        }
    if(!ok) return false;
    for(int i=0;i<10;i++)
      for(int j=0;j<20;j++)
        for(int k=0;k<30;k++) if(m_1[i][j][k]!=rhs.m_1[i][j][k]) return false;
    //if(m_2!=rhs.m_2) return false;
    //if(m_innData!=rhs.m_innData) return false;
    return true;
  }
  bool operator!=(const SA& rhs){
    return !operator==(rhs);
  }

  unsigned int m_intData;
  unsigned int m_0_i1_i2_i3_m_id;
  std::string m_stringData012345678012345678012345678012345678012345678;
  //SM m_objectData;
  //std::vector<int> m_arrayData;
  int m_carra[3];
  int m_carra0[5][4];
  int m_carra1[150];
  int m_carra2[110][2];
  SimpleStruct m_ocarra[30];
  std::map<std::string,std::string> m_maps[10];
  std::vector<int> m_vecs[2];
  SM m_0[2][3][4];
  SM m_1[10][20][30];
  //std::vector<std::vector<std::vector<int> > > m_2;
  //Inner m_innData;
};

struct Metadata{
    float EtMin,EtMax;
    int signaltks;
    Metadata():EtMin(0),EtMax(0),signaltks(0){}
    Metadata(const Metadata& rhs):EtMin(rhs.EtMin),EtMax(rhs.EtMax),signaltks(rhs.signaltks){}
    Metadata& operator=(const Metadata& rhs){
      EtMin = rhs.EtMin;
      EtMax = rhs.EtMax;
      signaltks = rhs.signaltks;
      return *this;
    }
    bool operator==(const Metadata& rhs) const {
      if(EtMin!=rhs.EtMin) return false;
      if(EtMax!=rhs.EtMax) return false;
      if(signaltks!=rhs.signaltks) return false;
      return true;
    }
  bool operator!=(const Metadata& rhs) const {
    return !operator==(rhs);
  }
    
};

struct Entry{
    Metadata metadata;
    std::vector<double> m_histogram;
    Entry():metadata(),m_histogram(){}
    Entry(const Entry& rhs):metadata(rhs.metadata),m_histogram(rhs.m_histogram){}
    Entry& operator=(const Entry& rhs){
      metadata = rhs.metadata;
      m_histogram = rhs.m_histogram;
      return *this;
    }
    bool operator==(const Entry& rhs) const {
      if(metadata!=rhs.metadata) return false;
      if(m_histogram!=rhs.m_histogram) return false;
      return true;
    }
  bool operator!=(const Entry& rhs) const {
    return !operator==(rhs);
  }
};

class SB {

 public:

  SB():m_intData(0),m_stringData(""),m_objectData(),m_arrayData(),m_nestedArrayData(),m_entries(){}
  SB(unsigned int id):m_intData(id),m_stringData(""),m_objectData(),m_arrayData(),m_nestedArrayData(),m_entries(){
    m_intData = id;
    std::ostringstream os;
    os << "SimpleClass_"<<id; 
    m_stringData = os.str();
    m_objectData.m_flag = true;
    m_objectData.m_id = id;
    for(int i=0;i<(int)id+1;i++){
      m_arrayData.push_back(i);
    }
    for(int j=0;j<(int)id+1;j++){
      std::vector<double> vec;
      for(int i=0;i<100;i++){
        double v = 0.+i;
        vec.push_back(v);
      }
      m_nestedArrayData.push_back(vec);
    }
    for(int i=0;i<(int)id+1;i++){
      Entry entry;
      entry.metadata.EtMax = float(0.+i);
      entry.metadata.EtMin = entry.metadata.EtMax-float(1.);
      if(entry.metadata.EtMin < 0.) entry.metadata.EtMin = float(0.);
      entry.metadata.signaltks = i;
      for(int j=0;j<10;j++){
        double v = 0.+j;
        entry.m_histogram.push_back(v);
      }
      m_entries.push_back(entry);
    }
    /**
    for(int j=0;j<2;j++){
      for(int i=0;i<20;i++){
        float v = float(0.+i);
        m_carrayData[j].push_back(v);
      }
    }
    **/
    
  }

    SB(const SB& obj):m_intData(obj.m_intData),m_stringData(obj.m_stringData),m_objectData(obj.m_objectData),m_arrayData(obj.m_arrayData),m_entries(obj.m_entries){}
      
  SB& operator=(const SB& obj){
    m_intData = obj.m_intData;
    m_stringData = obj.m_stringData;
    m_objectData = obj.m_objectData;
    m_arrayData = obj.m_arrayData;
    m_nestedArrayData = obj.m_nestedArrayData;
    m_entries = obj.m_entries;
    //for(int i=0;i<2;++i){
    //  m_carrayData[i] = obj.m_carrayData[i];
    //}
    return *this;
  }

  bool operator==(const SB& rhs){
    if(m_intData!=rhs.m_intData) return false;
    if(m_stringData!=rhs.m_stringData) return false;
    if(m_objectData.m_flag!=rhs.m_objectData.m_flag) return false;
    if(m_objectData.m_id!=rhs.m_objectData.m_id) return false;
    if(m_arrayData!=rhs.m_arrayData) return false;
    if(m_nestedArrayData!=rhs.m_nestedArrayData) {
      std::cout << "Size expected="<<m_nestedArrayData.size()<<" - found="<<rhs.m_nestedArrayData.size()<<std::endl;
      return false;
    }
    if(m_entries!=rhs.m_entries) return false;
    //for(int i=0;i<2;++i){
    //  if(m_carrayData[i]!=rhs.m_carrayData[i]) return false;
    //}
    return true;
  }
  bool operator!=(const SB& rhs){
    return !operator==(rhs);
  }

  unsigned int m_intData;
  std::string m_stringData;
  SimpleMember m_objectData;
  std::vector<int> m_arrayData;
  std::vector<std::vector<double> > m_nestedArrayData;
  std::vector<Entry> m_entries;
  //std::vector<float> m_carrayData[2];
};


class SiStripNoises {
  public:
  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
    bool operator==(const DetRegistry& rhs) const {
      if(detid!=rhs.detid)return false;
      if(ibegin!=rhs.ibegin)return false;
      if(iend!=rhs.iend)return false;
      return true;
    }
    bool operator!=(const DetRegistry& rhs) const {
      return !operator==(rhs);
    }
  };

  SiStripNoises():m_id(0),v_noises(),indexes(){}
    SiStripNoises(unsigned int id):m_id(id),v_noises(),indexes(){
    short id_s = (short)id;
    for(short i=0;i<id_s+1;i++) v_noises.push_back(i);
    for(unsigned int i=0;i<id+1;i++) {
      DetRegistry reg;
      reg.detid = i;
      reg.ibegin = i;
      reg.iend = i;
      indexes.push_back(reg);
    }
  }
  bool operator==(const SiStripNoises& rhs) const {
    if(m_id!=rhs.m_id){
      return false;
    }
    if(v_noises!=rhs.v_noises){
      return false;
    }
    if(indexes!=rhs.indexes){
      return false;
    }
    return true;
  }
  bool operator!=(const SiStripNoises& rhs) const {
    return !operator==(rhs);
  }
  unsigned int m_id;
  //std::vector<int16_t> v_noises;
  std::vector<short> v_noises;
  std::vector<DetRegistry> indexes;
};

class RefBase : public ora::Reference {
  public:
  RefBase():
    ora::Reference(),m_db(0){
  }
  RefBase( const RefBase& rhs):
    ora::Reference(rhs),m_db(rhs.m_db){
  }
  virtual ~RefBase(){
  }
  RefBase& operator=( const RefBase& rhs){
    ora::Reference::operator=( rhs );
    m_db = rhs.m_db;
    return *this;
  } 
  void setDb( ora::Database& db ){
    m_db = &db;
  }
  ora::Database* db(){
    return m_db;
  }
  
  private:
  ora::Database* m_db;
};

template <typename T> class Ref : public RefBase {
  public:
  Ref():
    RefBase(),m_data(){
  }
  Ref( const Ref<T>& rhs ):
    RefBase( rhs ),m_data( rhs.m_data ){
  }
  Ref& operator=( const Ref<T>& rhs ){
    RefBase::operator=( rhs );
    m_data = rhs.m_data;
    return *this;
  }
  void load(){
    ora::Database* database = db();
    if( database ){
      m_data = database->fetch<T>( oid() );
    }
  }
    
  boost::shared_ptr<T> m_data;
};

class SC {

  public:
  
  SC():m_intData(0),m_ref(),m_refVec(){}
    SC(unsigned int id):m_intData(id),m_ref(),m_refVec(){
  }
    
  bool operator==(const SC& rhs){
    if(m_intData!=rhs.m_intData) return false;
    if(*m_ref.m_data != *rhs.m_ref.m_data ) return false;
    if(m_refVec.size() != rhs.m_refVec.size()) {
      std::cout << "## size is different exp="<<m_refVec.size()<<" found="<<rhs.m_refVec.size()<<std::endl;
      return false;
    }
    for( size_t i=0;i<m_refVec.size();i++ ){
      std::cout << "## checking differecne for i="<<i<<std::endl;
      if( !m_refVec[i].m_data ){
        std::cout << "## current data null."<<std::endl;
        return false;
      }
      if( !rhs.m_refVec[i].m_data ){
        std::cout << "## found data null."<<std::endl;
        return false;
      }
      
      if(*(m_refVec[i].m_data) != *(rhs.m_refVec[i].m_data) ) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const SC& rhs){
    return !operator==(rhs);
  }
  
  unsigned int m_intData;
  Ref<SimpleCl> m_ref;
  std::vector<Ref<SimpleCl> > m_refVec;
};

class SD {

  public:
  
  SD():m_intData(0),m_ptr(),m_ptrVec(){}
    SD(unsigned int id):m_intData(id),m_ptr(),m_ptrVec(){
  }

  ~SD(){
  }
    
    
  bool operator==(const SD& rhs){
    if(m_intData!=rhs.m_intData) return false;
    if(*m_ptr != *rhs.m_ptr ) return false;
    if( m_ptrVec.size() != rhs.m_ptrVec.size() ) {
      std::cout << "### error; exp size="<<m_ptrVec.size()<<" found size="<<rhs.m_ptrVec.size()<<std::endl;
      return false;
    }
    bool ret = true;
    for( size_t i=0;i<m_ptrVec.size();i++){
      if(*m_ptrVec[i] != *rhs.m_ptrVec[i] ) {
        std::cout << "### error: diff in index="<<i<<std::endl;
        ret = false;
      }
    }
    if( !ret ) return false;
    return true;
  }
  bool operator!=(const SD& rhs){
    return !operator==(rhs);
  }
  
  unsigned int m_intData;
  ora::Ptr<SimpleCl> m_ptr;
  std::vector<ora::Ptr<SimpleMember> > m_ptrVec;
};

class SE {

  public:
  
  SE():m_intData(0),m_vec(){}
    SE(unsigned int id):m_intData(id),m_vec(){
      for(unsigned int i=0;i<id;i++){
        m_vec.push_back( SM(i) );
      }
  }

  ~SE(){
  }
    
    
  bool operator==(const SE& rhs){
    if(m_intData!=rhs.m_intData) return false;
    if( m_vec != rhs.m_vec ) {
      return false;
    }
    return true;
  }
  bool operator!=(const SE& rhs){
    return !operator==(rhs);
  }
  
  unsigned int m_intData;
  ora::PVector<SM> m_vec;
};

class SF {

  public:
  
  SF():m_intData(0),m_vec(){}
    SF(unsigned int id):m_intData(id),m_vec(){
      for(unsigned int i=0;i<id;i++){
        m_vec.push_back( SM(i) );
      }
  }

  ~SF(){
  }
    
    
  bool operator==(const SF& rhs){
    if(m_intData!=rhs.m_intData) return false;
    if( m_vec != rhs.m_vec ) {
      return false;
    }
    return true;
  }
  bool operator!=(const SF& rhs){
    return !operator==(rhs);
  }
  
  unsigned int m_intData;
  ora::QueryableVector<SM> m_vec;
};

class IBase {
  public:
  IBase():mt(""){
  }
  explicit IBase(const std::string& s):mt(s){
  }
    
  virtual ~IBase(){
  }

  virtual bool equals( const IBase& rhs ){
    if( mt != rhs.mt ) return false;
    return true;
  }
  
  std::string mt;
};

class D0: public IBase {
  public:
  D0():
    IBase("D0"),mid(0),ms(""),msm(),mv(){
  }
    
  D0( unsigned int i):
    IBase("D0"),mid(i),ms(""),msm(i),mv(){
    std::ostringstream os;
    os << "D0_"<<i;    
    ms = os.str();
    for(int j=0;j<(int)i+1;j++){
      mv.push_back(j);
    }
  }

    bool operator==(const D0& rhs){
      if(mid != rhs.mid) return false;
      if(ms != rhs.ms ) return false;
      if(msm != rhs.msm ) return false;
      if( mv != rhs.mv ) return false;
      return true;
    }
    bool operator!=(const D0& rhs){
      return !operator==(rhs);
    }

    bool equals( const IBase& rhs ){
      const D0& comp = dynamic_cast<const D0&>(rhs);
    if( !IBase::equals( rhs )) return false;
    if( *this != comp ) return false;
    return true;
  }
  
  unsigned int mid;
  std::string ms;
  SimpleMember msm;
  std::vector<int> mv;
};

class D1 : public IBase {
  public:
  D1():
    IBase("D1"),mid(0),ms(""),msm(),mm(){
  }
    
  D1( unsigned int i):
    IBase("D1"),mid(i),ms(""),msm(i),mm(){
    std::ostringstream os;
    os << "D1_"<<i;    
    ms = os.str();
    for(int j=0;j<(int)i+1;j++){
      std::ostringstream os;
      os << "D1_K"<<j;    
      std::string k = os.str();
      mm.insert(std::make_pair(k,j));
    }
  }

    bool operator==(const D1& rhs){
      if(mid != rhs.mid) return false;
      if(ms != rhs.ms ) return false;
      if(msm != rhs.msm ) return false;
      if( mm != rhs.mm ) return false;
      return true;
    }
    bool operator!=(const D1& rhs){
      return !operator==(rhs);
    }
    
    bool equals( const IBase& rhs ){
      const D1& comp = dynamic_cast<const D1&>(rhs);
    if( !IBase::equals( rhs )) return false;
    if( *this != comp ) return false;
    return true;
  }
  
  unsigned int mid;
  std::string ms;
  SM msm;
  std::map<std::string,unsigned int> mm;
};

class D2: public IBase {
  public:
  D2():IBase("D2"),md(){
  }
  D2( unsigned int i ):IBase("D2"),md(i){
    for( unsigned j=0;j<100;j++ ){
      mca[j] = SM(j+i);
    }
  }
    
  virtual ~D2(){
  }

    bool operator==(const D2& rhs){
      if(md != rhs.md) return false;
      for( unsigned j=0;j<100;j++ ){
        if( mca[j] != rhs.mca[j] ) return false;
      }
      return true;
    }
    bool operator!=(const D2& rhs){
      return !operator==(rhs);
    }
  virtual bool equals( const IBase& rhs ){
    const D2& comp = dynamic_cast<const D2&>(rhs);
    if( !IBase::equals( rhs )) return false;
    if( *this != comp ) return false;
    return true;
  }

  SM md;
  SM mca[100];
};


class SG {

  public:
  
  SG():m_intData(0),m_ref(),m_ref2(){}
  SG(unsigned int id):m_intData(id),m_ref(),m_ref2(){
  }

  ~SG(){
  }
    
    
  bool operator==(const SG& rhs){
    if(m_intData!=rhs.m_intData) return false;
    if( !m_ref->equals( *rhs.m_ref )) {
      return false;
    }
    if( !m_ref2->equals( *rhs.m_ref2 )) {
      return false;
    }
    return true;
  }
  bool operator!=(const SG& rhs){
    return !operator==(rhs);
  }
  
  unsigned int m_intData;
  ora::UniqueRef<IBase> m_ref;
  ora::UniqueRef<IBase> m_ref2;
};



namespace {
   std::vector<double>::iterator dummy1;
   std::vector<int>::iterator dummy2;
   std::vector<int16_t>::iterator dummy3;
   std::vector<short>::iterator dummy4;
   std::vector<float>::iterator dummy5;
   std::vector<Entry>::iterator dummy6;
   std::vector<SiStripNoises::DetRegistry>::iterator dummy7;
   std::vector<std::vector<double> >::iterator dummy8;
   std::pair<unsigned int,SM> dummy9;
}

#endif
