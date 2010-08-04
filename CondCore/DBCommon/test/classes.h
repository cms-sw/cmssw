#include "CondCore/DBCommon/test/testCondObj.h"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <map>


namespace testDBCommon {

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
  BaseClass():m_f(0.),m_s("-"),m_objectData(){
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

  SimpleClass():BaseClass(),m_intData(0),m_stringData("-"),m_objectData(),m_arrayData(),m_map(),m_code(ZERO){}
  SimpleClass(unsigned int id):BaseClass(id),m_intData(id),m_stringData(""),m_objectData(),m_arrayData(),m_map(),m_code(ZERO){
  //SimpleClass():BaseClass(),m_intData(0),m_stringData(""),m_objectData(),m_code(ZERO){}
  //SimpleClass(unsigned int id):BaseClass(id),m_intData(id),m_stringData(""),m_objectData(),m_code(ZERO){
    m_intData = id;
    std::ostringstream os;
    os << "SimpleClass_"<<id; 
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

    SimpleClass(const SimpleClass& obj):
      BaseClass( obj ){
    m_intData = obj.m_intData;
    m_stringData = obj.m_stringData;
    m_objectData = obj.m_objectData;
    m_arrayData = obj.m_arrayData;
    m_map = obj.m_map;
    m_code = obj.m_code;
  }
      SimpleClass& operator=(const SimpleClass& obj){
        BaseClass::operator=( obj );
    m_intData = obj.m_intData;
    m_stringData = obj.m_stringData;
    m_objectData = obj.m_objectData;
    m_arrayData = obj.m_arrayData;
    m_map = obj.m_map;
    m_code = obj.m_code;
    return *this;
  }

  bool operator==(const SimpleClass& rhs){
    if(BaseClass::operator!=(rhs)) return false;
    if(m_intData!=rhs.m_intData) return false;
    if(m_stringData!=rhs.m_stringData) return false;
    if(m_objectData.m_flag!=rhs.m_objectData.m_flag) return false;
    if(m_objectData.m_id!=rhs.m_objectData.m_id) return false;
    if(m_arrayData!=rhs.m_arrayData) return false;
    if(m_map!=rhs.m_map) return false;
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
  std::vector<int> m_arrayData;
  std::map<unsigned long long,unsigned long long> m_map;
  MySimpleClassCode m_code;
};



}  // namespace testDBCommon

using namespace testDBCommon;
