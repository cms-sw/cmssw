#ifndef CondCore_ORA_AnyData_H
#define CondCore_ORA_AnyData_H
#include<string>

namespace ora {
  union AnyData {
    
    // size is 8 byte no matter which architecture
    char v[8];
    bool b;
    char c;
    unsigned char uc;
    short s;
    unsigned short us;
    int i;
    unsigned int ui;
    long long l;
    unsigned long long ul;
    float f;
    double d;
    void * p;
    char * ss;
    // std::string ss;
    void * address() { return v;}
    void const * address() const { return v;}

    // for generic type T better be the pointer to it...
    template<typename T> 
    inline T & data() { return reinterpret_cast<T&>(p);}   

    template<typename T> 
    inline T data() const { return reinterpret_cast<T>(p);}   
    
  };


    template<> 
    inline int & AnyData::data<int>() { return i;}   

    template<> 
    inline int AnyData::data<int>() const { return i;}   


  
}

#endif //  CondCore_ORA_AnyData_H
