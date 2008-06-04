#ifndef CondFormats_simpleInheritance_h
#define CondFormats_simpleInheritance_h
class mybase{
 public:
  mybase(){}
};
class child:public mybase{
 public:
  child(){}
  int b;
};
#endif 
