#ifndef Cond_IOVDescription_h
#define Cond_IOVDescription_h

namespace cond {

  class IOVDescription {
  public:
    IOVDescription(){}
    virtual ~ IOVDescription(){}
    virtual IOVDescription * clone() const { return new  IOVDescription(*this);}

  private:
  };


}

#endif
