#ifndef Cond_IOVUserMetaData_h
#define Cond_IOVUserMetaData_h

namespace cond {

  class IOVUserMetaData {
  public:
    IOVUserMetaData(){}
    virtual ~ IOVUserMetaData(){}
    virtual IOVUserMetaData * clone() const { return new  IOVUserMetaData(*this);}

  private:
  };


}

#endif
