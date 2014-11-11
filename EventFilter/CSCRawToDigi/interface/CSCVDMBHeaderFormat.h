#ifndef CSCVDMBHeaderFormat_h
#define CSCVDMBHeaderFormat_h

#include <cassert>
#include <iosfwd>
#include <string.h> // bzero

class CSCVDMBHeaderFormat  {
public:

  virtual ~CSCVDMBHeaderFormat() {};
/*
  void init() {
    bzero(this, sizeInWords()*2);
  }
*/
  
  virtual bool cfebAvailable(unsigned icfeb) = 0;

  virtual void addCFEB(int icfeb) = 0;
  virtual void addNCLCT() = 0;  
  virtual void addNALCT() = 0;
  virtual void setBXN(int bxn) = 0;
  virtual void setL1A(int l1a) = 0;
  virtual void setL1A24(int l1a) = 0;
  virtual void setCrateAddress(int crate, int dmbId) = 0;
  virtual void setdmbID(int newDMBID) = 0;
  virtual void setdmbVersion(unsigned int version) = 0;

  virtual unsigned cfebActive() const = 0;
  virtual unsigned crateID() const = 0;
  virtual unsigned dmbID() const = 0;
  virtual unsigned bxn() const = 0;
  virtual unsigned bxn12() const = 0;
  virtual unsigned l1a() const = 0;
  virtual unsigned l1a24() const = 0;
  virtual unsigned cfebAvailable() const = 0;
  virtual unsigned nalct() const = 0;
  virtual unsigned nclct() const = 0;
  virtual unsigned cfebMovlp() const = 0;
  virtual unsigned dmbCfebSync() const = 0;
  virtual unsigned activeDavMismatch() const = 0;
  virtual unsigned format_version() const = 0;

  virtual unsigned sizeInWords() const = 0;
 
  virtual bool check() const = 0;

  virtual unsigned short * data() = 0;
  virtual unsigned short * data() const = 0;

  //ostream & operator<<(ostream &, const CSCVDMBHeaderFormat &);

};

#endif

