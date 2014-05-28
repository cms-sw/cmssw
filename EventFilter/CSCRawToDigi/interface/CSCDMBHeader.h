#ifndef CSCDMBHeader_h
#define CSCDMBHeader_h

#include <cassert>
#include <iosfwd>
#include <string.h> // bzero
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCVDMBHeaderFormat.h"
#include <boost/shared_ptr.hpp>

class CSCDMBHeader2005;
class CSCDMBHeader2013;

class CSCDMBHeader  {
public:
  
  CSCDMBHeader(uint16_t firmware_version = 2005);
  
  CSCDMBHeader(unsigned short * buf, uint16_t firmware_version = 2005);

  CSCDMBHeader(const CSCDMBStatusDigi & digi)
    {
      memcpy(this, digi.header(), sizeInWords()*2);
    }



  bool cfebAvailable(unsigned icfeb) { 
	assert (icfeb < (theFirmwareVersion==2013)?7:5);
	return (theHeaderFormat->cfebAvailable() >> icfeb) & 1;
    }

  void addCFEB(int icfeb) { theHeaderFormat->addCFEB(icfeb); }
  void addNCLCT() { theHeaderFormat->addNCLCT();};  

  void addNALCT() { theHeaderFormat->addNALCT(); };
  void setBXN(int bxn) { theHeaderFormat->setBXN(bxn); };
  void setL1A(int l1a) { theHeaderFormat->setL1A(l1a); };
  void setL1A24(int l1a) { theHeaderFormat->setL1A24(l1a); };
  void setCrateAddress(int crate, int dmbId) { theHeaderFormat->setCrateAddress(crate, dmbId); };
  void setdmbID(int newDMBID) { theHeaderFormat->setdmbID(newDMBID); };
  void setdmbVersion(unsigned int version) { theHeaderFormat->setdmbVersion(version); };

  unsigned cfebActive() const { return theHeaderFormat->cfebActive(); }; 
  unsigned crateID() const { return theHeaderFormat->crateID(); };
  unsigned dmbID() const { return theHeaderFormat->dmbID(); };
  unsigned bxn() const { return theHeaderFormat->bxn(); };
  unsigned bxn12() const { return theHeaderFormat->bxn12(); };
  unsigned l1a() const { return theHeaderFormat->l1a(); };
  unsigned l1a24() const { return theHeaderFormat->l1a24(); };
  unsigned cfebAvailable() const { return theHeaderFormat->cfebAvailable(); };
  unsigned nalct() const { return theHeaderFormat->nalct(); };
  unsigned nclct() const { return theHeaderFormat->nclct(); };
  unsigned cfebMovlp() const { return theHeaderFormat->cfebMovlp(); };
  unsigned dmbCfebSync() const { return theHeaderFormat->dmbCfebSync(); };
  unsigned activeDavMismatch() const { return theHeaderFormat->activeDavMismatch(); };

  unsigned sizeInWords() const { return theHeaderFormat->sizeInWords(); };
  unsigned format_version() const { return theHeaderFormat->format_version(); };
 
  bool check() const { return theHeaderFormat->check(); };

  unsigned short * data() {return theHeaderFormat->data(); };
  unsigned short * data() const {return theHeaderFormat->data(); };


  //ostream & operator<<(ostream &, const CSCDMBHeader &);
  
  /// will throw if the cast fails
  CSCDMBHeader2005 dmbHeader2005()   const;
  CSCDMBHeader2013 dmbHeader2013()   const;


 private:

  boost::shared_ptr<CSCVDMBHeaderFormat> theHeaderFormat;
  int theFirmwareVersion;


};

#endif

