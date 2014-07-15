#ifndef CSCDMBTrailer_h
#define CSCDMBTrailer_h

#include <cassert>
#include <iosfwd>
#include <string.h> // bzero
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/CSCDigi/interface/CSCDMBStatusDigi.h"
#include "EventFilter/CSCRawToDigi/interface/CSCVDMBTrailerFormat.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include <boost/shared_ptr.hpp>


// class CSCDMBHeader;
class CSCDMBTrailer2005;
class CSCDMBTrailer2013;

class CSCDMBTrailer {
public:

  CSCDMBTrailer(uint16_t firmware_version = 2005);

  CSCDMBTrailer(unsigned short * buf, uint16_t firmware_version = 2005);
  
  CSCDMBTrailer(const CSCDMBStatusDigi & digi) 
    {
      memcpy(this, digi.trailer(), sizeInWords()*2);
    }


  ///@@ NEEDS TO BE DONE
  void setEventInformation(const CSCDMBHeader & header) { return theTrailerFormat->setEventInformation(header); };

  unsigned crateID() const { return theTrailerFormat->crateID(); };
  unsigned dmbID() const { return theTrailerFormat->dmbID(); };

  unsigned dmb_l1a() const { return theTrailerFormat->dmb_l1a(); };
  unsigned dmb_bxn() const { return theTrailerFormat->dmb_bxn(); };

  unsigned alct_endtimeout() const { return theTrailerFormat->alct_endtimeout(); };
  unsigned tmb_endtimeout() const { return theTrailerFormat->tmb_endtimeout(); };
  unsigned cfeb_endtimeout() const { return theTrailerFormat->cfeb_endtimeout(); };

  unsigned alct_starttimeout() const { return theTrailerFormat->alct_starttimeout(); };
  unsigned tmb_starttimeout() const { return theTrailerFormat->tmb_starttimeout(); };
  unsigned cfeb_starttimeout() const { return theTrailerFormat->cfeb_starttimeout(); };

  unsigned cfeb_movlp() const { return theTrailerFormat->cfeb_movlp(); };
  unsigned dmb_l1pipe() const { return theTrailerFormat->dmb_l1pipe(); };

  unsigned alct_empty() const { return theTrailerFormat->alct_empty(); };
  unsigned tmb_empty() const { return theTrailerFormat->tmb_empty(); };
  unsigned cfeb_empty() const { return theTrailerFormat->cfeb_empty(); };

  unsigned alct_half() const { return theTrailerFormat->alct_half(); };
  unsigned tmb_half() const { return theTrailerFormat->tmb_half(); };
  unsigned cfeb_half() const { return theTrailerFormat->cfeb_half(); };

  unsigned alct_full() const { return theTrailerFormat->alct_full(); };
  unsigned tmb_full() const {return theTrailerFormat->tmb_full(); };
  unsigned cfeb_full() const { return theTrailerFormat->cfeb_full(); };
  
  unsigned crc22() const { return theTrailerFormat->crc22(); };
  unsigned crc_lo_parity() const { return theTrailerFormat->crc_lo_parity(); };
  unsigned crc_hi_parity() const { return theTrailerFormat->crc_hi_parity(); };
 
  unsigned short * data() { return theTrailerFormat->data(); };
  unsigned short * data() const { return theTrailerFormat->data(); };

  unsigned sizeInWords() const { return theTrailerFormat->sizeInWords(); };

  bool check() const { return theTrailerFormat->check(); };

  /// will throw if the cast fails
  CSCDMBTrailer2005 dmbTrailer2005() const;
  CSCDMBTrailer2013 dmbTrailer2013() const;  


 private:
  
  boost::shared_ptr<CSCVDMBTrailerFormat> theTrailerFormat;
  int theFirmwareVersion;

};

#endif

