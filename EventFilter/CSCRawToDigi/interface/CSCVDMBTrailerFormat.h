#ifndef CSCVDMBTrailerFormat_h
#define CSCVDMBTrailerFormat_h

#include <cassert>
#include <iosfwd>
#include <string.h> // bzero

class CSCDMBHeader;

class CSCVDMBTrailerFormat  {
public:

  virtual ~CSCVDMBTrailerFormat() {};
/*
  void init() {
    bzero(this, sizeInWords()*2);
  }
*/
  virtual void setEventInformation(const CSCDMBHeader &) = 0;

  virtual unsigned crateID() const = 0;
  virtual unsigned dmbID() const = 0;

  virtual unsigned dmb_l1a() const = 0;
  virtual unsigned dmb_bxn() const = 0;

  virtual unsigned alct_endtimeout() const = 0;
  virtual unsigned tmb_endtimeout() const = 0;
  virtual unsigned cfeb_endtimeout() const = 0;

  virtual unsigned alct_starttimeout() const = 0;
  virtual unsigned tmb_starttimeout() const = 0;
  virtual unsigned cfeb_starttimeout() const = 0;

  virtual unsigned cfeb_movlp() const = 0;
  virtual unsigned dmb_l1pipe() const = 0;

  virtual unsigned alct_empty() const = 0;
  virtual unsigned tmb_empty() const = 0;
  virtual unsigned cfeb_empty() const = 0;

  virtual unsigned alct_half() const = 0;
  virtual unsigned tmb_half() const = 0;
  virtual unsigned cfeb_half() const = 0;

  virtual unsigned alct_full() const = 0;
  virtual unsigned tmb_full() const = 0;
  virtual unsigned cfeb_full() const = 0;

  virtual unsigned crc22() const = 0;
  virtual unsigned crc_lo_parity() const = 0;
  virtual unsigned crc_hi_parity() const = 0;


  virtual unsigned short * data() = 0;
  virtual unsigned short * data() const = 0;

  virtual bool check() const = 0;

  virtual unsigned sizeInWords() const = 0;
  
  //ostream & operator<<(ostream &, const CSCVDMBTrailerFormat &);

};

#endif

