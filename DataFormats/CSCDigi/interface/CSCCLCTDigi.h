#ifndef CSCCLCTDigi_CSCCLCTDigi_h
#define CSCCLCTDigi_CSCCLCTDigi_h

/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives. 
 *
 * $Date:$
 * $Revision:$
 *
 * \author N. Terentiev, CMU
 */

class CSCCLCTDigi{

public:

  /// Constructors
  /// for consistency with DQM
  explicit CSCCLCTDigi (int valid, int quality, int patshape, int striptype,int bend,  int strip, int cfeb, int bx); 
  explicit CSCCLCTDigi (int valid, int quality, int patshape, int striptype, int bend,  int strip, int cfeb, int bx, int trknmb);  
  /// copy
  CSCCLCTDigi (const CSCCLCTDigi& digi);
  /// default
  CSCCLCTDigi ();

  /// Assignment operator
  CSCCLCTDigi& operator=(const CSCCLCTDigi& digi);

  /// return CLCT validity, 1 - valid ALCT
  int getValid() const {return valid_ ;}
  /// check CLCT validity (for consistency with ORCA)
  bool isValid()     const {return valid_ ;}
  /// return quality of a pattern
  int getQuality()   const {return quality_ ;}
  /// return pattern
  int getPattern()   const {return patshape_ ;}
  /// return striptype (obsolete, use getStripType() instead)
  int getStriptype() const {return striptype_ ;}   
  /// return striptype (compatible with ORCA)
  int getStripType() const {return striptype_ ;}
  /// return bend
  int getBend()      const {return bend_ ;}
  /// return strip     
  int getStrip()     const {return strip_ ;}

  /// Convert strip and CFEB to keyStrip. Each CFEB has up to 16 strips
  /// (32 halfstrips). There are 5 cfebs.  The "strip" variable is one
  /// of 32 halfstrips on the keylayer of a single CFEB, so that
  /// Distrip   = (cfeb*32 + strip)/4.
  /// Halfstrip = (cfeb*32 + strip).
  int getKeyStrip()  const {
                int keyStrip = 0;
                if (striptype_ == 1)
                        keyStrip = cfeb_ * 32 + strip_;
                else
                        keyStrip = cfeb_ * 8  + strip_/4;
                return keyStrip;
  }
  /// return Key CFEB ID (obsolete, use getCFEB()  instead)
  int getCfeb()      const {return cfeb_ ;}
  /// return Key CFEB ID (compatible with ORCA)
  int getCFEB()      const {return cfeb_ ;} 
  /// return BX (obsolete, use  getBX())
  int getBx()        const {return bx_ ;}
  /// return BX (compatible with ORCA)
  int getBX()        const {return bx_ ;}
  /// return track number (1,2)
  int getTrknmb()    const {return trknmb_ ;}

  /// Print content of digi
  void print() const;

private:
  friend class testCSCDigis;
  unsigned int valid_      ;
  unsigned int quality_    ;
  unsigned int patshape_   ;
  unsigned int striptype_  ;
  unsigned int bend_       ;
  unsigned int strip_      ;
  unsigned int cfeb_       ;
  unsigned int bx_         ;
  unsigned int trknmb_     ;
};

#include<iostream>
  /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCCLCTDigi& digi) {
  return o << " CSC CLCT isValid "      << digi.isValid()
           << " CSC CLCT Quality "      << digi.getQuality()
           << " CSC CLCT Pattern "      << digi.getPattern()
           << " CSC CLCT StripType "    << digi.getStripType()
           << " CSC CLCT Bend "         << digi.getBend()
           << " CSC CLCT Strip "        << digi.getStrip()
           << " CSC CLCT KeyStrip "     << digi.getKeyStrip()
           << " CSC CLCT CFEB "         << digi.getCFEB()
	   << " CSC CLCT BX "           << digi.getBX()
           << " CSC CLCT track number " << digi.getTrknmb();
}
#endif
