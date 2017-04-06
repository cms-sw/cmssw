#ifndef GEMDigi_ME0TriggerDigi_h
#define GEMDigi_ME0TriggerDigi_h

/**\class ME0TriggerDigi
 *
 * Digi for ME0 LCT trigger primitives. 
 *
 * \author Sven Dildick (TAMU)
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class ME0TriggerDigi 
{
 public:
  
  /// Constructors
  ME0TriggerDigi(const int trknmb, const int quality,
	     const int strip, const int pattern,
	     const int bend, const int bx);

  ME0TriggerDigi();                               /// default

  /// clear this LCT
  void clear();

  /// Print content of LCT digi
  void print() const;

  ///Comparison
  bool operator == (const ME0TriggerDigi &) const;
  bool operator != (const ME0TriggerDigi &rhs) const
  { return !(this->operator==(rhs)); }

  /// return track number
  int getTrknmb()  const { return trknmb; }

  /// return the Quality
  int getQuality() const { return quality; }

  /// return the key strip
  int getStrip()   const { return strip; }

  /// return pattern
  int getPattern() const { return pattern; }

  /// return bend
  int getBend()    const { return bend; }

  /// return BX
  int getBX()      const { return bx; }
	
  /// is valid?
  bool isValid() const { return pattern!=0; }

  /// Set track number.
  void setTrknmb(const uint16_t number) {trknmb = number;}

  /// set quality code
  void setQuality(unsigned int q) {quality=q;}

  /// set strip
  void setStrip(unsigned int s) {strip=s;}

  /// set pattern
  void setPattern(unsigned int p) {pattern=p;}

  /// set bend
  void setBend(unsigned int b) {bend=b;}

  /// set bx
  void setBX(unsigned int b) {bx=b;}

 private:
  uint16_t trknmb;
  uint16_t quality;
  uint16_t strip;
  uint16_t pattern;
  uint16_t bend;
  uint16_t bx;
};

std::ostream & operator<<(std::ostream & o, const ME0TriggerDigi& digi);

#endif
