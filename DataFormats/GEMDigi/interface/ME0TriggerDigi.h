#ifndef GEMDigi_ME0TriggerDigi_h
#define GEMDigi_ME0TriggerDigi_h

/**\class ME0TriggerDigi
 *
 * Digi for local ME0 trigger system
 *
 * \author Sven Dildick (TAMU)
 */

#include <cstdint>
#include <iosfwd>

class ME0TriggerDigi 
{
 public:
  
  /// Constructors
  ME0TriggerDigi(const int trknmb, const int quality,
		 const int strip, const int partition, 
		 const int pattern,
		 const int bend, const int bx);
  
  /// default
  ME0TriggerDigi();                               

  /// clear this Trigger
  void clear();

  ///Comparison
  bool operator == (const ME0TriggerDigi &) const;
  bool operator != (const ME0TriggerDigi &rhs) const
  { return !(this->operator==(rhs)); }

  /// return track number
  int getTrknmb()  const { return trknmb_; }

  /// return the Quality
  int getQuality() const { return quality_; }

  /// return the key strip
  int getStrip()   const { return strip_; }

  /// return the key "partition"
  int getPartition()   const { return partition_; }

  /// return pattern
  int getPattern() const { return pattern_; }

  /// return bend
  int getBend()    const { return bend_; }

  /// return BX
  int getBX()      const { return bx_; }
	
  /// is valid?
  bool isValid() const { return pattern_!=0; }

  /// Set track number.
  void setTrknmb(const uint16_t number) {trknmb_ = number;}

  /// set quality code
  void setQuality(unsigned int q) {quality_=q;}

  /// set strip
  void setStrip(unsigned int s) {strip_=s;}

  /// set partition
  void setPartition(unsigned int s) {partition_=s;}

  /// set pattern
  void setPattern(unsigned int p) {pattern_=p;}

  /// set bend
  void setBend(unsigned int b) {bend_=b;}

  /// set bx
  void setBX(unsigned int b) {bx_=b;}

 private:
  uint16_t trknmb_;
  uint16_t quality_;
  uint16_t strip_;
  uint16_t partition_;
  uint16_t pattern_;
  uint16_t bend_;
  uint16_t bx_;
};

std::ostream & operator<<(std::ostream & o, const ME0TriggerDigi& digi);

#endif
