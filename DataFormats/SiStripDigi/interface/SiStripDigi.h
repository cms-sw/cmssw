#ifndef DataFormats_SiStripDigi_SiStripDigi_H
#define DataFormats_SiStripDigi_SiStripDigi_H

#include <iosfwd>
#include <cstdint>

/**  
     @brief A Digi for the silicon strip detector, containing both
     strip and adc information, and suitable for storing
     zero-suppresed hit information.
*/
class SiStripDigi {
public:
  SiStripDigi(const uint16_t& strip, const uint16_t& adc) : strip_(strip), adc_(adc) { ; }

  SiStripDigi() : strip_(0), adc_(0) { ; }
  ~SiStripDigi() { ; }

  inline const uint16_t& strip() const;
  inline const uint16_t& adc() const;
  inline const uint16_t& channel() const;

  inline bool operator<(const SiStripDigi& other) const;

private:
  uint16_t strip_;
  uint16_t adc_;
};

std::ostream& operator<<(std::ostream& o, const SiStripDigi& digi);

// inline methods
const uint16_t& SiStripDigi::strip() const { return strip_; }
const uint16_t& SiStripDigi::adc() const { return adc_; }
const uint16_t& SiStripDigi::channel() const { return strip(); }
bool SiStripDigi::operator<(const SiStripDigi& other) const { return strip() < other.strip(); }

#endif  // DataFormats_SiStripDigi_SiStripDigi_H
