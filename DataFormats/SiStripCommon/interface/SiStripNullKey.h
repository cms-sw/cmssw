
#ifndef DataFormats_SiStripCommon_SiStripNullKey_H
#define DataFormats_SiStripCommon_SiStripNullKey_H

#include "DataFormats/SiStripCommon/interface/ConstantsForGranularity.h"
#include "DataFormats/SiStripCommon/interface/SiStripKey.h"

class SiStripNullKey;

/** Debug info for SiStripNullKey class. */
std::ostream& operator<< ( std::ostream&, const SiStripNullKey& );

/**
   @class SiStripNullKey
   @author R.Bainbridge

   @brief Concrete implementation of abstract base, signifying null
   values or an "unknown" position or view.
*/
class SiStripNullKey : public SiStripKey {
  
 public:
  
  // ---------- Constructors ----------
  
  /** Default constructor */
  SiStripNullKey();

  // ---------- Utility methods ---------- 

  /** Identifies key objects with identical member data. */
  bool isEqual( const SiStripKey& ) const override;
  
  /** "Consistent" means identical and/or null (ie, "all") data. */
  bool isConsistent( const SiStripKey& ) const override;

  /** Identifies all member data as being "valid" or null ("all"). */
  bool isValid() const override;
  
  /** All member data to level of "Granularity" are valid. If
      sistrip::Granularity is "undefined", returns false. */
  bool isValid( const sistrip::Granularity& ) const override;
  
  /** Identifies all member data as being invalid. */
  bool isInvalid() const override;

  /** All member data to level of "Granularity" are invalid. If
      sistrip::Granularity is "undefined", returns true.  */
  bool isInvalid( const sistrip::Granularity& ) const override;

  // ---------- Print methods ----------
  
  /** Print member data of the key  */
  void print( std::stringstream& ss ) const override;
  
  /** A terse summary of the key  */
  void terse( std::stringstream& ss ) const override {;}
  
 private: 

  // ---------- Private methods ----------

  void initFromValue() override;
  void initFromKey() override;
  void initFromPath() override;
  void initGranularity() override;
  
};

#endif // DataFormats_SiStripCommon_SiStripNullKey_H



