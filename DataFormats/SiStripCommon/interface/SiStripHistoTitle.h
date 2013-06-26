// Last commit: $Id: SiStripHistoTitle.h,v 1.4 2007/07/31 15:20:24 ratnik Exp $

#ifndef DataFormats_SiStripCommon_SiStripHistoTitle_H
#define DataFormats_SiStripCommon_SiStripHistoTitle_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <ostream>
#include <sstream>
#include <string>

class SiStripKey;
class SiStripHistoTitle;

/** Debug info for class. */
std::ostream& operator<< ( std::ostream&, const SiStripHistoTitle& );

/** 
    @author R.Bainbridge
    @brief Utility class that holds histogram title.
*/
class SiStripHistoTitle {

 public:
  
  // ---------- Constructors, destructors ----------
  
  /** Constructs histogram title from key object. */
  SiStripHistoTitle( const sistrip::HistoType& histo_type, 
		     const sistrip::RunType&   run_type, 
		     const SiStripKey&         key,
		     const std::string&        extra_info = "" );
  
  /** Constructs histogram title from various data. */
  SiStripHistoTitle( const sistrip::HistoType&   histo_type, 
		     const sistrip::RunType&     run_type, 
		     const sistrip::KeyType&     key_type,
		     const uint32_t&             key_value,
		     const sistrip::Granularity& gran,
		     const uint16_t&             channel,
		     const std::string&          extra_info = "" );
  
  /** Extracts individual components from histogram title. */
  SiStripHistoTitle( const std::string& histo_title );

  // ---------- Public interface ----------
  
  /** Returns the histogram title. */ 
  inline const std::string& title() const;
  
  /** Returns the histogram type. */ 
  inline const sistrip::HistoType& histoType() const;
  
  /** Returns the run type. */ 
  inline const sistrip::RunType& runType() const;
  
  /** Defines key type used to form the histogram title. */
  inline const sistrip::KeyType& keyType() const; 
  
  /** Returns values of the 32-bit key. */
  inline const uint32_t& keyValue() const;

  /** Returns granularity of histogram. */
  inline const sistrip::Granularity& granularity() const;
  
  /** Returns channel for histogram granularity. */
  inline const uint16_t& channel() const;
  
  /** Extra information attached to histogram title. */
  inline const std::string& extraInfo() const;
  
 private:

  // ---------- Private methods ----------
  
  /** Private default constructor. */
  SiStripHistoTitle() {;}

  /** Constructs histogram title. */
  void setTitle();

  /** Extracts member data values from title. */
  void extractTitle();
  
  // ---------- Private member data ----------

  /** Histogram title. */
  std::string title_;

  /** Defines histo type. */
  sistrip::HistoType histoType_;

  /** Defines run type. */
  sistrip::RunType runType_;

  /** Defines key type. */
  sistrip::KeyType keyType_; 

  /** Key value. */
  uint32_t keyValue_;

  /** Granularity of histogram. */
  sistrip::Granularity granularity_;

  /**Channel number for granularity. */
  uint16_t channel_;

  /** Extra information to be attached to title. */
  std::string extraInfo_;
  
};

// ---------- inline methods ----------

const std::string& SiStripHistoTitle::title() const { return title_; }
const sistrip::HistoType& SiStripHistoTitle::histoType() const { return histoType_; }
const sistrip::RunType& SiStripHistoTitle::runType() const { return runType_; }
const sistrip::KeyType& SiStripHistoTitle::keyType() const { return keyType_; } 
const uint32_t& SiStripHistoTitle::keyValue() const { return keyValue_; }
const sistrip::Granularity& SiStripHistoTitle::granularity() const { return granularity_; }
const uint16_t& SiStripHistoTitle::channel() const { return channel_; }
const std::string& SiStripHistoTitle::extraInfo() const { return extraInfo_; }

#endif // DataFormats_SiStripCommon_SiStripHistoTitle_H


