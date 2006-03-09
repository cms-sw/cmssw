#ifndef DataFormats_SiStripEventSummary_SiStripEventSummary_H
#define DataFormats_SiStripEventSummary_SiStripEventSummary_H

#include "boost/cstdint.hpp"
#include <string>

using namespace std;

/**  
     @brief A simple container class for generic event-related info.
*/
class SiStripEventSummary {

 public:

  SiStripEventSummary();
  ~SiStripEventSummary();

  // getters
  inline const string& task() const { return task_; }
  inline const uint16_t& apveAddress() const { return apveAddress_; }
  inline const uint32_t& nApvsInSync() const { return nApvsInSync_; }
  inline const uint32_t& nApvsOutOfSync() const { return nApvsOutOfSync_; }
  inline const uint32_t& nApvsErrors() const { return nApvsErrors_; }

  // setters
  inline void task( string& task ) { task_ = task; }
  inline void apveAddress( uint16_t& addr ) { apveAddress_ = addr; }
  inline void nApvsInSync( uint32_t& in_sync ) { nApvsInSync_ = in_sync; }
  inline void nApvsOutOfSync( uint32_t& out_of_sync ) { nApvsOutOfSync_ = out_of_sync; }
  inline void nApvsErrors( uint32_t& errors ) { nApvsErrors_ = errors; }
  
 private:

  // Commissioning task
  string task_;

  // APV synchronization and errors
  uint16_t apveAddress_;
  uint32_t nApvsInSync_;
  uint32_t nApvsOutOfSync_;
  uint32_t nApvsErrors_;
  
};

#endif // DataFormats_SiStripEventSummary_SiStripEventSummary_H



