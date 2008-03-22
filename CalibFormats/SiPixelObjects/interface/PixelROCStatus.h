#ifndef PixelROCStatus_h
#define PixelROCStatus_h
//
// This class keeps the possible non-standard
// status a ROC can have.
//
//
//

#include <stdint.h>
#include <set>
#include <string>

namespace pos{

  class PixelROCStatus {

    //Insert new status before nStatus
    enum ROCstatus {off=0, noHits, nStatus};

  public:

    PixelROCStatus();
    PixelROCStatus(const std::set<ROCstatus>& stat);
    virtual ~PixelROCStatus();

    std::string statusName(ROCstatus stat);

    void set(ROCstatus stat);
    void clear(ROCstatus stat);
    void set(ROCstatus stat, bool mode);
    void set(const std::string& statName);
    bool get(ROCstatus stat);
    

  private:

    uint32_t bits_;
 
  };
}
#endif
