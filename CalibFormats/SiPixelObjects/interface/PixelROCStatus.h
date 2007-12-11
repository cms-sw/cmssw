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
    enum status {off=0, noHits, nStatus};

  public:

    PixelROCStatus();
    PixelROCStatus(const std::set<status>& stat);
    virtual ~PixelROCStatus();

    std::string statusName(status stat);

    void set(status stat);
    void clear(status stat);
    void set(status stat, bool mode);
    void set(std::string stutus);
    bool get(status stat);
    

  private:

    uint32_t bits_;
 
  };
}
#endif
