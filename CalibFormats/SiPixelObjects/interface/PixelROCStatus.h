#ifndef PixelROCStatus_h
#define PixelROCStatus_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelROCStatus.h
*   \brief This class keeps the possible non-standard status a ROC can have.
*
*    A longer explanation will be placed here later
*/

#include <cstdint>
#include <set>
#include <string>

namespace pos {

  /*! \class PixelROCStatus PixelROCStatus.h "interface/PixelROCStatus.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelROCStatus {
  private:
    uint32_t bits_;

  public:
    //Insert new status before nStatus
    enum ROCstatus { off = 0, noHits, noInit, noAnalogSignal, nStatus };

    PixelROCStatus();
    PixelROCStatus(const std::set<ROCstatus>& stat);
    virtual ~PixelROCStatus();

    std::string statusName(ROCstatus stat) const;
    std::string statusName() const;

    void set(ROCstatus stat);
    void clear(ROCstatus stat);
    void set(ROCstatus stat, bool mode);
    void set(const std::string& statName);
    bool get(ROCstatus stat) const;

    // Added by Dario (March 4th 2008)
    void reset(void);
  };
}  // namespace pos
#endif
