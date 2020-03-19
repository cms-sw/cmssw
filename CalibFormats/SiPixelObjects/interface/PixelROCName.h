#ifndef PixelROCName_h
#define PixelROCName_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelROCName.h
*   \brief This class stores the name and related hardware mappings for a ROC
*
*    A longer explanation will be placed here later
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

namespace pos {
  /*! \class PixelROCName PixelROCName.h "interface/PixelROCName.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelROCName;
  std::ostream& operator<<(std::ostream& s, const PixelROCName& pixelroc);

  class PixelROCName {
  public:
    PixelROCName();

    explicit PixelROCName(std::string rocname);

    explicit PixelROCName(std::ifstream& s);

    std::string rocname() const;

    char detsub() const { return (id_ & 0x80000000) ? 'B' : 'F'; }
    char mp() const { return id_ & 0x40000000 ? 'p' : 'm'; }
    char IO() const { return id_ & 0x20000000 ? 'I' : 'O'; }
    int roc() const { return id_ & 0xf; }

    //These methods only for FPix
    int disk() const {
      assert((id_ & 0x80000000) == 0);
      return (id_ >> 12) & 0x3;
    }
    int blade() const {
      assert((id_ & 0x80000000) == 0);
      return (id_ >> 7) & 0x1f;
    }
    int panel() const {
      assert((id_ & 0x80000000) == 0);
      return ((id_ >> 6) & 0x1) + 1;
    }
    int plaquet() const {
      assert((id_ & 0x80000000) == 0);
      return ((id_ >> 4) & 0x3) + 1;
    }

    //These methods only for BPix
    int sec() const {
      assert((id_ & 0x80000000) != 0);
      return ((id_ >> 14) & 0x7) + 1;
    }
    int layer() const {
      assert((id_ & 0x80000000) != 0);
      return (id_ >> 12) & 0x3;
    }
    int ladder() const {
      assert((id_ & 0x80000000) != 0);
      return (id_ >> 6) & 0x1f;
    }
    char HF() const {
      assert((id_ & 0x80000000) != 0);
      return id_ & 0x00000800 ? 'F' : 'H';
    }
    int module() const {
      assert((id_ & 0x80000000) != 0);
      return ((id_ >> 4) & 0x3) + 1;
    }

    friend std::ostream& pos::operator<<(std::ostream& s, const PixelROCName& pixelroc);

    const PixelROCName& operator=(const PixelROCName& aROC);

    const bool operator<(const PixelROCName& aROC) const { return id_ < aROC.id_; }

    const bool operator==(const PixelROCName& aROC) const { return id_ == aROC.id_; }

    unsigned int id() const { return id_; }

  private:
    void parsename(std::string name);

    void check(bool check, const std::string& name);

    void setIdFPix(char np, char LR, int disk, int blade, int panel, int plaquet, int roc);

    void setIdBPix(char np, char LR, int sec, int layer, int ladder, char HF, int module, int roc);

    //BPix_BpI_SEC1_LYR1_LDR3F_MOD1_ROC0

    //The id_ holds the following values for BPix
    //bit [0,1,2,3] the ROC #
    //bit [4,5] the module#
    //bit [6,7,8,9,10] the ladder#
    //bit [11] H or F (0 or 1)#
    //bit [12,13] the layer#
    //bit [14,15,16] the section#
    //bit [29] I or 0 (0 or 1)
    //bit [30] m or p (0 or 1)
    //bit [31] = 1

    //FPix_BpI_D1_BLD1_PNL1_PLQ1_ROC1

    //The id_ holds the following values for FPix
    //bit [0,1,2,3] the ROC #
    //bit [4,5] the plaquet#
    //bit [6] the panel#
    //bit [7,8,9,10,11] the blade#
    //bit [12,13] the disk#
    //bit [29] I or O (0 or 1)
    //bit [30] m or p (0 or 1)
    //bit [31] = 0

    unsigned int id_;
  };
}  // namespace pos
#endif
