#ifndef SiPixelObjects_PixelROC_H
#define SiPixelObjects_PixelROC_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/SiPixelObjects/interface/FrameConversion.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include <string>
#include <cstdint>

/** \class PixelROC
 * Represents ReadOut Chip of DetUnit. 
 * Converts pixel coordinates from Local (in ROC) to Global (in DetUnit).
 * The Local coordinates are double column (dcol) and pixel index in dcol.
 * The Global coordinates are row and column in DetUnit.
 */

//class TrackerTopology;

namespace sipixelobjects {

  class PixelROC {
  public:
    /// dummy
    PixelROC() : theDetUnit(0), theIdDU(0), theIdLk(0) {}

    /// ctor with DetUnit id,
    /// ROC number in DU (given by token passage),
    /// ROC number in Link (given by token passage),
    PixelROC(uint32_t du, int idInDU, int idLk);

    /// return the DetUnit to which this ROC belongs to.
    uint32_t rawId() const { return theDetUnit; }

    /// id of this ROC in DetUnit etermined by token path
    unsigned int idInDetUnit() const { return theIdDU; }

    /// id of this ROC in parent Link.
    unsigned int idInLink() const { return theIdLk; }

    /// converts DU position to local.
    /// If GlobalPixel is outside ROC the resulting LocalPixel is not inside ROC.
    /// (call to inside(..) recommended)
    LocalPixel toLocal(const GlobalPixel& glo) const {
      int rocRow = theFrameConverter.row().inverse(glo.row);
      int rocCol = theFrameConverter.collumn().inverse(glo.col);

      LocalPixel::RocRowCol rocRowCol = {rocRow, rocCol};
      return LocalPixel(rocRowCol);
    }

    /// converts LocalPixel in ROC to DU coordinates.
    /// LocalPixel must be inside ROC. Otherwise result is meaningless
    GlobalPixel toGlobal(const LocalPixel& loc) const {
      GlobalPixel result;
      result.col = theFrameConverter.collumn().convert(loc.rocCol());
      result.row = theFrameConverter.row().convert(loc.rocRow());
      return result;
    }

    // recognise the detector side and layer number
    // this methods use hardwired constants
    // if the numberg changes the methods have to be modified
    int bpixSidePhase0(uint32_t rawId) const;
    int fpixSidePhase0(uint32_t rawId) const;
    int bpixSidePhase1(uint32_t rawId) const;
    int fpixSidePhase1(uint32_t rawId) const;
    static int bpixLayerPhase1(uint32_t rawId);

    /// printout for debug
    std::string print(int depth = 0) const;

    void initFrameConversion();
    void initFrameConversionPhase1();
    //void initFrameConversion(const TrackerTopology *tt, bool phase1=false);
    // Frame conversion compatible with CMSSW_9_0_X Monte Carlo samples
    void initFrameConversionPhase1_CMSSW_9_0_X();

  private:
    uint32_t theDetUnit;
    unsigned int theIdDU, theIdLk;
    FrameConversion theFrameConverter COND_TRANSIENT;

    COND_SERIALIZABLE;
  };

}  // namespace sipixelobjects

#endif
