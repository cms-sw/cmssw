#ifndef RecoTracker_MkFitCMS_interface_LayerNumberConverter_h
#define RecoTracker_MkFitCMS_interface_LayerNumberConverter_h

#include <stdexcept>

namespace mkfit {

  enum struct TkLayout { phase0 = 0, phase1 = 1, phase2 = 2 };

  class LayerNumberConverter {
  public:
    LayerNumberConverter(TkLayout layout) : lo_(layout) {}
    void reset(TkLayout layout) { lo_ = layout; }
    unsigned int nLayers() const {
      if (lo_ == TkLayout::phase0)
        return 69;
      if (lo_ == TkLayout::phase1)
        return 72;
      if (lo_ == TkLayout::phase2)
        return 60;  // 4 + 12 + 2*(12 + 10) = 16 + 22 + 22 = 60
      return 10;
    }
    TkLayout getEra() const { return lo_; }
    int convertLayerNumber(int det, int lay, bool useMatched, int isStereo, bool posZ) const {
      if (lo_ == TkLayout::phase2) {
        if (det == 1)
          return lay - 1;
        if (det == 2)
          return 16 + lay - 1 + (posZ ? 0 : 22);
        if (det == 5)
          return 4 + (2 * (lay - 1)) + isStereo;
        if (det == 4)
          return 16 + 12 + (2 * (lay - 1)) + isStereo + (posZ ? 0 : 22);
        throw std::runtime_error("bad subDet");
      }

      if (det == 1 || det == 3 || det == 5) {
        return convertBarrelLayerNumber(det, lay, useMatched, isStereo);
      } else {
        int disk = convertDiskNumber(det, lay, useMatched, isStereo);
        if (disk < 0)
          return -1;

        int lOffset = 0;
        if (lo_ == TkLayout::phase1)
          lOffset = 1;
        disk += 17 + lOffset;
        if (!posZ)
          disk += 25 + 2 * lOffset;
        return disk;
      }
      return -1;
    }
    bool doesHaveStereo(int det, int lay) const {
      if (lo_ == TkLayout::phase2) {
        if (det == 1 || det == 2)
          return false;
        if (det == 4 || det == 5)
          return true;
        throw std::runtime_error("bad subDet");
      }
      if (det == 3 || det == 5) {
        return lay == 1 || lay == 2;
      }
      if (det == 4 || det == 6) {
        return true;
      }
      return false;
    }

    int convertBarrelLayerNumber(int cmsswdet, int cmsswlay, bool useMatched, int isStereo) const {
      int lOffset = 0;
      if (lo_ == TkLayout::phase1)
        lOffset = 1;
      if (cmsswdet == 2 || cmsswdet == 4 || cmsswdet == 6)
        return -1;  //FPIX, TID, TEC
      if (cmsswdet == 1)
        return cmsswlay - 1;  //BPIX
      if (useMatched) {
        //TIB
        if (cmsswdet == 3) {
          if (cmsswlay == 1 && isStereo == -1)
            return 3 + lOffset;
          else if (cmsswlay == 2 && isStereo == -1)
            return 4 + lOffset;
          else if (cmsswlay == 3 && isStereo == 0)
            return 5 + lOffset;
          else if (cmsswlay == 4 && isStereo == 0)
            return 6 + lOffset;
        }
        //TOB
        else if (cmsswdet == 5) {
          if (cmsswlay == 1 && isStereo == -1)
            return 7 + lOffset;
          else if (cmsswlay == 2 && isStereo == -1)
            return 8 + lOffset;
          else if (cmsswlay >= 3 && cmsswlay <= 6 && isStereo == 0)
            return 6 + cmsswlay + lOffset;
        }
        return -1;
      } else {
        //TIB
        if (cmsswdet == 3) {
          if ((cmsswlay == 1 || cmsswlay == 2) && (isStereo == 0 || isStereo == 1)) {
            return 1 + cmsswlay * 2 + isStereo + lOffset;
          } else if (cmsswlay == 3 && isStereo == 0)
            return 7 + lOffset;
          else if (cmsswlay == 4 && isStereo == 0)
            return 8 + lOffset;
        }
        //TOB
        else if (cmsswdet == 5) {
          if ((cmsswlay == 1 || cmsswlay == 2) && (isStereo == 0 || isStereo == 1)) {
            return 7 + cmsswlay * 2 + isStereo + lOffset;
          } else if (cmsswlay >= 3 && cmsswlay <= 6 && isStereo == 0)
            return 10 + cmsswlay + lOffset;
        }
        return -1;
      }
    }
    int convertDiskNumber(int cmsswdet, int cmsswdisk, bool useMatched, int isStereo) const {
      if (cmsswdet == 1 || cmsswdet == 3 || cmsswdet == 5)
        return -1;  //BPIX, TIB, TOB
      if (cmsswdet == 2)
        return cmsswdisk - 1;  //FPIX
      int lOffset = 0;
      if (lo_ == TkLayout::phase1)
        lOffset = 1;
      if (useMatched) {
        return -1;
      } else {
        if ((isStereo != 0 && isStereo != 1) || cmsswdisk < 1)
          return -1;
        //TID
        if (cmsswdet == 4 && cmsswdisk <= 3)
          return cmsswdisk * 2 + isStereo + lOffset;
        //TEC
        else if (cmsswdet == 6 && cmsswdisk <= 9)
          return 6 + cmsswdisk * 2 + isStereo + lOffset;
        return -1;
      }
    }

  private:
    TkLayout lo_;
  };

}  // end namespace mkfit

#endif
