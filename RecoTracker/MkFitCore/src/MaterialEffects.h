#ifndef RecoTracker_MkFitCore_src_MaterialEffects_h
#define RecoTracker_MkFitCore_src_MaterialEffects_h

#include "RecoTracker/MkFitCore/interface/Config.h"

#include <cmath>

namespace mkfit {

  namespace Config {
    // config for material effects in cmssw
    constexpr float rangeZME = 300.;
    constexpr int nBinsZME = 300;
    constexpr float rangeRME = 120.;
    constexpr int nBinsRME = 120;
  }  // namespace Config

  class MaterialEffects {
  public:
    MaterialEffects();

    int __attribute__((optimize("no-inline"))) getZbin(const float z) const {
      return (std::abs(z) * Config::nBinsZME) / (Config::rangeZME);
    }
    int __attribute__((optimize("no-inline"))) getRbin(const float r) const {
      return (r * Config::nBinsRME) / (Config::rangeRME);
    }
    float getRlVal(const int zb, const int rb) const { return mRlgridME[zb][rb]; }
    float getXiVal(const int zb, const int rb) const { return mXigridME[zb][rb]; }

    /// (z,r) grid to material bin/det mapping for Rad length (Rl) and Xi arrays
    // see https://indico.cern.ch/event/924564/contributions/3885164/attachments/2097314/
    int getDetId(const float zin, const float r) const {
      const float z = std::abs(zin);

      //pixel
      if (r < 17) {
        //pixel barrel
        if (z < 28) {
          if (r < 4) {
            if (z < 20)
              return 0;
            else
              return 1;
          }
          if (r < 8) {
            if (z < 20)
              return 2;
            else
              return 3;
          }
          if (r < 12) {
            if (z < 20)
              return 4;
            else
              return 5;
          }
          if (z < 20)
            return 6;
          else
            return 7;
        }

        //pixel endcap
        if (z < 36) {
          if (r > 9.5 && z < 32.5)
            return 8;
          else
            return 9;
        }
        if (z < 45) {
          if (r > 9.5 && z < 40)
            return 10;
          else
            return 11;
        }
        if (z >= 45) {
          if (r > 9.5 && z < 49)
            return 12;
          else
            return 13;
        }
      }

      //TIB & TID
      if (r < 55) {
        //TIB
        if (z < 70) {
          if (r < 29) {
            if (z < 22)
              return 14;
            else
              return 15;
          }
          if (r < 38) {
            if (z < 25)
              return 16;
            else
              return 17;
          }
          if (r < 46) {
            if (z < 44)
              return 18;
            else
              return 19;
          }
          if (z < 50)
            return 20;
          else
            return 21;
        }

        //TID
        if (z > 70 && z < 120) {
          if (r > 35 && z < 80)
            return 22;
          else if (z < 86)
            return 23;
          else if (r > 35 && z < 92)
            return 24;
          else if (z < 98)
            return 25;
          else if (r > 35 && z < 104)
            return 26;
          else
            return 27;
        }
      }

      //TOB
      if (r < 120 && z < 120) {
        if (r < 65) {
          if (z < 17)
            return 28;
          else if (z < 70)
            return 29;
          else
            return 30;
        }
        if (r < 75) {
          if (z < 17)
            return 31;
          else if (z < 70)
            return 32;
          else
            return 33;
        }
        if (r < 82) {
          if (z < 17)
            return 34;
          else if (z < 70)
            return 35;
          else
            return 36;
        }
        if (r < 90) {
          if (z < 17)
            return 37;
          else if (z < 70)
            return 38;
          else
            return 39;
        }
        if (r < 100) {
          if (z < 17)
            return 40;
          else if (z < 70)
            return 41;
          else
            return 42;
        }
        if (z < 17)
          return 43;
        else if (z < 70)
          return 44;
        else
          return 45;
      }

      //TEC
      if (z > 120 && r < 120) {
        if (z < 128) {
          if (r < 35)
            return 46;
          else if (r < 55)
            return 47;
          else if (r < 80)
            return 48;
          else
            return 49;
        }
        if (z < 132) {
          if (r < 45)
            return 50;
          else if (r < 70)
            return 51;
          else
            return 52;
        }
        if (z < 136) {
          if (r < 35)
            return 53;
          else if (r < 55)
            return 54;
          else if (r < 80)
            return 55;
          else
            return 56;
        }
        if (z < 138) {
          if (r < 45)
            return 57;
          else if (r < 70)
            return 58;
          else
            return 59;
        }
        if (z < 142) {
          if (r < 35)
            return 60;
          else if (r < 55)
            return 61;
          else if (r < 80)
            return 62;
          else
            return 63;
        }
        if (z < 146) {
          if (r < 45)
            return 64;
          else
            return 65;
        }
        if (z < 150) {
          if (r < 35)
            return 66;
          else if (r < 55)
            return 67;
          else if (r < 80)
            return 68;
          else
            return 69;
        }
        if (z < 153) {
          if (r < 45)
            return 70;
          else
            return 71;
        }
        if (z < 156) {
          if (r < 35)
            return 72;
          else if (r < 55)
            return 73;
          else if (r < 80)
            return 74;
          else
            return 75;
        }
        if (z < 160) {
          if (r < 45)
            return 76;
          else
            return 77;
        }
        if (z < 164) {
          if (r < 35)
            return 78;
          else if (r < 55)
            return 79;
          else if (r < 80)
            return 80;
          else
            return 81;
        }
        if (z < 167) {
          if (r < 45)
            return 82;
          else
            return 83;
        }

        if (z < 170) {
          if (r < 55)
            return 84;
          else if (r < 80)
            return 85;
          else
            return 86;
        }
        if (z < 174) {
          if (r < 45)
            return 87;
          else
            return 88;
        }

        if (z < 177.3) {
          if (r < 55)
            return 89;
          else if (r < 80)
            return 90;
          else
            return 91;
        }
        if (z < 181) {
          if (r < 45)
            return 92;
          else
            return 93;
        }

        if (z < 185) {
          if (r < 55)
            return 94;
          else if (r < 80)
            return 95;
          else
            return 96;
        }
        if (z < 188.5) {
          if (r < 45)
            return 97;
          else
            return 98;
        }

        if (z < 192) {
          if (r < 55)
            return 99;
          else if (r < 80)
            return 100;
          else
            return 101;
        }
        if (z < 195) {
          if (r < 45)
            return 102;
          else
            return 103;
        }

        if (z < 202) {
          if (r < 55)
            return 104;
          else if (r < 80)
            return 105;
          else
            return 106;
        }
        if (z < 206) {
          if (r < 45)
            return 107;
          else
            return 108;
        }

        if (z < 210) {
          if (r < 55)
            return 109;
          else if (r < 80)
            return 110;
          else
            return 111;
        }
        if (z < 212) {
          if (r < 45)
            return 112;
          else
            return 113;
        }

        if (z < 222) {
          if (r < 55)
            return 114;
          else if (r < 80)
            return 115;
          else
            return 116;
        }
        if (z < 224)
          return 117;

        if (z < 228) {
          if (r < 55)
            return 118;
          else if (r < 80)
            return 119;
          else
            return 120;
        }
        if (z < 232)
          return 121;

        if (z < 241) {
          if (r < 55)
            return 122;
          else if (r < 80)
            return 123;
          else
            return 124;
        }
        if (z < 245)
          return 125;

        if (z < 248) {
          if (r < 55)
            return 126;
          else if (r < 80)
            return 127;
          else
            return 128;
        }
        if (z < 252)
          return 129;

        if (z < 264) {
          if (r < 80)
            return 130;
          else
            return 131;
        }
        if (z < 267)
          return 132;

        if (z < 270) {
          if (r < 80)
            return 133;
          else
            return 134;
        }
        if (z < 280)
          return 135;
      }
      return -1;
    }

  private:
    float mRlgridME[Config::nBinsZME][Config::nBinsRME];
    float mXigridME[Config::nBinsZME][Config::nBinsRME];
  };  // class MaterialEffects

  namespace Config {
    extern const MaterialEffects materialEff;
  }
}  // end namespace mkfit
#endif
