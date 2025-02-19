#ifndef DataFormats_SiStripCommon_ConstantsForCondObjects_H
#define DataFormats_SiStripCommon_ConstantsForCondObjects_H


namespace sistrip {
  static const uint32_t FirstBadStripMask_  = 0x3FF;
  static const uint32_t RangeBadStripMask_  = 0x3FF;
  static const uint32_t FlagBadStripMask_   = 0xFFF;

  static const uint32_t FirstThStripMask_  = 0x3FF;
  static const uint32_t HighThStripMask_   = 0x3F;
  static const uint32_t LowThStripMask_    = 0x3F;

  static const uint32_t FirstBadStripShift_ = 22;
  static const uint32_t RangeBadStripShift_ = 12;
  static const uint32_t FlagBadStripShift_  = 0;

  static const uint32_t FirstThStripShift_ = 6;
  static const uint32_t HighThStripShift_  = 0;
  static const uint32_t LowThStripShift_   = 0;
}

#endif // DataFormats_SiStripCommon_ConstantsForCondObjects_H
