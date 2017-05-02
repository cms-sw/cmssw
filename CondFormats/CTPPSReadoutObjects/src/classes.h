#include "CondFormats/CTPPSReadoutObjects/src/headers.h"

namespace CondFormats_CTPPSPixelObjects {
   struct dictionary {
   std::map<CTPPSPixelFramePosition, CTPPSPixelROCInfo> ROCMapping; 
   std::map<uint32_t, CTPPSPixelROCAnalysisMask> analysisMask;
 };
}     
