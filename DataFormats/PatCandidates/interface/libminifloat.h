#ifndef libminifloat_h
#define libminifloat_h
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include <cstdint>

// ftp://ftp.fox-toolkit.org/pub/fasthalffloatconversion.pdf
class MiniFloatConverter {
    public:
        MiniFloatConverter() ;
        inline static float float16to32(uint16_t h) {
            union { float flt; uint32_t i32; } conv;
            conv.i32 = mantissatable[offsettable[h>>10]+(h&0x3ff)]+exponenttable[h>>10];
            return conv.flt;
        }
        inline static uint16_t float32to16(float x) {
            return float32to16round(x);
        }
        /// Fast implementation, but it crops the number so it biases low
        inline static uint16_t float32to16crop(float x) {
            union { float flt; uint32_t i32; } conv;
            conv.flt = x;
            return basetable[(conv.i32>>23)&0x1ff]+((conv.i32&0x007fffff)>>shifttable[(conv.i32>>23)&0x1ff]);
        }
        /// Slower implementation, but it rounds to avoid biases
        inline static uint16_t float32to16round(float x) {
            union { float flt; uint32_t i32; } conv;
            conv.flt = x;
            uint8_t shift = shifttable[(conv.i32>>23)&0x1ff];
            if (shift == 13) {
                uint16_t base2 = (conv.i32&0x007fffff)>>12;
                uint16_t base = base2 >> 1;
                if (((base2 & 1) != 0) && (base < 1023)) base++;
                return basetable[(conv.i32>>23)&0x1ff]+base; 
            } else {
                return basetable[(conv.i32>>23)&0x1ff]+((conv.i32&0x007fffff)>>shifttable[(conv.i32>>23)&0x1ff]);
            }
        }
        template<int bits>
        inline static float reduceMantissaToNbits(const float &f)
        {
            static_assert(bits <= 23,"max mantissa size is 23 bits");
            constexpr uint32_t mask = (0xFFFFFFFF >> (23-bits)) << (23-bits);
            union { float flt; uint32_t i32; } conv;
            conv.flt=f;
            conv.i32&=mask;
            return conv.flt;
        }
    private:
        CMS_THREAD_SAFE static uint32_t mantissatable[2048];
        CMS_THREAD_SAFE static uint32_t exponenttable[64];
        CMS_THREAD_SAFE static uint16_t offsettable[64];
        CMS_THREAD_SAFE static uint16_t basetable[512];
        CMS_THREAD_SAFE static uint8_t  shifttable[512];
        static void filltables() ;
};
#endif
