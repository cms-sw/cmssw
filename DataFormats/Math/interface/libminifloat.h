#ifndef libminifloat_h
#define libminifloat_h
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include <cstdint>
#include <cassert>
#include <algorithm>

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
        inline static float reduceMantissaToNbits(const float &f, int bits)
        {
            uint32_t mask = (0xFFFFFFFF >> (23-bits)) << (23-bits);
            union { float flt; uint32_t i32; } conv;
            conv.flt=f;
            conv.i32&=mask;
            return conv.flt;
        }

        class ReduceMantissaToNbitsRounding {
            public:
                ReduceMantissaToNbitsRounding(int bits) : 
                    shift(23-bits), mask((0xFFFFFFFF >> (shift)) << (shift)), 
                    test(1 << (shift-1)), maxn((1<<bits)-2) {
                        assert(bits <= 23); // "max mantissa size is 23 bits"
                    }
                float operator()(float f) const {
                    constexpr uint32_t low23 = (0x007FFFFF); // mask to keep lowest 23 bits = mantissa
                    constexpr uint32_t  hi9  = (0xFF800000); // mask to keep highest 9 bits = the rest
                    union { float flt; uint32_t i32; } conv;
                    conv.flt=f;
                    if (conv.i32 & test) { // need to round
                        uint32_t mantissa = (conv.i32 & low23) >> shift;
                        if (mantissa < maxn) mantissa++;
                        conv.i32 = (conv.i32 & hi9) | (mantissa << shift);
                    } else {
                        conv.i32 &= mask;
                    }
                    return conv.flt;
                }
            private:
                const int shift;
                const uint32_t mask, test, maxn;           
        };

        template<int bits>
        inline static float reduceMantissaToNbitsRounding(const float &f)
        {
            static const ReduceMantissaToNbitsRounding reducer(bits);
            return reducer(f);
        }



        inline static float reduceMantissaToNbitsRounding(float f, int bits) 
        {
            return ReduceMantissaToNbitsRounding(bits)(f);
        }

        template<typename InItr, typename OutItr>
        static void reduceMantissaToNbitsRounding(int bits, InItr begin, InItr end, OutItr out) 
        {
            std::transform(begin, end, out, ReduceMantissaToNbitsRounding(bits));
        }
        

        inline static float max() {
            union { float flt; uint32_t i32; } conv;
            conv.i32 = 0x477fe000; // = mantissatable[offsettable[0x1e]+0x3ff]+exponenttable[0x1e]
            return conv.flt;
        }

        // Maximum float32 value that gets rounded to max()
        inline static float max32RoundedToMax16() {
            union { float flt; uint32_t i32; } conv;
            // 2^16 in float32 is the first to result inf in float16, so
            // 2^16-1 is the last float32 to result max() in float16
            conv.i32 = (0x8f<<23) - 1;
            return conv.flt;
        }

        inline static float min() {
            union { float flt; uint32_t i32; } conv;
            conv.i32 = 0x38800000; // = mantissatable[offsettable[1]+0]+exponenttable[1]
            return conv.flt;
        }

        // Minimum float32 value that gets rounded to min()
        inline static float min32RoundedToMin16() {
            union { float flt; uint32_t i32; } conv;
            // 2^-14-1 in float32 is the first to result denormalized in float16, so
            // 2^-14 is the first float32 to result min() in float16
            conv.i32 = (0x71<<23);
            return conv.flt;
        }

        inline static float denorm_min() {
            union { float flt; uint32_t i32; } conv;
            conv.i32 = 0x33800000; // mantissatable[offsettable[0]+1]+exponenttable[0]
            return conv.flt;
        }

        inline static bool isdenorm(uint16_t h) {
            // if exponent is zero (sign-bit excluded of course) and mantissa is not zero
            return ((h >> 10) & 0x1f) == 0 && (h & 0x3ff) != 0;
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
