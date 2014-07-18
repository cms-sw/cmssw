#include "DataFormats/PatCandidates/interface/libminifloat.h"

namespace {
    MiniFloatConverter dummy; // so the constructor is called
}
uint32_t MiniFloatConverter::mantissatable[2048];
uint32_t MiniFloatConverter::exponenttable[64];
uint16_t MiniFloatConverter::offsettable[64];
uint16_t MiniFloatConverter::basetable[512];
uint8_t  MiniFloatConverter::shifttable[512];


MiniFloatConverter::MiniFloatConverter() {
    static bool once = false;
    if (!once) { filltables(); once = true; }
}

void MiniFloatConverter::filltables() {
    // ==== mantissatable ===
    // -- zero --
    mantissatable[0] = 0;
    // -- denorm --
    for (unsigned int i = 1; i <= 1023; ++i) {
        unsigned int m =(i<<13), e=0;
        while(!(m&0x00800000)){ // While not normalized
            e-=0x00800000; // Decrement exponent (1<<23)
            m<<=1; // Shift mantissa
        }
        m&=~0x00800000; // Clear leading 1 bit
        e+= 0x38800000; // Adjust bias ((127-14)<<23)
        mantissatable[i] = m | e; 
    }
    // -- norm --
    for (unsigned int i = 1024; i <= 2047; ++i) {
        mantissatable[i] = 0x38000000 + ((i-1024)<<13);
    }
    // ==== exponenttable ===
    exponenttable[0] = 0;
    for (unsigned int i = 1; i <= 30; ++i) exponenttable[i] = i<<23;
    exponenttable[31] = 0x47800000;
    exponenttable[32] = 0x80000000u;
    for (unsigned int i = 33; i <= 62; ++i) exponenttable[i] = 0x80000000u | ((i-32)<<23);
    exponenttable[63] = 0xC7800000;

    // ==== offsettable ====
    for (unsigned int i = 0; i <= 63; ++i) offsettable[i] = ((i == 0 || i == 32) ? 0 : 1024);

    // ==== basetable, shifttable ===
    for (unsigned i=0; i<256; ++i){
        int e = int(i)-127;
        if(e<-24){ // Very small numbers map to zero
            basetable[i|0x000]=0x0000;
            basetable[i|0x100]=0x8000;
            shifttable[i|0x000]=24;
            shifttable[i|0x100]=24;
        }
        else if(e<-14){ // Small numbers map to denorms
            basetable[i|0x000]=(0x0400>>(-e-14));
            basetable[i|0x100]=(0x0400>>(-e-14)) | 0x8000;
            shifttable[i|0x000]=-e-1;
            shifttable[i|0x100]=-e-1;
        }
        else if(e<=15){ // Normal numbers just lose precision
            basetable[i|0x000]=((e+15)<<10);
            basetable[i|0x100]=((e+15)<<10) | 0x8000;
            shifttable[i|0x000]=13;
            shifttable[i|0x100]=13;
        }
        else if(e<128){ // Large numbers map to Infinity
            basetable[i|0x000]=0x7C00;
            basetable[i|0x100]=0xFC00;
            shifttable[i|0x000]=24;
            shifttable[i|0x100]=24;
        }
        else{ // Infinity and NaN's stay Infinity and NaN's
            basetable[i|0x000]=0x7C00;
            basetable[i|0x100]=0xFC00;
            shifttable[i|0x000]=13;
            shifttable[i|0x100]=13;
        }
    }
}
