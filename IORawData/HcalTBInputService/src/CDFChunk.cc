#include "IORawData/HcalTBInputService/src/CDFChunk.h"

CDFChunk::CDFChunk()
    : TObject(), fChunkName(), fHeaderSize(0), fTrailerSize(0), fChunkLength(0), fChunk(new ULong64_t[1]) {}

CDFChunk::CDFChunk(const char* name)
    : TObject(), fChunkName(name), fHeaderSize(0), fTrailerSize(0), fChunkLength(0), fChunk(nullptr) {}
