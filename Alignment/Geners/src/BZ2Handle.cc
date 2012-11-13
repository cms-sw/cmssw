#include <cassert>

#include "Alignment/Geners/interface/BZ2Handle.hh"

namespace gs {
    BZ2InflateHandle::BZ2InflateHandle(bz_stream& strm)
        : strm_(&strm)
    {
        strm_->bzalloc = 0;
        strm_->bzfree = 0;
        strm_->opaque = 0;
        strm_->avail_in = 0;
        strm_->next_in = 0;
        assert(BZ2_bzDecompressInit(strm_, 0, 0) == BZ_OK);
    }

    BZ2InflateHandle::~BZ2InflateHandle()
    {
        assert(BZ2_bzDecompressEnd(strm_) == BZ_OK);
    }

    BZ2DeflateHandle::BZ2DeflateHandle(bz_stream& strm)
        : strm_(&strm)
    {
        strm_->bzalloc = 0;
        strm_->bzfree = 0;
        strm_->opaque = 0;
        strm_->avail_in = 0;
        strm_->next_in = 0;
        assert(BZ2_bzCompressInit(strm_, 9, 0, 0) == BZ_OK);
    }

    BZ2DeflateHandle::~BZ2DeflateHandle()
    {
        assert(BZ2_bzCompressEnd(strm_) == BZ_OK);
    }
}
