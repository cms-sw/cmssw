#include <cassert>

#include "zlib.h"

#include "Alignment/Geners/interface/ZlibHandle.hh"

namespace gs {
    ZlibInflateHandle::ZlibInflateHandle()
    {
        strm_ = new z_stream_s();
        strm_->zalloc = Z_NULL;
        strm_->zfree = Z_NULL;
        strm_->opaque = Z_NULL;
        strm_->avail_in = 0;
        strm_->next_in = Z_NULL;
        assert(inflateInit(strm_) == Z_OK);
    }

    ZlibInflateHandle::~ZlibInflateHandle()
    {
        inflateEnd(strm_);
        delete strm_;
    }

    ZlibDeflateHandle::ZlibDeflateHandle(const int lev)
        : level_(lev)
    {
        strm_ = new z_stream_s();
        strm_->zalloc = Z_NULL;
        strm_->zfree = Z_NULL;
        strm_->opaque = Z_NULL;
        strm_->avail_in = 0;
        strm_->next_in = Z_NULL;
        assert(deflateInit(strm_, lev) == Z_OK);
    }

    ZlibDeflateHandle::~ZlibDeflateHandle()
    {
        deflateEnd(strm_);
        delete strm_;
    }
}
