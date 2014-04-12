#ifndef GENERS_ZLIBHANDLE_HH_
#define GENERS_ZLIBHANDLE_HH_

extern "C" {
    struct z_stream_s;
}

namespace gs {
    class ZlibInflateHandle
    {
    public:
        ZlibInflateHandle();
        ~ZlibInflateHandle();

        inline z_stream_s& strm() {return *strm_;}

    private:
        ZlibInflateHandle(const ZlibInflateHandle&);
        ZlibInflateHandle& operator=(const ZlibInflateHandle&);

        z_stream_s *strm_;
    };

    class ZlibDeflateHandle
    {
    public:
        explicit ZlibDeflateHandle(int level);
        ~ZlibDeflateHandle();

        inline z_stream_s& strm() {return *strm_;}
        inline int level() {return level_;}

    private:
        ZlibDeflateHandle();
        ZlibDeflateHandle(const ZlibDeflateHandle&);
        ZlibDeflateHandle& operator=(const ZlibDeflateHandle&);

        z_stream_s *strm_;
        int level_;
    };
}

#endif // GENERS_ZLIBHANDLE_HH_

