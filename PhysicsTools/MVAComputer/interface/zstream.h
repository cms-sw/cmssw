#ifndef PhysicsTools_MVAComputer_zstream_h
#define PhysicsTools_MVAComputer_zstream_h
// -*- C++ -*-
//
// Package:     MVAComputer
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: zstream.h,v 1.4 2012/04/02 12:37:11 davidlt Exp $
//

#include <iostream>
#include <vector>

#include <zlib.h>

namespace ext {

// implements STL stream wrappers around other streams
// to provide transparent gzip compression/decompression
//
// constructors take reference to an existing stream, rest is straightforward

template<typename Item_t, typename Traits_t = std::char_traits<Item_t>,
         typename Allocator_t = std::allocator<Item_t> >
class basic_ozstreambuf : public std::basic_streambuf<Item_t, Traits_t> {
    public:
	typedef std::basic_ostream<Item_t, Traits_t>	OStream_t;
	typedef std::basic_streambuf<Item_t, Traits_t>	StreamBuf_t;

	typedef Item_t					char_type;
	typedef typename Traits_t::int_type		int_type;
	typedef typename Traits_t::pos_type		pos_type;
	typedef typename Traits_t::off_type		off_type;
	typedef unsigned char				byte_type;
	typedef Traits_t				traits_type;

	basic_ozstreambuf(OStream_t *os, int level);
	~basic_ozstreambuf();

	using StreamBuf_t::pbase;
	using StreamBuf_t::pptr;
	using StreamBuf_t::epptr;

	int sync();
	int_type overflow(int_type c);
	std::streamsize flush();

    private:
	bool zipToStream(char_type *buf, std::streamsize size);
	size_t fillInputBuffer();

	OStream_t				*os;
	z_stream				zipStream;
	int					err;
	std::vector<byte_type>			outputBuffer;
	std::vector<char_type, Allocator_t>	buffer;
};

template<typename Item_t, typename Traits_t = std::char_traits<Item_t>,
         typename Allocator_t = std::allocator<Item_t> >
class basic_izstreambuf : public std::basic_streambuf<Item_t, Traits_t> {
    public:
	typedef std::basic_istream<Item_t, Traits_t>	IStream_t;
	typedef std::basic_streambuf<Item_t, Traits_t>	StreamBuf_t;

	typedef Item_t					char_type;
	typedef typename Traits_t::int_type		int_type;
	typedef typename Traits_t::pos_type		pos_type;
	typedef typename Traits_t::off_type		off_type;
	typedef unsigned char				byte_type;
	typedef Traits_t				traits_type;

	basic_izstreambuf(IStream_t *is);
	~basic_izstreambuf();

	using StreamBuf_t::gptr;
	using StreamBuf_t::egptr;
	using StreamBuf_t::eback;

	int_type underflow();

    private:
	void putbackFromZStream();
	std::streamsize unzipFromStream(char_type *buf, std::streamsize size);
	size_t fillInputBuffer();

	IStream_t				*is;
	z_stream				zipStream;
	int					err;
	std::vector<byte_type>			inputBuffer;
	std::vector<char_type, Allocator_t>	buffer;
};

template<typename Item_t, typename Traits_t = std::char_traits<Item_t>,
         typename Allocator_t = std::allocator<Item_t> >
class basic_ozstreambase : virtual public std::basic_ios<Item_t, Traits_t> {
    public:
	typedef std::basic_ostream<Item_t, Traits_t> OStream_t;
	typedef basic_ozstreambuf<Item_t, Traits_t, Allocator_t> ZOStreamBuf_t;

	basic_ozstreambase(OStream_t *os, int level) :
		buffer(os, level) { this->init(&buffer); }

	ZOStreamBuf_t *rdbuf() { return &buffer; }

    private:
	ZOStreamBuf_t buffer;
};

template<typename Item_t, typename Traits_t = std::char_traits<Item_t>,
         typename Allocator_t = std::allocator<Item_t> >
class basic_izstreambase : virtual public std::basic_ios<Item_t, Traits_t> {
    public:
	typedef std::basic_istream<Item_t, Traits_t> IStream_t;
	typedef basic_izstreambuf<Item_t, Traits_t, Allocator_t> ZIStreamBuf_t;

	basic_izstreambase(IStream_t *is) : buffer(is) { this->init(&buffer); }

	ZIStreamBuf_t *rdbuf() { return &buffer; }

    private:
	ZIStreamBuf_t buffer;
};

template<typename Item_t, typename Traits_t = std::char_traits<Item_t>,
         typename Allocator_t = std::allocator<Item_t> >
class basic_ozstream : public basic_ozstreambase<Item_t, Traits_t, Allocator_t>,
                       public std::basic_ostream<Item_t, Traits_t> {
    public:
	typedef std::basic_ostream<Item_t, Traits_t> OStream_t;
	typedef basic_ozstreambase<Item_t, Traits_t, Allocator_t> ZOStreamBase_t;

	basic_ozstream(OStream_t *os, int open_mode = std::ios::out,
	               int level = 9) :
		ZOStreamBase_t(os, level),
		OStream_t(ZOStreamBase_t::rdbuf()) {}
	~basic_ozstream() {}
};

template<typename Item_t, typename Traits_t = std::char_traits<Item_t>,
         typename Allocator_t = std::allocator<Item_t> >
class basic_izstream : public basic_izstreambase<Item_t, Traits_t, Allocator_t>,
                       public std::basic_istream<Item_t, Traits_t> {
    public:
	typedef std::basic_istream<Item_t, Traits_t> IStream_t;
	typedef basic_izstreambase<Item_t, Traits_t, Allocator_t> ZIStreamBase_t;

	basic_izstream(IStream_t *is, int open_mode = std::ios::in) :
		ZIStreamBase_t(is), IStream_t(ZIStreamBase_t::rdbuf()) {}
	~basic_izstream() {}
};

typedef basic_ozstream<char> ozstream;
typedef basic_ozstream<wchar_t> wozstream;
typedef basic_izstream<char> izstream;
typedef basic_izstream<wchar_t> wizstream;

} // namespace ext

#include "PhysicsTools/MVAComputer/interface/zstream.icc"

#endif // PhysicsTools_MVAComputer_zstream_h
