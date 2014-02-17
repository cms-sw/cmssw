#ifndef PhysicsTools_MVAComputer_memstream_h
#define PhysicsTools_MVAComputer_memstream_h
// -*- C++ -*-
//
// Package:     MVAComputer
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: memstream.h,v 1.4 2011/05/23 13:41:03 eulisse Exp $
//

#include <iostream>

namespace ext {

// provide STL stream to read/write into memory buffers
//
// constructors both take pointer to a buffer and buffer size

template<typename Item_t, typename Traits_t = std::char_traits<Item_t>,
         typename Allocator_t = std::allocator<Item_t> >
class basic_omemstream : private std::basic_streambuf<Item_t, Traits_t>,
                         public std::basic_ostream<Item_t, Traits_t> {
    public:
	typedef typename Traits_t::char_type	char_type;
	typedef typename Traits_t::int_type	int_type;
	typedef Traits_t			traits_type;

	basic_omemstream(char_type *buf, size_t size) :
			std::basic_ostream<Item_t, Traits_t>(this),
			buffer(buf), cur(buf), last(buf + size)
	{ this->exceptions(std::ios_base::badbit); }

	char_type* begin() const { return buffer; }
	char_type* end() const { return cur; }
	size_t size() const { return cur - buffer; }
	bool empty() const { return cur == buffer; }

    private:
	std::streamsize xsputn(char_type const *data, std::streamsize size) {
		size_t n = std::min<size_t>(last - cur, size);
		traits_type::copy(cur, data, n);
		cur += n;
		return n;
	}

	int_type overflow(int_type c)
	{
		if (!traits_type::eq_int_type(c, traits_type::eof())) {
			char_type t = traits_type::to_char_type(c);
			if (xsputn(&t, 1) < 1)
				return traits_type::eof();
		}

		return c;
	}

	int sync() { return 0; }

	char_type	*buffer, *cur, *last;
};

template<typename Item_t, typename Traits_t = std::char_traits<Item_t>,
         typename Allocator_t = std::allocator<Item_t> >
class basic_imemstream : private std::basic_streambuf<Item_t, Traits_t>,
                         public std::basic_istream<Item_t, Traits_t> {
    public:
	typedef typename Traits_t::char_type	char_type;
	typedef typename Traits_t::int_type	int_type;
	typedef Traits_t			traits_type;

	basic_imemstream(const char_type *buf, size_t size) :
			std::basic_istream<Item_t, Traits_t>(this)
	{
		this->exceptions(std::ios_base::badbit);
		this->setg(const_cast<char_type*>(buf),
		     const_cast<char_type*>(buf),
		     const_cast<char_type*>(buf + size));
	}

    private:
	int_type underflow()
	{
		if (this->gptr() && this->gptr() < this->egptr())
			return traits_type::to_int_type(*this->gptr());

		return traits_type::eof();
	}
};

typedef basic_omemstream<char>		omemstream;
typedef basic_omemstream<wchar_t>	womemstream;
typedef basic_imemstream<char>		imemstream;
typedef basic_imemstream<wchar_t>	wimemstream;

} // namespace ext

#endif // PhysicsTools_MVAComputer_memstream_h
