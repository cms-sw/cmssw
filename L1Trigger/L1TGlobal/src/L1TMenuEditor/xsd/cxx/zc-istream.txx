// file      : xsd/cxx/zc-istream.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    // zc_streambuf
    //
    template <typename C>
    zc_streambuf<C>::
    zc_streambuf (const ro_string<C>& str)
        : str_ (str.data (), str.size ())
    {
      init ();
    }

    template <typename C>
    zc_streambuf<C>::
    zc_streambuf (const std::basic_string<C>& str)
        : str_ (str)
    {
      init ();
    }

    template <typename C>
    void zc_streambuf<C>::
    init ()
    {
      C* b (const_cast<C*> (str_.data ()));
      C* e (b + str_.size ());

      this->setg (b, b, e);
    }

    template <typename C>
    std::streamsize zc_streambuf<C>::
    showmanyc ()
    {
      return static_cast<std::streamsize> (
        this->egptr () - this->gptr ());
    }

    template <typename C>
    typename zc_streambuf<C>::int_type zc_streambuf<C>::
    underflow ()
    {
      int_type r = traits_type::eof ();

      if (this->gptr () < this->egptr ())
        r = traits_type::to_int_type (*this->gptr ());

      return r;
    }


    // zc_istream_base
    //
    template <typename C>
    zc_istream_base<C>::
    zc_istream_base (const ro_string<C>& str)
        : buf_ (str)
    {
    }

    template <typename C>
    zc_istream_base<C>::
    zc_istream_base (const std::basic_string<C>& str)
        : buf_ (str)
    {
    }


    // zc_istream
    //
    template <typename C>
    zc_istream<C>::
    zc_istream (const ro_string<C>& str)
        : zc_istream_base<C> (str),
          std::basic_istream<C> (&this->buf_)
    {
    }

    template <typename C>
    zc_istream<C>::
    zc_istream (const std::basic_string<C>& str)
        : zc_istream_base<C> (str),
          std::basic_istream<C> (&this->buf_)
    {
    }
  }
}
