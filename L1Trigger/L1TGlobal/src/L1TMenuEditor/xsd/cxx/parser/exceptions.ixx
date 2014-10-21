// file      : xsd/cxx/parser/exceptions.ixx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#if defined(XSD_CXX_PARSER_USE_CHAR) || !defined(XSD_CXX_PARSER_USE_WCHAR)

#ifndef XSD_CXX_PARSER_EXCEPTIONS_IXX_CHAR
#define XSD_CXX_PARSER_EXCEPTIONS_IXX_CHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // error
      //
      inline
      std::basic_ostream<char>&
      operator<< (std::basic_ostream<char>& os, const error<char>& e)
      {
        return os << e.id () << ':' << e.line () << ':' << e.column ()
                  << (e.severity () == severity::error
                      ? " error: "
                      : " warning: ") << e.message ();
      }


      // diagnostics
      //
      inline
      std::basic_ostream<char>&
      operator<< (std::basic_ostream<char>& os, const diagnostics<char>& d)
      {
        for (diagnostics<char>::const_iterator b (d.begin ()), i (b);
             i != d.end ();
             ++i)
        {
          if (i != b)
            os << "\n";

          os << *i;
        }

        return os;
      }

      // parsing
      //
      template<>
      inline
      void parsing<char>::
      print (std::basic_ostream<char>& os) const
      {
        if (diagnostics_.empty ())
          os << "instance document parsing failed";
        else
          os << diagnostics_;
      }
    }
  }
}

#endif // XSD_CXX_PARSER_EXCEPTIONS_IXX_CHAR
#endif // XSD_CXX_PARSER_USE_CHAR


#if defined(XSD_CXX_PARSER_USE_WCHAR) || !defined(XSD_CXX_PARSER_USE_CHAR)

#ifndef XSD_CXX_PARSER_EXCEPTIONS_IXX_WCHAR
#define XSD_CXX_PARSER_EXCEPTIONS_IXX_WCHAR

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // error
      //
      inline
      std::basic_ostream<wchar_t>&
      operator<< (std::basic_ostream<wchar_t>& os, const error<wchar_t>& e)
      {
        return os << e.id () << L':' << e.line () << L':' << e.column ()
                  << (e.severity () == severity::error
                      ? L" error: "
                      : L" warning: ") << e.message ();
      }

      // diagnostics
      //
      inline
      std::basic_ostream<wchar_t>&
      operator<< (std::basic_ostream<wchar_t>& os,
                  const diagnostics<wchar_t>& d)
      {
        for (diagnostics<wchar_t>::const_iterator b (d.begin ()), i (b);
             i != d.end ();
             ++i)
        {
          if (i != b)
            os << L"\n";

          os << *i;
        }

        return os;
      }

      // parsing
      //
      template<>
      inline
      void parsing<wchar_t>::
      print (std::basic_ostream<wchar_t>& os) const
      {
        if (diagnostics_.empty ())
          os << L"instance document parsing failed";
        else
          os << diagnostics_;
      }
    }
  }
}

#endif // XSD_CXX_PARSER_EXCEPTIONS_IXX_WCHAR
#endif // XSD_CXX_PARSER_USE_WCHAR
