// file      : xsd/cxx/parser/expat/elements.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <new>     // std::bad_alloc
#include <istream>
#include <fstream>
#include <cstring> // std::strchr
#include <cassert>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/bits/literals.hxx> // xml::bits::{xml_prefix, etc}

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      namespace expat
      {

        // document
        //

        template <typename C>
        document<C>::
        document (parser_base<C>& p,
                  const std::basic_string<C>& name,
                  bool polymorphic)
            : cxx::parser::document<C> (p, std::basic_string<C> (), name),
              xml_parser_ (0),
              eh_ (0),
              polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        document<C>::
        document (parser_base<C>& p,
                  const C* name,
                  bool polymorphic)
            : cxx::parser::document<C> (p, std::basic_string<C> (), name),
              xml_parser_ (0),
              eh_ (0),
              polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        document<C>::
        document (parser_base<C>& p,
                  const C* ns,
                  const C* name,
                  bool polymorphic)
            : cxx::parser::document<C> (p, ns, name),
              xml_parser_ (0),
              eh_ (0),
              polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        document<C>::
        document (parser_base<C>& p,
                  const std::basic_string<C>& ns,
                  const std::basic_string<C>& name,
                  bool polymorphic)
            : cxx::parser::document<C> (p, ns, name),
              xml_parser_ (0),
              eh_ (0),
              polymorphic_ (polymorphic)
        {
        }

        template <typename C>
        document<C>::
        document (bool polymorphic)
            : xml_parser_ (0),
              eh_ (0),
              polymorphic_ (polymorphic)
        {
        }

        // file
        //

        template <typename C>
        void document<C>::
        parse (const std::basic_string<C>& file)
        {
          std::ifstream ifs;
          ifs.exceptions (std::ios_base::badbit | std::ios_base::failbit);
          ifs.open (file.c_str (), std::ios_base::in | std::ios_base::binary);

          parse (ifs, file);
        }

        template <typename C>
        void document<C>::
        parse (const std::basic_string<C>& file, xml::error_handler<C>& eh)
        {
          std::ifstream ifs;
          ifs.exceptions (std::ios_base::badbit | std::ios_base::failbit);
          ifs.open (file.c_str (), std::ios_base::in | std::ios_base::binary);

          parse (ifs, file, eh);
        }


        // istream
        //

        template <typename C>
        void document<C>::
        parse (std::istream& is)
        {
          parse (is, 0, 0, default_eh_);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is, xml::error_handler<C>& eh)
        {
          if (!parse (is, 0, 0, eh))
            throw parsing<C> ();
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is, const std::basic_string<C>& system_id)
        {
          default_eh_.reset ();
          parse (is, &system_id, 0, default_eh_);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               xml::error_handler<C>& eh)
        {
          if (!parse (is, &system_id, 0, eh))
            throw parsing<C> ();
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               const std::basic_string<C>& public_id)
        {
          default_eh_.reset ();
          parse (is, &system_id, &public_id, default_eh_);
        }

        template <typename C>
        void document<C>::
        parse (std::istream& is,
               const std::basic_string<C>& system_id,
               const std::basic_string<C>& public_id,
               xml::error_handler<C>& eh)
        {
          if (!parse (is, &system_id, &public_id, eh))
            throw parsing<C> ();
        }

        // data
        //

        template <typename C>
        void document<C>::
        parse (const void* data, std::size_t size, bool last)
        {
          default_eh_.reset ();
          parse (data, size, last, 0, 0, default_eh_);
        }

        template <typename C>
        void document<C>::
        parse (const void* data, std::size_t size, bool last,
               xml::error_handler<C>& eh)
        {
          if (!parse (data, size, last, 0, 0, eh))
            throw parsing<C> ();
        }

        template <typename C>
        void document<C>::
        parse (const void* data, std::size_t size, bool last,
               const std::basic_string<C>& system_id)
        {
          default_eh_.reset ();
          parse (data, size, last, &system_id, 0, default_eh_);
        }

        template <typename C>
        void document<C>::
        parse (const void* data, std::size_t size, bool last,
               const std::basic_string<C>& system_id,
               xml::error_handler<C>& eh)
        {
          if (!parse (data, size, last, &system_id, 0, eh))
            throw parsing<C> ();
        }

        template <typename C>
        void document<C>::
        parse (const void* data, std::size_t size, bool last,
               const std::basic_string<C>& system_id,
               const std::basic_string<C>& public_id)
        {
          default_eh_.reset ();
          parse (data, size, last, &system_id, &public_id, default_eh_);
        }

        template <typename C>
        void document<C>::
        parse (const void* data, std::size_t size, bool last,
               const std::basic_string<C>& system_id,
               const std::basic_string<C>& public_id,
               xml::error_handler<C>& eh)
        {
          if (!parse (data, size, last, &system_id, &public_id, eh))
            throw parsing<C> ();
        }

        // Implementation details.
        //

        namespace bits
        {
          struct stream_exception_controller
          {
            ~stream_exception_controller ()
            {
              std::ios_base::iostate s = is_.rdstate ();
              s &= ~std::ios_base::failbit;

              // If our error state (sans failbit) intersects with the
              // exception state then that means we have an active
              // exception and changing error/exception state will
              // cause another to be thrown.
              //
              if (!(old_state_ & s))
              {
                // Clear failbit if it was caused by eof.
                //
                if (is_.fail () && is_.eof ())
                  is_.clear (s);

                is_.exceptions (old_state_);
              }
            }

            stream_exception_controller (std::istream& is)
                : is_ (is), old_state_ (is_.exceptions ())
            {
              is_.exceptions (old_state_ & ~std::ios_base::failbit);
            }

          private:
            stream_exception_controller (const stream_exception_controller&);

            stream_exception_controller&
            operator= (const stream_exception_controller&);

          private:
            std::istream& is_;
            std::ios_base::iostate old_state_;
          };
        };

        template <typename C>
        bool document<C>::
        parse (std::istream& is,
               const std::basic_string<C>* system_id,
               const std::basic_string<C>* public_id,
               xml::error_handler<C>& eh)
        {
          parser_auto_ptr parser (XML_ParserCreateNS (0, XML_Char (' ')));

          if (parser == 0)
            throw std::bad_alloc ();

          if (system_id || public_id)
            parse_begin (parser, system_id ? *system_id : *public_id, eh);
          else
            parse_begin (parser, eh);

          // Temporarily unset the exception failbit. Also clear the
          // fail bit when we reset the old state if it was caused
          // by eof.
          //
          bits::stream_exception_controller sec (is);

          char buf[16384]; // 4 x page size.

          bool r (true);

          do
          {
            is.read (buf, sizeof (buf));

            if (is.bad () || (is.fail () && !is.eof ()))
            {
              // If the stream is not using exceptions then the user
              // will have to test for stream failures before calling
              // post.
              //
              break;
            }

            if (XML_Parse (
                  parser, buf, is.gcount (), is.eof ()) == XML_STATUS_ERROR)
            {
              r = false;
              break;
            }
          } while (!is.eof ());

          parse_end ();
          return r;
        }

        template <typename C>
        bool document<C>::
        parse (const void* data,
               std::size_t size,
               bool last,
               const std::basic_string<C>* system_id,
               const std::basic_string<C>* public_id,
               xml::error_handler<C>& eh)
        {
          // First call.
          //
          if (auto_xml_parser_ == 0)
          {
            auto_xml_parser_ = XML_ParserCreateNS (0, XML_Char (' '));

            if (auto_xml_parser_ == 0)
              throw std::bad_alloc ();

            if (system_id || public_id)
              parse_begin (auto_xml_parser_,
                           system_id ? *system_id : *public_id, eh);
            else
              parse_begin (auto_xml_parser_, eh);
          }

          bool r (XML_Parse (xml_parser_,
                             static_cast<const char*> (data),
                             static_cast<int> (size),
                             last) != XML_STATUS_ERROR);
          parse_end ();
          return r;
        }

        // XML_Parser
        //

        template <typename C>
        void document<C>::
        parse_begin (XML_Parser parser)
        {
          xml_parser_ = parser;
          eh_ = &default_eh_;
          public_id_.clear ();
          set ();
        }

        template <typename C>
        void document<C>::
        parse_begin (XML_Parser parser,
                     const std::basic_string<C>& public_id)
        {
          xml_parser_ = parser;
          eh_ = &default_eh_;
          public_id_ = public_id;
          set ();
        }

        template <typename C>
        void document<C>::
        parse_begin (XML_Parser parser, xml::error_handler<C>& eh)
        {
          xml_parser_ = parser;
          eh_ = &eh;
          public_id_.clear ();
          set ();
        }

        template <typename C>
        void document<C>::
        parse_begin (XML_Parser parser,
                     const std::basic_string<C>& public_id,
                     xml::error_handler<C>& eh)
        {
          xml_parser_ = parser;
          eh_ = &eh;
          public_id_ = public_id;
          set ();
        }

        template <typename C>
        void document<C>::
        parse_end ()
        {
          XML_Error e (XML_GetErrorCode (xml_parser_));

          if (e == XML_ERROR_NONE || e == XML_ERROR_ABORTED)
          {
            clear ();
            xml_parser_ = 0;
            auto_xml_parser_ = 0;
          }
          else
          {
            unsigned long l = XML_GetCurrentLineNumber (xml_parser_);
            unsigned long c = XML_GetCurrentColumnNumber (xml_parser_);
            std::basic_string<C> message (XML_ErrorString (e));

            eh_->handle (public_id_,
                         l, c,
                         xml::error_handler<C>::severity::fatal,
                         message);

            clear ();
            xml_parser_ = 0;
            auto_xml_parser_ = 0;

            // We don't want to throw an empty parsing exception here
            // since the user probably already knows about the error.
          }

          if (eh_ == &default_eh_)
            default_eh_.throw_if_failed ();
        }

        //
        //
        template <typename C>
        void document<C>::
        set ()
        {
          assert (xml_parser_ != 0);

          XML_SetUserData(xml_parser_, this);

          XML_SetStartElementHandler (xml_parser_, start_element_thunk_);
          XML_SetEndElementHandler (xml_parser_, end_element_thunk_);
          XML_SetCharacterDataHandler (xml_parser_, characters_thunk_);

          if (polymorphic_)
          {
            XML_SetNamespaceDeclHandler (xml_parser_,
                                         start_namespace_decl_thunk_,
                                         end_namespace_decl_thunk_);
          }
        }

        template <typename C>
        void document<C>::
        clear ()
        {
          assert (xml_parser_ != 0);

          XML_SetUserData (xml_parser_, 0);
          XML_SetStartElementHandler (xml_parser_, 0);
          XML_SetEndElementHandler (xml_parser_, 0);
          XML_SetCharacterDataHandler (xml_parser_, 0);

          if (polymorphic_)
            XML_SetNamespaceDeclHandler (xml_parser_, 0, 0);
        }

        template <typename C>
        void document<C>::
        translate_schema_exception (const schema_exception<C>& e)
        {
          unsigned long l = XML_GetCurrentLineNumber (xml_parser_);
          unsigned long c = XML_GetCurrentColumnNumber (xml_parser_);

          eh_->handle (public_id_,
                       l, c,
                       xml::error_handler<C>::severity::fatal,
                       e.message ());

          XML_StopParser (xml_parser_, false);
        }

        // Event routing.
        //

        // Expat thunks.
        //
        template <typename C>
        void XMLCALL document<C>::
        start_element_thunk_ (void* data,
                              const XML_Char* ns_name,
                              const XML_Char** atts)
        {
          document& d (*reinterpret_cast<document*> (data));
          d.start_element_ (ns_name, atts);
        }

        template <typename C>
        void XMLCALL document<C>::
        end_element_thunk_ (void* data, const XML_Char* ns_name)
        {
          document& d (*reinterpret_cast<document*> (data));
          d.end_element_ (ns_name);
        }

        template <typename C>
        void XMLCALL document<C>::
        characters_thunk_ (void* data, const XML_Char* s, int n)
        {
          document& d (*reinterpret_cast<document*> (data));
          d.characters_ (s, static_cast<std::size_t> (n));
        }

        template <typename C>
        void XMLCALL document<C>::
        start_namespace_decl_thunk_ (void* data,
                                     const XML_Char* prefix,
                                     const XML_Char* ns)
        {
          document& d (*reinterpret_cast<document*> (data));
          d.start_namespace_decl_ (prefix, ns);
        }

        template <typename C>
        void XMLCALL document<C>::
        end_namespace_decl_thunk_ (void* data, const XML_Char* prefix)
        {
          document& d (*reinterpret_cast<document*> (data));
          d.end_namespace_decl_ (prefix);
        }

        namespace bits
        {
          inline void
          split_name (const XML_Char* s,
                      const char*& ns, std::size_t& ns_s,
                      const char*& name, std::size_t& name_s)
          {
            const char* p (std::strchr (s, ' '));

            if (p)
            {
              ns = s;
              ns_s = p - s;
              name = p + 1;
            }
            else
            {
              ns = s;
              ns_s = 0;
              name = s;
            }

            name_s = std::char_traits<char>::length (name);
          }
        }

        template <typename C>
        void document<C>::
        start_element_ (const XML_Char* ns_name, const XML_Char** atts)
        {
          // Current Expat (2.0.0) has a (mis)-feature of a possibility of
	  // calling callbacks even after the non-resumable XML_StopParser
          // call. The following code accounts for this.
          //
          {
            XML_ParsingStatus s;
            XML_GetParsingStatus (xml_parser_, &s);
            if (s.parsing == XML_FINISHED)
              return;
          }

          typedef std::basic_string<C> string;

          const char* ns_p;
          const char* name_p;
          size_t ns_s, name_s;

          bits::split_name (ns_name, ns_p, ns_s, name_p, name_s);

          {
            const ro_string<C> ns (ns_p, ns_s), name (name_p, name_s);

            if (!polymorphic_)
            {
              try
              {
                this->start_element (ns, name, 0);
              }
              catch (const schema_exception<C>& e)
              {
                translate_schema_exception (e);
                return;
              }
            }
            else
            {
              // Search for the xsi:type attribute.
              //
              const XML_Char** p = atts; // VC8 can't handle p (atts)
              for (; *p != 0; p += 2)
              {
                bits::split_name (*p, ns_p, ns_s, name_p, name_s);
                const ro_string<C> ns (ns_p, ns_s), name (name_p, name_s);

                if (name == xml::bits::type<C> () &&
                    ns == xml::bits::xsi_namespace<C> ())
                  break;
              }

              if (*p == 0)
              {
                try
                {
                  this->start_element (ns, name, 0);
                }
                catch (const schema_exception<C>& e)
                {
                  translate_schema_exception (e);
                  return;
                }
              }
              else
              {
                // @@ Need proper QName validation.
                //
                // Get the qualified type name and try to resolve it.
                //
                ro_string<C> qn (*(p + 1));

                ro_string<C> tp, tn;
                typename ro_string<C>::size_type pos (qn.find (C (':')));

                try
                {
                  if (pos != ro_string<C>::npos)
                  {
                    tp.assign (qn.data (), pos);
                    tn.assign (qn.data () + pos + 1);

                    if (tp.empty ())
                      throw dynamic_type<C> (qn);
                  }
                  else
                    tn.assign (qn.data (), qn.size ());

                  if (tn.empty ())
                    throw dynamic_type<C> (qn);

                  // Search our namespace declaration stack. Note that
                  // we need to do this even if prefix is empty. Sun CC
                  // 5.7 blows if we use const_reverse_iterator.
                  //
                  ro_string<C> tns;
                  for (typename ns_decls::reverse_iterator
                         it (ns_decls_.rbegin ()), e (ns_decls_.rend ());
                       it != e; ++it)
                  {
                    if (it->prefix == tp)
                    {
                      tns.assign (it->ns);
                      break;
                    }
                  }

                  if (!tp.empty () && tns.empty ())
                  {
                    // The 'xml' prefix requires special handling.
                    //
                    if (tp == xml::bits::xml_prefix<C> ())
                      tns.assign (xml::bits::xml_namespace<C> ());
                    else
                      throw dynamic_type<C> (qn);
                  }

                  // Construct the compound type id.
                  //
                  string id (tn.data (), tn.size ());

                  if (!tns.empty ())
                  {
                    id += C (' ');
                    id.append (tns.data (), tns.size ());
                  }

                  ro_string<C> ro_id (id);
                  this->start_element (ns, name, &ro_id);
                }
                catch (const schema_exception<C>& e)
                {
                  translate_schema_exception (e);
                  return;
                }
              }
            }
          }

          for (; *atts != 0; atts += 2)
          {
            bits::split_name (*atts, ns_p, ns_s, name_p, name_s);

            const ro_string<C> ns (ns_p, ns_s), name (name_p, name_s);
            const ro_string<C> value (*(atts + 1));

            try
            {
              this->attribute (ns, name, value);
            }
            catch (const schema_exception<C>& e)
            {
              translate_schema_exception (e);
              break;
            }
          }
        }

        template <typename C>
        void document<C>::
        end_element_ (const XML_Char* ns_name)
        {
          // Current Expat (2.0.0) has a (mis)-feature of a possibility of
	  // calling callbacks even after the non-resumable XML_StopParser
          // call. The following code accounts for this.
          //
          {
            XML_ParsingStatus s;
            XML_GetParsingStatus (xml_parser_, &s);
            if (s.parsing == XML_FINISHED)
              return;
          }

          const char* ns_p;
          const char* name_p;
          size_t ns_s, name_s;

          bits::split_name (ns_name, ns_p, ns_s, name_p, name_s);

          const ro_string<C> ns (ns_p, ns_s), name (name_p, name_s);

          try
          {
            this->end_element (ns, name);
          }
          catch (const schema_exception<C>& e)
          {
            translate_schema_exception (e);
          }
        }

        template <typename C>
        void document<C>::
        characters_ (const XML_Char* s, std::size_t n)
        {
          // Current Expat (2.0.0) has a (mis)-feature of a possibility of
	  // calling callbacks even after the non-resumable XML_StopParser
          // call. The following code accounts for this.
          //
          {
            XML_ParsingStatus s;
            XML_GetParsingStatus (xml_parser_, &s);
            if (s.parsing == XML_FINISHED)
              return;
          }

          if (n != 0)
          {
            const ro_string<C> str (s, n);

            try
            {
              this->characters (str);
            }
            catch (const schema_exception<C>& e)
            {
              translate_schema_exception (e);
            }
          }
        }

        template <typename C>
        void document<C>::
        start_namespace_decl_ (const XML_Char* p, const XML_Char* ns)
        {
          // prefix is 0 for default namespace
          // namespace is 0 when unsetting default namespace
          //
          if (polymorphic_)
            ns_decls_.push_back (ns_decl ((p ? p : ""), (ns ? ns : "")));
        }

        template <typename C>
        void document<C>::
        end_namespace_decl_ (const XML_Char* p)
        {
          // prefix is 0 for default namespace
          //
          if (polymorphic_)
          {
            // Here we assume the prefixes are removed in the reverse
            // order of them being added. This appears to how every
            // sensible implementation works.
            //
            assert (p
                    ? ns_decls_.back ().prefix == p
                    : ns_decls_.back ().prefix.empty ());

            ns_decls_.pop_back ();
          }
        }
      }
    }
  }
}
