// file      : xsd/cxx/parser/document.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <cassert>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/parser/schema-exceptions.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace parser
    {
      // document
      //
      template <typename C>
      document<C>::
      ~document ()
      {
      }

      template <typename C>
      document<C>::
      document (parser_base<C>& root,
                const std::basic_string<C>& ns,
                const std::basic_string<C>& name)
          : root_ (&root), ns_ (ns), name_ (name), depth_ (0)
      {
      }

      template <typename C>
      document<C>::
      document ()
          : root_ (0), depth_ (0)
      {
      }

      template <typename C>
      void document<C>::
      start_element (const ro_string<C>& ns,
                     const ro_string<C>& name,
                     const ro_string<C>* type)
      {
        if (depth_++ > 0)
        {
          if (root_)
            root_->_start_element (ns, name, type);
        }
        else
        {
          root_ = start_root_element (ns, name, type);

          if (root_)
          {
            // pre () is called by the user.
            //
            root_->_pre_impl ();
          }
        }
      }

      template <typename C>
      void document<C>::
      end_element (const ro_string<C>& ns, const ro_string<C>& name)
      {
        assert (depth_ > 0);

        if (--depth_ > 0)
        {
          if (root_)
            root_->_end_element (ns, name);
        }
        else
        {
          if (root_)
          {
	    root_->_post_impl ();
            //
            // post() is called by the user.
          }

          end_root_element (ns, name, root_);
        }
      }

      template <typename C>
      void document<C>::
      attribute (const ro_string<C>& ns,
                 const ro_string<C>& name,
                 const ro_string<C>& value)
      {
        if (root_)
          root_->_attribute (ns, name, value);
      }

      template <typename C>
      void document<C>::
      characters (const ro_string<C>& s)
      {
        if (root_)
          root_->_characters (s);
      }

      template <typename C>
      parser_base<C>* document<C>::
      start_root_element (const ro_string<C>& ns,
                          const ro_string<C>& name,
                          const ro_string<C>*)
      {
        if (name_ == name && ns_ == ns)
        {
          return root_;
        }
        else
          throw expected_element<C> (ns_, name_, ns, name);
      }

      template <typename C>
      void document<C>::
      end_root_element (const ro_string<C>&,
                        const ro_string<C>&,
                        parser_base<C>*)
      {
      }
    }
  }
}
