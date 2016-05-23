// file      : xsd/cxx/tree/elements.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

/**
 * @file
 *
 * @brief Contains C++ class definitions for XML Schema anyType and
 * anySimpleType types as well as supporting code.
 *
 * This is an internal header and is included by the generated code. You
 * normally should not include it directly.
 *
 */

#ifndef XSD_CXX_TREE_ELEMENTS_HXX
#define XSD_CXX_TREE_ELEMENTS_HXX

#include <map>
#include <string>
#include <memory>  // std::auto_ptr
#include <istream>
#include <sstream>
#include <cassert>

#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/elements.hxx> // xml::properties
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/auto-ptr.hxx> // dom::auto_ptr

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/facet.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/exceptions.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/istream-fwd.hxx>

namespace xsd
{
  namespace cxx
  {
    /**
     * @brief C++/Tree mapping runtime namespace.
     *
     * This is an internal namespace and normally should not be referenced
     * directly. Instead you should use the aliases for types in this
     * namespaces that are created in the generated code.
     *
     */
    namespace tree
    {
      /**
       * @brief Parsing and %serialization %flags.
       *
       * Flags are used to modify the default behavior of %parsing and
       * %serialization functions as well as %parsing constructors.
       *
       * @nosubgrouping
       */
      class flags
      {
      public:
        /**
         * @name Flag constants
         */
        //@{

        /**
         * @brief Keep DOM association in the resulting tree.
         */
        static const unsigned long keep_dom = 0x00000100UL;

        /**
         * @brief Assume ownership of the DOM document.
         *
         * This flag only makes sense together with the @c keep_dom
         * flag in the call to the %parsing function with the
         * @c dom::auto_ptr<DOMDocument> argument.
         *
         */
        static const unsigned long own_dom = 0x00000200UL;

        /**
         * @brief Turn off XML Schema validation in the underlying XML
         * parser.
         */
        static const unsigned long dont_validate = 0x00000400UL;

        /**
         * @brief Do not initialize the Xerces-C++ runtime.
         */
        static const unsigned long dont_initialize = 0x00000001UL;

        /**
         * @brief Do not write XML declaration during %serialization.
         */
        static const unsigned long no_xml_declaration = 0x00010000UL;

        /**
         * @brief Do not add extra spaces or new lines that make the
         * resulting XML easier to read.
         */
        static const unsigned long dont_pretty_print = 0x00020000UL;

        //@cond

        // The following flags are for internal use.
        //
        static const unsigned long base = 0x01000000UL;

        //@endcond

        // Notes on flag blocks:
        //
        // 0x000000FF - common (applicable to both parsing and serialization)
        // 0x0000FF00 - parsing (values aligned with XML parsing)
        // 0x00FF0000 - serialization (values aligned with XML serialization)
        // 0xFF000000 - internal

        //@}

      public:
        /**
         * @brief Initialize an instance with an integer value.
         *
         * @param x A %flags value as an integer.
         */
        flags (unsigned long x = 0)
            : x_ (x)
        {
        }

        /**
         * @brief Convert an instance to an integer value.
         *
         * @return An integer %flags value.
         */
        operator unsigned long () const
        {
          return x_;
        }

        /**
         * @brief Combine two %flags.
         *
         * @return A %flags object that is a combination of the arguments.
         */
        friend flags
        operator| (const flags& a, const flags& b)
        {
          return flags (a.x_ | b.x_);
        }

        /**
         * @brief Combine two %flags.
         *
         * @return A %flags object that is a combination of the arguments.
         */
        friend flags
        operator| (const flags& a, unsigned long b)
        {
          return flags (a.x_ | b);
        }

        /**
         * @brief Combine two %flags.
         *
         * @return A %flags object that is a combination of the arguments.
         */
        friend flags
        operator| (unsigned long a, const flags& b)
        {
          return flags (a | b.x_);
        }

      private:
        unsigned long x_;
      };


      // Parsing properties. Refer to xsd/cxx/xml/elements.hxx for XML-
      // related properties.
      //
      template <typename C>
      class properties: public xml::properties<C>
      {
      };

      //@cond

      // DOM user data keys.
      //
      template <int dummy>
      struct user_data_keys_template
      {
        // Back pointers to tree nodes.
        //
        static const XMLCh node[21];
      };

      typedef user_data_keys_template<0> user_data_keys;

      // HP aCC3 complains about unresolved symbols without an explicit
      // instantiation.
      //
#if defined(__HP_aCC) && __HP_aCC <= 39999
      template struct user_data_keys_template<0>;
#endif
      //
      //
      struct identity
      {
        virtual
        ~identity ()
        {
        }

        identity ()
        {
        }

        virtual bool
        before (const identity&) const = 0;

        virtual void
        throw_duplicate_id () const = 0;

      private:
        identity (const identity&);

        identity&
        operator= (const identity&);
      };

      //@endcond


      // anyType. VC++ has a name injection bug that makes it impossible
      // to have a member with the same name as a base type. To address
      // that we will have to choose some unique name for the definition
      // and typedef it to 'type'.
      //
      class _type;

      /**
       * @brief Class corresponding to the XML Schema anyType built-in type.
       *
       */
      typedef _type type;

      /**
       * @brief Container type.
       *
       */
      typedef _type container;

      /**
       * @brief Class corresponding to the XML Schema anyType built-in type.
       *
       * This class is a base for every generated and built-in type in the
       * C++/Tree mapping.
       *
       * @nosubgrouping
       */
      class _type
      {
      public:
        virtual
        ~_type ()
        {
          // Everything should have been unregistered by now.
          //
          assert (map_.get () == 0 || map_->size () == 0);
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor.
         */
        _type ()
            : container_ (0)
        {
        }

      public:
        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the @c _clone function instead.
         */
        _type (const type& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual type*
        _clone (flags f = 0, container* c = 0) const
        {
          return new type (*this, f, c);
        }

      public:
        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        _type (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        _type (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        _type (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename C>
        _type (const std::basic_string<C>& s,
               const xercesc::DOMElement* e,
               flags f = 0,
               container* c = 0);
        //@}

      public:
        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        type&
        operator= (const type& x) = delete;

        // Container API.
        //
      public:
        /**
         * @brief Get a constant pointer to container, an object model
         * node that contains this instance.
         *
         * @return A constant pointer to container, or 0 if this instance
         * is not contained.
         */
        const container*
        _container () const
        {
          return container_;
        }

        /**
         * @brief Get a pointer to container, an object model node that
         * contains this instance.
         *
         * @return A pointer to container, or 0 if this instance is not
         * contained.
         */
        container*
        _container ()
        {
          return container_;
        }

        /**
         * @brief Set this instance's new container, an object model node
         * that contains this instance.
         *
         * @param c A pointer to container.
         */
        virtual void
        _container (container* c)
        {
          container* dr (0);

          if (c != 0)
          {
            dr = c->_root ();

            if (dr == 0)
              dr = c;
          }

          std::auto_ptr<map>& m (dr ? dr->map_ : map_);

          if (container_ == 0)
          {
            if (c != 0 && map_.get () != 0)
            {
              // Transfer our IDs to the new root.
              //
              if (m.get () != 0)
              {
                m->insert (map_->begin (), map_->end ());
                std::auto_ptr<map> tmp (0);
                map_ = tmp;
              }
              else
                m = map_;
            }
          }
          else
          {
            container* sr (_root ());

            if (sr->map_.get () != 0)
            {
              // Transfer IDs that belong to this subtree.
              //
              for (map::iterator i (sr->map_->begin ()), e (sr->map_->end ());
                   i != e;)
              {
                type* x (i->second);
                for (; x != this && x != sr; x = x->_container ()) ;

                if (x != sr)
                {
                  // Part of our subtree.
                  //
                  if (m.get () == 0)
                  {
                    std::auto_ptr<map> tmp (new map);
                    m = tmp;
                  }

                  m->insert (*i);
                  sr->map_->erase (i++);
                }
                else
                  ++i;
              }
            }
          }

          container_ = c;
        }

        /**
         * @brief Get a constant pointer to object model's root node.
         *
         * @return A constant pointer to root node, or 0 if this instance
         * is not contained.
         */
        const container*
        _root () const
        {
          const container* r (container_);

          for (const container* c (r); c != 0; c = c->container_)
            r = c;

          return r;
        }

        /**
         * @brief Get a pointer to object model's root node.
         *
         * @return A pointer to root node, or 0 if this instance is not
         * contained.
         */
        container*
        _root ()
        {
          container* r (container_);

          for (container* c (r); c != 0; c = c->container_)
            r = c;

          return r;
        }

        // DOM association.
        //
      public:
        /**
         * @brief Get a constant pointer to a DOM node associated with
         * this object model node.
         *
         * @return A constant pointer to DOM node, or 0 if none associated.
         */
        const xercesc::DOMNode*
        _node () const
        {
          return dom_info_.get () ? dom_info_->node() : 0;
        }

        /**
         * @brief Get a pointer to a DOM node associated with this object
         * model node.
         *
         * @return A pointer to DOM node, or 0 if none associated.
         */
        xercesc::DOMNode*
        _node ()
        {
          return dom_info_.get () ? dom_info_->node () : 0;
        }

        /**
         * @brief Exception indicating that a DOM node cannot be associated
         * with an object model node.
         */
        class bad_dom_node_type: public std::exception //@@ Inherit exception.
        {
        public:
          /**
           * @brief Get %exception description.
           *
           * @return A C %string describing the %exception.
           */
          virtual const char*
          what () const throw ()
          {
            return "DOM node is not an attribute node or element node";
          }
        };

        /**
         * @brief Manually set a DOM node associated with this object
         * model node.
         *
         * The DOM node should be a child of the parent's DOM node. If
         * this object model node is a root of the tree, then it will
         * assume the ownership of the whole DOM document to which this
         * DOM node belongs.
         *
         * @param n A pointer to DOM node (should be either an element or
         * an attribute).
         */
        void
        _node (xercesc::DOMNode* n)
        {
          switch (n->getNodeType ())
          {
          case xercesc::DOMNode::ELEMENT_NODE:
            {
              if (container_ != 0)
              {
                // @@ Should be a throw.
                //
                assert (_root ()->_node () != 0);
                assert (_root ()->_node ()->getOwnerDocument () ==
                        n->getOwnerDocument ());
              }

              std::auto_ptr<dom_info> r (
                dom_info_factory::create (
                  *static_cast<xercesc::DOMElement*> (n),
                  *this,
                  container_ == 0));

              dom_info_ = r;
              break;
            }
          case xercesc::DOMNode::ATTRIBUTE_NODE:
            {
              //@@ Should be a throw.
              //
              assert (container_ != 0);
              assert (_root ()->_node () != 0);
              assert (_root ()->_node ()->getOwnerDocument () ==
                      n->getOwnerDocument ());

              std::auto_ptr<dom_info> r (
                dom_info_factory::create (
                  *static_cast<xercesc::DOMAttr*> (n),
                  *this));

              dom_info_ = r;
              break;
            }
          default:
            {
              throw bad_dom_node_type ();
            }
          }
        }

      public:
        //@cond

        void
        _register_id (const identity& id, type* t)
        {
          // We should be the root.
          //
          assert (container_ == 0);

          if (map_.get () == 0)
          {
            std::auto_ptr<map> tmp (new map);
            map_ = tmp;
          }

          if (!map_->insert (
                std::pair<const identity*, type*> (&id, t)).second)
          {
            id.throw_duplicate_id ();
          }
        }

        //@@ Does not inherit from exception.
        //
        struct not_registered: std::exception
        {
          virtual const char*
          what () const throw ()
          {
            return "attempt to unregister non-existent id";
          }
        };

        void
        _unregister_id (const identity& id)
        {
          // We should be the root.
          //
          assert (container_ == 0);

          if (map_.get () == 0 || map_->erase (&id) == 0)
            throw not_registered ();
        }

        type*
        _lookup_id (const identity& id) const
        {
          if (map_.get ())
          {
            map::const_iterator it (map_->find (&id));

            if (it != map_->end ())
              return it->second;
          }

          return 0;
        }

        //@endcond

      private:
        //@cond

        struct dom_info
        {
          virtual
          ~dom_info ()
          {
          }

          dom_info ()
          {
          }

          virtual std::auto_ptr<dom_info>
          clone (type& tree_node, container*) const = 0;

          virtual xercesc::DOMNode*
          node () = 0;

        private:
          dom_info (const dom_info&);

          dom_info&
          operator= (const dom_info&);
        };


        struct dom_element_info: public dom_info
        {
          dom_element_info (xercesc::DOMElement& e, type& n, bool root)
              : doc_ (0), e_ (e)
          {
            e_.setUserData (user_data_keys::node, &n, 0);

            if (root)
            {
              // The caller should have associated a dom::auto_ptr object
              // that owns this document with the document node using the
              // xml_schema::dom::tree_node_key key.
              //
              xml::dom::auto_ptr<xercesc::DOMDocument>* pd (
                reinterpret_cast<xml::dom::auto_ptr<xercesc::DOMDocument>*> (
                  e.getOwnerDocument ()->getUserData (user_data_keys::node)));

              assert (pd != 0);
              assert (pd->get () == e.getOwnerDocument ());

              doc_ = *pd; // Transfer ownership.
            }
          }

          virtual std::auto_ptr<dom_info>
          clone (type& tree_node, container* c) const
          {
            using std::auto_ptr;

            // Check if we are a document root.
            //
            if (c == 0)
            {
              // We preserver DOM associations only in complete
              // copies from root.
              //
              if (doc_.get () == 0)
                return auto_ptr<dom_info> (0);

              return auto_ptr<dom_info> (
                new dom_element_info (*doc_, tree_node));
            }

            // Check if our container does not have DOM association (e.g.,
            // because it wasn't a complete copy of the tree).
            //
            using xercesc::DOMNode;

            DOMNode* cn (c->_node ());

            if (cn == 0)
              return auto_ptr<dom_info> (0);


            // Now we are going to find the corresponding element in
            // the new tree.
            //
            {
              using xercesc::DOMElement;

              DOMNode& pn (*e_.getParentNode ());
              assert (pn.getNodeType () == DOMNode::ELEMENT_NODE);

              DOMNode* sn (pn.getFirstChild ()); // Source.
              DOMNode* dn (cn->getFirstChild ()); // Destination.

              // We should have at least one child.
              //
              assert (sn != 0);

              // Move in parallel until we get to the needed node.
              //
              for (; sn != 0 && !e_.isSameNode (sn);)
              {
                sn = sn->getNextSibling ();
                dn = dn->getNextSibling ();
              }

              // e_ should be on the list.
              //
              assert (sn != 0);

              assert (dn->getNodeType () == DOMNode::ELEMENT_NODE);

              return auto_ptr<dom_info> (
                new dom_element_info (static_cast<DOMElement&> (*dn),
                                      tree_node,
                                      false));
            }
          }

          virtual xercesc::DOMNode*
          node ()
          {
            return &e_;
          }

        private:
          dom_element_info (const xercesc::DOMDocument& d, type& n)
              : doc_ (static_cast<xercesc::DOMDocument*> (
                        d.cloneNode (true))),
                e_ (*doc_->getDocumentElement ())
          {
            e_.setUserData (user_data_keys::node, &n, 0);
          }

        private:
          xml::dom::auto_ptr<xercesc::DOMDocument> doc_;
          xercesc::DOMElement& e_;
        };


        struct dom_attribute_info: public dom_info
        {
          dom_attribute_info (xercesc::DOMAttr& a, type& n)
              : a_ (a)
          {
            a_.setUserData (user_data_keys::node, &n, 0);
          }

          virtual std::auto_ptr<dom_info>
          clone (type& tree_node, container* c) const
          {
            using std::auto_ptr;

            // Check if we are a document root.
            //
            if (c == 0)
            {
              // We preserver DOM associations only in complete
              // copies from root.
              //
              return auto_ptr<dom_info> (0);
            }

            // Check if our container does not have DOM association (e.g.,
            // because it wasn't a complete copy of the tree).
            //
            using xercesc::DOMNode;

            DOMNode* cn (c->_node ());

            if (cn == 0)
              return auto_ptr<dom_info> (0);

            // We are going to find the corresponding attribute in
            // the new tree.
            //
            using xercesc::DOMAttr;
            using xercesc::DOMElement;
            using xercesc::DOMNamedNodeMap;

            DOMElement& p (*a_.getOwnerElement ());
            DOMNamedNodeMap& nl (*p.getAttributes ());

            XMLSize_t size (nl.getLength ()), i (0);

            // We should have at least one child.
            //
            assert (size != 0);

            for ( ;i < size && !a_.isSameNode (nl.item (i)); ++i)/*noop*/;

            // a_ should be in the list.
            //
            assert (i < size);

            DOMNode& n (*cn->getAttributes ()->item (i));
            assert (n.getNodeType () == DOMNode::ATTRIBUTE_NODE);

            return auto_ptr<dom_info> (
              new dom_attribute_info (static_cast<DOMAttr&> (n), tree_node));
          }

          virtual xercesc::DOMNode*
          node ()
          {
            return &a_;
          }

        private:
          xercesc::DOMAttr& a_;
        };

        // For Sun C++ 5.6.
        //
        struct dom_info_factory;
        friend struct _type::dom_info_factory;

        struct dom_info_factory
        {
          static std::auto_ptr<dom_info>
          create (const xercesc::DOMElement& e, type& n, bool root)
          {
            return std::auto_ptr<dom_info> (
              new dom_element_info (
                const_cast<xercesc::DOMElement&> (e), n, root));
          }

          static std::auto_ptr<dom_info>
          create (const xercesc::DOMAttr& a, type& n)
          {
            return std::auto_ptr<dom_info> (
              new dom_attribute_info (
                const_cast<xercesc::DOMAttr&> (a), n));
          }
        };

        //@endcond

        std::auto_ptr<dom_info> dom_info_;


        // ID/IDREF map.
        //
      private:

        //@cond

        struct identity_comparator
        {
          bool operator () (const identity* x, const identity* y) const
          {
            return x->before (*y);
          }
        };

        //@endcond

        typedef
        std::map<const identity*, type*, identity_comparator>
        map;

        std::auto_ptr<map> map_;

      private:
        container* container_;
      };

      inline _type::
      _type (const type& x, flags, container* c)
          : container_ (c)
      {
        if (x.dom_info_.get ())
        {
          std::auto_ptr<dom_info> r (x.dom_info_->clone (*this, c));
          dom_info_ = r;
        }
      }


      /**
       * @brief Class corresponding to the XML Schema anySimpleType built-in
       * type.
       *
       * @nosubgrouping
       */
      template <typename B>
      class simple_type: public B
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor.
         */
        simple_type ();

        /**
         * @brief Create an instance from a string.
         *
         * @param s A string to initialize the instance with.
         */
        template <typename C>
        simple_type (const C* s);

      public:
        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the @c _clone function instead.
         */
        simple_type (const simple_type& x, flags f = 0, container* c = 0);

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual simple_type*
        _clone (flags f = 0, container* c = 0) const;

      public:
        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        simple_type (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        simple_type (const xercesc::DOMElement& e,
                     flags f = 0,
                     container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        simple_type (const xercesc::DOMAttr& a,
                     flags f = 0,
                     container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename C>
        simple_type (const std::basic_string<C>& s,
                     const xercesc::DOMElement* e,
                     flags f = 0,
                     container* c = 0);
        //@}
      };


      /**
       * @brief Base class for element types.
       *
       * This class is a base for every generated element type.
       *
       * @nosubgrouping
       */
      template <typename C, typename T>
      class element_type
      {
      public:
        virtual
        ~element_type ()
        {
        }

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual element_type*
        _clone (flags f = 0) const = 0;

        /**
         * @brief Return the element name.
         *
         * @return A read-only string reference containing the element
         * name.
         */
        virtual const std::basic_string<C>&
        _name () const = 0;

        /**
         * @brief Return the element namespace.
         *
         * @return A read-only string reference containing the element
         * namespace. Empty string is returned if the element is
         * unqualified.
         */
        virtual const std::basic_string<C>&
        _namespace () const = 0;

        /**
         * @brief Return the element value.
         *
         * @return A pointer to the element value or 0 if the element
         * is of a fundamental type.
         */
        virtual T*
        _value () = 0;

        /**
         * @brief Return the element value.
         *
         * @return A read-only pointer to the element value or 0 if the
         * element is of a fundamental type.
         */
        virtual const T*
        _value () const = 0;
      };


      //@cond

      // Extra schema type id to disambiguate certain cases where
      // different XML Schema types (e.g., double and decimal) are
      // mapped to the same fundamental C++ type (e.g., double).
      //
      struct schema_type
      {
        enum value
        {
          other,
          double_,
          decimal
        };
      };

      //@endcond


      //@cond
      template <typename T,
                typename C,
                schema_type::value ST = schema_type::other>
      struct traits
      {
        typedef T type;

        static std::auto_ptr<T>
        create (const xercesc::DOMElement& e, flags f, container* c)
        {
          return std::auto_ptr<T> (new T (e, f, c));
        }

        static std::auto_ptr<T>
        create (const xercesc::DOMAttr& a, flags f, container* c)
        {
          return std::auto_ptr<T> (new T (a, f, c));
        }

        static std::auto_ptr<T>
        create (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f,
                container* c)
        {
          return std::auto_ptr<T> (new T (s, e, f, c));
        }
      };

      //@endcond


      /**
       * @brief Class template that emulates inheritance from a
       * fundamental C++ type.
       *
       * @nosubgrouping
       */
      template <typename T,
                typename C,
                typename B,
                schema_type::value ST = schema_type::other>
      class fundamental_base: public B
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor.
         */
        fundamental_base ()
            : facet_table_ (0), x_ ()
        {
        }

        /**
         * @brief Initialize an instance with an underlying type value.
         *
         * @param x An underlying type value.
         */
        fundamental_base (T x)
            : facet_table_ (0), x_ (x)
        {
        }

      public:
        /**
         * @brief Copy constructor.
         *
         * @param x An instance to make a copy of.
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         *
         * For polymorphic object models use the @c _clone function instead.
         */
        fundamental_base (const fundamental_base& x,
                          flags f = 0,
                          container* c = 0)
            : B (x, f, c), facet_table_ (0), x_ (x.x_)
        {
        }

        /**
         * @brief Copy the instance polymorphically.
         *
         * @param f Flags to create the copy with.
         * @param c A pointer to the object that will contain the copy.
         * @return A pointer to the dynamically allocated copy.
         *
         * This function ensures that the dynamic type of the instance
         * is used for copying and should be used for polymorphic object
         * models instead of the copy constructor.
         */
        virtual fundamental_base*
        _clone (flags f = 0, container* c = 0) const;

      public:
        /**
         * @brief Create an instance from a data representation
         * stream.
         *
         * @param s A stream to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        template <typename S>
        fundamental_base (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        fundamental_base (const xercesc::DOMElement& e,
                          flags f = 0,
                          container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        fundamental_base (const xercesc::DOMAttr& a,
                          flags f = 0,
                          container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        fundamental_base (const std::basic_string<C>& s,
                          const xercesc::DOMElement* e,
                          flags f = 0,
                          container* c = 0);
        //@}

      public:
        /**
         * @brief Assign an underlying type value to the instance.
         *
         * @param x An underlying type value.
         * @return A reference to the instance.
         */
        fundamental_base&
        operator= (const T& x)
        {
          if (&x_ != &x)
            x_ = x;

          return *this;
        }

      public:
        /**
         * @brief Implicitly convert the instance to constant reference to
         * the underlying type.
         *
         * @return A constant reference to the underlying type.
         */
        operator const T& () const
        {
          return x_;
        }

        /**
         * @brief Implicitly convert the instance to reference to the
         * underlying type.
         *
         * @return A reference to the underlying type.
         */
        operator T& ()
        {
          return x_;
        }

        // The following extra conversion operators causes problems on
        // some compilers (notably VC 7.1 and 9.0) and are disabled by
        // default.
        //
#ifdef XSD_TREE_EXTRA_FUND_CONV
        /**
         * @brief Implicitly convert the instance to another type (const
         * version).
         *
         * @return A value converted to the target type.
         */
        template <typename T2>
        operator T2 () const
        {
          return x_;
        }

        /**
         * @brief Implicitly convert the instance to another type.
         *
         * @return A value converted to the target type.
         */
        template <typename T2>
        operator T2 ()
        {
          return x_;
        }
#endif // XSD_TREE_EXTRA_FUND_CONV

      public:
        /**
         * @brief Get the facet table associated with this type.
         *
         * @return A pointer to read-only facet table or 0.
         */
        const facet*
        _facet_table () const
        {
          return facet_table_;
        }

      protected:
        /**
         * @brief Set the facet table associated with this type.
         *
         * @param ft A pointer to read-only facet table.
         */
        void
        _facet_table (const facet* ft)
        {
          facet_table_ = ft;
        }

      private:
        const facet* facet_table_;
        T x_;
      };

      // While thse operators are not normally necessary, they
      // help resolve ambiguities between implicit conversion and
      // construction.
      //

      /**
       * @brief %fundamental_base comparison operator.
       *
       * @return True if the underlying values are equal, false otherwise.
       */
      template <typename T, typename C, typename B, schema_type::value ST>
      inline bool
      operator== (const fundamental_base<T, C, B, ST>& x,
                  const fundamental_base<T, C, B, ST>& y)
      {
        T x_ (x);
        T y_ (y);
        return x_ == y_;
      }

      /**
       * @brief %fundamental_base comparison operator.
       *
       * @return True if the underlying values are not equal, false otherwise.
       */
      template <typename T, typename C, typename B, schema_type::value ST>
      inline bool
      operator!= (const fundamental_base<T, C, B, ST>& x,
                  const fundamental_base<T, C, B, ST>& y)
      {
        T x_ (x);
        T y_ (y);
        return x_ != y_;
      }


      //@cond

      // Comparator for enum tables.
      //
      template <typename C>
      struct enum_comparator
      {
        enum_comparator (const C* const* table)
            : table_ (table)
        {
        }

        bool
        operator() (std::size_t i, const std::basic_string<C>& s) const
        {
          return table_[i] < s;
        }

        bool
        operator() (const std::basic_string<C>& s, std::size_t i) const
        {
          return s < table_[i];
        }

        bool
        operator() (std::size_t i, std::size_t j) const
        {
          return std::basic_string<C> (table_[i]) < table_[j];
        }

      private:
        const C* const* table_;
      };

      //@endcond
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.ixx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.txx>

#endif  // XSD_CXX_TREE_ELEMENTS_HXX
