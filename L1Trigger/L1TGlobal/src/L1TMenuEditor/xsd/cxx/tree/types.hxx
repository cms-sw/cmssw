// file      : xsd/cxx/tree/types.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

/**
 * @file
 *
 * @brief Contains C++ class definitions for XML Schema built-in types.
 *
 * This is an internal header and is included by the generated code. You
 * normally should not include it directly.
 *
 */

#ifndef XSD_CXX_TREE_TYPES_HXX
#define XSD_CXX_TREE_TYPES_HXX

#include <string>
#include <cstddef> // std::size_t

#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMElement.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/elements.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/list.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/buffer.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/istream-fwd.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/date-time.hxx>

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
       * @brief Class corresponding to the XML Schema %string built-in
       * type.
       *
       * The %string class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class string: public B, public std::basic_string<C>
      {
        typedef std::basic_string<C> base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor creates an empty %string.
         */
        string ()
        {
        }

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        string (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        string (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        string (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        string (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        string (const std::basic_string<C>& s,
                std::size_t pos,
                std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        string (const string& x, flags f = 0, container* c = 0)
            : B (x, f, c), base_type (x)
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
        virtual string*
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
        string (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        string (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        string (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        string (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f = 0,
                container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %string has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        string&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %string contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        string&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %string contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        string&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        string&
        operator= (const string& x)
        {
          base () = x;
          return *this;
        }
      };


      /**
       * @brief Class corresponding to the XML Schema normalizedString
       * built-in type.
       *
       * The %normalized_string class publicly inherits from and has
       * the same set of constructors as @c std::basic_string. It
       * therefore can be used as @c std::string (or @c std::wstring
       * if you are using @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class normalized_string: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor creates an empty %normalized_string.
         */
        normalized_string ()
        {
        }

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        normalized_string (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        normalized_string (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        normalized_string (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        normalized_string (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        normalized_string (const std::basic_string<C>& s,
                           std::size_t pos,
                           std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        normalized_string (const normalized_string& x,
                           flags f = 0,
                           container* c = 0)
            : base_type (x, f, c)
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
        virtual normalized_string*
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
        normalized_string (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        normalized_string (const xercesc::DOMElement& e,
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
        normalized_string (const xercesc::DOMAttr& a,
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
        normalized_string (const std::basic_string<C>& s,
                           const xercesc::DOMElement* e,
                           flags f = 0,
                           container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %normalized_string has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        normalized_string&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %normalized_string contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        normalized_string&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %normalized_string contains a copy of the standard
         * %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        normalized_string&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        normalized_string&
        operator= (const normalized_string& x)
        {
          base () = x;
          return *this;
        }

      protected:
        //@cond

        void
        normalize ();

        //@endcond
      };


      /**
       * @brief Class corresponding to the XML Schema %token built-in
       * type.
       *
       * The %token class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class token: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor creates an empty %token.
         */
        token ()
        {
        }

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        token (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        token (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        token (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        token (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        token (const std::basic_string<C>& s,
               std::size_t pos,
               std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        token (const token& x, flags f = 0, container* c = 0)
            : base_type (x, f, c)
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
        virtual token*
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
        token (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        token (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        token (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        token (const std::basic_string<C>& s,
               const xercesc::DOMElement* e,
               flags f = 0,
               container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %token has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        token&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %token contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        token&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %token contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        token&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        token&
        operator= (const token& x)
        {
          base () = x;
          return *this;
        }

      protected:
        //@cond

        void
        collapse ();

        //@endcond
      };


      /**
       * @brief Class corresponding to the XML Schema NMTOKEN built-in
       * type.
       *
       * The %nmtoken class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class nmtoken: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        nmtoken (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        nmtoken (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        nmtoken (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        nmtoken (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        nmtoken (const std::basic_string<C>& s,
                 std::size_t pos,
                 std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        nmtoken (const nmtoken& x, flags f = 0, container* c = 0)
            : base_type (x, f, c)
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
        virtual nmtoken*
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
        nmtoken (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        nmtoken (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        nmtoken (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        nmtoken (const std::basic_string<C>& s,
                 const xercesc::DOMElement* e,
                 flags f = 0,
                 container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %nmtoken has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        nmtoken&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %nmtoken contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        nmtoken&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %nmtoken contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        nmtoken&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        nmtoken&
        operator= (const nmtoken& x)
        {
          base () = x;
          return *this;
        }

      protected:
        //@cond

        nmtoken ()
            : base_type ()
        {
        }

        //@endcond
      };


      /**
       * @brief Class corresponding to the XML Schema NMTOKENS built-in
       * type.
       *
       * The %nmtokens class is a vector (or %list in XML Schema terminology)
       * of nmtoken elements. It is implemented in terms of the list class
       * template.
       *
       * @nosubgrouping
       */
      template <typename C, typename B, typename nmtoken>
      class nmtokens: public B, public list<nmtoken, C>
      {
        typedef list<nmtoken, C> base_type;

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor creates no elements.
         */
        nmtokens ()
            : base_type (0, this)
        {
        }

        /**
         * @brief Initialize the instance with copies of an exemplar elements.
         *
         * @param n A number of elements to copy.
         * @param x An exemplar element to copy.
         */
        nmtokens (typename base_type::size_type n, const nmtoken& x)
            : base_type (n, x, this)
        {
        }

        /**
         * @brief Initialize the instance with copies of elements from an
         * iterator range.
         *
         * @param begin An iterator pointing to the first element.
         * @param end An iterator pointing to the one past the last element.
         */
        template <typename I>
        nmtokens (const I& begin, const I& end)
            : base_type (begin, end, this)
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
        nmtokens (const nmtokens& x, flags f, container* c = 0)
            : B (x, f, c), base_type (x, f, this)
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
        virtual nmtokens*
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
        nmtokens (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        nmtokens (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        nmtokens (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        nmtokens (const std::basic_string<C>& s,
                  const xercesc::DOMElement* e,
                  flags f = 0,
                  container* c = 0);
        //@}
      };


      /**
       * @brief Class corresponding to the XML Schema Name built-in
       * type.
       *
       * The %name class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class name: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        name (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        name (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        name (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        name (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        name (const std::basic_string<C>& s,
              std::size_t pos,
              std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        name (const name& x, flags f = 0, container* c = 0)
            : base_type (x, f, c)
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
        virtual name*
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
        name (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        name (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        name (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        name (const std::basic_string<C>& s,
              const xercesc::DOMElement* e,
              flags f = 0,
              container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %name has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        name&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %name contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        name&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %name contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        name&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        name&
        operator= (const name& x)
        {
          base () = x;
          return *this;
        }

      protected:
        //@cond

        name ()
            : base_type ()
        {
        }

        //@endcond
      };


      // Forward declaration for Sun CC.
      //
      template <typename C, typename B, typename uri, typename ncname>
      class qname;


      /**
       * @brief Class corresponding to the XML Schema NCame built-in
       * type.
       *
       * The %ncname class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class ncname: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        ncname (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        ncname (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        ncname (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        ncname (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        ncname (const std::basic_string<C>& s,
                std::size_t pos,
                std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        ncname (const ncname& x, flags f = 0, container* c = 0)
            : base_type (x, f, c)
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
        virtual ncname*
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
        ncname (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        ncname (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        ncname (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        ncname (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f = 0,
                container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %ncname has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        ncname&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %ncname contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        ncname&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %ncname contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        ncname&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        ncname&
        operator= (const ncname& x)
        {
          base () = x;
          return *this;
        }

      protected:
        //@cond

        ncname ()
            : base_type ()
        {
        }

        //@endcond

        template <typename, typename, typename, typename>
        friend class qname;
      };


      /**
       * @brief Class corresponding to the XML Schema %language built-in
       * type.
       *
       * The %language class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class language: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        language (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        language (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        language (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        language (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        language (const std::basic_string<C>& s,
                  std::size_t pos,
                  std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        language (const language& x, flags f = 0, container* c = 0)
            : base_type (x, f, c)
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
        virtual language*
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
        language (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        language (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        language (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        language (const std::basic_string<C>& s,
                  const xercesc::DOMElement* e,
                  flags f = 0,
                  container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %language has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        language&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %language contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        language&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %language contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        language&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        language&
        operator= (const language& x)
        {
          base () = x;
          return *this;
        }

      protected:
        //@cond

        language ()
            : base_type ()
        {
        }

        //@endcond
      };


      //@cond

      template <typename C, typename ncname>
      struct identity_impl: identity
      {
        identity_impl (const ncname& id)
            : id_ (id)
        {
        }

        virtual bool
        before (const identity& y) const;

        virtual void
        throw_duplicate_id () const;

      private:
        const ncname& id_;
      };

      //@endcond


      /**
       * @brief Class corresponding to the XML Schema ID built-in
       * type.
       *
       * The %id class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class id: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        ~id()
        {
          unregister_id ();
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        id (const C* s)
            : base_type (s), identity_ (*this)
        {
          register_id ();
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        id (const C* s, std::size_t n)
            : base_type (s, n), identity_ (*this)
        {
          register_id ();
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        id (std::size_t n, C c)
            : base_type (n, c), identity_ (*this)
        {
          register_id ();
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        id (const std::basic_string<C>& s)
            : base_type (s), identity_ (*this)
        {
          register_id ();
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        id (const std::basic_string<C>& s,
            std::size_t pos,
            std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n), identity_ (*this)
        {
          register_id ();
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
        id (const id& x, flags f = 0, container* c = 0)
            : base_type (x, f, c), identity_ (*this)
        {
          register_id ();
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
        virtual id*
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
        id (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        id (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        id (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        id (const std::basic_string<C>& s,
            const xercesc::DOMElement* e,
            flags f = 0,
            container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %id has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        id&
        operator= (C c);


        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %id contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        id&
        operator= (const C* s);

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %id contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        id&
        operator= (const std::basic_string<C>& s);

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        id&
        operator= (const id& x);

      public:
        //@cond

        virtual void
        _container (container*);

        // The above override also hides other _container versions. We
        // also cannot do using-declarations because of bugs in HP aCC3.
        //
        const container*
        _container () const
        {
          return B::_container ();
        }

        container*
        _container ()
        {
          return B::_container ();
        }

        //@endcond

      protected:
        //@cond

        id ()
            : base_type (), identity_ (*this)
        {
          register_id ();
        }

        //@endcond

      private:
        void
        register_id ();

        void
        unregister_id ();

      private:
        identity_impl<C, B> identity_;
      };


      /**
       * @brief Class corresponding to the XML Schema IDREF built-in
       * type.
       *
       * The %idref class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * The %idref class also provides an autopointer-like interface
       * for resolving referenced objects. By default the object is
       * returned as type (mapping for anyType) but statically-typed
       * %idref can be created using the XML Schema extension. See the
       * C++/Tree Mapping User Manual for more information.
       *
       * @nosubgrouping
       */
      template <typename C, typename B, typename T>
      class idref: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @brief Referenced type.
         */
        typedef T ref_type;

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        idref (const C* s)
            : base_type (s), identity_ (*this)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        idref (const C* s, std::size_t n)
            : base_type (s, n), identity_ (*this)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        idref (std::size_t n, C c)
            : base_type (n, c), identity_ (*this)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        idref (const std::basic_string<C>& s)
            : base_type (s), identity_ (*this)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        idref (const std::basic_string<C>& s,
               std::size_t pos,
               std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n), identity_ (*this)
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
        idref (const idref& x, flags f = 0, container* c = 0)
            : base_type (x, f, c), identity_ (*this)
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
        virtual idref*
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
        idref (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        idref (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        idref (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        idref (const std::basic_string<C>& s,
               const xercesc::DOMElement* e,
               flags f = 0,
               container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %idref has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        idref&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %idref contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        idref&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %idref contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        idref&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        idref&
        operator= (const idref& x)
        {
          base () = x;
          return *this;
        }

      public:
        /**
         * @brief Call referenced object.
         *
         * @return A constant pointer to the referenced object.
         */
        const ref_type*
        operator-> () const
        {
          return get ();
        }

        /**
         * @brief Call referenced object.
         *
         * @return A pointer to the referenced object.
         */
        ref_type*
        operator-> ()
        {
          return get ();
        }

        /**
         * @brief Dereference referenced object.
         *
         * @return A constant C++ reference to the referenced object.
         */
        const ref_type&
        operator* () const
        {
          return *(get ());
        }

        /**
         * @brief Dereference referenced object.
         *
         * @return A C++ reference to the referenced object.
         */
        ref_type&
        operator* ()
        {
          return *(get ());
        }

        /**
         * @brief Get a constant pointer to the referenced object.
         *
         * @return A constant pointer to the referenced object or 0 if
         * the object is not found.
         */
        const ref_type*
        get () const
        {
          return dynamic_cast<const ref_type*> (get_ ());
        }

        /**
         * @brief Get a pointer to the referenced object.
         *
         * @return A pointer to the referenced object or 0 if the object
         * is not found.
         */
        ref_type*
        get ()
        {
          return dynamic_cast<ref_type*> (get_ ());
        }

        /**
         * @brief Opaque type that can be evaluated as true or false.
         */
        typedef void (idref::*bool_convertible)();

        /**
         * @brief Implicit conversion to boolean type.
         *
         * @return True if the referenced object is found, false otherwise.
         */
        operator bool_convertible () const
        {
          return get_ () ? &idref::true_ : 0;
        }

      protected:
        //@cond

        idref ()
            : base_type (), identity_ (*this)
        {
        }

        //@endcond

      private:
        const _type*
        get_ () const;

        _type*
        get_ ();

        void
        true_ ();

      private:
        identity_impl<C, B> identity_;
      };


      /**
       * @brief Class corresponding to the XML Schema IDREFS built-in
       * type.
       *
       * The %idrefs class is a vector (or %list in XML Schema terminology)
       * of idref elements. It is implemented in terms of the list class
       * template.
       *
       * @nosubgrouping
       */
      template <typename C, typename B, typename idref>
      class idrefs: public B, public list<idref, C>
      {
        typedef list<idref, C> base_type;

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor creates no elements.
         */
        idrefs ()
            : base_type (0, this)
        {
        }

        /**
         * @brief Initialize the instance with copies of an exemplar elements.
         *
         * @param n A number of elements to copy.
         * @param x An exemplar element to copy.
         */
        idrefs (typename base_type::size_type n, const idref& x)
            : base_type (n, x, this)
        {
        }

        /**
         * @brief Initialize the instance with copies of elements from an
         * iterator range.
         *
         * @param begin An iterator pointing to the first element.
         * @param end An iterator pointing to the one past the last element.
         */
        template <typename I>
        idrefs (const I& begin, const I& end)
            : base_type (begin, end, this)
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
        idrefs (const idrefs& x, flags f = 0, container* c = 0)
            : B (x, f, c), base_type (x, f, this)
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
        virtual idrefs*
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
        idrefs (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        idrefs (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        idrefs (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        idrefs (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f = 0,
                container* c = 0);
        //@}
      };


      /**
       * @brief Class corresponding to the XML Schema anyURI built-in
       * type.
       *
       * The %uri class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class uri: public B, public std::basic_string<C>
      {
        typedef std::basic_string<C> base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        uri (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        uri (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        uri (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        uri (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        uri (const std::basic_string<C>& s,
             std::size_t pos,
             std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        uri (const uri& x, flags f = 0, container* c = 0)
            : B (x, f, c), base_type (x)
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
        virtual uri*
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
        uri (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        uri (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        uri (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        uri (const std::basic_string<C>& s,
             const xercesc::DOMElement* e,
             flags f = 0,
             container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %uri has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        uri&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %uri contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        uri&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %uri contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        uri&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        uri&
        operator= (const uri& x)
        {
          base () = x;
          return *this;
        }

      protected:
        //@cond

        uri ()
            : base_type ()
        {
        }

        //@endcond

        template <typename, typename, typename, typename>
        friend class qname;
      };


      /**
       * @brief Class corresponding to the XML Schema QName built-in
       * type.
       *
       * The %qname class represents a potentially namespace-qualified
       * XML %name.
       *
       * @nosubgrouping
       */
      template <typename C, typename B, typename uri, typename ncname>
      class qname: public B
      {
      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a %name only.
         *
         * The resulting %qname is unqualified.
         *
         * @param n An XML %name (ncname).
         */
        qname (const ncname& n)
            : ns_ (), name_ (n)
        {
        }

        /**
         * @brief Initialize an instance with a %name and a namespace.
         *
         * The resulting %qname is qualified.
         *
         * @param ns An XML namespace (uri).
         * @param n  An XML %name (ncname).
         */
        qname (const uri& ns, const ncname& n)
            : ns_ (ns), name_ (n)
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
        qname (const qname& x, flags f = 0, container* c = 0)
            : B (x, f, c),
              ns_ (x.ns_),
              name_ (x.name_)
        {
          // Note that ns_ and name_ have no DOM association.
          //
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
        virtual qname*
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
        qname (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        qname (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        qname (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        qname (const std::basic_string<C>& s,
               const xercesc::DOMElement* e,
               flags f = 0,
               container* c = 0);
        //@}

      public:
        /**
         * @brief Determine if the %name is qualified.
         *
         * @return True if the %name is qualified, false otherwise.
         */
        bool
        qualified () const
        {
          return !ns_.empty ();
        }

        /**
         * @brief Get XML namespace.
         *
         * @return A constant reference to qualifying XML namespace.
         */
        const uri&
        namespace_ () const
        {
          return ns_;
        }

        /**
         * @brief Get XML %name.
         *
         * @return A constant reference to unqualified XML %name.
         */
        const ncname&
        name () const
        {
          return name_;
        }

      protected:
        //@cond

        qname ()
            : ns_ (), name_ ()
        {
        }

        //@endcond

      private:
        static uri
        resolve (const std::basic_string<C>&, const xercesc::DOMElement*);

      private:
        uri ns_;
        ncname name_;
      };

      /**
       * @brief %qname comparison operator.
       *
       * @return True if the names are equal, false otherwise.
       */
      template <typename C, typename B, typename uri, typename ncname>
      inline bool
      operator== (const qname<C, B, uri, ncname>& a,
                  const qname<C, B, uri, ncname>& b)
      {
        return a.name () == b.name () && a.namespace_ () == b.namespace_ ();
      }

      /**
       * @brief %qname comparison operator.
       *
       * @return True if the names are not equal, false otherwise.
       */
      template <typename C, typename B, typename uri, typename ncname>
      inline bool
      operator!= (const qname<C, B, uri, ncname>& a,
                  const qname<C, B, uri, ncname>& b)
      {
        return !(a == b);
      }


      /**
       * @brief Class corresponding to the XML Schema base64Binary
       * built-in type.
       *
       * The %base64_binary class is a binary %buffer abstraction with
       * base64-encoded representation in XML. It publicly inherits from
       * the buffer class which provides the %buffer functionality.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class base64_binary: public B, public buffer<C>
      {
      public:
        typedef typename buffer<C>::size_t size_t;

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Allocate a %buffer of the specified size.
         *
         * The resulting %buffer has the same size and capacity.
         *
         * @param size A %buffer size in bytes.
         */
        explicit
        base64_binary (size_t size = 0);

        /**
         * @brief Allocate a %buffer of the specified size and capacity.
         *
         * @param size A %buffer size in bytes.
         * @param capacity A %buffer capacity in bytes.
         * @throw bounds If @a size exceeds @a capacity
         */
        base64_binary (size_t size, size_t capacity);

        /**
         * @brief Allocate a %buffer of the specified size and copy
         * the data.
         *
         * The resulting %buffer has the same size and capacity with
         * @a size bytes copied from @a data.
         *
         * @param data A %buffer to copy the data from.
         * @param size A %buffer size in bytes.
         */
        base64_binary (const void* data, size_t size);

        /**
         * @brief Allocate a %buffer of the specified size and capacity
         * and copy the data.
         *
         * @a size bytes are copied from @a data to the resulting
         * %buffer.
         *
         * @param data A %buffer to copy the data from.
         * @param size A %buffer size in bytes.
         * @param capacity A %buffer capacity in bytes.
         * @throw bounds If @a size exceeds @a capacity
         */
        base64_binary (const void* data, size_t size, size_t capacity);

        /**
         * @brief Reuse an existing %buffer.
         *
         * If the @a assume_ownership argument is true, the %buffer will
         * assume ownership of @a data and will release the memory
         * by calling @c operator @c delete().
         *
         * @param data A %buffer to reuse.
         * @param size A %buffer size in bytes.
         * @param capacity A %buffer capacity in bytes.
         * @param assume_ownership A boolean value indication whether to
         * assume ownership.
         * @throw bounds If @a size exceeds @a capacity
         */
        base64_binary (void* data,
                       size_t size,
                       size_t capacity,
                       bool assume_ownership);
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
        base64_binary (const base64_binary& x,
                       flags f = 0,
                       container* c = 0)
            : B (x, f, c), buffer<C> (x)
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
        virtual base64_binary*
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
        base64_binary (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        base64_binary (const xercesc::DOMElement& e,
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
        base64_binary (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        base64_binary (const std::basic_string<C>& s,
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
        base64_binary&
        operator= (const base64_binary& x)
        {
          buffer<C>& b (*this);
          b = x;
          return *this;
        }

      public:
        /**
         * @brief Encode the %buffer in base64 encoding.
         *
         * @return A %string with base64-encoded data.
         */
        std::basic_string<C>
        encode () const;

      private:
        void
        decode (const XMLCh*);
      };


      /**
       * @brief Class corresponding to the XML Schema hexBinary
       * built-in type.
       *
       * The %hex_binary class is a binary %buffer abstraction with
       * hex-encoded representation in XML. It publicly inherits from
       * the buffer class which provides the %buffer functionality.
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class hex_binary: public B, public buffer<C>
      {
      public:
        typedef typename buffer<C>::size_t size_t;

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Allocate a %buffer of the specified size.
         *
         * The resulting %buffer has the same size and capacity.
         *
         * @param size A %buffer size in bytes.
         */
        explicit
        hex_binary (size_t size = 0);

        /**
         * @brief Allocate a %buffer of the specified size and capacity.
         *
         * @param size A %buffer size in bytes.
         * @param capacity A %buffer capacity in bytes.
         * @throw bounds If @a size exceeds @a capacity
         */
        hex_binary (size_t size, size_t capacity);

        /**
         * @brief Allocate a %buffer of the specified size and copy
         * the data.
         *
         * The resulting %buffer has the same size and capacity with
         * @a size bytes copied from @a data.
         *
         * @param data A %buffer to copy the data from.
         * @param size A %buffer size in bytes.
         */
        hex_binary (const void* data, size_t size);

        /**
         * @brief Allocate a %buffer of the specified size and capacity
         * and copy the data.
         *
         * @a size bytes are copied from @a data to the resulting
         * %buffer.
         *
         * @param data A %buffer to copy the data from.
         * @param size A %buffer size in bytes.
         * @param capacity A %buffer capacity in bytes.
         * @throw bounds If @a size exceeds @a capacity
         */
        hex_binary (const void* data, size_t size, size_t capacity);

        /**
         * @brief Reuse an existing %buffer..
         *
         * If the @a assume_ownership argument is true, the %buffer will
         * assume ownership of @a data and will release the memory
         * by calling @c operator @c delete().
         *
         * @param data A %buffer to reuse.
         * @param size A %buffer size in bytes.
         * @param capacity A %buffer capacity in bytes.
         * @param assume_ownership A boolean value indication whether to
         * assume ownership.
         * @throw bounds If @a size exceeds @a capacity
         */
        hex_binary (void* data,
                    size_t size,
                    size_t capacity,
                    bool assume_ownership);

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
        hex_binary (const hex_binary& x, flags f = 0, container* c = 0)
            : B (x, f, c), buffer<C> (x)
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
        virtual hex_binary*
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
        hex_binary (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        hex_binary (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        hex_binary (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        hex_binary (const std::basic_string<C>& s,
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
        hex_binary&
        operator= (const hex_binary& x)
        {
          buffer<C>& b (*this);
          b = x;
          return *this;
        }

      public:
        /**
         * @brief Encode the %buffer in hex encoding.
         *
         * @return A %string with hex-encoded data.
         */
        std::basic_string<C>
        encode () const;

      private:
        void
        decode (const XMLCh*);
      };


      /**
       * @brief Class corresponding to the XML Schema ENTITY built-in
       * type.
       *
       * The %entity class publicly inherits from and has the same set
       * of constructors as @c std::basic_string. It therefore can be
       * used as @c std::string (or @c std::wstring if you are using
       * @c wchar_t as the character type).
       *
       * @nosubgrouping
       */
      template <typename C, typename B>
      class entity: public B
      {
        typedef B base_type;

        base_type&
        base ()
        {
          return *this;
        }

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Initialize an instance with a copy of a C %string.
         *
         * @param s A C %string to copy.
         */
        entity (const C* s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a character array.
         *
         * @param s A character array to copy.
         * @param n A number of character to copy.
         */
        entity (const C* s, std::size_t n)
            : base_type (s, n)
        {
        }

        /**
         * @brief Initialize an instance with multiple copies of the same
         * character.
         *
         * @param n A number of copies to create.
         * @param c A character to copy.
         */
        entity (std::size_t n, C c)
            : base_type (n, c)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a standard %string.
         *
         * @param s A standard %string to copy.
         */
        entity (const std::basic_string<C>& s)
            : base_type (s)
        {
        }

        /**
         * @brief Initialize an instance with a copy of a substring.
         *
         * @param s   A standard %string to copy the substring from.
         * @param pos An index of the first character to copy from.
         * @param n   A number of characters to copy.
         */
        entity (const std::basic_string<C>& s,
                std::size_t pos,
                std::size_t n = std::basic_string<C>::npos)
            : base_type (s, pos, n)
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
        entity (const entity& x, flags f = 0, container* c = 0)
            : base_type (x, f, c)
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
        virtual entity*
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
        entity (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        entity (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        entity (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        entity (const std::basic_string<C>& s,
                const xercesc::DOMElement* e,
                flags f = 0,
                container* c = 0);
        //@}

      public:
        /**
         * @brief Assign a character to the instance.
         *
         * The resulting %entity has only one character.
         *
         * @param c A character to assign.
         * @return A reference to the instance.
         */
        entity&
        operator= (C c)
        {
          base () = c;
          return *this;
        }

        /**
         * @brief Assign a C %string to the instance.
         *
         * The resulting %entity contains a copy of the C %string.
         *
         * @param s A C %string to assign.
         * @return A reference to the instance.
         */
        entity&
        operator= (const C* s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Assign a standard %string to the instance.
         *
         * The resulting %entity contains a copy of the standard %string.
         *
         * @param s A standard %string to assign.
         * @return A reference to the instance.
         */
        entity&
        operator= (const std::basic_string<C>& s)
        {
          base () = s;
          return *this;
        }

        /**
         * @brief Copy assignment operator.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        entity&
        operator= (const entity& x)
        {
          base () = x;
          return *this;
        }

      protected:
        //@cond

        entity ()
            : base_type ()
        {
        }

        //@endcond
      };


      /**
       * @brief Class corresponding to the XML Schema ENTITIES built-in
       * type.
       *
       * The %entities class is a vector (or %list in XML Schema terminology)
       * of entity elements. It is implemented in terms of the list class
       * template.
       *
       * @nosubgrouping
       */
      template <typename C, typename B, typename entity>
      class entities: public B, public list<entity, C>
      {
        typedef list<entity, C> base_type;

      public:
        /**
         * @name Constructors
         */
        //@{

        /**
         * @brief Default constructor creates no elements.
         */
        entities ()
            : base_type (0, this)
        {
        }

        /**
         * @brief Initialize the instance with copies of an exemplar elements.
         *
         * @param n A number of elements to copy.
         * @param x An exemplar element to copy.
         */
        entities (typename base_type::size_type n, const entity& x)
            : base_type (n, x, this)
        {
        }

        /**
         * @brief Initialize the instance with copies of elements from an
         * iterator range.
         *
         * @param begin An iterator pointing to the first element.
         * @param end An iterator pointing to the one past the last element.
         */
        template <typename I>
        entities (const I& begin, const I& end)
            : base_type (begin, end, this)
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
        entities (const entities& x, flags f = 0, container* c = 0)
            : B (x, f, c), base_type (x, f, this)
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
        virtual entities*
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
        entities (istream<S>& s, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM element.
         *
         * @param e A DOM element to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        entities (const xercesc::DOMElement& e, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a DOM Attribute.
         *
         * @param a A DOM attribute to extract the data from.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        entities (const xercesc::DOMAttr& a, flags f = 0, container* c = 0);

        /**
         * @brief Create an instance from a %string fragment.
         *
         * @param s A %string fragment to extract the data from.
         * @param e A pointer to DOM element containing the %string fragment.
         * @param f Flags to create the new instance with.
         * @param c A pointer to the object that will contain the new
         * instance.
         */
        entities (const std::basic_string<C>& s,
                  const xercesc::DOMElement* e,
                  flags f = 0,
                  container* c = 0);
        //@}
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/types.txx>

#endif  // XSD_CXX_TREE_TYPES_HXX
