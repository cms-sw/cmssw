// file      : xsd/cxx/tree/exceptions.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

/**
 * @file
 *
 * @brief Contains exception definitions for the C++/Tree mapping.
 *
 * This is an internal header and is included by the generated code.
 * You normally should not include it directly.
 *
 */

#ifndef XSD_CXX_TREE_EXCEPTIONS_HXX
#define XSD_CXX_TREE_EXCEPTIONS_HXX

#include <string>
#include <vector>
#include <ostream>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/exceptions.hxx> // xsd::cxx::exception

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
       * @brief Root of the C++/Tree %exception hierarchy.
       *
       * You can catch this %exception in order to handle all C++/Tree
       * errors.
       *
       * @nosubgrouping
       */
      template <typename C>
      class exception: public xsd::cxx::exception
      {
      public:
        /**
         * @brief Stream insertion operator for %exception.
         */
        friend
        std::basic_ostream<C>&
        operator<< (std::basic_ostream<C>& os, const exception& e)
        {
          e.print (os);
          return os;
        }

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const = 0;

        //@endcond
      };


      /**
       * @brief Error %severity.
       *
       * @nosubgrouping
       */
      class severity
      {
      public:
        /**
         * @brief Underlying enum type.
         */
        enum value
        {
          /**
           * @brief Indicates the warning condition.
           */
          warning,

          /**
           * @brief Indicates the %error condition.
           */
          error
        };

        /**
         * @brief Initialize an instance with the underlying enum value.
         *
         * @param v An underlying enum value.
         */
        severity (value v) : v_ (v) {}

        /**
         * @brief Implicitly convert the instance to the underlying enum
         * value.
         *
         * @return The underlying enum value.
         */
        operator value () const { return v_; }

      private:
        value v_;
      };

      /**
       * @brief Error condition.
       *
       * @nosubgrouping
       */
      template <typename C>
      class error
      {
      public:
        /**
         * @brief Initialize an instance with %error description.
         *
         * @param s An %error %severity.
         * @param res_id A resource %id where the %error occurred.
         * @param line A line number where the %error occurred.
         * @param column A column number where the %error occurred.
         * @param message A message describing the %error.
         */
        error (tree::severity s,
               const std::basic_string<C>& res_id,
               unsigned long line,
               unsigned long column,
               const std::basic_string<C>& message);

        /**
         * @brief Get %error %severity.
         *
         * @return The %severity of this %error.
         */
        tree::severity
        severity () const
        {
          return severity_;
        }

        /**
         * @brief Get resource %id.
         *
         * @return The %id of the resource where this %error occurred.
         */
        const std::basic_string<C>&
        id () const
        {
          return id_;
        }

        /**
         * @brief Get %error line.
         *
         * @return The line number where this %error occurred.
         */
        unsigned long
        line () const
        {
          return line_;
        }

        /**
         * @brief Get %error column.
         *
         * @return The column number where this %error occurred.
         */
        unsigned long
        column () const
        {
          return column_;
        }

        /**
         * @brief Get %error message.
         *
         * @return The message for this %error.
         */
        const std::basic_string<C>&
        message () const
        {
          return message_;
        }

        //@cond

        // Default c-tor that shouldn't be. Needed when we completely
        // instantiate std::vector in diagnostics below.
        //
        error ();

        //@endcond


      private:
        tree::severity severity_;
        std::basic_string<C> id_;
        unsigned long line_;
        unsigned long column_;
        std::basic_string<C> message_;
      };

      // See exceptions.ixx for operator<< (error).


      /**
       * @brief List of %error conditions.
       *
       * @nosubgrouping
       */
      template <typename C>
      class diagnostics: public std::vector<error<C> >
      {
      };

      // See exceptions.ixx for operator<< (diagnostics).

      /**
       * @brief Exception indicating a %parsing failure.
       *
       * @nosubgrouping
       */
      template <typename C>
      class parsing: public exception<C>
      {
      public:
        virtual
        ~parsing () throw ();

        /**
         * @brief Default constructor.
         */
        parsing ();

        /**
         * @brief Initialize an instance with a %list of %error conditions.
         *
         * @param d A %list of %error conditions.
         */
        parsing (const tree::diagnostics<C>& d);

      public:
        /**
         * @brief Get the %list of %error conditions.
         *
         * @return The %list of %error conditions.
         */
        const tree::diagnostics<C>&
        diagnostics () const
        {
          return diagnostics_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        tree::diagnostics<C> diagnostics_;
      };


      /**
       * @brief Exception indicating that an expected element was not
       * encountered.
       *
       * @nosubgrouping
       */
      template <typename C>
      class expected_element: public exception<C>
      {
      public:
        virtual
        ~expected_element () throw ();

        /**
         * @brief Initialize an instance with the expected element
         * description.
         *
         * @param name A name of the expected element.
         * @param ns A namespace of the expected element.
         */
        expected_element (const std::basic_string<C>& name,
                          const std::basic_string<C>& ns);


      public:
        /**
         * @brief Get the name of the expected element.
         *
         * @return The name of the expected element.
         */
        const std::basic_string<C>&
        name () const
        {
          return name_;
        }

        /**
         * @brief Get the namespace of the expected element.
         *
         * @return The namespace of the expected element.
         */
        const std::basic_string<C>&
        namespace_ () const
        {
          return namespace__;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> name_;
        std::basic_string<C> namespace__;
      };


      /**
       * @brief Exception indicating that an unexpected element was
       * encountered.
       *
       * @nosubgrouping
       */
      template <typename C>
      class unexpected_element: public exception<C>
      {
      public:
        virtual
        ~unexpected_element () throw ();

        /**
         * @brief Initialize an instance with the encountered and expected
         * element descriptions.
         *
         * @param encountered_name A name of the encountered element.
         * @param encountered_ns A namespace of the encountered element.
         * @param expected_name A name of the expected element.
         * @param expected_ns A namespace of the expected element.
         */
        unexpected_element (const std::basic_string<C>& encountered_name,
                            const std::basic_string<C>& encountered_ns,
                            const std::basic_string<C>& expected_name,
                            const std::basic_string<C>& expected_ns);

      public:
        /**
         * @brief Get the name of the encountered element.
         *
         * @return The name of the encountered element.
         */
        const std::basic_string<C>&
        encountered_name () const
        {
          return encountered_name_;
        }

        /**
         * @brief Get the namespace of the encountered element.
         *
         * @return The namespace of the encountered element.
         */
        const std::basic_string<C>&
        encountered_namespace () const
        {
          return encountered_namespace_;
        }

        /**
         * @brief Get the name of the expected element.
         *
         * @return The name of the expected element.
         */
        const std::basic_string<C>&
        expected_name () const
        {
          return expected_name_;
        }

        /**
         * @brief Get the namespace of the expected element.
         *
         * @return The namespace of the expected element.
         */
        const std::basic_string<C>&
        expected_namespace () const
        {
          return expected_namespace_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> encountered_name_;
        std::basic_string<C> encountered_namespace_;
        std::basic_string<C> expected_name_;
        std::basic_string<C> expected_namespace_;
      };


      /**
       * @brief Exception indicating that an expected attribute was not
       * encountered.
       *
       * @nosubgrouping
       */
      template <typename C>
      class expected_attribute: public exception<C>
      {
      public:
        virtual
        ~expected_attribute () throw ();

        /**
         * @brief Initialize an instance with the expected attribute
         * description.
         *
         * @param name A name of the expected attribute.
         * @param ns A namespace of the expected attribute.
         */
        expected_attribute (const std::basic_string<C>& name,
                            const std::basic_string<C>& ns);

      public:
        /**
         * @brief Get the name of the expected attribute.
         *
         * @return The name of the expected attribute.
         */
        const std::basic_string<C>&
        name () const
        {
          return name_;
        }

        /**
         * @brief Get the namespace of the expected attribute.
         *
         * @return The namespace of the expected attribute.
         */
        const std::basic_string<C>&
        namespace_ () const
        {
          return namespace__;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> name_;
        std::basic_string<C> namespace__;
      };


      /**
       * @brief Exception indicating that an unexpected enumerator was
       * encountered.
       *
       * @nosubgrouping
       */
      template <typename C>
      class unexpected_enumerator: public exception<C>
      {
      public:
        virtual
        ~unexpected_enumerator () throw ();

        /**
         * @brief Initialize an instance with the encountered enumerator.
         *
         * @param e A value of the encountered enumerator.
         */
        unexpected_enumerator (const std::basic_string<C>& e);

      public:
        /**
         * @brief Get the value of the encountered enumerator.
         *
         * @return The value of the encountered enumerator.
         */
        const std::basic_string<C>&
        enumerator () const
        {
          return enumerator_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> enumerator_;
      };


      /**
       * @brief Exception indicating that the text content was expected
       * for an element.
       *
       * @nosubgrouping
       */
      template <typename C>
      class expected_text_content: public exception<C>
      {
      public:
        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond
      };


      /**
       * @brief Exception indicating that the type information is not
       * available for a type.
       *
       * @nosubgrouping
       */
      template <typename C>
      class no_type_info: public exception<C>
      {
      public:
        virtual
        ~no_type_info () throw ();

        /**
         * @brief Initialize an instance with the type description.
         *
         * @param type_name A name of the type.
         * @param type_ns A namespace of the type.
         */
        no_type_info (const std::basic_string<C>& type_name,
                      const std::basic_string<C>& type_ns);

      public:
        /**
         * @brief Get the type name.
         *
         * @return The type name.
         */
        const std::basic_string<C>&
        type_name () const
        {
          return type_name_;
        }

        /**
         * @brief Get the type namespace.
         *
         * @return The type namespace.
         */
        const std::basic_string<C>&
        type_namespace () const
        {
          return type_namespace_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> type_name_;
        std::basic_string<C> type_namespace_;
      };

      /**
       * @brief Exception indicating that %parsing or %serialization
       * information is not available for an element.
       *
       * @nosubgrouping
       */
      template <typename C>
      class no_element_info: public exception<C>
      {
      public:
        virtual
        ~no_element_info () throw ();

        /**
         * @brief Initialize an instance with the element description.
         *
         * @param element_name An element name.
         * @param element_ns An element namespace.
         */
        no_element_info (const std::basic_string<C>& element_name,
                         const std::basic_string<C>& element_ns);

      public:
        /**
         * @brief Get the element name.
         *
         * @return The element name.
         */
        const std::basic_string<C>&
        element_name () const
        {
          return element_name_;
        }

        /**
         * @brief Get the element namespace.
         *
         * @return The element namespace.
         */
        const std::basic_string<C>&
        element_namespace () const
        {
          return element_namespace_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> element_name_;
        std::basic_string<C> element_namespace_;
      };

      /**
       * @brief Exception indicating that the types are not related by
       * inheritance.
       *
       * @nosubgrouping
       */
      template <typename C>
      class not_derived: public exception<C>
      {
      public:
        virtual
        ~not_derived () throw ();

        //@cond

        // @@ tmp
        //
        not_derived ()
        {
        }

        //@endcond

        /**
         * @brief Initialize an instance with the type descriptions.
         *
         * @param base_type_name A name of the base type.
         * @param base_type_ns A namespace of the base type.
         * @param derived_type_name A name of the derived type.
         * @param derived_type_ns A namespace of the derived type.
         */
        not_derived (const std::basic_string<C>& base_type_name,
                     const std::basic_string<C>& base_type_ns,
                     const std::basic_string<C>& derived_type_name,
                     const std::basic_string<C>& derived_type_ns);

      public:
        /**
         * @brief Get the base type name.
         *
         * @return The base type name.
         */
        const std::basic_string<C>&
        base_type_name () const
        {
          return base_type_name_;
        }

        /**
         * @brief Get the base type namespace.
         *
         * @return The base type namespace.
         */
        const std::basic_string<C>&
        base_type_namespace () const
        {
          return base_type_namespace_;
        }

        /**
         * @brief Get the derived type name.
         *
         * @return The derived type name.
         */
        const std::basic_string<C>&
        derived_type_name () const
        {
          return derived_type_name_;
        }

        /**
         * @brief Get the derived type namespace.
         *
         * @return The derived type namespace.
         */
        const std::basic_string<C>&
        derived_type_namespace () const
        {
          return derived_type_namespace_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> base_type_name_;
        std::basic_string<C> base_type_namespace_;
        std::basic_string<C> derived_type_name_;
        std::basic_string<C> derived_type_namespace_;
      };


      /**
       * @brief Exception indicating that a duplicate ID value was
       * encountered in the object model.
       *
       * @nosubgrouping
       */
      template <typename C>
      class duplicate_id: public exception<C>
      {
      public:
        virtual
        ~duplicate_id () throw ();

        /**
         * @brief Initialize an instance with the offending ID value.
         *
         * @param id An offending ID value.
         */
        duplicate_id (const std::basic_string<C>& id);

      public:
        /**
         * @brief Get the offending ID value.
         *
         * @return The offending ID value.
         */
        const std::basic_string<C>&
        id () const
        {
          return id_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> id_;
      };


      /**
       * @brief Exception indicating a %serialization failure.
       *
       * @nosubgrouping
       */
      template <typename C>
      class serialization: public exception<C>
      {
      public:
        virtual
        ~serialization () throw ();

        /**
         * @brief Default constructor.
         */
        serialization ();

        /**
         * @brief Initialize an instance with a %list of %error conditions.
         *
         * @param d A %list of %error conditions.
         */
        serialization (const tree::diagnostics<C>& d);

      public:
        /**
         * @brief Get the %list of %error conditions.
         *
         * @return The %list of %error conditions.
         */
        const tree::diagnostics<C>&
        diagnostics () const
        {
          return diagnostics_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        tree::diagnostics<C> diagnostics_;
      };


      /**
       * @brief Exception indicating that a prefix-namespace mapping was
       * not provided.
       *
       * @nosubgrouping
       */
      template <typename C>
      class no_prefix_mapping: public exception<C>
      {
      public:
        virtual
        ~no_prefix_mapping () throw ();

        /**
         * @brief Initialize an instance with the prefix for which the
         * prefix-namespace mapping was not provided.
         *
         * @param prefix A prefix.
         */
        no_prefix_mapping (const std::basic_string<C>& prefix);

      public:
        /**
         * @brief Get the prefix for which the prefix-namespace mapping was
         * not provided.
         *
         * @return The prefix.
         */
        const std::basic_string<C>&
        prefix () const
        {
          return prefix_;
        }

        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond

      private:
        std::basic_string<C> prefix_;
      };


      /**
       * @brief Exception indicating that the size argument exceeds
       * the capacity argument.
       *
       * See the buffer class for details.
       *
       * @nosubgrouping
       */
      template <typename C>
      class bounds: public exception<C>
      {
      public:
        /**
         * @brief Get %exception description.
         *
         * @return A C %string describing the %exception.
         */
        virtual const char*
        what () const throw ();

      protected:
        //@cond

        virtual void
        print (std::basic_ostream<C>&) const;

        //@endcond
      };
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/exceptions.txx>

#endif  // XSD_CXX_TREE_EXCEPTIONS_HXX
