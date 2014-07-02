// file      : xsd/cxx/tree/buffer.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

/**
 * @file
 *
 * @brief Contains a simple binary buffer abstraction that is used to
 * implement the base64Binary and hexBinary XML Schema built-in types.
 *
 * This is an internal header and is included by the generated code. You
 * normally should not include it directly.
 *
 */

#ifndef XSD_CXX_TREE_BUFFER_HXX
#define XSD_CXX_TREE_BUFFER_HXX

#include <new>     // operator new/delete
#include <cstddef> // std::size_t
#include <cstring> // std::memcpy, std::memcmp

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/exceptions.hxx>

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
      //@cond

      class buffer_base
      {
      protected:
        virtual
        ~buffer_base ()
        {
          if (free_ && data_)
            operator delete (data_);
        }

        buffer_base ()
            : data_ (0), size_ (0), capacity_ (0), free_ (true)
        {
        }

      protected:
        char* data_;
        size_t size_;
        size_t capacity_;
        bool free_;
      };

      //@endcond

      /**
       * @brief Simple binary %buffer abstraction
       *
       * The %buffer class manages a continuous binary %buffer. The base
       * concepts are data (actual memory region), size (the portion of
       * the %buffer that contains useful information), and capacity (the
       * actual size of the underlying memory region). The bounds
       * %exception is thrown from the constructors and modifier functions
       * if the (size <= capacity) constraint is violated.
       *
       * Note that the template parameter is only used to instantiate
       * %exception types. The underlying %buffer type is always @c char.
       *
       * @nosubgrouping
       */
      template<typename C>
      class buffer: protected buffer_base
      {
      public:
        /**
         * @brief Size type
         */
        typedef std::size_t size_t;

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
        buffer (size_t size = 0);

        /**
         * @brief Allocate a %buffer of the specified size and capacity.
         *
         * @param size A %buffer size in bytes.
         * @param capacity A %buffer capacity in bytes.
         * @throw bounds If @a size exceeds @a capacity
         */
        buffer (size_t size, size_t capacity);

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
        buffer (const void* data, size_t size);

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
        buffer (const void* data, size_t size, size_t capacity);

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
        buffer (void* data,
                size_t size,
                size_t capacity,
                bool assume_ownership);

        /**
         * @brief Copy constructor.
         *
         * The copy constructor performs a deep copy of the underlying
         * memory %buffer.
         *
         * @param x An instance to make a copy of.
         */
        buffer (const buffer& x);

        //@}

      public:
        /**
         * @brief Copy assignment operator.
         *
         * The copy assignment operator changes the buffer's capacity
         * to @c x.capacity() and copies @c x.size() bytes from @a x.
         *
         * @param x An instance to assign.
         * @return A reference to the instance.
         */
        buffer&
        operator= (const buffer& x);

      public:
        /**
         * @brief Get buffer's capacity.
         *
         * @return A number of bytes that the %buffer can hold without
         * reallocation.
         */
        size_t
        capacity () const
	{
	  return capacity_;
	}

        /**
         * @brief Set buffer's capacity.
         *
         * @param c The new capacity in bytes.
         * @return True if the underlying %buffer has moved, false otherwise.
         */
        bool
        capacity (size_t c)
        {
          return this->capacity (c, true);
        }

      public:
        /**
         * @brief Get buffer's size.
         *
         * @return A number of bytes that the %buffer holds.
         */
        size_t
        size () const {return size_;}

        /**
         * @brief Set buffer's size.
         *
         * @param s The new size in bytes.
         * @return True if the underlying %buffer has moved, false otherwise.
         */
        bool
        size (size_t s)
        {
          bool r (false);

          if (s > capacity_)
            r = capacity (s);

          size_ = s;

          return r;
        }

      public:
        /**
         * @brief Get the underlying memory region.
         *
         * @return A constant pointer to the underlying memory region.
         */
        const char*
        data () const {return data_;}

        /**
         * @brief Get the underlying memory region.
         *
         * @return A pointer to the underlying memory region.
         */
        char*
        data () {return data_;}

        /**
         * @brief Get the beginning of the underlying memory region.
         *
         * @return A constant pointer to the first byte of the underlying
         * memory region.
         */
        const char*
        begin () const {return data_;}

        /**
         * @brief Get the beginning of the underlying memory region.
         *
         * @return A pointer to the first byte of the underlying memory
         * region.
         */
        char*
        begin () {return data_;}

        /**
         * @brief Get the end of the underlying memory region.
         *
         * @return A constant pointer to the one past last byte of the
         * underlying memory region (that is @c %begin() @c + @c %size() ).
         */
        const char*
        end () const {return data_ + size_;}

        /**
         * @brief Get the end of the underlying memory region.
         *
         * @return A pointer to the one past last byte of the underlying
         * memory region (that is @c %begin() @c + @c %size() ).
         */
        char*
        end () {return data_ + size_;}

      public:
        /**
         * @brief Swap data with another %buffer.
         *
         * @param x A %buffer to swap with.
         */
        void
        swap (buffer& x);

      private:
        bool
        capacity (size_t capacity, bool copy);
      };

      /**
       * @brief %buffer comparison operator.
       *
       * @return True if the buffers have the same sizes and the same
       * data.
       */
      template <typename C>
      inline bool
      operator== (const buffer<C>& a, const buffer<C>& b)
      {
        return a.size () == b.size () &&
          std::memcmp (a.data (), b.data (), a.size ()) == 0;
      }

      /**
       * @brief %buffer comparison operator.
       *
       * @return True if the buffers have different sizes or different
       * data.
       */
      template <typename C>
      inline bool
      operator!= (const buffer<C>& a, const buffer<C>& b)
      {
        return !(a == b);
      }
    }
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/buffer.txx>

#endif  // XSD_CXX_TREE_BUFFER_HXX
