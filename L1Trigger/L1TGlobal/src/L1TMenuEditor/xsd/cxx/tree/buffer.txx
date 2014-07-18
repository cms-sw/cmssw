// file      : xsd/cxx/tree/buffer.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C>
      buffer<C>::
      buffer (size_t size)
      {
        capacity (size);
        size_ = size;
      }

      template <typename C>
      buffer<C>::
      buffer (size_t size, size_t capacity)
      {
        if (size > capacity)
          throw bounds<C> ();

        this->capacity (capacity);
        size_ = size;
      }

      template <typename C>
      buffer<C>::
      buffer (const void* data, size_t size)
      {
        capacity (size);
        size_ = size;

        if (size_)
          std::memcpy (data_, data, size_);
      }

      template <typename C>
      buffer<C>::
      buffer (const void* data, size_t size, size_t capacity)
      {
        if (size > capacity)
          throw bounds<C> ();

        this->capacity (capacity);
        size_ = size;

        if (size_)
          std::memcpy (data_, data, size_);
      }

      template <typename C>
      buffer<C>::
      buffer (void* data, size_t size, size_t capacity, bool own)
      {
        if (size > capacity)
          throw bounds<C> ();

        data_ = reinterpret_cast<char*> (data);
        size_ = size;
        capacity_ = capacity;
        free_ = own;
      }

      template <typename C>
      buffer<C>::
      buffer (const buffer& other)
          : buffer_base ()
      {
        capacity (other.capacity_);
        size_ = other.size_;

        if (size_)
          std::memcpy (data_, other.data_, size_);
      }

      template <typename C>
      buffer<C>& buffer<C>::
      operator= (const buffer& other)
      {
        if (this != &other)
        {
          capacity (other.capacity_, false);
          size_ = other.size_;

          if (size_)
            std::memcpy (data_, other.data_, size_);
        }

        return *this;
      }

      template <typename C>
      void buffer<C>::
      swap (buffer& other)
      {
        char* tmp_data (data_);
        size_t tmp_size (size_);
        size_t tmp_capacity (capacity_);
        bool tmp_free (free_);

        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        free_ = other.free_;

        other.data_ = tmp_data;
        other.size_ = tmp_size;
        other.capacity_ = tmp_capacity;
        other.free_ = tmp_free;
      }

      template <typename C>
      bool buffer<C>::
      capacity (size_t capacity, bool copy)
      {
        if (size_ > capacity)
          throw bounds<C> ();

        if (capacity <= capacity_)
        {
          return false; // Do nothing if shrinking is requested.
        }
        else
        {
          char* data (reinterpret_cast<char*> (operator new (capacity)));

          if (copy && size_ > 0)
            std::memcpy (data, data_, size_);

          if (free_ && data_)
            operator delete (data_);

          data_ = data;
          capacity_ = capacity;
          free_ = true;

          return true;
        }
      }
    }
  }
}
