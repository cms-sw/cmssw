#ifndef DataFormatsCommonDebuggingAllocator
#define DataFormatsCommonDebuggingAllocator

//---------------------------------------------------------------------
//
// This file declares and defines an allocator class template.
// This allocator is intended for use with the memory checking tool
// valgrind. It is a minimum conformant implementation which makes sure
// not use use any unitialized memory, and so it causes no spurious error
// reports from valgrind.
//
// The intended use is in the declarations of objects from STL templates,
// e.g.
//    typedef vector<int, edm::debugging_allocator<int> > vint;
// etc.
//
//---------------------------------------------------------------------

#include <limits>
#include <cstddef>

namespace edm
{
  template <class T>
  class debugging_allocator
  {
  public:
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef T*        pointer;
    typedef T const*  const_pointer;
    typedef T&        reference;
    typedef T const&  const_reference;
    typedef T         value_type;

    template <class U> struct rebind { typedef debugging_allocator<U> other; };


    debugging_allocator() noexcept : dummy('x') { }

    debugging_allocator(debugging_allocator const&) noexcept : dummy('c') { }

    template <class U> debugging_allocator(debugging_allocator<U> const&) noexcept : dummy('u') { }

    ~debugging_allocator() noexcept { };

    pointer address(reference value) const {return &value;}

    const_pointer address(const_reference value) const {return &value; }    

    size_type max_size() const noexcept { return std::numeric_limits<size_type>::max()/sizeof(T); }

    pointer allocate(size_type num, void const* hint = 0)
    {
      // allocate objects with global new
      return (pointer)(::operator new(num*sizeof(T)));
    }

    void construct(pointer p, T const& value) { new((void*)p)T(value); }

    void destroy(pointer p) { p->~T(); }

    void deallocate(pointer p, size_type num) { ::operator delete((void*)p); }

  private:
    char dummy;

  };

  // instances of all specializations of this allocator are equal
  template <class X, class Y>
  bool operator==  (debugging_allocator<X> const&, debugging_allocator<Y> const&) noexcept { return true; }

  template <class X, class Y>
  bool operator!=  (debugging_allocator<X> const&, debugging_allocator<Y> const&) noexcept { return false; }

} //namespace edm


#endif
