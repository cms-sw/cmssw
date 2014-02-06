#ifndef CondCore_CondDB_Binary_h
#define CondCore_CondDB_Binary_h

#include <string>
#include <memory>
// temporarely
#include <boost/shared_ptr.hpp>
// 

namespace coral {
  class Blob;
}

namespace cond {

  struct Nodelete {
    Nodelete(){}
    void operator()( void* ptr ){}
  };
  
  class Binary {
  public:
    Binary();

    template <typename T> explicit Binary( const T& object );

    explicit Binary( const boost::shared_ptr<void>& objectPtr );

    Binary( const void* data, size_t size  );

    explicit Binary( const coral::Blob& data );

    Binary( const Binary& rhs );

    Binary& operator=( const Binary& rhs );

    const coral::Blob& get() const;

    void copy( const std::string& source );

    const void* data() const;

    void* data();

    size_t size() const;

    boost::shared_ptr<void> share() const;

  private:
    std::shared_ptr<coral::Blob> m_data;
    //
    boost::shared_ptr<void> m_object;
  };

  template <typename T> Binary::Binary( const T& object ):
    m_object( &const_cast<T&>(object), Nodelete() ){
  }
}

#endif

