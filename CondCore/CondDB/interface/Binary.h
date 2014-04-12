#ifndef CondCore_CondDB_Binary_h
#define CondCore_CondDB_Binary_h

// for the old system, this will go away
#include "CondCore/ORA/interface/Object.h"
#include <string>
#include <memory>
// temporarely
#include <boost/shared_ptr.hpp>
// 

namespace coral {
  class Blob;
}

namespace cond {

  class Binary {
  public:
    Binary();

    template <typename T> explicit Binary( const T& object );

    Binary( const void* data, size_t size  );

    explicit Binary( const coral::Blob& data );

    Binary( const Binary& rhs );

    Binary& operator=( const Binary& rhs );

    const coral::Blob& get() const;

    void copy( const std::string& source );

    const void* data() const;

    void* data();

    size_t size() const;

    ora::Object oraObject() const;

    void fromOraObject( const ora::Object& object );

  private:
    std::shared_ptr<coral::Blob> m_data;
    //
    // workaround to support the non-streamed, packed objects ( the old system )
    ora::Object m_object;
  };

  template <typename T> Binary::Binary( const T& object ):
    m_object( object ){
  }
}

#endif

