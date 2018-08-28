#ifndef DETECTOR_DESCRIPTION_DD_SINGLETON_H
#define DETECTOR_DESCRIPTION_DD_SINGLETON_H

#include <memory>

namespace dd {

  template< typename T, typename CONTEXT >
    class DDSingleton
  {
  public:
    T* operator->() { return m_instance.get(); }
    const T* operator->() const { return m_instance; }
    T& operator*() { return *m_instance; }
    const T& operator*() const { return *m_instance; }
    
  protected:
    DDSingleton() {
      static bool static_init __attribute__(( unused )) = []()->bool {
	m_instance = std::make_unique<T>();
	return true;
      }();
    }

    DDSingleton( int ) {
      static bool static_init __attribute__(( unused )) = []()->bool {
	m_instance = CONTEXT::init();
	return true;
      }();
    }
    
  private:
    static std::unique_ptr<T> m_instance;
  };
}

template < typename T, typename CONTEXT >
  std::unique_ptr<T> dd::DDSingleton< T, CONTEXT >::m_instance;
  
#endif
