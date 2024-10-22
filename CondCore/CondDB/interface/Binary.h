#ifndef CondCore_CondDB_Binary_h
#define CondCore_CondDB_Binary_h

#include <string>
#include <memory>
// temporarely
//

namespace coral {
  class Blob;
}

namespace cond {

  class Binary {
  public:
    Binary();

    Binary(const void* data, size_t size);

    explicit Binary(const coral::Blob& data);

    Binary(const Binary& rhs);

    Binary& operator=(const Binary& rhs);

    const coral::Blob& get() const;

    void copy(const std::string& source);

    const void* data() const;

    void* data();

    size_t size() const;

  private:
    std::shared_ptr<coral::Blob> m_data;
  };

}  // namespace cond

#endif
