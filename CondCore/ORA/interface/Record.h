#ifndef CondCore_ORA_Record_H
#define CondCore_ORA_Record_H

#include <typeinfo>
#include <vector>
#include<string>
#include<boost/shared_ptr.hpp>

namespace ora {
  union AnyData;
  struct TypeHandler;
  struct RecordSpecImpl;
  class Record;

  class RecordSpec {
  public:
    RecordSpec();
    ~RecordSpec();

    size_t add(std::string const & name, std::type_info const & type);

  private:
    friend class Record;
    boost::shared_ptr<RecordSpecImpl> specs;
  };


  class Record {
  public:
    Record();
    
    explicit Record(RecordSpec ispecs);
    void init(RecordSpec ispecs);

    ~Record();
    void destroy();
    
    void swap(Record & lh);

    size_t size() const { return m_null.size();}

    template<typename T>
    T & data(int i) {
      // assert???
      return
	*reinterpret_cast<T*>(const_cast<void*>(address(i)));
    }

    template<typename T>
    T const & data(int i) const {
       // assert???
     return 
	*reinterpret_cast<T const*>(address(i));
    }

    void setNull(int i) {  m_null[i]=true;  } 
    void setNotNull(int i) {  m_null[i]=false;  } 

    int index(std::string const & iname) const;
    std::type_info const * type(int i) const;
    void const * address(int i) const;
    void const * get(int i) const;
    void set(int i, void * p);
    std::string const & name(int i) const;
    bool isNull(int i) const { return m_null[i];}

    boost::shared_ptr<RecordSpecImpl> specs;
    std::vector<AnyData> m_field;
    std::vector<bool> m_null;
  };

}

inline void swap(ora::Record & rh, ora::Record & lh) {
  rh.swap(lh);
}

#endif //  CondCore_ORA_Record_H
