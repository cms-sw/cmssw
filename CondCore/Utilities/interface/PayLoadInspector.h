#ifndef  PayLoadInspector_H
#define  PayLoadInspector_H
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"

#include <string>
#include <vector>
#include <sstream>


namespace cond {
  // to be moved elsewhere
  class PoolTransactionSentry {
  public:
    PoolTransactionSentry(){}
    explicit PoolTransactionSentry(cond::DbSession & db) : 
      elem(new Elem(db)){}      
  private:
    struct Elem {
      Elem(cond::DbSession & db) : pooldb(db){
	pooldb.transaction().start(true);
      }
      ~Elem() { pooldb.transaction().commit();}
      cond::DbSession pooldb;
    };
    boost::shared_ptr<Elem> elem;
      
  };

  template<typename T>
  class BaseValueExtractor {
  public:
    typedef T Class;

    BaseValueExtractor(){}
    
    virtual ~BaseValueExtractor(){}
    void computeW(Class const &o){
      reset();
      compute(o);
    }
    std::vector<float> const & values() const { return m_values;}
  protected:
    void add(float v) { m_values.push_back(v); }
    void swap(std::vector<float> & v) {m_values.swap(v);}
  private:
    void reset() { m_values.clear();}
    virtual void compute(Class const &){}
    
    
  private:
    std::vector<float> m_values;
  };


  // a collection of enumerators, strings, ints
  template<typename T>
  struct ExtractWhat {

  };



  template<typename T>
  class ValueExtractor : public  BaseValueExtractor<T> {
  public:
    typedef T Class;
    typedef ExtractWhat<Class> What;
    ValueExtractor(){}
    ValueExtractor(What const &){}
    static What what() { return What();}
  private:
    void compute(Class const &){}
  private:
  };

  template<typename DataT>
  class PayLoadInspector {
  public:
    typedef DataT Class;
    typedef ValueExtractor<DataT> Extractor;

    PayLoadInspector() {}

    PayLoadInspector(const cond::IOVElementProxy & elem) {
      cond::DbSession db = elem.db();
      db.transaction().start(true);
      load(db,elem.token());
      db.transaction().commit();
    }

    std::string dump() const { return ""; }

    // specialize...
    std::string summary() const {
      std::ostringstream os;
      os << std::endl;
      return os.str();
    }

    // return the real name of the file including extension...
    std::string plot(std::string const & /* filename */,
		     std::string const &, 
		     std::vector<int> const&, std::vector<float> const& ) const {return "";}

    void extract(Extractor & extractor) const {extractor.computeW(object()); }

    Class const & object() const { 
      return *m_Data; 
    }
    
  private:
    bool load( cond::DbSession & db, std::string const & token) {
      bool ok = false;
      m_Data =  db.getTypedObject<DataT>(token);
      if (m_Data.get()) ok =  true;
    
      return ok;
  }
    
  private:
    boost::shared_ptr<DataT> m_Data;
  };

}

#endif //   PayLoadInspector_H
