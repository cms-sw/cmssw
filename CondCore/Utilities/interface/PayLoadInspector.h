#ifndef  PayLoadInspector_H
#define  PayLoadInspector_H
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include <string>
#include <vector>
#include<sstream>

// to be moved in src
#include "CondCore/DBCommon/interface/PoolTransaction.h"

#include "CondFormats/Common/interface/PayloadWrapper.h"

namespace cond {
  // to be moved elsewhere
  class PoolTransactionSentry {
  public:
    PoolTransactionSentry(){}
    PoolTransactionSentry(cond::PoolTransaction & db) : 
      elem(new Elem(db)){}      
  private:
    struct Elem {
      Elem(cond::PoolTransaction & db) : pooldb(db){
	pooldb.start(true);
      }
      ~Elem() { pooldb.commit();}
      cond::PoolTransaction & pooldb;
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
    typedef cond::DataWrapper<DataT> DataWrapper;

    PayLoadInspector() {}

    PayLoadInspector(const cond::IOVElementProxy & elem) {
      cond::PoolTransaction & db = *elem.db();
      db.start(true);
      load(&db.poolDataSvc(),elem.token());
      db.commit();
    }

    std::string dump() const { return ""; }

    // specialize in case of no-wrapper!
    std::string summary() const {
      std::ostringstream os;
      os << wrapper->summary();
      os << std::endl;
      return os.str();
    }

    // return the real name of the file including extension...
    std::string plot(std::string const & /* filename */,
		     std::string const &, 
		     std::vector<int> const&, std::vector<float> const& ) const {return "";}

    void extract(Extractor & extractor) const {extractor.computeW(object()); }

    Class const & object() const { 
      return old ? *m_OldData : m_data->data(); 
    }

  private:
    virtual bool load(pool::IDataSvc * svc, std::string const & token) {
      old = false;
     bool ok = false;
      // try wrapper, if not try plain
      pool::Ref<DataWrapper> ref(svc,token);
      if (ref) {
	m_data.copyShallow(ref);
	m_data->data();
	ok= true;
      } else {
	pool::Ref<DataT> refo(svc,token);
	if (refo) {
	  old = true;
	  m_OldData.copyShallow(refo);
	  ok =  true;
	}
      }
      return ok;
    }
    
    

  private:
    bool old;
    pool::Ref<DataWrapper> m_data;
    // Backward compatibility
    pool::Ref<DataT> m_OldData;
  };

}

#endif //   PayLoadInspector_H
