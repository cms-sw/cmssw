#ifndef Cond_PayloadWrapper_h
#define Cond_PayloadWrapper_h


#include <typeinfo>
#include "POOLCore/Ptr.h"
#include "CondFormats/Common/interface/Summary.h"

namespace cond {

  /** base class of IOV payload wapper
      the final class will include ptrs for payload and its summary
  */
  class  PayloadWrapper {
  public:
    
    // load DOES NOT throw!
    virtual void loadAll() const {
      loadData();
      loadSummary();
    }
    
    virtual bool loadData() const =0;
    
    
    //    virtual bool loadSummary() const =0;
    
    
    //-- summary part (concrete)
    typedef cond::Summary summary_type;
    
    
    PayloadWrapper(Summary * sum=0) :
      m_summary(sum){}
    
    virtual ~PayloadWrapper() {
      if (m_summary.isLoaded()) delete m_summary.get();
    }    
    
    
    Summary const & summary() const { return *m_summary;}
    
    bool loadSummary() const {
      return m_summary.get();
    }
    
  private:
    
    pool::Ptr<Summary> m_summary;
    
  };
  
  
  /** base class of IOV payload wrapper (no summary)
   */
  template<typename O> 
  class DataWrapper : public PayloadWrapper {
  public:
    typedef PayloadWrapper base;
    typedef O Object; 
    typedef Object value_type; 
    typedef DataWrapper<value_type> self;
    typedef base::summary_type summary_type;
    
    
    explicit DataWrapper(Object * obj=0, Summary * sum=0) : 
      base(sum), m_data(obj){}
    
    virtual ~DataWrapper() {
      if (m_data.isLoaded()) delete m_data.get();
    }    
    
    
    Object const & data() const { return *m_data;}
    
    bool loadData() const {
      return m_data.get();
    }
    
  private:
    
    pool::Ptr<Object> m_data;
  };
  
} // ns

#endif //Cond_PayloadWrapper_h
