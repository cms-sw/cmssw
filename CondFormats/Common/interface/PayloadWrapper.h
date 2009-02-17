#ifndef Cond_PayloadWrapper_h
#define Cond_PayloadWrapper_h


#include "POOLCore/Ptr.h"

namespace cond {

  /** base class of IOV payload wapper
      the final class will include ptrs for payload and its summary
   */
  class  PayloadWrapper {
  public:
    virtual ~PayloadWrapper(){}
    

    // load DOES NOT throw!
    virtual void loadAll() const {
      loadData();
      loadSummary();
    }

    virtual bool loadData() const =0;
    virtual bool loadSummary() const =0;



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
 
    
    explicit DataWrapper(Object * obj=0) : m_data(obj){}

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

  /** base class of IOV payload wrapper (with summary)
   */
  template<typename O, typename S> 
  class DataAndSummaryWrapper : public DataWrapper<O> {
  public: 
    typedef DataWrapper<O> ObjectWrapper;
    typedef typename ObjectWrapper::base base;
    typedef typename ObjectWrapper::Object Object;
    typedef typename ObjectWrapper::value_type value_type;
    typedef S Summary;
    typedef Summary summary_type;
    
    typedef DataAndSummaryWrapper<value_type, summary_type> self;
    
    DataAndSummaryWrapper(Object * obj=0, Summary * sum=0) :
      ObjectWrapper(obj), m_summary(sum){}

    virtual ~DataAndSummaryWrapper() {
      if (m_summary.isLoaded()) delete m_summary.get();
    }    

    
    Summary const & summary() const { return *m_summary;}

    bool loadSummary() const {
      return m_summary.get();
    }

  private:

    pool::Ptr<Summary> m_summary;


  };


} // ns

#endif //Cond_PayloadWrapper_h
