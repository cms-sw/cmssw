#ifndef CondIter_CondBasicIter_h
#define CondIter_CondBasicIter_h
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"

class CondBasicIter{

    private:
        
  cond::IOVIterator* ioviterator;
  std::string payloadContainer;
  cond::PoolTransaction *pooldb;
  cond::Connection * myconnection;
  /*minimum and maximum of the interval where search IOVs*/     
  unsigned int iter_Min;
  unsigned int iter_Max;
  /*start time of each IOV*/
  unsigned int m_startTime;
  /*stop time of each IOV*/
  unsigned int m_stopTime;
  unsigned int m_time;
  
  
 public:
  
  CondBasicIter();
  ~CondBasicIter();    
  
  template <class A> friend class CondIter;
  
  
  /**
     tell Iter to point to a database. After this call Iter can be used.
     Direct Access to database through frontier
     It needs:
     \li \c NameDB -> name of the database
     \li \c File -> Tag human-readable of the content of the database
     \li \c User -> name of the User (if you don't need to authenticate don't write anything here)
     \li \c Pass -> Password to access database (if you don't need to authenticate don't write anything here)
     \li \c nameBlob -> to handle blob type of data (if it is not needed this field has to be left empty)
  */
  
  void create(const std::string & NameDB,
	      const std::string & File,
	      const std::string & User = "",
	      const std::string & Pass = "",
	      const std::string & nameBlob = ""
	      );
  
  
  /**
     Set the range of interest of the Iterator of the IOVs.
  */ 
  void setRange(unsigned int min,unsigned int max);
  void setRange(int min,int max); 
  
  
  /**
     Set the minimum of the range of interest of the Iterator of the IOVs.
  */  
  
  void setMin(unsigned int min);
  void setMin(int min);
  
  /**
     Set the maximum of the range of interest of the Iterator of the IOVs.
  */  
  
  void setMax(unsigned int max);
  void setMax(int max);
  
  /**
     Get the mean time of the Iterval of Validity.
  */  
  unsigned int getTime();
  
  /**
     Get the SINCE TIME of the Interval of Validity.
  */
  unsigned int getStartTime();
  
  /**
     Get the TILL TIME of the Interval of Validity.
  */
  unsigned int getStopTime();
  
  /**
     Get the minimum of the range of interest of the Iterator of the IOVs.
  */
  unsigned int getMin();
  
  /**
     Get the maximum of the range of interest of the Iterator of the IOVs.
  */     
  unsigned int getMax();
  
  
  /**
     Get the minimum and the maximum of the range of interest of the Iterator of the IOVs.
  */ 
  void getRange(unsigned int*,unsigned int*);
  
  /**
     Set the SINCE TIME of the IOV just analized inside the class CondBasicIter.
  */
  void setStartTime(unsigned int start);
  
  /**
     Set the TILL TIME of the IOV just analized inside the class CondBasicIter.
  */
  void setStopTime(unsigned int stop); 
  
  /**
     Set the avarage time of the IOV just analized inside the class CondBasicIter.
  */      
  void setTime(unsigned int time); 
  
};


#endif
