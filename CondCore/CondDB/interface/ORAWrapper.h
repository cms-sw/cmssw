#ifndef CondCore_CondDB_ORAWrapper_h
#define CondCore_CondDB_ORAWrapper_h

#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/TagMetadata.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Configuration.h"

namespace cond {

  namespace ora_wrapper {

    typedef cond::persistency::SessionConfiguration SessionConfiguration;

    class Switch {
    public:
      typedef std::map<std::string,cond::DbSession> OraPool;
    public:
      Switch();
      Switch( const Switch& rhs );
      Switch& operator=( const Switch& rhs );
      bool open( const std::string& connectionString, bool readOnly=false  );
      void close();
      bool isOra() const;
      
      std::shared_ptr<OraPool> oraImpl;
      cond::DbSession oraCurrent;
      cond::persistency::Session impl;
    };

    class Transaction {
    public:
      explicit Transaction( Switch& s );
      Transaction( const Transaction& rhs );
      Transaction& operator=( const Transaction& rhs );

      void start( bool readOnly=true );

      void commit();

      void rollback();

      bool isActive(); 
    private:
      Switch* m_switch;
    };

    class IOVProxy {
    public:
      class Iterator  : public std::iterator<std::input_iterator_tag, cond::Iov_t> {
      public:
        //
        Iterator();
        explicit Iterator( cond::persistency::IOVProxy::Iterator impl );
        explicit Iterator( cond::IOVProxy::const_iterator impl );
        Iterator( const Iterator& rhs );

        //
        Iterator& operator=( const Iterator& rhs );

	//
        cond::Iov_t operator*();

        //
        Iterator& operator++();
        Iterator operator++(int);

        //
        bool operator==( const Iterator& rhs ) const;
        bool operator!=( const Iterator& rhs ) const;

      private:
        cond::persistency::IOVProxy::Iterator m_impl;
        cond::IOVProxy::const_iterator m_oraImpl;
        bool m_isOra = false;
      };

    public:
      explicit IOVProxy( const cond::persistency::IOVProxy& impl );
      explicit IOVProxy( cond::DbSession& ORASession );

      //
      IOVProxy();
      //
      IOVProxy( const IOVProxy& rhs );

      //
      IOVProxy& operator=( const IOVProxy& rhs );

      void load( const std::string& tag, bool full=false );

      void reload();

      void reset();

      std::string tag() const; 

      cond::TimeType timeType() const;

      std::string payloadObjectType() const;

      cond::Time_t endOfValidity() const;

      cond::Time_t lastValidatedTime() const;

      Iterator begin() const;

      Iterator end() const;

      Iterator find(cond::Time_t time);

      cond::Iov_t getInterval( cond::Time_t time );
    
      int size() const;

    private:
      cond::persistency::IOVProxy m_impl;
      cond::IOVProxy m_oraImpl;
      std::shared_ptr<std::string> m_oraTag;
    };

    class IOVEditor {
    public:

      IOVEditor();
      explicit IOVEditor( const cond::persistency::IOVEditor& impl );
      explicit IOVEditor( const cond::IOVEditor& ORAImpl );

      IOVEditor( const IOVEditor& rhs );

      //
      IOVEditor& operator=( const IOVEditor& rhs );

      void load( const std::string& tag );

      std::string tag() const;
      cond::TimeType timeType() const;
      std::string payloadType() const;
      cond::SynchronizationType synchronizationType() const;

      cond::Time_t endOfValidity() const;
      void setEndOfValidity( cond::Time_t validity );

      std::string description() const;
      void setDescription( const std::string& description );
      
      cond::Time_t lastValidatedTime() const;
      void setLastValidatedTime( cond::Time_t time );  

      void insert( cond::Time_t since, const cond::Hash& payloadHash, bool checkType=false );
      void insert( cond::Time_t since, const cond::Hash& payloadHash, const boost::posix_time::ptime& insertionTime, bool checkType=false ); 

      bool flush();
      bool flush( const boost::posix_time::ptime& operationTime );

    private:
      cond::persistency::IOVEditor m_impl;
      cond::IOVEditor m_oraImpl;
      std::shared_ptr<std::string> m_oraTag;
    };

    class GTProxy {
    public: 
      typedef std::map<std::string,cond::TagMetadata> OraTagMap;
    public:
      class Iterator  : public std::iterator<std::input_iterator_tag, cond::GTEntry_t> {
      public:
	//
	Iterator();
	explicit Iterator( cond::persistency::GTProxy::Iterator impl );
	explicit Iterator( OraTagMap::const_iterator impl );
	Iterator( const Iterator& rhs );

	Iterator& operator=( const Iterator& rhs );

	cond::GTEntry_t operator*();

	Iterator& operator++();
	Iterator operator++(int);

	bool operator==( const Iterator& rhs ) const;
	bool operator!=( const Iterator& rhs ) const;

      private:
        cond::persistency::GTProxy::Iterator m_impl;
        OraTagMap::const_iterator m_oraImpl;
        bool m_isOra = false;
      };
    
    public:

      GTProxy();
      explicit GTProxy( const cond::persistency::GTProxy& impl );
      explicit GTProxy( const cond::DbSession& oraSession );

      GTProxy( const GTProxy& rhs );

      GTProxy& operator=( const GTProxy& rhs );
      
      void load( const std::string& gtName );

      void reload();

      void reset();

      std::string name() const; 

      cond::Time_t validity() const;

      boost::posix_time::ptime snapshotTime() const;

      Iterator begin() const;

      Iterator end() const;
    
      int size() const;

    private:
      cond::persistency::GTProxy m_impl;
      cond::DbSession m_oraSession;
      struct OraData {
	std::string gt;
	OraTagMap gtData;
      };
      std::shared_ptr<OraData> m_oraData;
    };


    class Session {
    public:
      Session();

      Session( const Session& rhs );

      virtual ~Session(){}

      Session& operator=( const Session& rhs );

      void open( const std::string& connectionString, bool readOnly=false );

      //Session switchSchema( const std::string& connectionString );

      void close();

      SessionConfiguration& configuration();

      Transaction& transaction();

      bool existsDatabase();

      IOVProxy readIov( const std::string& tag, bool full=false );//,const boost::posix_time::ptime& snapshottime )  

      bool existIov( const std::string& tag );

      template <typename T>
      IOVEditor createIov( const std::string& tag,cond::TimeType timeType, 
			   cond::SynchronizationType synchronizationType=cond::OFFLINE );
      IOVEditor createIov( const std::string& tag, cond::TimeType timeType, const std::string& payloadType, 
			   cond::SynchronizationType synchronizationType=cond::OFFLINE );

      IOVEditor editIov( const std::string& tag );

      template <typename T> cond::Hash storePayload( const T& payload, const boost::posix_time::ptime& creationTime );
      template <typename T> boost::shared_ptr<T> fetchPayload( const cond::Hash& payloadHash );

      IOVProxy iovProxy();

      GTProxy readGlobalTag( const std::string& name );

      bool isOra() const;

    private:

      Switch m_switch;
      Transaction m_transaction;
    };

    template <typename T> inline IOVEditor Session::createIov( const std::string& tag, cond::TimeType timeType, cond::SynchronizationType synchronizationType ){
      return createIov( tag, timeType, cond::demangledName( typeid(T) ), synchronizationType );
    }

    template <typename T> inline cond::Hash Session::storePayload( const T& payload, const boost::posix_time::ptime& creationTime ){
      cond::Hash hashOrToken("");
      if( m_switch.isOra() ){
	hashOrToken = m_switch.oraCurrent.storeObject( &payload, cond::demangledName(typeid(payload)) );
      } else {
	hashOrToken = m_switch.impl.storePayload( payload, creationTime );
      }
      return hashOrToken;
    }

    template <typename T> inline boost::shared_ptr<T> Session::fetchPayload( const cond::Hash& payloadHash ){
      boost::shared_ptr<T> obj;
      if( m_switch.isOra() ) {
	obj = m_switch.oraCurrent.getTypedObject<T>( payloadHash );
      } else {
	obj = m_switch.impl.fetchPayload<T>( payloadHash );
      }
      return obj;
    }

  }
}

#endif
