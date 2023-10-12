#ifndef CondCore_CondDB_GTProxy_h
#define CondCore_CondDB_GTProxy_h
//
// Package:     CondDB
// Class  :     GTProxy
//
/**\class GTProxy GTProxy.h CondCore/CondDB/interface/GTProxy.h
   Description: service for read/only access to the condition Global Tags.  
*/
//
// Author:      Giacomo Govi
// Created:     Jul 2013
//

#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    // required to handle the tag overriding
    std::string fullyQualifiedTag(const std::string& tag, const std::string& connectionString);
    std::pair<std::string, std::string> parseTag(const std::string& tag);

    class SessionImpl;
    class GTProxyData;

    // value semantics. to be used WITHIN the parent session transaction ( therefore the session should be kept alive ).
    class GTProxy {
    public:
      typedef std::vector<std::tuple<std::string, std::string, std::string> > GTContainer;

    public:
      // more or less compliant with typical iterator semantics...
      class Iterator {
      public:
        // C++17 compliant iterator definition
        using iterator_category = std::input_iterator_tag;
        using value_type = cond::GTEntry_t;
        using difference_type = void;  // Not used
        using pointer = void;          // Not used
        using reference = void;        // Not used

        //
        Iterator();
        explicit Iterator(GTContainer::const_iterator current);
        Iterator(const Iterator& rhs);

        //
        Iterator& operator=(const Iterator& rhs);

        // returns a VALUE not a reference!
        cond::GTEntry_t operator*();

        //
        Iterator& operator++();
        Iterator operator++(int);

        //
        bool operator==(const Iterator& rhs) const;
        bool operator!=(const Iterator& rhs) const;

      private:
        GTContainer::const_iterator m_current;
      };

    public:
      GTProxy();
      // the only way to construct it from scratch...
      explicit GTProxy(const std::shared_ptr<SessionImpl>& session);

      //
      GTProxy(const GTProxy& rhs);

      //
      GTProxy& operator=(const GTProxy& rhs);

      // loads in memory the gtag information and the tags. to be resurrected after the changeover.
      //void load( const std::string& gtName );

      // overloading for pre- and post-fix. Used in the ORA implementation
      void load(const std::string& gtName, const std::string& preFix = "", const std::string& postFix = "");

      // reset the data in memory and execute again the queries for the current tag
      void reload();

      // clear all the iov data in memory
      void reset();

      std::string name() const;

      cond::Time_t validity() const;

      boost::posix_time::ptime snapshotTime() const;

      // start the iteration.
      Iterator begin() const;

      //
      Iterator end() const;

      //
      int size() const;

    private:
      void checkTransaction(const std::string& ctx);

    private:
      std::shared_ptr<GTProxyData> m_data;
      std::shared_ptr<SessionImpl> m_session;
    };

  }  // namespace persistency
}  // namespace cond

#endif
