#ifndef CondCore_CondDB_IOVProxy_h
#define CondCore_CondDB_IOVProxy_h
//
// Package:     CondDB
// Class  :     IOVProxy
//
/**\class IOVProxy IOVProxy.h CondCore/CondDB/interface/IOVProxy.h
   Description: service for read/only access to the condition IOVs.  
*/
//
// Author:      Giacomo Govi
// Created:     Apr 2013
//

#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <boost/date_time/posix_time/posix_time.hpp>

namespace cond {

  namespace persistency {

    class SessionImpl;
    class IOVProxyData;

    typedef std::vector<std::tuple<cond::Time_t, cond::Hash> > IOVContainer;

    class IOVArray {
    public:
      IOVArray();
      IOVArray(const IOVArray& rhs);
      IOVArray& operator=(const IOVArray& rhs);

    public:
      // more or less compliant with typical iterator semantics...
      class Iterator : public std::iterator<std::input_iterator_tag, cond::Iov_t> {
      public:
        //
        Iterator();
        Iterator(IOVContainer::const_iterator current, const IOVArray* parent);

        Iterator(const Iterator& rhs);

        //
        Iterator& operator=(const Iterator& rhs);

        // returns a VALUE not a reference!
        cond::Iov_t operator*();

        //
        Iterator& operator++();
        Iterator operator++(int);

        //
        bool operator==(const Iterator& rhs) const;
        bool operator!=(const Iterator& rhs) const;

      private:
        IOVContainer::const_iterator m_current;
        const IOVArray* m_parent;
      };
      friend class Iterator;

    public:
      const cond::Tag_t& tagInfo() const;

      // start the iteration. it referes to the LOADED iov sequence subset, which consists in two consecutive groups - or the entire sequence if it has been requested.
      // returns data only when a find or a load( tag, true ) have been at least called once.
      Iterator begin() const;

      // the real end of the LOADED iov sequence subset.
      Iterator end() const;

      // searches the array for a valid iov containing the specified time.
      // if the available iov sequence subset contains the target time, it does not issue a new query.
      // otherwise, a new query will be executed using the resolved group boundaries.
      Iterator find(cond::Time_t time) const;

      size_t size() const;

      // returns true if at least one IOV is in the sequence.
      bool isEmpty() const;

    private:
      friend class IOVProxy;
      std::unique_ptr<IOVContainer> m_array;
      cond::Tag_t m_tagInfo;
    };

    // value semantics. to be used WITHIN the parent session transaction ( therefore the session should be kept alive ).
    class IOVProxy {
    public:
      //
      IOVProxy();

      // the only way to construct it from scratch...
      explicit IOVProxy(const std::shared_ptr<SessionImpl>& session);

      //
      IOVProxy(const IOVProxy& rhs);

      //
      IOVProxy& operator=(const IOVProxy& rhs);

      IOVArray selectAll();
      IOVArray selectAll(const boost::posix_time::ptime& snapshottime);

      IOVArray selectRange(const cond::Time_t& begin, const cond::Time_t& end);
      IOVArray selectRange(const cond::Time_t& begin,
                           const cond::Time_t& end,
                           const boost::posix_time::ptime& snapshottime);

      cond::Tag_t tagInfo() const;

      cond::TagInfo_t iovSequenceInfo() const;

      // searches the DB for a valid iov containing the specified time.
      // if the available iov sequence subset contains the target time, it does not issue a new query.
      // otherwise, a new query will be executed using the resolved group boundaries.
      // throws if the target time cannot be found.
      cond::Iov_t getInterval(cond::Time_t time);
      cond::Iov_t getInterval(cond::Time_t time, cond::Time_t defaultIovSize);

      std::tuple<std::string, boost::posix_time::ptime, boost::posix_time::ptime> getMetadata() const;

      // loads in memory the tag information and the iov groups
      // full=true load the full iovSequence
      void load(const std::string& tag);

      // loads in memory the tag information and the iov groups
      void load(const std::string& tag, const boost::posix_time::ptime& snapshottime);

      // clear all the iov data in memory
      void reset();

      // it does NOT use the cache, every time it performs a new query.
      cond::Iov_t getLast();

      // the size of the LOADED iov sequence subset.
      int loadedSize() const;

      // the size of the entire iov sequence. Peforms a query at every call.
      int sequenceSize() const;

      // for reporting
      size_t numberOfQueries() const;

      // for debugging
      std::pair<cond::Time_t, cond::Time_t> loadedGroup() const;

      // maybe will be removed with a re-design of the top level interface (ESSources )
      const std::shared_ptr<SessionImpl>& session() const;

    private:
      void checkTransaction(const std::string& ctx) const;
      void resetIOVCache();
      void loadGroups();
      void fetchSequence(cond::Time_t lowerGroup, cond::Time_t higherGroup);

    private:
      std::shared_ptr<IOVProxyData> m_data;
      std::shared_ptr<SessionImpl> m_session;
    };

  }  // namespace persistency
}  // namespace cond

#endif
