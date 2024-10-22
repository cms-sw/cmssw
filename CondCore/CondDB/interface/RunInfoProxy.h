#ifndef CondCore_CondDB_RunInfoProxy_h
#define CondCore_CondDB_RunInfoProxy_h
//
// Package:     CondDB
// Class  :     RunInfoProxy
//
/**\class RunInfoProxy RunInfoProxy.h CondCore/CondDB/interface/RunInfoProxy.h
   Description: service for read/only access to the Run information.  
*/
//
// Author:      Giacomo Govi
// Created:     Dec 2016
//

#include "CondCore/CondDB/interface/Types.h"
//

namespace cond {

  namespace persistency {

    class SessionImpl;
    class RunInfoProxyData;

    // value semantics. to be used WITHIN the parent session transaction ( therefore the session should be kept alive ).
    class RunInfoProxy {
    public:
      typedef std::vector<std::tuple<Time_t, boost::posix_time::ptime, boost::posix_time::ptime> > RunInfoData;

    public:
      // more or less compliant with typical iterator semantics...
      class Iterator {
      public:
        // C++17 compliant iterator definition
        using iterator_category = std::input_iterator_tag;
        using value_type = cond::RunInfo_t;
        using difference_type = void;  // Not used
        using pointer = void;          // Not used
        using reference = void;        // Not used

        //
        Iterator();
        explicit Iterator(RunInfoData::const_iterator current);
        Iterator(const Iterator& rhs);

        //
        Iterator& operator=(const Iterator& rhs);

        // returns a VALUE not a reference!
        cond::RunInfo_t operator*();

        //
        Iterator& operator++();
        Iterator operator++(int);

        //
        bool operator==(const Iterator& rhs) const;
        bool operator!=(const Iterator& rhs) const;

      private:
        RunInfoData::const_iterator m_current;
      };

    public:
      RunInfoProxy();

      //
      explicit RunInfoProxy(const std::shared_ptr<SessionImpl>& session);

      //
      RunInfoProxy(const RunInfoProxy& rhs);

      //
      RunInfoProxy& operator=(const RunInfoProxy& rhs);

      // loads in memory the RunInfo data for the specified run range
      void load(Time_t low, Time_t up);

      // loads in memory the RunInfo data for the specified run range
      void load(const boost::posix_time::ptime& low, const boost::posix_time::ptime& up);

      // clear all the iov data in memory
      void reset();

      // start the iteration.
      Iterator begin() const;

      //
      Iterator end() const;

      //
      Iterator find(Time_t target) const;

      //
      Iterator find(const boost::posix_time::ptime& target) const;

      //
      cond::RunInfo_t get(Time_t target) const;

      //
      cond::RunInfo_t get(const boost::posix_time::ptime& target) const;

      //
      int size() const;

    private:
      void checkTransaction(const std::string& ctx);

    private:
      std::shared_ptr<RunInfoProxyData> m_data;
      std::shared_ptr<SessionImpl> m_session;
    };

  }  // namespace persistency
}  // namespace cond

#endif
