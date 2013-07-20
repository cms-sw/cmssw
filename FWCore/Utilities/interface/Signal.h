#ifndef FWCore_ServiceRegistry_Signal_h
#define FWCore_ServiceRegistry_Signal_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     Signal
// 
/**\class Signal Signal.h FWCore/ServiceRegistry/interface/Signal.h

 Description: A simple implementation of the signal/slot pattern

 Usage:
    This is a simple version of the signal/slot pattern and is used by the Framework. It is safe
 to call 'emit' from multiple threads simultaneously.
 Assumptions:
 -The attached slots have a life-time greater than the last 'emit' call issued from the Signal.
 -'connect' is not called simultaneously with any other methods of the class.
*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jan 17 16:03:51 CST 2013
// $Id: Signal.h,v 1.1 2013/01/20 16:56:21 chrjones Exp $
//

// system include files
#include <vector>
#include <functional>

// user include files

// forward declarations

namespace edm {
  namespace signalslot {
    template <typename T>
    class Signal
    {
      
    public:
      typedef std::function<T> slot_type;
      typedef std::vector<slot_type> slot_list_type;
      
      Signal() = default;
      ~Signal() = default;
      
      // ---------- const member functions ---------------------
      template<typename... Args>
      void emit(Args&&... args) const {
        for(auto& slot:m_slots) {
          slot(std::forward<Args>(args)...);
        }
      }
      
      template<typename... Args>
      void operator()(Args&&... args) const {
        emit(std::forward<Args>(args)...);
      }
      
      slot_list_type const& slots() const {return m_slots;}
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      template<typename U>
      void connect(U iFunc) {
        m_slots.push_back(std::function<T>(iFunc));
      }

      template<typename U>
      void connect_front(U iFunc) {
        m_slots.insert(m_slots.begin(),std::function<T>(iFunc));
      }

    private:
      Signal(const Signal&) = delete; // stop default
      
      const Signal& operator=(const Signal&) = delete; // stop default
      
      // ---------- member data --------------------------------
      slot_list_type m_slots;
      
    };
  }
}

#endif
