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
//

// system include files
#include <exception>
#include <vector>
#include <functional>

// user include files
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// forward declarations

namespace edm {
  namespace signalslot {
    template <typename T>
    class Signal {
    public:
      typedef std::function<T> slot_type;
      typedef std::vector<slot_type> slot_list_type;

      Signal() = default;
      ~Signal() = default;
      Signal(Signal&&) = default;
      Signal(const Signal&) = delete;
      Signal& operator=(const Signal&) = delete;

      // ---------- const member functions ---------------------
      template <typename... Args>
      void emit(Args&&... args) const {
        std::exception_ptr exceptionPtr;
        for (auto& slot : m_slots) {
          CMS_SA_ALLOW try { slot(std::forward<Args>(args)...); } catch (...) {
            if (!exceptionPtr) {
              exceptionPtr = std::current_exception();
            }
          }
        }
        if (exceptionPtr) {
          std::rethrow_exception(exceptionPtr);
        }
      }

#ifdef FWCORE_SIGNAL_OPERATOR_PARENTHESIS_PRIVATE
    private:
#endif
      template <typename... Args>
      void operator()(Args&&... args) const {
        emit(std::forward<Args>(args)...);
      }

#ifdef FWCORE_SIGNAL_OPERATOR_PARENTHESIS_PRIVATE
    public:
#endif
      slot_list_type const& slots() const { return m_slots; }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      // Utility helpers to connect slots when cheking that emission of signals is done via emit().
      // Check is done at compile time by calling scram b with
      // -DFWCORE_SIGNAL_OPERATOR_PARENTHESIS_PRIVATE=1:
      //
      // USER_CXXFLAGS="-DFWCORE_SIGNAL_OPERATOR_PARENTHESIS_PRIVATE=1" scram b
      //
      // Allow connecting another Signal<T> via std::reference_wrapper<const Signal<T>>
      // std::function cannot be constructed from std::reference_wrapper<const Signal>
      // when we make the Signal::operator() private, so wrap with a lambda that calls emit().
#ifdef FWCORE_SIGNAL_OPERATOR_PARENTHESIS_PRIVATE
      void connect(std::reference_wrapper<const Signal> iFunc) {
        m_slots.emplace_back([iFunc](auto&&... args) { iFunc.get().emit(std::forward<decltype(args)>(args)...); });
      }

      void connect_front(std::reference_wrapper<const Signal> iFunc) {
        m_slots.insert(m_slots.begin(),
                       slot_type([iFunc](auto&&... args) { iFunc.get().emit(std::forward<decltype(args)>(args)...); }));
      }
#endif

      template <typename U>
      void connect(U iFunc) {
        m_slots.push_back(std::function<T>(iFunc));
      }

      template <typename U>
      void connect_front(U iFunc) {
        m_slots.insert(m_slots.begin(), std::function<T>(iFunc));
      }

    private:
      // ---------- member data --------------------------------
      slot_list_type m_slots;
    };
  }  // namespace signalslot
}  // namespace edm

#endif
