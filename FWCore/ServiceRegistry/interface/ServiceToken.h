#ifndef ServiceRegistry_ServiceToken_h
#define ServiceRegistry_ServiceToken_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceToken
//
/**\class ServiceToken ServiceToken.h FWCore/ServiceRegistry/interface/ServiceToken.h

 Description: Token used to denote a 'service set' 

 Usage:
    When you request a new 'service set' to be created from the ServiceRegistry, 
  the ServiceRegistry will return a ServiceToken.  When you want this 'service set' to be used,
  create a ServiceRegistry::Operate by passing the ServiceToken via the constructor.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Sep  6 18:31:44 EDT 2005
//

// system include files
#include <memory>

// user include files

// forward declarations
namespace edm {
  class ServiceRegistry;
  class ActivityRegistry;

  namespace serviceregistry {
    class ServicesManager;
  }

  class OwningServiceToken;
  class WeakServiceToken;

  /// Non-owning handle to a 'service set': it behaves like a pointer to the services, without sharing their
  /// ownership. It can be copied as cheaply as a raw pointer, with no reference counting, so that it can be passed
  /// to and stored in the tasks that run concurrently on many threads.
  /// A ServiceToken can only be obtained from an OwningServiceToken, or from ServiceRegistry::presentToken() while
  /// the services are in use; it must not outlive the OwningServiceToken that keeps the services alive. The framework
  /// guarantees this for the tokens it hands out: the services outlive all the modules and tasks of the process.
  class ServiceToken {
    friend class edm::ServiceRegistry;
    friend class edm::serviceregistry::ServicesManager;
    friend class edm::OwningServiceToken;
    friend class edm::WeakServiceToken;

  public:
    ServiceToken() = default;

    // ---------- const member functions ---------------------
    explicit operator bool() const noexcept { return manager_ != nullptr; }
    friend bool operator==(ServiceToken const&, ServiceToken const&) = default;

  private:
    explicit ServiceToken(edm::serviceregistry::ServicesManager* iManager) : manager_(iManager) {}

    // ---------- member data --------------------------------
    edm::serviceregistry::ServicesManager* manager_ = nullptr;
  };

  /// Owning handle to a 'service set': the services are kept alive as long as an OwningServiceToken refers to them.
  /// It is held by the owners of the services (e.g. the EventProcessor) and converts to a non-owning ServiceToken.
  class OwningServiceToken {
    friend class edm::ServiceRegistry;
    friend class edm::WeakServiceToken;

  public:
    OwningServiceToken() = default;
    /// share the ownership of the services referred to by iToken
    explicit OwningServiceToken(ServiceToken iToken);

    // ---------- const member functions ---------------------
    explicit operator bool() const noexcept { return bool(manager_); }

    ServiceToken token() const& noexcept { return ServiceToken(manager_.get()); }
    operator ServiceToken() const& noexcept { return token(); }
    // a non-owning token to a temporary would dangle as soon as the temporary is destroyed
    ServiceToken token() && = delete;
    operator ServiceToken() && = delete;

    // ---------- member functions ---------------------------
    ///the argument's signals will forward the token's signals
    void connect(ActivityRegistry&);

    ///copy our Service's slots to the argument's signals
    void copySlotsTo(ActivityRegistry&);

  private:
    OwningServiceToken(std::shared_ptr<edm::serviceregistry::ServicesManager> iManager)
        : manager_(std::move(iManager)) {}

    // ---------- member data --------------------------------
    std::shared_ptr<edm::serviceregistry::ServicesManager> manager_;
  };

  /// Weak handle to a 'service set', for objects that may outlive the services (e.g. a TFile kept open by ROOT):
  /// lock() returns an empty token if the services have been destroyed.
  class WeakServiceToken {
  public:
    WeakServiceToken(ServiceToken iToken);
    WeakServiceToken() = default;

    WeakServiceToken& operator=(ServiceToken iToken) { return *this = WeakServiceToken(iToken); }

    OwningServiceToken lock() const { return OwningServiceToken(manager_.lock()); }

  private:
    std::weak_ptr<edm::serviceregistry::ServicesManager> manager_;
  };
}  // namespace edm

#endif
