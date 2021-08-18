#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Auth.h"
#include "CondCore/CondDB/interface/DecodingKey.h"
#include "SessionImpl.h"

#include <memory>

#include "DbConnectionString.h"
//
//
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"

namespace cond {

  namespace persistency {

    class CondDBTransaction : public ITransaction {
    public:
      CondDBTransaction(const std::shared_ptr<coral::ISessionProxy>& coralSession) : m_session(coralSession) {}
      ~CondDBTransaction() override {}

      void commit() override { m_session->transaction().commit(); }

      void rollback() override { m_session->transaction().rollback(); }

      bool isActive() override { return m_session->transaction().isActive(); }

    private:
      std::shared_ptr<coral::ISessionProxy> m_session;
    };

    SessionImpl::SessionImpl() : coralSession() {}

    SessionImpl::SessionImpl(std::shared_ptr<coral::ISessionProxy>& session,
                             const std::string& connectionStr,
                             const std::string& principalNm)
        : coralSession(session), sessionHash(""), connectionString(connectionStr), principalName(principalNm) {
      cond::auth::KeyGenerator kg;
      sessionHash = kg.make(cond::auth::COND_SESSION_HASH_SIZE);
    }

    SessionImpl::~SessionImpl() { close(); }

    void SessionImpl::close() {
      if (isActive()) {
        if (coralSession->transaction().isActive()) {
          rollbackTransaction();
        }
        if (!lockedTags.empty()) {
          startTransaction(false);
          releaseTagLocks();
          commitTransaction();
        }
        coralSession.reset();
      }
      transaction.reset();
    }

    bool SessionImpl::isActive() const { return coralSession.get(); }

    void SessionImpl::startTransaction(bool readOnly) {
      std::unique_lock<std::recursive_mutex> lock(transactionMutex);
      if (!transaction.get()) {
        coralSession->transaction().start(readOnly);
        iovSchemaHandle = std::make_unique<IOVSchema>(coralSession->nominalSchema());
        gtSchemaHandle = std::make_unique<GTSchema>(coralSession->nominalSchema());
        runInfoSchemaHandle = std::make_unique<RunInfoSchema>(coralSession->nominalSchema());
        transaction = std::make_unique<CondDBTransaction>(coralSession);
      } else {
        if (!readOnly)
          throwException("An update transaction is already active.", "SessionImpl::startTransaction");
      }
      transaction->clients++;
      transactionLock.swap(lock);
    }

    void SessionImpl::commitTransaction() {
      std::unique_lock<std::recursive_mutex> lock;
      lock.swap(transactionLock);
      if (transaction) {
        transaction->clients--;
        if (!transaction->clients) {
          transaction->commit();
          transaction.reset();
          iovSchemaHandle.reset();
          gtSchemaHandle.reset();
          runInfoSchemaHandle.reset();
        }
      }
    }

    void SessionImpl::rollbackTransaction() {
      std::unique_lock<std::recursive_mutex> lock;
      lock.swap(transactionLock);
      if (transaction) {
        transaction->rollback();
        transaction.reset();
        iovSchemaHandle.reset();
        gtSchemaHandle.reset();
        runInfoSchemaHandle.reset();
      }
    }

    bool SessionImpl::isTransactionActive(bool deep) const {
      if (!transaction)
        return false;
      if (!deep)
        return true;
      return transaction->isActive();
    }

    void SessionImpl::releaseTagLocks() {
      iovSchema().tagAccessPermissionTable().removeEntriesForCredential(sessionHash,
                                                                        cond::auth::COND_SESSION_HASH_CODE);
      std::string lt("-");
      std::string action("Lock removed by session ");
      action += sessionHash;
      for (const auto& tag : lockedTags) {
        iovSchema().tagTable().unsetProtectionCode(tag, cond::auth::COND_DBTAG_LOCK_ACCESS_CODE);
        iovSchema().tagLogTable().insert(tag,
                                         boost::posix_time::microsec_clock::universal_time(),
                                         cond::getUserName(),
                                         cond::getHostName(),
                                         cond::getCommand(),
                                         action,
                                         lt);
      }
    }

    void SessionImpl::openIovDb(SessionImpl::FailureOnOpeningPolicy policy) {
      if (!transaction.get())
        throwException("The transaction is not active.", "SessionImpl::openIovDb");
      if (!transaction->iovDbOpen) {
        transaction->iovDbExists = iovSchemaHandle->exists();
        transaction->iovDbOpen = true;
      }
      if (!transaction->iovDbExists) {
        if (policy == CREATE) {
          iovSchemaHandle->create();
          transaction->iovDbExists = true;
        } else {
          if (policy == THROW)
            throwException("IOV Database does not exist.", "SessionImpl::openIovDb");
        }
      }
    }

    void SessionImpl::openGTDb(SessionImpl::FailureOnOpeningPolicy policy) {
      if (!transaction.get())
        throwException("The transaction is not active.", "SessionImpl::openGTDb");
      if (!transaction->gtDbOpen) {
        transaction->gtDbExists = gtSchemaHandle->exists();
        transaction->gtDbOpen = true;
      }
      if (!transaction->gtDbExists) {
        if (policy == CREATE) {
          gtSchemaHandle->create();
          transaction->gtDbExists = true;
        } else {
          if (policy == THROW)
            throwException("GT Database does not exist.", "SessionImpl::openGTDb");
        }
      }
    }

    void SessionImpl::openRunInfoDb() {
      if (!transaction.get())
        throwException("The transaction is not active.", "SessionImpl::openRunInfoDb");
      if (!transaction->runInfoDbOpen) {
        transaction->runInfoDbExists = runInfoSchemaHandle->exists();
        transaction->runInfoDbOpen = true;
      }
      if (!transaction->runInfoDbExists) {
        throwException("RunInfo Database does not exist.", "SessionImpl::openRunInfoDb");
      }
    }

    void SessionImpl::openDb() {
      if (!transaction.get())
        throwException("The transaction is not active.", "SessionImpl::openIovDb");
      if (!transaction->iovDbOpen) {
        transaction->iovDbExists = iovSchemaHandle->exists();
        transaction->iovDbOpen = true;
      }
      if (!transaction->gtDbOpen) {
        transaction->gtDbExists = gtSchemaHandle->exists();
        transaction->gtDbOpen = true;
      }
      if (!transaction->iovDbExists) {
        iovSchemaHandle->create();
        transaction->iovDbExists = true;
        if (!transaction->gtDbExists) {
          gtSchemaHandle->create();
          transaction->gtDbExists = true;
        }
      }
    }

    IIOVSchema& SessionImpl::iovSchema() { return *iovSchemaHandle; }

    IGTSchema& SessionImpl::gtSchema() { return *gtSchemaHandle; }

    IRunInfoSchema& SessionImpl::runInfoSchema() { return *runInfoSchemaHandle; }
  }  // namespace persistency
}  // namespace cond
