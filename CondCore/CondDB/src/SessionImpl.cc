#include "CondCore/CondDB/interface/Exception.h"
#include "SessionImpl.h"
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

    SessionImpl::SessionImpl(std::shared_ptr<coral::ISessionProxy>& session, const std::string& connectionStr)
        : coralSession(session), connectionString(connectionStr) {}

    SessionImpl::~SessionImpl() { close(); }

    void SessionImpl::close() {
      if (coralSession.get()) {
        if (coralSession->transaction().isActive()) {
          coralSession->transaction().rollback();
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
        iovSchemaHandle.reset(new IOVSchema(coralSession->nominalSchema()));
        gtSchemaHandle.reset(new GTSchema(coralSession->nominalSchema()));
        runInfoSchemaHandle.reset(new RunInfoSchema(coralSession->nominalSchema()));
        transaction.reset(new CondDBTransaction(coralSession));
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
