#include "CondCore/DBCommon/interface/ITransaction.h"
#include "ITransactionObserver.h"
//#include <iostream>
void 
cond::ITransaction::attach( cond::ITransactionObserver* observer ){
  m_observers.push_back( observer );
}
void
cond::ITransaction::NotifyStartOfTransaction(){
  std::vector< cond::ITransactionObserver* >::iterator it;
  std::vector< cond::ITransactionObserver* >::iterator itBeg=m_observers.begin();
  std::vector< cond::ITransactionObserver* >::iterator itEnd=m_observers.end();
  for (it=itBeg; it!=itEnd; ++it){
    (*it)->reactOnStartOfTransaction(this);
  } 
}
void
cond::ITransaction::NotifyEndOfTransaction(){
  std::vector< cond::ITransactionObserver* >::iterator it;
  std::vector< cond::ITransactionObserver* >::iterator itBeg=m_observers.begin();
  std::vector< cond::ITransactionObserver* >::iterator itEnd=m_observers.end();
  for (it=itBeg; it!=itEnd; ++it){
    (*it)->reactOnEndOfTransaction(this);
  } 
}
