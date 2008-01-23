#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/SequenceManager.h"

#include <string>
#include <iostream>
//#include <stdio.h>
//#include <time.h>

int main(){
  cond::DBSession* session=new cond::DBSession;
  session->configuration().setMessageLevel( cond::Error );
  session->configuration().setAuthenticationMethod(cond::XML);
  cond::Connection con("sqlite_file:mydata.db",-1);
  session->open();
  con.connect(session);
  cond::CoralTransaction& coralTransaction=con.coralTransaction();
  coralTransaction.start(false);
  cond::SequenceManager sequenceGenerator(coralTransaction,"mysequenceDepot");
  if( !sequenceGenerator.existSequencesTable() ){
    sequenceGenerator.createSequencesTable();
  }
  unsigned long long targetId=sequenceGenerator.incrementId("MYLOGDATA");
  std::cout<<"targetId for table MYLOGDATA "<<targetId<<std::endl;
  sequenceGenerator.clear();
  coralTransaction.commit();
  con.disconnect();
  delete session;
}
