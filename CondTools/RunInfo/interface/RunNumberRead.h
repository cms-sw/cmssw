#ifndef RUNNUMBEREAD_H
#define RUNNUMBEREAD_H


#include "TestBase.h"
#include "CoralBase/TimeStamp.h"
#include "CondFormats/RunInfo/interface/RunNumber.h"
/*
 *  \class RunNumberRead
 *  
 *  coral application for reading  runnumber info from Run Control account in OMDS and filling the object
 *  
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN (June-25-2008)
 *
*/

class RunNumberRead : virtual public TestBase
{
public:
  RunNumberRead(
	 const std::string& connectionString,
         const std::string& user,
         const std::string& pass);
	      
   virtual ~RunNumberRead();
   void run();

   void dropTable(const std::string & table );
   
   std::vector<RunNumber::Item>  readData(const std::string & table, const std::string & column, const int r_number );
 
private:
  std::string m_tableToDrop;  
  std::string m_tableToRead;
  std::string m_columnToRead;
  std::string m_connectionString;
  std::string m_user;
  std::string m_pass;
};

#endif
