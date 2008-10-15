#ifndef RUNINFOREAD_H
#define RUNINFOREAD_H


#include "TestBase.h"
#include "CoralBase/TimeStamp.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"



class RunInfoRead : virtual public TestBase
{
public:
  RunInfoRead(
	 const std::string& connectionString,
         const std::string& user,
         const std::string& pass);
	      
   virtual ~RunInfoRead();
   void run();

   RunInfo::RunInfo  readData(const std::string & table, const std::string & column, const int r_number );
 
private:
   std::string m_tableToRead;
  std::string m_columnToRead;
  std::string m_connectionString;
  std::string m_user;
  std::string m_pass;
};

#endif
