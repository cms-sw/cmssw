#ifndef WRITEDQMSUMMARYINTOOMDS_H
#define WRITEDQMSUMMARYINTOOMDS_H

/*
 *  \class WriteDQMSummaryIntoRDB
 *  
 *  write data from DQM summary file into OMDS (relational db) using Coral 
 *  
 *  further feature provided: dropping table and reading  columns to std
 *
 *  \author Michele de Gruttola (degrutto) - INFN Naples (June-12-2008)
 *
*/

#include "TestBase.h"
#include "CoralBase/TimeStamp.h"

struct Item {
    Item(){}
    ~Item(){}
    int m_run;
    std::string m_subsystem;
    std::string m_reportcontent;
    float m_status;

  };

  typedef std::vector<Item>::const_iterator ItemIterator;


class WriteDQMSummaryIntoOMDS : virtual public TestBase
{
public:
  WriteDQMSummaryIntoOMDS(
	 const std::string& connectionString,
         const std::string& user,
         const std::string& pass);
	      
   virtual ~WriteDQMSummaryIntoOMDS();
   void run();
   std::vector<Item>  readData(const std::string & file );
   void dropTable(const std::string & table );
   void dropView(const std::string & view );
   void writeData(const std::string & tableToRead );
 
  

 
private:
  std::vector<Item>  m_itemvec;

  std::string m_file;
  std::string m_tableToDrop;  
  std::string m_viewToDrop;  

  std::string m_tableToAppend;

  std::string m_connectionString;
  std::string m_user;
  std::string m_pass;
};

#endif
