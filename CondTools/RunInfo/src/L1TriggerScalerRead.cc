#include "CondTools/RunInfo/interface/L1TriggerScalerRead.h"

#include "RelationalAccess/ISession.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/TableDescription.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
//#include "SealBase/TimeInfo.h"

#include "CondCore/CondDB/interface/Time.h"

#include "CoralBase/TimeStamp.h"

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

L1TriggerScalerRead::L1TriggerScalerRead(

    const std::string& connectionString, const std::string& user, const std::string& pass)
    : TestBase(),

      m_connectionString(connectionString),
      m_user(user),
      m_pass(pass) {
  m_tableToDrop = "";
  //  m_tableToRead="";
  //m_columnToRead="";
}

L1TriggerScalerRead::~L1TriggerScalerRead() {}

void L1TriggerScalerRead::run() {}

void L1TriggerScalerRead::dropTable(const std::string& table) {
  m_tableToDrop = table;
  coral::ISession* session = this->connect(m_connectionString, m_user, m_pass);
  session->transaction().start();
  std::cout << "connected succesfully to omds" << std::endl;
  coral::ISchema& schema = session->nominalSchema();
  schema.dropIfExistsTable(m_tableToDrop);
}

std::vector<L1TriggerScaler::Lumi> L1TriggerScalerRead::readData(const int r_number) {
  //  m_tableToRead = table; // to be  cms_runinfo.runsession_parameter
  // m_columnToRead= column;  // to be string_value;

  /* query to execute
select * from runsession_parameter where string_value like '%[%' and name like 'CMS.TR%' order by time;

% gets you the parameters of l1 trg
select * from runsession_parameter where string_value like '%[%' and name like 'CMS.TR%' and runnumber=51384 order by time;

%what is the type?
select * from runsession_parameter_meta where name like 'CMS.TRG:GTLumiSegInfo';

%what is the table for that type?
select * from runsession_type_meta where type like 'rcms.fm.fw.parameter.type.VectorT';

% give me the data... value_id is the parent_id for the element
select * from runsession_vector where runsession_parameter_id = 3392114;

% where is the table for that element type?
select * from runsession_type_meta where type like 'rcms.fm.fw.parameter.type.IntegerT';

% what is the value of the element (of the vector)
select * from runsession_integer where parent_id = 3392118;

*/

  /* let's begin with a query to obtain the number of lumi section and the corresponding string value:
select string_value from runsession_parameter where string_value like '%[51384]' and name='CMS.TRG:GTLumiSegInfo' and runnumber=51384 order by time;   

*/

  std::cout << "entering readData" << std::endl;
  coral::ISession* session = this->connect(m_connectionString, m_user, m_pass);
  session->transaction().start();
  std::cout << "starting session " << std::endl;
  coral::ISchema& schema = session->nominalSchema();
  std::cout << " accessing schema " << std::endl;

  // queryI to access the string_format_value
  coral::IQuery* query0 = schema.tableHandle("RUNSESSION_PARAMETER").newQuery();
  std::cout << "table handling " << std::endl;
  query0->addToOutputList("RUNSESSION_PARAMETER.STRING_VALUE", "STRING_VALUE");

  std::string condition0 =
      "RUNSESSION_PARAMETER.RUNNUMBER=:n_run  AND  RUNSESSION_PARAMETER.NAME='CMS.TRG:GTLumiSegInfo_format'";
  coral::AttributeList conditionData0;
  conditionData0.extend<int>("n_run");
  query0->setCondition(condition0, conditionData0);
  conditionData0[0].data<int>() = r_number;
  coral::ICursor& cursor0 = query0->execute();

  std::vector<L1TriggerScaler::Lumi> l1triggerscaler_array;

  std::string string_format;
  while (cursor0.next() != 0) {
    const coral::AttributeList& row = cursor0.currentRow();
    std::cout << " entering the query == " << std::endl;
    string_format = row["STRING_VALUE"].data<std::string>();

    std::cout << " string value extracted == " << string_format << std::endl;
  }

  // queryI to access the id_value
  coral::IQuery* queryI = schema.tableHandle("RUNSESSION_PARAMETER").newQuery();

  //  queryI->addToTableList( m_tableToRead );
  std::cout << "table handling " << std::endl;
  // implemating the query here.......
  //  queryI->addToOutputList("RUNSESSION_PARAMETER.STRING_VALUE" , "STRING_VALUE");
  queryI->addToOutputList("RUNSESSION_PARAMETER.ID", "ID");
  // to add the starting time of the lumisection
  queryI->addToOutputList("RUNSESSION_PARAMETER.TIME", "TIME");

  //  condition
  std::string condition =
      "RUNSESSION_PARAMETER.RUNNUMBER=:n_run  AND  RUNSESSION_PARAMETER.NAME LIKE 'CMS.TRG:GTLumiSegInfo%' ORDER BY "
      "TIME ";  //AND RUNSESSION_PARAMETER.STRING_VALUE LIKE '%[%'  ORDER BY TIME";
  // controllare...................
  coral::AttributeList conditionData;
  conditionData.extend<int>("n_run");
  queryI->setCondition(condition, conditionData);
  conditionData[0].data<int>() = r_number;
  coral::ICursor& cursorI = queryI->execute();

  std::vector<std::pair<int, long long> >
      v_vid;  // parent_id for value_id corresponding to each memeber of the lumi object, paired with his corresponding index to diceded the type

  if (cursorI.next() == 0) {
    std::cout << " run " << r_number << " not full  " << std::endl;
  }
  while (cursorI.next() != 0) {
    // 1 cicle for each lumi
    L1TriggerScaler::Lumi Itemp;
    // if cursor is null  setting null values
    Itemp.m_runnumber = r_number;
    // initializing the date string ...
    Itemp.m_date = "0";
    Itemp.m_string_format = string_format;
    const coral::AttributeList& row = cursorI.currentRow();
    std::cout << " entering the query == " << std::endl;
    //      Itemp.m_string_value= row["STRING_VALUE"].data<std::string>();

    //  std::cout<< " string value extracted == " << Itemp.m_string_value  << std::endl;
    Itemp.m_lumi_id = row["ID"].data<long long>();
    std::cout << " id value extracted == " << Itemp.m_lumi_id
              << std::endl;  // now we have the id and we  can search and fill the lumi odject
    // retrieving all the value_id one for each member of the lumi scaler

    coral::TimeStamp st = row["TIME"].data<coral::TimeStamp>();
    int year = st.year();
    int month = st.month();
    int day = st.day();
    int hour = st.hour();

    int minute = st.minute();
    int second = st.second();
    long nanosecond = st.nanosecond();
    std::cout << "  start time time extracted == "
              << "-->year " << year << "-- month " << month << "-- day " << day << "-- hour " << hour << "-- minute "
              << minute << "-- second " << second << std::endl;
    boost::gregorian::date dt(year, month, day);
    boost::posix_time::time_duration td(hour, minute, second, nanosecond / 1000);
    boost::posix_time::ptime pt(dt, td);
    Itemp.m_start_time = boost::posix_time::to_iso_extended_string(pt);
    std::cout << " time extracted == " << Itemp.m_start_time << std::endl;

    coral::IQuery* queryII = schema.newQuery();
    queryII->addToOutputList("RUNSESSION_VECTOR.VALUE_ID", "VALUE_ID");
    queryII->addToOutputList("RUNSESSION_VECTOR.VALUE_INDEX", "VALUE_INDEX");

    queryII->addToTableList("RUNSESSION_VECTOR");
    std::string condition2 = "RUNSESSION_VECTOR.PARENT_ID=:n_vid";
    coral::AttributeList conditionData2;
    conditionData2.extend<long>("n_vid");
    queryII->setCondition(condition2, conditionData2);
    conditionData2[0].data<long>() = Itemp.m_lumi_id + 1;
    coral::ICursor& cursorII = queryII->execute();
    while (cursorII.next() != 0) {
      const coral::AttributeList& row = cursorII.currentRow();
      std::cout << " entering the queryII == " << std::endl;
      long long vid_val = row["VALUE_ID"].data<long long>();
      int vid_id = (int)row["VALUE_INDEX"].data<long long>();
      v_vid.push_back(std::make_pair(vid_id, vid_val));
      std::cout << " value_id index extracted == " << v_vid.back().first << std::endl;
      std::cout << " value_id value extracted == " << v_vid.back().second << std::endl;
      // depending from the index, fill the object....

      coral::AttributeList conditionData3;
      conditionData3.extend<int>("n_vid_val");
      conditionData3[0].data<int>() = vid_val;
      switch (vid_id) {
        case 0: {
          std::cout << " caso 0" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_INTEGER.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_INTEGER");
          std::string condition3 = "RUNSESSION_INTEGER.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();

          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            //   std::cout<< " entering the queryIII  " << std::endl;
            Itemp.m_rn = row["VALUE"].data<long long>();
            std::cout << " run extracted == " << Itemp.m_rn << std::endl;
          }
          delete queryIII;
        } break;
        case 1: {
          std::cout << " caso 1" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_INTEGER.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_INTEGER");
          std::string condition3 = "RUNSESSION_INTEGER.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            Itemp.m_lumisegment = row["VALUE"].data<long long>();
            std::cout << " lumisegment extracted == " << Itemp.m_lumisegment << std::endl;
          }
          delete queryIII;
        } break;
        case 2: {
          std::cout << " caso 2" << std::endl;
          /*
	       coral::IQuery* queryIII = schema.newQuery(); 
	       queryIII->addToOutputList("RUNSESSION_STRING.VALUE" , "VALUE");
	       queryIII->addToTableList("RUNSESSION_STRING");
	       std::string  condition3 =  "RUNSESSION_STRING.PARENT_ID=:n_vid_val";
	       queryIII->setCondition( condition3, conditionData3 );
	       coral::ICursor& cursorIII = queryIII->execute();
	       while ( cursorIII.next()!=0 ) {
	       const coral::AttributeList& row = cursorIII.currentRow();
	       std::cout<< " entering the queryIII  " << std::endl;
	       Itemp.m_version = row["VALUE"].data<std::string>();
	       std::cout<< "version extracted == " << Itemp.m_version << std::endl;
	       
	       }
	       delete queryIII;
	     */
        } break;
        case 3: {
          std::cout << " caso 3" << std::endl;
          /*
               coral::IQuery* queryIII = schema.newQuery(); 
	       queryIII->addToOutputList("RUNSESSION_STRING.VALUE" , "VALUE");
	       queryIII->addToTableList("RUNSESSION_STRING");
	       std::string  condition3 =  "RUNSESSION_STRING.PARENT_ID=:n_vid_val";
	       queryIII->setCondition( condition3, conditionData3 );
	       coral::ICursor& cursorIII = queryIII->execute();
	       while ( cursorIII.next()!=0 ) {
	       const coral::AttributeList& row = cursorIII.currentRow();
	       std::cout<< " entering the queryIII  " << std::endl;
	       Itemp.m_context = row["VALUE"].data<std::string>();
	       std::cout<< " context extracted == " << Itemp.m_context << std::endl;
	       
	       }
	       delete queryIII;
	     */
        } break;
        case 4: {
          std::cout << " caso 4" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_DATE.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_DATE");
          std::string condition3 = "RUNSESSION_DATE.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          if (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            coral::TimeStamp ts = row["VALUE"].data<coral::TimeStamp>();
            int year = ts.year();
            int month = ts.month();
            int day = ts.day();
            int hour = ts.hour();

            int minute = ts.minute();
            int second = ts.second();
            long nanosecond = ts.nanosecond();
            std::cout << "  start time time extracted == "
                      << "-->year " << year << "-- month " << month << "-- day " << day << "-- hour " << hour
                      << "-- minute " << minute << "-- second " << second << std::endl;
            boost::gregorian::date dt(year, month, day);
            boost::posix_time::time_duration td(hour, minute, second, nanosecond / 1000);
            boost::posix_time::ptime pt(dt, td);
            Itemp.m_date = boost::posix_time::to_iso_extended_string(pt);
            std::cout << " date extracted == " << Itemp.m_date << std::endl;
          } else {
            std::cout << "date  extracted == " << Itemp.m_date << std::endl;
          }
          delete queryIII;
        } break;
        case 5: {
          std::cout << " caso 5" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_INTEGER.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_INTEGER");
          std::string condition3 = "RUNSESSION_INTEGER.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            int v = (int)row["VALUE"].data<long long>();
            Itemp.m_GTAlgoCounts.push_back(v);
          }

          delete queryIII;

        }

        break;
        case 6: {
          std::cout << " caso 6" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_FLOAT.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_FLOAT");
          std::string condition3 = "RUNSESSION_FLOAT.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            float v = (float)row["VALUE"].data<double>();
            Itemp.m_GTAlgoRates.push_back(v);
          }

          delete queryIII;

        }

        break;

        case 7: {
          std::cout << " caso 7" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_INTEGER.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_INTEGER");
          std::string condition3 = "RUNSESSION_INTEGER.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            int v = (int)row["VALUE"].data<long long>();
            Itemp.m_GTAlgoPrescaling.push_back(v);
          }

          delete queryIII;

        }

        break;
        case 8: {
          std::cout << " caso 8" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_INTEGER.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_INTEGER");
          std::string condition3 = "RUNSESSION_INTEGER.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            int v = (int)row["VALUE"].data<long long>();
            Itemp.m_GTTechCounts.push_back(v);
          }

          delete queryIII;

        }

        break;
        case 9: {
          std::cout << " caso 9" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_FLOAT.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_FLOAT");
          std::string condition3 = "RUNSESSION_FLOAT.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            float v = (float)row["VALUE"].data<double>();
            Itemp.m_GTTechRates.push_back(v);
          }

          delete queryIII;

        }

        break;
        case 10: {
          std::cout << " caso 10" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_INTEGER.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_INTEGER");
          std::string condition3 = "RUNSESSION_INTEGER.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            int v = (int)row["VALUE"].data<long long>();
            Itemp.m_GTTechPrescaling.push_back(v);
          }

          delete queryIII;

        }

        break;
        case 11: {
          std::cout << " caso 11" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_INTEGER.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_INTEGER");
          std::string condition3 = "RUNSESSION_INTEGER.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            int v = (int)row["VALUE"].data<long long>();
            Itemp.m_GTPartition0TriggerCounts.push_back(v);
          }

          delete queryIII;

        }

        break;
        case 12: {
          std::cout << " caso 12" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_FLOAT.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_FLOAT");
          std::string condition3 = "RUNSESSION_FLOAT.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            float v = (float)row["VALUE"].data<double>();
            Itemp.m_GTPartition0TriggerRates.push_back(v);
          }

          delete queryIII;

        }

        break;
        case 13: {
          std::cout << " caso 13" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_INTEGER.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_INTEGER");
          std::string condition3 = "RUNSESSION_INTEGER.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            int v = (int)row["VALUE"].data<long long>();
            Itemp.m_GTPartition0DeadTime.push_back(v);
          }

          delete queryIII;

        }

        break;
        case 14: {
          std::cout << " caso 14" << std::endl;
          coral::IQuery* queryIII = schema.newQuery();
          queryIII->addToOutputList("RUNSESSION_FLOAT.VALUE", "VALUE");
          queryIII->addToTableList("RUNSESSION_FLOAT");
          std::string condition3 = "RUNSESSION_FLOAT.PARENT_ID=:n_vid_val";
          queryIII->setCondition(condition3, conditionData3);
          coral::ICursor& cursorIII = queryIII->execute();
          while (cursorIII.next() != 0) {
            const coral::AttributeList& row = cursorIII.currentRow();
            std::cout << " entering the queryIII  " << std::endl;
            float v = (float)row["VALUE"].data<double>();
            Itemp.m_GTPartition0DeadTimeRatio.push_back(v);
          }

          delete queryIII;

        }

        break;

        default:
          std::cout << "index out of range" << std::endl;
          break;
      }

      // l1triggerscaler_array.push_back(Itemp);
    }
    delete queryII;

    l1triggerscaler_array.push_back(Itemp);
  }

  delete queryI;

  session->transaction().commit();
  delete session;

  // std::cout<<"filling the lumi array to pass to the object" << std::endl;

  return l1triggerscaler_array;
}
