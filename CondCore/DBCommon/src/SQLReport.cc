#include "CondCore/DBCommon/interface/SQLReport.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "RelationalAccess/IMonitoringReporter.h"
#include <fstream>

static std::string SQLREPORT_DEFAULT_FILENAME("sqlreport.out");

void cond::SQLReport::reportForConnection(const std::string& connectionString){
  m_report << "-- connection: "<< connectionString << std::endl;
  m_connection.monitoringReporter().reportToOutputStream( connectionString, m_report );
}

bool cond::SQLReport::putOnFile(std::string fileName){
  std::ofstream outFile;
  if(fileName.empty()) fileName.append(SQLREPORT_DEFAULT_FILENAME);
  outFile.open(fileName.c_str());
  if(!outFile.good()){
    std::stringstream msg;
    msg << "Cannot open the output file \""<<fileName<<"\"";
    outFile.close();
    throw cond::Exception(msg.str());
  }
  outFile << m_report.str();
  outFile.flush();
  outFile.close();
  return true;
}
