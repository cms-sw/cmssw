#include <iostream>
#include <exception>
#include "DQM/Integration/interface/WriteDQMSummaryIntoOMDS.h"
#include "CoralBase/Exception.h"

/*
 *  \script 
 *  
 *  write data from DQM summery file into OMDS (relational db) using Coral, after having read data from a file (output of python script) 
 *  
 *  further feature provided: dropping table and reading
 *
 *  \author Michele de Gruttola (degrutto) - INFN Naples  (June-12-2008)
 *
*/

int main( int, char** )
{
  try {
   WriteDQMSummaryIntoOMDS  app("oracle://cms_omds_lb/CMS_DQM_SUMMARY", "CMS_DQM_SUMMARY", "****");
   // app.dropTable("SUMMARYCONTENT"); app.dropView("SUMMARY");

        app.readData("tmp.txt");
        app.writeData("SUMMARYCONTENT");
    
  }

  catch ( coral::Exception& e ) {
    std::cerr << "CORAL Exception : " << e.what() << std::endl;
    return 1;
  }
  catch ( std::exception& e ) {
    std::cerr << "C++ Exception : " << e.what() << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "Unknown exception ..." << std::endl;
    return 1;
  }
  std::cout << "[OVAL] Success" << std::endl;
  return 0;
}
