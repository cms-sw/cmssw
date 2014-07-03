#include <Python.h>
#include <exception>
#include "DQM/Integration/interface/WriteDQMSummaryIntoOMDS.h"
#include "CoralBase/Exception.h"
#include <iostream>
#include <exception>
#include <boost/python.hpp>
#include "Python.h"


/*
 *  \script 
 *  
 *  exctracting first dqm summary content from a root file with a python script provided by Yuri Gotra (embedded in the main) 
 *  write data from DQM summary file  into OMDS using Coral
 *  
 * 
 *  \author Michele de Gruttola (degrutto) - INFN Naples / CERN
 *
*/


// argv[1] is the .root file, imput for the python script, 
// argv[2] the .txt output file from the py script and imput to WriteDQMSummaryIntoOMDS::readData 

// i.g.: GetAndWriteDQMSummaryIntoOMDS DQM_R000043434.root tmp.txt
  

int  main(int argc, char *argv[])
 
{
  using namespace boost::python;
  
    Py_Initialize();
    PySys_SetArgv(argc, argv   );
    
    
   object main_module = import("__main__").attr("__dict__");
   
   std::string file = "getDQMSummary.py";
   exec_file(file.c_str(), main_module, main_module);
   Py_Finalize();
 
   // now invocking the module that fist read data from the .txt file(argv[2]) and then build SUMMARYCONTENT table on oracle://cms_omds_lb/CMS_DQM_SUMMARY account 
     
   try {
     WriteDQMSummaryIntoOMDS  app("oracle://cms_omds_lb/CMS_DQM_SUMMARY", "CMS_DQM_SUMMARY", "****");
     // app.dropTable("SUMMARYCONTENT"); app.dropView("SUMMARY");
     app.readData(argv[2]);
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


