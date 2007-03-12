// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/12/07

#include <iostream>
#include <vector>
#include <sstream>

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutorQTest.h"

std::string 
  SiStripActionExecutorQTest::getQTestSummary( const MonitorUserInterface 
                                                 *poMUI) const {

  std::string oSummary;

  std::vector<std::string> oVContents;

  // Extract available MonitoringElements. For example:
  //
  // Tree -+- TopBranch1 -+- Branch1 -+- ME1
  //       |              |           +- ME2
  //       |              |           +- ME3
  //       |              |
  //       |              +- Branch2 -+- ...
  //       |
  //       +- TobBranch2 --- Branch1 -+- ME1
  //                                  +- ME2
  //
  // getContents will fill vector vith string values of format:
  //
  //   <path>:ME1,ME2,ME3,...
  //
  // where <path> should be (as in example above):
  //   Tree/TopBranch1/Branch1
  //
  // [Note: Branches, MEs delimeter symbols may change.]
  poMUI->getContents( oVContents);

  std::string oStrWarnings;
  std::string oStrErrors;

  // Loop over available paths
  for( std::vector<std::string>::const_iterator oPATH_ITER = oVContents.begin();
       oPATH_ITER != oVContents.end();
       ++oPATH_ITER) {

    std::vector<std::string> oVSubContents;
    // Get list of MEs in current path. getMEList will return number of
    // extracted MEs or 0 on any errror or if there are no MEs available
    int nVal = SiStripUtility::getMEList( *oPATH_ITER, oVSubContents);
    if( 0 == nVal) continue;

    std::string oStrPathWarnings;
    std::string oStrPathErrors;
    std::string oPath;

    // Loop over MEs in current path
    for( std::vector<std::string>::const_iterator oME_ITER = 
           oVSubContents.begin();
         oME_ITER != oVSubContents.end();
         ++oME_ITER) {

      // Extract ME object
      MonitorElement *poME = poMUI->get( *oME_ITER);
      if( poME) {
        if( !oPath.size()) {
          oPath = poME->getPathname();
        }

        if( poME->hasError()) {
          // Some QTests failed
          std::vector<QReport *> oVQErrors = poME->getQErrors();
          for( std::vector<QReport *>::const_iterator oWARNINGS_ITER = 
                 oVQErrors.begin();
               oWARNINGS_ITER != oVQErrors.end();
               ++oWARNINGS_ITER) {

            std::stringstream oTmpStream;
            oTmpStream << "\t " << ( *oWARNINGS_ITER)->getQRName() << std::endl
                       << '\t' << ( *oWARNINGS_ITER)->getMessage() << std::endl;
            oStrPathErrors += oTmpStream.str();
          } // End loop over Errors
        } // End Check if ME has Errors

        if( poME->hasWarning()) {
          // Some QTests raised warnings
          std::vector<QReport *> oVQWarnings = poME->getQWarnings();
          for( std::vector<QReport *>::const_iterator oWARNINGS_ITER = 
                 oVQWarnings.begin();
               oWARNINGS_ITER != oVQWarnings.end();
               ++oWARNINGS_ITER) {

            std::stringstream oTmpStream;
            oTmpStream << "\t " << ( *oWARNINGS_ITER)->getQRName() << std::endl
                       << '\t' << ( *oWARNINGS_ITER)->getMessage() << std::endl;
            oStrPathWarnings += oTmpStream.str();
          } // End loop over Warnings
        } // End Check if ME has Warnings
      }
    } // End loop over MEs in current path

    if( oStrPathErrors.size()) {
      std::stringstream oSumStream;
      oSumStream << oPath << std::endl
                 << oStrPathErrors;
      oStrErrors += oSumStream.str();
    }

    if( oStrPathWarnings.size()) {
      std::stringstream oSumStream;
      oSumStream << oPath << std::endl
                 << oStrPathWarnings;
      oStrWarnings += oSumStream.str();
    }
  } // End loop over paths

  if( oStrWarnings.size()) {
    std::stringstream oSumStream;
    oSumStream << "--[ Warnings ]----------------------------------------------------------------"
               << std::endl
               << oStrWarnings << std::endl
               << "------------------------------------------------------------------------------"
               << std::endl;
    oSummary += oSumStream.str();
    oStrWarnings.empty();
  }

  if( oStrErrors.size()) {
    std::stringstream oSumStream;
    oSumStream << "--[ Errors ]------------------------------------------------------------------"
               << std::endl
               << oStrErrors << std::endl
               << "------------------------------------------------------------------------------"
               << std::endl;
    oSummary += oSumStream.str();
    oStrErrors.empty();
  }

  return oSummary;
}
