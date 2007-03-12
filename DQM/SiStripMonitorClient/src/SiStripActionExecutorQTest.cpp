// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/12/07

#include <iostream>
#include <map>
#include <vector>
#include <sstream>

#include "DQMServices/ClientConfig/interface/QTestConfigurationParser.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutorQTest.h"

SiStripActionExecutorQTest::SiStripActionExecutorQTest( const edm::ParameterSet 
                                                          &roPARAMETER_SET)
  : SiStripActionExecutor(),
    oQTEST_CONFIG_FILE_( 
      roPARAMETER_SET.getUntrackedParameter<std::string>( "oQTestsXMLConfig")) {
}

/** 
* @brief
*   This is an expert version of summary that should be of form:
*
*   --[ Warnings ]------------------------------------
*   <path to module>
*     <QTestName>
*     <QTest Message>
*     <QTestName>
*     <QTest Message>
*     ...
*   <path to module>
*     <QTestName>
*     <QTest Message>
*     <QTestName>
*     <QTest Message>
*     ...
*   ...
*   --------------------------------------------------
*   --[ Errors ]--------------------------------------
*   <path to module>
*     <QTestName>
*     <QTest Message>
*     <QTestName>
*     <QTest Message>
*     ...
*   <path to module>
*     <QTestName>
*     <QTest Message>
*     <QTestName>
*     <QTest Message>
*     ...
*   ...
*   --------------------------------------------------
*
*   It is a table of QTests that produced Warning or Error states.
* 
* @param poMUI 
*   poMui  MonitorUserInterface for which QTests are assigned
* 
* @return 
*   sumamry string
*/
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

  // Construct Summary
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

/** 
* @brief
*   This is a Lite version of summary that should be of form:
*
*   --[ Digi ]------------------------------------
*     Warnings: XX out of NNN modules
*     Errors  : XX out of NNN modules
*   ----------------------------------------------
*   --[ Cluster ]---------------------------------
*     Warnings: XX out of NNN modules
*     Errors  : XX out of NNN modules
*   ----------------------------------------------
*
*   It represents a brief summary table of modules for which at least one
*   QTest produced Warning or Error. Tests are groupped according to Plot
*   Name
*
*   [Note: given method is under development. Groupping should be done via
*          QTests config XML file. See code below for details]
* 
* @param poMUI 
*   poMui  MonitorUserInterface for which QTests are assigned
* 
* @return 
*   sumamry string
*/
std::string 
  SiStripActionExecutorQTest::getQTestSummaryLite( const MonitorUserInterface 
                                                     *poMUI) const {

  std::string oSummary;

  QTestConfigurationParser oParser;
  oParser.getDocument( edm::FileInPath( oQTEST_CONFIG_FILE_).fullPath());
  if( !oParser.parseQTestsConfiguration()) {
    /*
    --[ UNDER DEVELOPMENT ]---------------------------------------------------
    This part is left for future improvement: histograms selection between
    Digis, Clusters, etc. groups should be done via input QTESTS XML config
    [Note: simply add additional attribute to each histogramk, e.g.

      <LINK name="*NumberOfDigis__det__*"> 
        <TestName activate="true">MeanWithinExpected:RMS</TestName> 
      </LINK>

      should look like:

      <LINK name="*NumberOfDigis__det__*" group="Digi"> 
        <TestName activate="true">MeanWithinExpected:RMS</TestName> 
      </LINK>

      where group may take value: Digi, Cluster, etc. - this must (!)
      be strictly specified elsewhere]

    typedef std::map<std::string, std::vector<std::string> > METests;
    METests oMETests = oParser.meToTestsList();
    for( METests::const_iterator oMETESTS_ITER = oMETests.begin();
         oMETESTS_ITER != oMETests.end();
         ++oMETESTS_ITER) {

      std::ostringstream oOut;
      oOut << oMETESTS_ITER->first << std::endl;

      for( std::vector<std::string>::const_iterator oTESTS_ITER = oMETESTS_ITER->second.begin();
           oTESTS_ITER != oMETESTS_ITER->second.end();
           ++oTESTS_ITER) {

      } // End Loop over available tests

    } // End Loop over METests
    --------------------------------------------------------------------------
    */

    // Extract available paths
    std::vector<std::string> oVContents;
    poMUI->getContents( oVContents);

    // Warning/Error Status helper class
    // It is used only in given method - that's the reason it's declaration
    // and implementaion is put into method itself.
    // Class simply tracks for Warnings and Errors. Meanwhile it let user
    // to add only one Warning and Error unless object is not reset.
    // [Hint: Imagine number of histograms with warnings per module. We would
    //        like to calculate number of modules where at least one Warning
    //        appeared. On the other hand it is unknown apriory What group
    //        histogram did raise flag. Thus all histograms should be checked
    //        for different groups]
    class WEStatus {
      public:
        WEStatus(): bWUpdated_( false),
                    bEUpdated_( false),
                    nWarnings_( 0),
                    nErrors_( 0) {}
        ~WEStatus() {}

        unsigned int getWarnings() const { return nWarnings_; }
        unsigned int getErrors  () const { return nErrors_;   }

        void addWarning() { if( !bWUpdated_) { bWUpdated_ = true; 
                                               ++nWarnings_; } }
        void addError  () { if( !bEUpdated_) { bEUpdated_ = true; 
                                               ++nErrors_;   } }
        void reset     () { bWUpdated_ = false; bEUpdated_ = false; }

      private:
        bool bWUpdated_;
        bool bEUpdated_;

        unsigned int nWarnings_;
        unsigned int nErrors_;
    };

    WEStatus oStatusDigi, oStatusCluster;

    // Loop over available paths
    for( std::vector<std::string>::const_iterator oPATH_ITER = oVContents.begin();
         oPATH_ITER != oVContents.end();
         ++oPATH_ITER) {

      // Reset Helper Class
      oStatusDigi.reset();
      oStatusCluster.reset();

      // Extract path
      const std::string oPATH = oPATH_ITER->substr( 0, oPATH_ITER->find( ':'));

      // Check status of QTests in given Path
      switch( poMUI->getStatus( oPATH)) {
        case dqm::qstatus::ERROR:
          // FALL THROUGH
        case dqm::qstatus::WARNING: {
          // WARNING or ERROR fired up in current path. Take closer look at 
          // each plot and determine what group does it belong to: Digi, 
          // Cluster, etc.? Then increase corresponding number

          std::vector<std::string> oVSubContents;
          // Get list of MEs in current path. getMEList will return number of
          // extracted MEs or 0 on any errror or if there are no MEs available
          int nVal = SiStripUtility::getMEList( *oPATH_ITER, oVSubContents);
          if( 0 == nVal) continue;

          // Loop over MEs in current path
          for( std::vector<std::string>::const_iterator oME_ITER = 
                 oVSubContents.begin();
               oME_ITER != oVSubContents.end();
               ++oME_ITER) {

            // Extract ME object
            MonitorElement *poME = poMUI->get( *oME_ITER);
            if( poME) {
              std::cout << "here" << std::endl;
              if( poME->hasError()) {
                // Some QTests failed
                std::vector<QReport *> oVQErrors = poME->getQErrors();
                for( std::vector<QReport *>::const_iterator oWARNINGS_ITER = 
                       oVQErrors.begin();
                     oWARNINGS_ITER != oVQErrors.end();
                     ++oWARNINGS_ITER) {

                  if( std::string::npos != poME->getName().find( "Digi")) {
                    oStatusDigi.addError();
                  } else if( std::string::npos != 
                               poME->getName().find( "Cluster")) {
                    oStatusCluster.addError();
                  }
                } // End loop over Errors
              } // End Check if ME has Errors

              if( poME->hasWarning()) {
                // Some QTests raised warnings
                std::vector<QReport *> oVQWarnings = poME->getQWarnings();
                for( std::vector<QReport *>::const_iterator oWARNINGS_ITER = 
                       oVQWarnings.begin();
                     oWARNINGS_ITER != oVQWarnings.end();
                     ++oWARNINGS_ITER) {

                  if( std::string::npos != poME->getName().find( "Digi")) {
                    oStatusDigi.addWarning();
                  } else if( std::string::npos != 
                               poME->getName().find( "Cluster")) {
                    oStatusCluster.addWarning();
                  }
                } // End loop over Warnings
              } // End Check if ME has Warnings
            }
          } // End loop over MEs in current path

          break;
        }
        default:
          break;
      }
    } // End Loop over available paths

    const unsigned int nTOT_MODULES = oVContents.size();

    // Construct Summary
    std::ostringstream oOut;
    oOut << "--[ Digis ]--------------------------------------" << std::endl
         << "\tWarnings: " << oStatusDigi.getWarnings() 
           << " out of " << nTOT_MODULES
           << std::endl
         << "\tErrors  : " << oStatusDigi.getErrors() 
           << " out of " << nTOT_MODULES << " modules"
           << std::endl
         << "-------------------------------------------------" << std::endl;

    oOut << "--[ Clusters ]-----------------------------------" << std::endl
         << "\tWarnings: " << oStatusCluster.getWarnings() 
           << " out of " << nTOT_MODULES
           << std::endl
         << "\tErrors  : " << oStatusCluster.getErrors() 
           << " out of " << nTOT_MODULES << " modules"
           << std::endl
         << "-------------------------------------------------" << std::endl;

    oSummary = oOut.str();
  } else {
    // Failed to parse QTests configuration file
    oSummary = "Failed to parse QTests configuration file";
  }

  return oSummary;
}
