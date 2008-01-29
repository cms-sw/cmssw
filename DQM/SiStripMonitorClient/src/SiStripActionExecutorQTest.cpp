// Author : Samvel Khalatian ( samvel at fnal dot gov)
// Created: 03/12/07

#include <iostream>
#include <vector>
#include <sstream>

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutorQTest.h"

SiStripActionExecutorQTest::SiStripActionExecutorQTest()
  : SiStripActionExecutor(),
    bSummaryTagsNotRead_( true) {}

std::string SiStripActionExecutorQTest::getQTestSummary(
  const DaqMonitorBEInterface *poMUI) {

  std::ostringstream oSStream;

  getQTestSummary_( oSStream, poMUI, dqm::XMLTag::STRING);
  
  return oSStream.str();
}

std::string SiStripActionExecutorQTest::getQTestSummaryLite(
  const DaqMonitorBEInterface *poMUI) {

  std::ostringstream oSStream;

  getQTestSummary_( oSStream, poMUI, dqm::XMLTag::STRING_LITE);
  
  return oSStream.str();
}

std::string SiStripActionExecutorQTest::getQTestSummaryXML( 
  const DaqMonitorBEInterface *poMUI) {

  std::ostringstream oSStream;
  oSStream << "<?xml version='1.0' encoding='UTF-8'>" << std::endl;

  getQTestSummary_( oSStream, poMUI, dqm::XMLTag::XML);
  
  return oSStream.str();
}

std::string SiStripActionExecutorQTest::getQTestSummaryXMLLite( 
  const DaqMonitorBEInterface *poBEI) {

  std::ostringstream oSStream;

  oSStream << "<?xml version='1.0' encoding='UTF-8'>" << std::endl;

  getQTestSummary_( oSStream, poBEI, dqm::XMLTag::XML_LITE);
  
  return oSStream.str();
}

std::ostream &SiStripActionExecutorQTest::getQTestSummary_(
  std::ostream                &roOut,
  const DaqMonitorBEInterface  *poBEI,
  const dqm::XMLTag::TAG_MODE &reMODE) {

  if( bSummaryTagsNotRead_) {
    createQTestSummary_( poBEI);
    bSummaryTagsNotRead_ = false;
  }

  poXMLTagWarnings_->setMode( reMODE);
  poXMLTagErrors_->setMode( reMODE);

  roOut << *poXMLTagWarnings_ << std::endl;
  roOut << *poXMLTagErrors_   << std::endl;

  return  roOut;
}

void SiStripActionExecutorQTest::createQTestSummary_(
  const DaqMonitorBEInterface *poBEI) {

  typedef std::vector<std::string> VContents;
  typedef std::vector<QReport *>   VReports;

  VContents oVContents;
  poBEI->getContents( oVContents);

  poXMLTagWarnings_ = std::auto_ptr<dqm::XMLTagWarnings>( 
    new dqm::XMLTagWarnings());
  poXMLTagErrors_ = std::auto_ptr<dqm::XMLTagErrors>( 
    new dqm::XMLTagErrors());

  dqm::XMLTagDigis    *poXMLTagDigisWarnings =
    poXMLTagWarnings_->createChild<dqm::XMLTagDigis>();

  dqm::XMLTagClusters *poXMLTagClustersWarnings =
    poXMLTagWarnings_->createChild<dqm::XMLTagClusters>();

  dqm::XMLTagDigis    *poXMLTagDigisErrors =
    poXMLTagErrors_->createChild<dqm::XMLTagDigis>();

  dqm::XMLTagClusters *poXMLTagClustersErrors =
    poXMLTagErrors_->createChild<dqm::XMLTagClusters>();

  for( VContents::const_iterator oPATH_ITER = oVContents.begin();
       oPATH_ITER != oVContents.end();
       ++oPATH_ITER) {

    VContents oVSubContents;
    if( !SiStripUtility::getMEList( *oPATH_ITER, oVSubContents)) {
      continue;
    }

    const std::string oPATH = oPATH_ITER->substr( 0, oPATH_ITER->find( ':'));

    poXMLTagDigisWarnings->unlock();
    poXMLTagClustersWarnings->unlock();
    poXMLTagDigisErrors->unlock();
    poXMLTagClustersErrors->unlock();

    for( VContents::const_iterator oME_ITER = oVSubContents.begin();
         oME_ITER != oVSubContents.end();
         ++oME_ITER) {

      if( MonitorElement *poME = poBEI->get( *oME_ITER)) {
        dqm::XMLTagModule *poXMLTagModuleWarnings = 0;
        dqm::XMLTagModule *poXMLTagModuleErrors   = 0;

        if( poME->hasError()) {
          VReports oVErrors = poME->getQErrors();
          for( VReports::const_iterator oERROR_ITER = oVErrors.begin();
               oERROR_ITER != oVErrors.end();
               ++oERROR_ITER) {

            if( !poXMLTagModuleErrors) {
              poXMLTagModuleErrors = 
                poXMLTagErrors_->createChild<dqm::XMLTagModule>();

              poXMLTagModuleErrors->createChild<dqm::XMLTagPath>()->setPath( 
                oPATH);
            }

            if( std::string::npos != poME->getName().find( "Digi")) {
              ++( *poXMLTagDigisErrors);
            } else if( std::string::npos != poME->getName().find( "Cluster")) {
              ++( *poXMLTagClustersErrors);
            }

            dqm::XMLTagQTest *poXMLTagQTest =
              poXMLTagModuleErrors->createChild<dqm::XMLTagQTest>();

            poXMLTagQTest->setName   ( ( *oERROR_ITER)->getQRName());
            poXMLTagQTest->setMessage( ( *oERROR_ITER)->getMessage());
          }
        } // End Check if ME has Errors

        if( poME->hasWarning()) { 
          VReports oVWarnings = poME->getQWarnings();
          for( VReports::const_iterator oWARNING_ITER = oVWarnings.begin();
               oWARNING_ITER != oVWarnings.end();
               ++oWARNING_ITER) {

            if( !poXMLTagModuleWarnings) {
              poXMLTagModuleWarnings = 
                poXMLTagWarnings_->createChild<dqm::XMLTagModule>();

              poXMLTagModuleWarnings->createChild<dqm::XMLTagPath>()->setPath( 
                oPATH);
            }

            if( std::string::npos != poME->getName().find( "Digi")) {
              ++( *poXMLTagDigisWarnings);
            } else if( std::string::npos != poME->getName().find( "Cluster")) {
              ++( *poXMLTagClustersWarnings);
            }

            dqm::XMLTagQTest *poXMLTagQTest =
              poXMLTagModuleWarnings->createChild<dqm::XMLTagQTest>();

            poXMLTagQTest->setName   ( ( *oWARNING_ITER)->getQRName());
            poXMLTagQTest->setMessage( ( *oWARNING_ITER)->getMessage());
          }
        } // End Check if ME has Warnings
      } // End Exract ME object
    } // End Loop over MEs in current Path
  } // End Loop over  available Paths

  poXMLTagDigisWarnings->setTotModules( oVContents.size());
  poXMLTagClustersWarnings->setTotModules( oVContents.size());

  poXMLTagDigisErrors->setTotModules( oVContents.size());
  poXMLTagClustersErrors->setTotModules( oVContents.size());
}
