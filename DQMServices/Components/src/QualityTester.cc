/*
 * \file QualityTester.cc
 *
 * $Date: 2010/07/20 02:58:28 $
 * $Revision: 1.19 $
 * \author M. Zanetti - CERN PH
 *
 */

#include "DQMServices/Components/interface/QualityTester.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

QualityTester::QualityTester(const ParameterSet& ps)
{
  prescaleFactor          = ps.getUntrackedParameter<int>("prescaleFactor", 1);
  getQualityTestsFromFile = ps.getUntrackedParameter<bool>("getQualityTestsFromFile", true);
  Label                   = ps.getUntrackedParameter<string>("label","");
  reportThreshold = ps.getUntrackedParameter<string>("reportThreshold", "");
  testInEventloop = ps.getUntrackedParameter<bool>("testInEventloop",false);
  qtestOnEndRun   = ps.getUntrackedParameter<bool>("qtestOnEndRun",true);
  qtestOnEndJob   = ps.getUntrackedParameter<bool>("qtestOnEndJob",false);
  qtestOnEndLumi  = ps.getUntrackedParameter<bool>("qtestOnEndLumi",false);
  verboseQT       = ps.getUntrackedParameter<bool>("verboseQT", true);

  bei = &*edm::Service<DQMStore>();

  qtHandler=new QTestHandle;

  // if you use this module, it's non-sense not to provide the QualityTests.xml
  if (getQualityTestsFromFile) {
    edm::FileInPath qtlist = ps.getUntrackedParameter<edm::FileInPath>("qtList");
    qtHandler->configureTests(FileInPath(qtlist).fullPath(), bei);
  }


  nEvents = 0;

}

void QualityTester::beginRun(const edm::Run& run , const edm::EventSetup& iSetup){

  // if getQualityTestsFromFile is False, it means that the end-user wants them from the Database
  if (!getQualityTestsFromFile) {
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DQMXMLFileRcd"));
    if(recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      throw cms::Exception ("Record not found") << "Record \"DQMXMLFileRcd" 
						<< "\" does not exist!" << std::endl;
    }
//     std::cout << "Reading XML from Database" << std::endl ;
    edm::ESHandle<FileBlob> xmlfile;
    iSetup.get<DQMXMLFileRcd>().get(Label,xmlfile);
    boost::scoped_ptr<std::vector<unsigned char> > vc( (*xmlfile).getUncompressedBlob() );
    std::string xmlstr="";
    for(std::vector<unsigned char>::iterator it=vc->begin();it!=vc->end();it++){
      xmlstr += *it;
    }

    qtHandler->configureTests(xmlstr,bei,1);

  }
}

QualityTester::~QualityTester()
{
  delete qtHandler;
}

void QualityTester::analyze(const edm::Event& e, const edm::EventSetup& c) 
{
  if (testInEventloop) {
    nEvents++;
    if (getQualityTestsFromFile
	&& prescaleFactor > 0 
	&& nEvents % prescaleFactor == 0)  {
      performTests();
    }
  }
}

void QualityTester::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context)
{
  if (!testInEventloop&&qtestOnEndLumi) {
    if (getQualityTestsFromFile
        && prescaleFactor > 0
        && lumiSeg.id().luminosityBlock() % prescaleFactor == 0) {
      performTests();
    }
  }
}

void QualityTester::endRun(const Run& r, const EventSetup& context){
  if (qtestOnEndRun) performTests();
}

void QualityTester::endJob(){
  if (qtestOnEndJob) performTests();
}

void QualityTester::performTests(void)
{
    // done here because new ME can appear while processing data
    qtHandler->attachTests(bei,verboseQT);

    edm::LogVerbatim ("QualityTester") << "Running the Quality Test";

    bei->runQTests();

    if (reportThreshold.size() != 0)
    {
      std::map< std::string, std::vector<std::string> > theAlarms
	= qtHandler->checkDetailedQTStatus(bei);

      for (std::map<std::string,std::vector<std::string> >::iterator itr = theAlarms.begin();
	   itr != theAlarms.end(); ++itr)
      {
        const std::string &alarmType = itr->first;
        const std::vector<std::string> &msgs = itr->second;
        if ((reportThreshold == "black")
	    || (reportThreshold == "orange" && (alarmType == "orange" || alarmType == "red"))
	    || (reportThreshold == "red" && alarmType == "red"))
	{
          std::cout << std::endl;
          std::cout << "Error Type: " << alarmType << std::endl;
          for (std::vector<std::string>::const_iterator msg = msgs.begin();
	       msg != msgs.end(); ++msg)
            std::cout << *msg << std::endl;
        }
      }
      std::cout << std::endl;
    }
}
