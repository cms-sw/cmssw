// -*- C++ -*-
//
// Package:    DQMServices/Daemon
// Class:      DQMMessageAnalyzer
// 
/**\class DQMMessageAnalyzer

  Description: Example DQM Source with multiple top level folders for
  testing in the Storage manager. This started from the DQMSourceExample.cc
  file in DQMServices/Daemon/test, but modified to include another top level
  folder, to remove the 1 sec wait, and to do the fitting without printout.

  $Id: DQMMessageAnalyzer.cc,v 1.12 2011/02/17 16:06:07 mommsen Exp $

*/


// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <math.h>
#include <cstdio>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include "IOPool/Streamer/interface/StreamDQMSerializer.h"
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"
#include "IOPool/Streamer/interface/StreamDQMOutputFile.h"
#include "IOPool/Streamer/interface/StreamDQMInputFile.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "TClass.h"
#include <TRandom.h> // this is just the random number generator

using std::cout; using std::endl;

//
// class declaration
//

class DQMMessageAnalyzer: public edm::EDAnalyzer {
public:
   explicit DQMMessageAnalyzer( const edm::ParameterSet& );
   ~DQMMessageAnalyzer();
   
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

  virtual void endJob(void);

private:
      // ----------member data ---------------------------

  void findMonitorElements(DQMEvent::TObjectTable&, std::string);

  bool useCompression_;
  int compressionLevel_;
  edm::StreamDQMSerializer serializeWorker_;
  edm::StreamDQMDeserializer deserializeWorker_;

  StreamDQMOutputFile * dqmOutputFile_;

  MonitorElement * h1;
  MonitorElement * h2;
  MonitorElement * h3;
  MonitorElement * h4;
  MonitorElement * h5;
  MonitorElement * h6;
  MonitorElement * h7;
  MonitorElement * h8;
  MonitorElement * h9;
  MonitorElement * i1;
  MonitorElement * f1;
  MonitorElement * s1;
  float XMIN; float XMAX;
  // event counter
  int counter;
  char host_name_[255];
  // back-end interface
  DQMStore * dbe;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DQMMessageAnalyzer::DQMMessageAnalyzer( const edm::ParameterSet& iConfig )
  : counter(0)
{

  std::cout <<" DQMMessageAnalyzer::DQMMessageAnalyzer CTOR" << std::endl;
  useCompression_ = iConfig.getParameter<bool>("useCompression");
  compressionLevel_ = iConfig.getParameter<int>("compressionLevel");

  int got_host = gethostname(host_name_, 255);
  if(got_host != 0) strcpy(host_name_, "noHostNameFoundOrTooLong");

  dqmOutputFile_ = new StreamDQMOutputFile("dqm_events.bin");


  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();
  
  const int NBINS = 5000; XMIN = 0; XMAX = 50;

  // book some histograms here

  // create and cd into new folder
  dbe->setCurrentFolder("C1");
  h1 = dbe->book1D("histo", "Example 1D histogram.", NBINS, XMIN, XMAX);
  h2 = dbe->book2D("histo2", "Example 2 2D histogram.", NBINS, XMIN, XMAX, 
		   NBINS, XMIN, XMAX);
  // create and cd into new folder
  dbe->setCurrentFolder("C1/C2");
  h3 = dbe->book1D("histo3", "Example 3 1D histogram.", NBINS, XMIN, XMAX);
  h4 = dbe->book1D("histo4", "Example 4 1D histogram.", NBINS, XMIN, XMAX);
  h5 = dbe->book1D("histo5", "Example 5 1D histogram.", NBINS, XMIN, XMAX);
  h6 = dbe->book1D("histo6", "Example 6 1D histogram.", NBINS, XMIN, XMAX);
  // create and cd into new folder
  dbe->setCurrentFolder("C1/C3");
  const int NBINS2 = 10;
  h7 = dbe->book1D("histo7", "Example 7 1D histogram.", NBINS2, XMIN, XMAX);
  char temp[1024];
  for(int i = 1; i <= NBINS2; ++i)
    {
      sprintf(temp, " bin no. %d", i);
      h7->setBinLabel(i, temp);
    }
  i1 = dbe->bookInt("int1");
  f1 = dbe->bookFloat("float1");
  s1 = dbe->bookString("s1", "my std::string");

  // create and cd into a new top level folder
  dbe->setCurrentFolder("D1");
  h8 = dbe->book1D("histo8", "Example 8 1D histogram.", NBINS, XMIN, XMAX);
  // create and cd into new sublevel folder
  dbe->setCurrentFolder("D1/D2");
  h9 = dbe->book2D("histo9", "Example 9 2D histogram.", NBINS, XMIN, XMAX, 
		   NBINS, XMIN, XMAX);

  h2->setAxisTitle("Customized x-axis", 1);
  h2->setAxisTitle("Customized y-axis", 2);

  // assign tag to MEs h1, h2 and h7
  const unsigned int detector_id = 17;
  dbe->tag(h1, detector_id);
  dbe->tag(h2, detector_id);
  dbe->tag(h7, detector_id);
  // tag full directory
  dbe->tagContents("C1/C3", detector_id);

  // assign tag to MEs h4 and h6
  const unsigned int detector_id2 = 25;
  const unsigned int detector_id3 = 50;
  dbe->tag(h4, detector_id2);
  dbe->tag(h6, detector_id3);

  // contents of h5 & h6 will be reset at end of monitoring cycle
  h5->setResetMe(true);
  h6->setResetMe(true);
  dbe->showDirStructure();

  std::cout <<" DQMMessageAnalyzer::DQMMessageAnalyzer CTOR Done" << std::endl;

}

DQMMessageAnalyzer::~DQMMessageAnalyzer()
{
   
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   delete dqmOutputFile_; 

}

void DQMMessageAnalyzer::endJob(void)
{

  std::cout << "DQMMessageAnalyzer::endJob" << std::endl;
  //dbe->save("test.root");  

  //Let us read the freshly writen dqm event file and display dqm events from there in
  StreamDQMInputFile dqm_file("dqm_events.bin");
  while(dqm_file.next()) {

      std::cout << "----------------DQM Event -------------" << std::endl;

      //Lets print it
      const DQMEventMsgView* dqmEventView = dqm_file.currentRecord();

      std::cout << "  DQM Message data:" << std::endl;
      std::cout << "    protocol version = "
                << dqmEventView->protocolVersion() << std::endl;
      std::cout << "    header size = "
                << dqmEventView->headerSize() << std::endl;
      std::cout << "    run number = "
                << dqmEventView->runNumber() << std::endl;
      std::cout << "    event number = "
                << dqmEventView->eventNumberAtUpdate() << std::endl;
      std::cout << "    lumi section = "
                << dqmEventView->lumiSection() << std::endl;
      std::cout << "    update number = "
                << dqmEventView->updateNumber() << std::endl;
      std::cout << "    checksum = "
                << dqmEventView->adler32_chksum() << std::endl;
      std::cout << "    host name = "
                << dqmEventView->hostName() << std::endl;
      std::cout << "    compression flag = "
                << dqmEventView->compressionFlag() << std::endl;
      std::cout << "    merge count = "
                << dqmEventView->mergeCount() << std::endl;
      std::cout << "    release tag = "
                << dqmEventView->releaseTag() << std::endl;
      std::cout << "    top folder name = "
                << dqmEventView->topFolderName() << std::endl;
      std::cout << "    sub folder count = "
                << dqmEventView->subFolderCount() << std::endl;
      std::cout << "    time stamp = "
                << dqmEventView->timeStamp().value() << std::endl;
      std::auto_ptr<DQMEvent::TObjectTable> toTablePtr =
        deserializeWorker_.deserializeDQMEvent(*dqmEventView);
      DQMEvent::TObjectTable::const_iterator toIter;
      for (toIter = toTablePtr->begin();
           toIter != toTablePtr->end(); toIter++) {
        std::string subFolderName = toIter->first;
        std::cout << "  folder = " << subFolderName << std::endl;
        std::vector<TObject *> toList = toIter->second;
        for (int tdx = 0; tdx < (int) toList.size(); tdx++) {
          TObject *toPtr = toList[tdx];
          std::string cls = toPtr->IsA()->GetName();
          std::string nm = toPtr->GetName();
          std::cout << "    TObject class = " << cls
                    << ", name = " << nm << std::endl;
        }
      }
    }
}

//
// member functions
//

void DQMMessageAnalyzer::findMonitorElements(DQMEvent::TObjectTable &toTable,
                                           std::string folderPath)
{
  if (dbe == NULL) {return;}
  
  // fetch the monitor elements in the specified directory
  std::vector<MonitorElement *> localMEList = dbe->getContents(folderPath);
  //MonitorElementRootFolder* folderPtr = dbe->getDirectory(folderPath);
  
  // add the MEs that should be updated to the table
  std::vector<TObject *> updateTOList;
  for (int idx = 0; idx < (int) localMEList.size(); idx++) {
    MonitorElement *mePtr = localMEList[idx];
    if (mePtr->wasUpdated()) {
      updateTOList.push_back(mePtr->getRootObject());
    }
  }
  if (updateTOList.size() > 0) {
    toTable[folderPath] = updateTOList;
  }

  // find the subdirectories in this folder
  // (checking if the directory exists is probably overkill,
  // but we really don't want to create new folders using
  // setCurrentFolder())
  if (dbe->dirExists(folderPath)) {
    dbe->setCurrentFolder(folderPath);
    std::vector<std::string> subDirList = dbe->getSubdirs();

    // loop over the subdirectories, find the MEs in each one
    std::vector<std::string>::const_iterator dirIter;
    for (dirIter = subDirList.begin(); dirIter != subDirList.end(); dirIter++) {
      std::string subDirPath = folderPath + "/" + (*dirIter);
      findMonitorElements(toTable, subDirPath);
    }
  }
}

// ------------ method called to produce the data  ------------
void DQMMessageAnalyzer::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup )
{   

  std::cout << "DQMMessageAnalyzer::analyze" << std::endl;


  // Filling the histogram with random data
  i1->Fill(4);
  f1->Fill(-3.14);

  srand( 0 );

  if(counter%1000 == 0)
    std::cout << " # of cycles = " << counter << std::endl;

  for(int i = 0; i != 20; ++i )
    {
      float x = gRandom->Uniform(XMAX);
      h1->Fill(x,1./log(x+1));
      h3->Fill(x, 1);
      h4->Fill(gRandom->Gaus(30, 3), 1.0);
      h5->Fill(gRandom->Poisson(15), 0.5);
      h6->Fill(gRandom->Gaus(25, 15), 1.0);
      h7->Fill(gRandom->Gaus(25, 8), 1.0);
      h8->Fill(gRandom->Gaus(25, 7), 1.0);
    }

  // fit h4 to gaussian
  h4->getTH1F()->Fit("gaus","Q");

  for ( int i = 0; i != 10; ++i )
    {
      float x = gRandom->Gaus(15, 7);
      float y = gRandom->Gaus(20, 5);
      h2->Fill(x,y);
      h9->Fill(x,y);
    }
  //      (*(*i1))++;
  //usleep(1000000);

  ++counter;

  std::cout << "Counter: " << counter << std::endl;


  if (dbe == NULL) {
    dbe = edm::Service<DQMStore>().operator->();
  }

  if (dbe == NULL) {
    throw cms::Exception("postEventProcessing", "FUShmDQMOutputService")
      << "Unable to lookup the DQMStore service!\n";
  }

  //Lets convert histograms into TObjects and throw them in DQMEvent::TObjectTable


  // determine the top level folders (these will be used for grouping
  // monitor elements into DQM Events)
  std::vector<std::string> topLevelFolderList;
  //std::cout << "### SenderService, pwd = " << dbe->pwd() << std::endl;
  dbe->cd();
  //std::cout << "### SenderService, pwd = " << dbe->pwd() << std::endl;
  topLevelFolderList = dbe->getSubdirs();

  // find the monitor elements under each top level folder (including
  // subdirectories)
  std::map< std::string, DQMEvent::TObjectTable > toMap;
  std::vector<std::string>::const_iterator dirIter;
  for (dirIter = topLevelFolderList.begin();
       dirIter != topLevelFolderList.end();
       dirIter++) {
    std::string dirName = *dirIter;
    DQMEvent::TObjectTable toTable;
    std::cout <<"dirName= "<<dirName<<std::endl; 
    // find the MEs
    findMonitorElements(toTable, dirName);

    // store the list in the map
    toMap[dirName] = toTable;
  }

  //Buffer where DQM Evenet Msg will be built 
  typedef std::vector<uint8> Buffer;
  //Lets go with a fix size buffer for now

  // create a DQMEvent message for each top-level folder
  // and write each to the shared memory
  for (dirIter = topLevelFolderList.begin();
       dirIter != topLevelFolderList.end();
       dirIter++) {
    std::string dirName = *dirIter;
    DQMEvent::TObjectTable toTable = toMap[dirName];
    if (toTable.size() == 0) {continue;}


    Buffer buf(1024);
  
    uint32_t lumiSection = 789;
    uint32_t updateNumber = 111;

    // serialize the monitor element data
    serializeWorker_.serializeDQMEvent(toTable, useCompression_,
                                       compressionLevel_);

    // resize the message buffer, if needed 
    unsigned int srcSize = serializeWorker_.currentSpaceUsed();
    unsigned int newSize = srcSize + 50000;  // allow for header
    if (buf.size() < newSize) buf.resize(newSize);


    // create the message
    DQMEventMsgBuilder dqmMsgBuilder(&buf[0], buf.size(),
                                        iEvent.id().run(), iEvent.id().event(),
                                        iEvent.time(),
                                        lumiSection, updateNumber,
                                        (uint32_t)serializeWorker_.adler32_chksum(),
                                        host_name_,
                                        edm::getReleaseVersion(), dirName,
                                        toTable);

    // copy the serialized data into the message
    unsigned char* src = serializeWorker_.bufferPointer();
    std::copy(src,src + srcSize, dqmMsgBuilder.eventAddress());
    dqmMsgBuilder.setEventLength(srcSize);
    if (useCompression_) {
      dqmMsgBuilder.setCompressionFlag(serializeWorker_.currentEventSize());
    }


   //Let us write this message into a File as well
   dqmOutputFile_->write(dqmMsgBuilder);


   /****
   //Lets print it
   DQMEventMsgView dqmEventView(&buf[0]);
      std::cout << "  DQM Message data:" << std::endl;
      std::cout << "    protocol version = "
                << dqmEventView.protocolVersion() << std::endl;
      std::cout << "    header size = "
                << dqmEventView.headerSize() << std::endl;
      std::cout << "    run number = "
                << dqmEventView.runNumber() << std::endl;
      std::cout << "    event number = "
                << dqmEventView.eventNumberAtUpdate() << std::endl;
      std::cout << "    lumi section = "
                << dqmEventView.lumiSection() << std::endl;
      std::cout << "    update number = "
                << dqmEventView.updateNumber() << std::endl;
      std::cout << "    checksum = "
                << dqmEventView->adler32_chksum() << std::endl;
      std::cout << "    host name = "
                << dqmEventView->hostName() << std::endl;
      std::cout << "    compression flag = "
                << dqmEventView.compressionFlag() << std::endl;
      std::cout << "    reserved word = "
                << dqmEventView.reserved() << std::endl;
      std::cout << "    release tag = "
                << dqmEventView.releaseTag() << std::endl;
      std::cout << "    top folder name = "
                << dqmEventView.topFolderName() << std::endl;
      std::cout << "    sub folder count = "
                << dqmEventView.subFolderCount() << std::endl;
      std::cout << "    time stamp = "
                << dqmEventView.timeStamp().value() << std::endl;

      std::auto_ptr<DQMEvent::TObjectTable> toTablePtr =
        deserializeWorker_.deserializeDQMEvent(dqmEventView);
      DQMEvent::TObjectTable::const_iterator toIter;
      for (toIter = toTablePtr->begin();
           toIter != toTablePtr->end(); toIter++) {
        std::string subFolderName = toIter->first;
        std::cout << "  folder = " << subFolderName << std::endl;
        std::vector<TObject *> toList = toIter->second;
        for (int tdx = 0; tdx < (int) toList.size(); tdx++) {
          TObject *toPtr = toList[tdx];
          std::string cls = toPtr->IsA()->GetName();
          std::string nm = toPtr->GetName();
          std::cout << "    TObject class = " << cls
                    << ", name = " << nm << std::endl;
        }
      }   ***/

  }//for loop


}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMMessageAnalyzer);




/**
  // create a DQMEvent message for each top-level folder
  // and write each to the shared memory
  for (dirIter = topLevelFolderList.begin();
       dirIter != topLevelFolderList.end();
       dirIter++) {
    std::string dirName = *dirIter;
    DQMEvent::TObjectTable toTable = toMap[dirName];
    if (toTable.size() == 0) {continue;}

    // serialize the monitor element data
    serializeWorker_.serializeDQMEvent(toTable, useCompression_,
                                       compressionLevel_);

    // resize the message buffer, if needed 
    unsigned int srcSize = serializeWorker_.currentSpaceUsed();
    unsigned int newSize = srcSize + 50000;  // allow for header
    if (messageBuffer_.size() < newSize) messageBuffer_.resize(newSize);

    // create the message
    DQMEventMsgBuilder dqmMsgBuilder(&messageBuffer_[0], messageBuffer_.size(),
                                     event.id().run(), event.id().event(),
                                     event.time(),
                                     //event.time().value(),
                                     lumiSectionTag, updateNumber,
                                     (uint32_t)serializeWorker_.adler32_chksum(),
                                     host_name_,
                                     edm::getReleaseVersion(), dirName,
                                     toTable);

    // copy the serialized data into the message
    unsigned char* src = serializeWorker_.bufferPointer();
    std::copy(src,src + srcSize, dqmMsgBuilder.eventAddress());
    dqmMsgBuilder.setEventLength(srcSize);
    if (useCompression_) {
      dqmMsgBuilder.setCompressionFlag(serializeWorker_.currentEventSize());
    }

    // send the message
    writeShmDQMData(dqmMsgBuilder);

**/

