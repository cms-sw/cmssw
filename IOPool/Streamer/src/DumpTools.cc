/** File contains simple tools to dump Init and Event 
    Messages on screen.
*/

#include "IOPool/Streamer/interface/DumpTools.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <iostream>
#include <iterator>
#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "TBufferFile.h"

using namespace edm;

void dumpInitHeader(const InitMsgView* view)
{
  std::cout
    << "code = " << view->code() << ", "
    << "size = " << view->size() << "\n"
    << "run = " << view->run() << ", "
    << "proto = " << view->protocolVersion() << "\n"
    << "release = " << view->releaseTag() << "\n"
    << "processName = " << view->processName() << "\n";
  if (view->protocolVersion() >= 5) {
    std::cout << "outModuleLabel = " << view->outputModuleLabel() << "\n";
  }
  if (view->protocolVersion() >= 6) {
    std::cout << "outputModuleId=0x" << std::hex << view->outputModuleId()
              << std::dec << std::endl;
  }
  if (view->protocolVersion() >= 8) {
    std::cout << "Checksum for Registry data = " << view->adler32_chksum()
              << " Hostname = " << view->hostName() << std::endl;
  }

  //PSet 16 byte non-printable representation, stored in message.
  uint8 vpset[16];
  view->pset(vpset); 

  //Lets convert it to printable hex form
  std::string pset_str(vpset, vpset+sizeof(vpset));
  pset_str += '\0';
  cms::Digest dig(pset_str);
  cms::MD5Result r1 = dig.digest();
  std::string hexy = r1.toString();
  std::cout << "PSetID= " << hexy << std::endl;

  Strings vhltnames,vhltselections,vl1names;
  view->hltTriggerNames(vhltnames);
  if (view->protocolVersion() >= 5) {
    view->hltTriggerSelections(vhltselections);
  }
  view->l1TriggerNames(vl1names);

  std::cout << "HLT names :- \n ";
  edm::copy_all(vhltnames,std::ostream_iterator<std::string>(std::cout,"\n"));

  if (view->protocolVersion() >= 5) {
    std::cout << "HLT selections :- \n ";
    edm::copy_all(vhltselections,std::ostream_iterator<std::string>(std::cout,"\n"));
  }

  std::cout << "L1 names :- \n ";
  edm::copy_all(vl1names,std::ostream_iterator<std::string>(std::cout,"\n"));
  std::cout << "\n";
  std::cout.flush();

}

void dumpInitView(const InitMsgView* view)
{


  dumpInitHeader(view);
  std::cout << "desc len = " << view->descLength() << "\n";
  //const uint8* pos = view->descData();
  //std::copy(pos,pos+view->descLength(),std::ostream_iterator<uint8>(std::cout,""));
  //std::cout << "\n";
  std::cout.flush();

}

void dumpStartMsg(const InitMsgView* view)
{
  dumpInitHeader(view);
  std::cout.flush();
}

void dumpInitVerbose(const InitMsgView* view)
{
  std::cout << ">>>>> INIT Message Dump (begin) >>>>>" << std::endl;
  dumpInitHeader(view);

  TClass* desc = getTClass(typeid(SendJobHeader));
  TBufferFile xbuf(TBuffer::kRead, view->descLength(),
               (char*)view->descData(), kFALSE);
  std::auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc));

  if (sd.get() == 0) {
    std::cout << "Unable to determine the product registry - "
              << "Registry deserialization error." << std::endl;
  }
  else {
    std::cout << "Branch Descriptions:" << std::endl;
    SendDescs const& descs = sd->descs();
    SendDescs::const_iterator iDesc(descs.begin()), eDesc(descs.end());
    while (iDesc != eDesc) {
      BranchDescription branchDesc = *iDesc;
      branchDesc.init();
      //branchDesc.write(std::cout);
      std::cout << branchDesc.branchName() << std::endl;
      iDesc++;
    }
  }

  std::cout << "<<<<< INIT Message Dump (end) <<<<<" << std::endl;
  std::cout.flush();
}

void dumpInit(uint8* buf)
{
  InitMsgView view(buf);
  dumpInitHeader(&view);
  
  std::cout << "desc len = " << view.descLength() << "\n";
  //const uint8* pos = view.descData();
  //std::copy(pos,pos+view.descLength(),std::ostream_iterator<uint8>(std::cout,""));
  //std::cout << "\n";
  std::cout.flush();
}

void printBits(unsigned char c){

        for (int i = 7; i >= 0; --i) {
            int bit = ((c >> i) & 1);
            std::cout << " "<<bit;
        }
}

void dumpEventHeader(const EventMsgView* eview)
{
  std::cout << "code=" << eview->code() << "\n"
       << "size=" << eview->size() << "\n"
       << "protocolVersion=" << eview->protocolVersion() << "\n"
       << "run=" << eview->run() << "\n"
       << "event=" << eview->event() << "\n"
       << "lumi=" << eview->lumi() << "\n"
       << "origDataSize=" << eview->origDataSize() << "\n"
       << "outModId=0x" << std::hex << eview->outModId() << std::dec << "\n"
       << "adler32 chksum= " << eview->adler32_chksum() << "\n"
       << "host name= " << eview->hostName() << "\n"
       << "event length=" << eview->eventLength() << "\n"
       << "droppedEventsCount=" << eview->droppedEventsCount() << "\n";

  std::vector<bool> l1_out;
  eview->l1TriggerBits(l1_out);

  std::cout << "\nl1 size= " << l1_out.size() << "\n l1 bits=\n";
  edm::copy_all(l1_out,std::ostream_iterator<bool>(std::cout," "));

  std::vector<unsigned char> hlt_out;
  if (eview->hltCount() > 0) {hlt_out.resize(1 + (eview->hltCount()-1)/4);}
  eview->hltTriggerBits(&hlt_out[0]);

  std::cout << "\nhlt Count:" << eview->hltCount();
  std::cout << "\nhlt bits=\n(";
  for(int i=(hlt_out.size()-1); i != -1 ; --i)
    printBits(hlt_out[i]);
  std::cout << ")\n";
  std::cout.flush();
}

void dumpEventView(const EventMsgView* eview)
  {
  dumpEventHeader(eview);
  //const uint8* edata = eview->eventData();
  //std::cout << "\nevent data=\n(";
  //std::copy(&edata[0],&edata[0]+eview->eventLength(),
  //     std::ostream_iterator<char>(std::cout,""));
  //std::cout << ")\n";
  std::cout.flush();

}

void dumpEventIndex(const EventMsgView* eview)
{
  dumpEventHeader(eview);
  std::cout.flush();
}

void dumpEvent(uint8* buf)
{
  EventMsgView eview(buf);

  dumpEventHeader(&eview);

  //const uint8* edata = eview.eventData();
  //std::cout << "\nevent data=\n(";
  //std::copy(&edata[0],&edata[0]+eview.eventLength(),
  //     std::ostream_iterator<char>(std::cout,""));
  //std::cout << ")\n";
  std::cout.flush();

}

void dumpDQMEventHeader(const DQMEventMsgView* dview)
{
  std::cout << "code = " << dview->code() << "\n"
       << "size = " << dview->size() << "\n"
       << "protocolVersion = " << dview->protocolVersion() << "\n"
       << "run = " << dview->runNumber() << "\n"
       << "event = " << dview->eventNumberAtUpdate() << "\n"
       << "lumi = " << dview->lumiSection() << "\n"
       << "update = " << dview->updateNumber() << "\n"
       << "checksum = " << dview->adler32_chksum() << "\n"
       << "host name= " << dview->hostName() << "\n"
       << "compressionFlag = " << dview->compressionFlag() << "\n"
       << "fuProcessId = " << dview->fuProcessId() << "\n"
       << "fuGuid = 0x" << std::hex << dview->fuGuid() << std::dec << "\n"
       << "mergeCount = " << dview->mergeCount() << "\n"
       << "release = " << dview->releaseTag() << "\n"
       << "topFolder = " << dview->topFolderName() << "\n"
       << "event length = " << dview->eventLength() << "\n";
  std::cout.flush();
 }

void dumpDQMEventView(const DQMEventMsgView* dview)
{
  std::cout << "\n>>>>> DQMEvent Message Dump (begin) >>>>>" << std::endl;
  dumpDQMEventHeader(dview);

  boost::shared_ptr< std::vector<std::string> > subFolders =
    dview->subFolderNames();
  for (uint32 idx = 0; idx < subFolders->size(); idx++) {
    std::string name = subFolders->at(idx);
    uint32 count = dview->meCount(name);
    std::cout << "Subfolder " << name << " has " << count
              << " monitor elements." << std::endl;
  }

  std::cout << ">>>>> DQMEvent Message Dump (end) >>>>>" << std::endl;
  std::cout.flush();
}

void dumpFRDEventView(const FRDEventMsgView* fview)
{
  std::cout << "\n>>>>> FRDEvent Message Dump (begin) >>>>>" << std::endl;
  std::cout.flush();

  std::cout << "size = " << fview->size() << "\n"
            << "version = " << fview->version() << "\n"
            << "run = " << fview->run() << "\n"
            << "lumi = " << fview->lumi() << "\n"
            << "event = " << fview->event() << "\n";
  std::cout.flush();

  std::cout << ">>>>> FRDEvent Message Dump (end) >>>>>" << std::endl;
  std::cout.flush();
}
