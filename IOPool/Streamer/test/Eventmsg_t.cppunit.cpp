
#include "IOPool/Streamer/interface/Messages.h"
#include "cppunit/extensions/HelperMacros.h"

#include <iostream>

typedef edm::MsgCode MsgCode;
typedef edm::InitMsg InitMsg;
typedef edm::EventMsg EventMsg;

// ----------------------------------------------
class testeventmsg: public CppUnit::TestFixture
{
 CPPUNIT_TEST_SUITE(testeventmsg);
 CPPUNIT_TEST(encodeDecode);
 CPPUNIT_TEST(msgCode);
 CPPUNIT_TEST(initMsg);
 CPPUNIT_TEST(eventMsg);
 CPPUNIT_TEST_SUITE_END();
 public:
  void setUp(){}
  void tearDown(){}
  void encodeDecode();
  void msgCode();
  void initMsg();
  void eventMsg();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventmsg);

struct DecodeStruct
{
  unsigned char junk[3];
  unsigned char data[4];
};

void testeventmsg::encodeDecode()
{
  unsigned int x = 42450876;
  DecodeStruct d;
  edm::encodeInt(x,d.data);
  unsigned int y = edm::decodeInt(d.data);

  CPPUNIT_ASSERT( x==y );
}

void testeventmsg::msgCode()
{
  char buf[51],bufI[51],bufE[51],bufD[51];
  const int sz = sizeof(buf);

  MsgCode code_size(buf,sz);
  MsgCode code_nosize(buf);
  MsgCode code_with_init(bufI,MsgCode::INIT);
  MsgCode code_with_event(bufE,MsgCode::EVENT);
  MsgCode code_with_done(bufD,MsgCode::DONE);

  code_nosize.setCode(MsgCode::DONE);

  CPPUNIT_ASSERT( code_size.codeSize() == 4 );
  CPPUNIT_ASSERT( code_size.totalSize() == sz );
  CPPUNIT_ASSERT( code_with_init.getCode() == MsgCode::INIT );
  CPPUNIT_ASSERT( code_with_event.getCode() == MsgCode::EVENT );
  CPPUNIT_ASSERT( code_with_done.getCode() == MsgCode::DONE );
  CPPUNIT_ASSERT( code_nosize.getCode() == MsgCode::DONE );
}

void testeventmsg::initMsg()
{
  char buf[51],out[51];
  char* in = out;
  int bufsz = (int)sizeof(buf);
  int outsz = (int)sizeof(buf)-7;

  MsgCode code(buf,bufsz,MsgCode::INIT);

  InitMsg from_code(code);
  InitMsg from_new(out,outsz,true);
  InitMsg from_old(in,sizeof(out),false);

  int sz = from_code.codeSize()  + sizeof(InitMsg::InitMsgHeader);

  // -----------
  CPPUNIT_ASSERT( from_code.getDataSize() == bufsz-sz );
  CPPUNIT_ASSERT( from_code.dataSize() == bufsz /*-from_code.codeSize()*/ );
  CPPUNIT_ASSERT( (char*)from_code.data() == buf+sz );
  CPPUNIT_ASSERT( from_code.msgSize() == bufsz );

  from_code.setDataSize(25);
  CPPUNIT_ASSERT(  from_code.getDataSize() == 25);

  // -----------
  CPPUNIT_ASSERT( from_old.getDataSize() == outsz-sz );
  CPPUNIT_ASSERT( from_old.dataSize() == bufsz /*-from_old.codeSize()*/ );
  CPPUNIT_ASSERT( (char*)from_old.data() == out+sz );
  CPPUNIT_ASSERT( from_old.msgSize() == outsz );

  // -----------
  CPPUNIT_ASSERT( from_new.getDataSize() == outsz-sz );
  CPPUNIT_ASSERT( from_new.dataSize() == outsz /*-from_new.codeSize()*/ );
  CPPUNIT_ASSERT( (char*)from_new.data() == out+sz );
  CPPUNIT_ASSERT( from_new.msgSize() == outsz );

}

void testeventmsg::eventMsg()
{
  char buf[51],out[51];
  char* in = out;
  int bufsz = (int)sizeof(buf);
  int outsz = (int)sizeof(buf)-7;

  EventMsg setup(buf,bufsz,13,3,3,7);
  MsgCode mc(buf,bufsz);

  edm::EventNumber_t evt = 12;
  edm::RunNumber_t run = 2;

  EventMsg from_code(mc);
  EventMsg from_new(out,outsz,evt,run,2,5);
  EventMsg from_old(in);

  int sz = from_code.codeSize() + sizeof(EventMsg::EventMsgHeader);
  int sz_out = from_new.codeSize() + sizeof(EventMsg::EventMsgHeader);

  // -----------
  CPPUNIT_ASSERT( setup.msgSize() == setup.msgSize() ); // gets rid of unused variable warning
  CPPUNIT_ASSERT( from_code.getDataSize() == bufsz-sz );
  CPPUNIT_ASSERT( from_code.dataSize() == bufsz-from_code.codeSize() );
  CPPUNIT_ASSERT( (char*)from_code.data() == buf+sz );
  CPPUNIT_ASSERT( from_code.msgSize() == bufsz );
  CPPUNIT_ASSERT( from_code.getWhichSeg() == 3 );
  CPPUNIT_ASSERT( from_code.getTotalSegs() == 7 );
  CPPUNIT_ASSERT( from_code.getEventNumber() == 13 );
  CPPUNIT_ASSERT( from_code.getRunNumber() == 3 );

  from_code.setDataSize(25);
  CPPUNIT_ASSERT(  from_code.getDataSize() == 25);

  from_code.setRunNumber(100);
  CPPUNIT_ASSERT( from_code.getRunNumber() == 100 );
  from_code.setEventNumber(10);
  CPPUNIT_ASSERT( from_code.getEventNumber() == 10 );
  from_code.setWhichSeg(6);
  CPPUNIT_ASSERT( from_code.getWhichSeg() == 6 );
  from_code.setTotalSegs(7);
  CPPUNIT_ASSERT( from_code.getTotalSegs() == 7 );


  // -----------
  CPPUNIT_ASSERT( from_old.getDataSize() == outsz-sz_out );
  CPPUNIT_ASSERT( from_old.dataSize() == outsz-from_old.codeSize() );
  CPPUNIT_ASSERT( (char*)from_old.data() == out+sz_out );
  CPPUNIT_ASSERT( from_old.msgSize() == outsz );
  CPPUNIT_ASSERT( from_old.getWhichSeg() == 2 );
  CPPUNIT_ASSERT( from_old.getTotalSegs() == 5 );
  CPPUNIT_ASSERT( from_old.getEventNumber() == 12 );
  CPPUNIT_ASSERT( from_old.getRunNumber() == 2 );


  // -----------
  CPPUNIT_ASSERT( from_new.getDataSize() == outsz-sz_out );
  CPPUNIT_ASSERT( from_new.dataSize() == outsz-from_old.codeSize() );
  CPPUNIT_ASSERT( (char*)from_new.data() == out+sz_out );
  CPPUNIT_ASSERT( from_new.msgSize() == outsz );
  CPPUNIT_ASSERT( from_new.getWhichSeg() == 2 );
  CPPUNIT_ASSERT( from_new.getTotalSegs() == 5 );
  CPPUNIT_ASSERT( from_new.getEventNumber() == 12 );
  CPPUNIT_ASSERT( from_new.getRunNumber() == 2 );

}
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
