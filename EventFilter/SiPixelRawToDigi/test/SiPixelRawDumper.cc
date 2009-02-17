/** \class SiPixelRawDumper_H
 *  Plug-in module that dump raw data file 
 *  for pixel subdetector
 *  Added class to interpret the data d.k. 30/10/08
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include <iostream>
using namespace std;

const bool errorsOnly = false;

// Include the helper decoding class
/////////////////////////////////////////////////////////////////////////////
class MyDecode {
public:
  MyDecode() {}
  ~MyDecode() {}
  static int error(int error);
  static int data(int error);
  static int header(unsigned long long word64);
  static int trailer(unsigned long long word64);
private:
};
/////////////////////////////////////////////////////////////////////////////
int MyDecode::header(unsigned long long word64) {
  int fed_id=(word64>>8)&0xfff;
  int event_id=(word64>>32)&0xffffff;
  unsigned int bx_id=(word64>>20)&0xfff;
  cout<<" Header "<<" for FED "
      <<fed_id<<" event "<<event_id<<" bx "<<bx_id<<endl;
  return 0;
}
//
int MyDecode::trailer(unsigned long long word64) {
  int slinkLength = int( (word64>>32) & 0xffffff );
  int crc         = int( (word64&0xffff0000)>>16 );
  int tts         = int( (word64&0xf0)>>4);
  int slinkError  = int( (word64&0xf00)>>8);
  cout<<" Trailer "<<" len "<<slinkLength
      <<" tts "<<tts<<" error "<<slinkError<<" crc "<<hex<<crc<<dec<<endl;
  return 0;
}
//
// Decode error FIFO
// Works for both, the error FIFO and the SLink error words. d.k. 25/04/07
int MyDecode::error(int word) {
  int status = -1;
  const unsigned int  errorMask      = 0x3e00000;
  const unsigned int  dummyMask      = 0x03600000;
  const unsigned int  gapMask        = 0x03400000;
  const unsigned int  timeOut        = 0x3a00000;
  const unsigned int  eventNumError  = 0x3e00000;
  const unsigned int  trailError     = 0x3c00000;
  const unsigned int  fifoError      = 0x3800000;

//  const unsigned int  timeOutChannelMask = 0x1f;  // channel mask for timeouts
  //const unsigned int  eventNumMask = 0x1fe000; // event number mask
  const unsigned int  channelMask = 0xfc000000; // channel num mask
  const unsigned int  tbmEventMask = 0xff;    // tbm event num mask
  const unsigned int  overflowMask = 0x100;   // data overflow
  const unsigned int  tbmStatusMask = 0xff;   //TBM trailer info
  const unsigned int  BlkNumMask = 0x700;   //pointer to error fifo #
  const unsigned int  FsmErrMask = 0x600;   //pointer to FSM errors
  const unsigned int  RocErrMask = 0x800;   //pointer to #Roc errors
  const unsigned int  ChnFifMask = 0x1f;   //channel mask for fifo error
  const unsigned int  Fif2NFMask = 0x40;   //mask for fifo2 NF
  const unsigned int  TrigNFMask = 0x80;   //mask for trigger fifo NF

 const int offsets[8] = {0,4,9,13,18,22,27,31};

 //cout<<"error word "<<hex<<word<<dec<<endl;

  if( (word&errorMask) == dummyMask ) { // DUMMY WORD
    //cout<<" Dummy word";
    return 0;
  } else if( (word&errorMask) == gapMask ) { // GAP WORD
    //cout<<" Gap word";
    return 0;
  } else if( (word&errorMask)==timeOut ) { // TIMEOUT
     // More than 1 channel within a group can have a timeout error
     unsigned int index = (word & 0x1F);  // index within a group of 4/5
     unsigned int chip = (word& BlkNumMask)>>8;
     int offset = offsets[chip];
     cout<<"Timeout Error- channels: ";
     for(int i=0;i<5;i++) {
       if( (index & 0x1) != 0) {
         int chan = offset + i + 1;
         cout<<chan<<" ";
       }
       index = index >> 1;
     }
     cout<<endl;
     //end of timeout  chip and channel decoding

   } else if( (word&errorMask) == eventNumError ) { // EVENT NUMBER ERROR
     unsigned int channel =  (word & channelMask) >>26;
     unsigned int tbm_event   =  (word & tbmEventMask);

     cout<<"Event Number Error- channel: "<<channel<<" tbm event nr. "
         <<tbm_event<<endl;

   } else if( ((word&errorMask) == trailError)) {
    unsigned int channel =  (word & channelMask) >>26;
    unsigned int tbm_status   =  (word & tbmStatusMask);
    if(word & RocErrMask)
      cout<<"Number of Rocs Error- "<<"channel: "<<channel<<" "<<endl;
    if(word & FsmErrMask)
      cout<<"Finite State Machine Error- "<<"channel: "<<channel
          <<" Error status:0x"<<hex<< ((word & FsmErrMask)>>9)<<dec<<" "<<endl;
    if(word & overflowMask)
      cout<<"Overflow Error- "<<"channel: "<<channel<<" "<<endl;
    //if(!((word & RocErrMask)|(word & FsmErrMask)|(word & overflowMask)))
    if(tbm_status!=0)
      cout<<"Trailer Error- "<<"channel: "<<channel<<" TBM status:0x"
          <<hex<<tbm_status<<dec<<" "<<endl;

  } else if((word&errorMask)==fifoError) {
    if(word & Fif2NFMask) cout<<"A fifo 2 is Nearly full- ";
    if(word & TrigNFMask) cout<<"The trigger fifo is nearly Full - ";
    if(word & ChnFifMask) cout<<"fifo-1 is nearly full for channel"<<(word & ChnFifMask);
    cout<<endl;

  } else {
    cout<<" Unknown error?";
  }

  //unsigned int event   =  (word & eventNumMask) >>13;
  //unsigned int tbm_status   =  (word & tbmStatusMask);

  //if(event>0) cout<<":event: "<<event;
  //cout<<endl;

  return status;
}
///////////////////////////////////////////////////////////////////////////
int MyDecode::data(int word) {
  const bool CHECK_PIXELS = true;
  bool PRINT_PIXELS = true;
  if(errorsOnly) PRINT_PIXELS = false;

  const int ROCMAX = 24;
  const unsigned int plsmsk = 0xff;   // pulse height
  const unsigned int pxlmsk = 0xff00; // pixel index
  const unsigned int dclmsk = 0x1f0000;
  const unsigned int rocmsk = 0x3e00000;
  const unsigned int chnlmsk = 0xfc000000;
  int status = 0;

  int roc = ((word&rocmsk)>>21);
  // Check for embeded special words
  if(roc>0 && roc<25) {  // valid ROCs go from 1-24
    //if(PRINT_PIXELS) cout<<"data "<<hex<<word<<dec;
    unsigned int channel = ((word&chnlmsk)>>26);
    if(channel>0 && channel<37) {  // valid channels 1-36
      //cout<<hex<<word<<dec;
      int dcol=(word&dclmsk)>>16;
      int pix=(word&pxlmsk)>>8;
      int adc=(word&plsmsk);
      // print the roc number according to the online 0-15 scheme
      if(PRINT_PIXELS) cout<<" Channel- "<<channel<<" ROC- "<<(roc-1)<<" DCOL- "<<dcol<<" Pixel- "
          <<pix<<" ADC- "<<adc<<endl;
      if(CHECK_PIXELS) {
        if(roc>ROCMAX)
          cout<<" wrong roc number "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
        if(dcol<0 || dcol>25)
          cout<<" wrong dcol number "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
        if(pix<2 || pix>181)
          cout<<" wrong pix number chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
      }
      status++;
    } else {
      cout<<"Wrong channel "<<channel<<endl;
      return -2;
    }
  } else {  // error word
    //cout<<"error word "<<hex<<word<<dec;
    status=error(word);
  }

  return status;
}
////////////////////////////////////////////////////////////////////////////


class SiPixelRawDumper : public edm::EDAnalyzer {
public:

  /// ctor
  explicit SiPixelRawDumper( const edm::ParameterSet& cfg) : theConfig(cfg) {} 

  /// dtor
  virtual ~SiPixelRawDumper() {}

  virtual void beginJob( const edm::EventSetup& ) {}

  /// dummy end of job 
  virtual void endJob() {}

  /// get data, convert to digis attach againe to Event
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  edm::ParameterSet theConfig;
};

void SiPixelRawDumper::analyze(const  edm::Event& ev, const edm::EventSetup& es) 
{
  bool printHeaders = true;

  edm::Handle<FEDRawDataCollection> buffers;
  static std::string label = theConfig.getUntrackedParameter<std::string>("InputLabel","source");
  static std::string instance = theConfig.getUntrackedParameter<std::string>("InputInstance","");
  
  ev.getByLabel( label, instance, buffers);

  std::pair<int,int> fedIds(FEDNumbering::MINSiPixelFEDID, FEDNumbering::MAXSiPixelFEDID);

  PixelDataFormatter formatter(0);
  bool dummyErrorBool;

  //typedef unsigned int Word32;
  //typedef long long Word64;
  typedef uint32_t Word32;
  typedef uint64_t Word64;
  int status=0;

  for (int fedId = fedIds.first; fedId <= fedIds.second; fedId++) {
    LogDebug("SiPixelRawDumper")<< " GET DATA FOR FED: " <<  fedId ;
    if(printHeaders) cout<<" For FED = "<<fedId<<endl;

    PixelDataFormatter::Digis digis;
    PixelDataFormatter::Errors errors;

    //get event data for this fed
    const FEDRawData& rawData = buffers->FEDData( fedId );

    //cout<<" size "<<rawData.size()<<endl;

    int nWords = rawData.size()/sizeof(Word64);

    // check headers
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); 
    //cout<<hex<<*header<<dec<<endl;
    if(printHeaders) status = MyDecode::header(*header);

    const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1);
    //cout<<hex<<*trailer<<dec<<endl;
    if(printHeaders) status = MyDecode::trailer(*trailer);

    //header--;
    for (const Word64* word = header+1; word != trailer; word++) {
      static const Word64 WORD32_mask  = 0xffffffff;
      Word32 w1 =  *word       & WORD32_mask;
      status = MyDecode::data(w1);
      Word32 w2 =  *word >> 32 & WORD32_mask;
      status = MyDecode::data(w2);
      //cout<<hex<<w1<<" "<<w2<<dec<<endl;

    }

    //convert data to digi (dummy for the moment)
    formatter.interpretRawData( dummyErrorBool, fedId, rawData, digis, errors);
    //cout<<dummyErrorBool<<" "<<digis.size()<<" "<<errors.size()<<endl;
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelRawDumper);
