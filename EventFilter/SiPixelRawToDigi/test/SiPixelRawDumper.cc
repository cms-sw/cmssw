/** \class SiPixelRawDumper_H
 *  Plug-in module that dump raw data file 
 *  for pixel subdetector
 *  Added class to interpret the data d.k. 30/10/08
 *  Add histograms. Add pix 0 detection.
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


// For L1  NOT IN RAW
//#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"


// To use root histos
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// For ROOT
#include <TROOT.h>
//#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>


#include <iostream>
using namespace std;

// #define L1  // L1 information not in RAW

const bool printErrors  = true;
const bool printData    = false;
const bool printHeaders = false;
const bool CHECK_PIXELS = true;

// Include the helper decoding class
/////////////////////////////////////////////////////////////////////////////
class MyDecode {
public:
  MyDecode() {}
  ~MyDecode() {}
  static int error(int error, int & fedChannel, int fed, bool print=false);
  static int data(int error, int & fedChannel, int fed, bool print=false);
  static int header(unsigned long long word64, int fed, bool print);
  static int trailer(unsigned long long word64, int fed, bool print);
private:
};
/////////////////////////////////////////////////////////////////////////////
int MyDecode::header(unsigned long long word64, int fed, bool print) {
  int fed_id=(word64>>8)&0xfff;
  int event_id=(word64>>32)&0xffffff;
  unsigned int bx_id=(word64>>20)&0xfff;
//   if(bx_id!=101) {
//     cout<<" Header "<<" for FED "
// 	<<fed_id<<" event "<<event_id<<" bx "<<bx_id<<endl;
//     int dummy=0;
//     cout<<" : ";
//     cin>>dummy;
//   }
  if(print) cout<<" Header "<<" for FED "
		<<fed_id<<" event "<<event_id<<" bx "<<bx_id<<endl;

  return event_id;
}
//
int MyDecode::trailer(unsigned long long word64, int fed, bool print) {
  int slinkLength = int( (word64>>32) & 0xffffff );
  int crc         = int( (word64&0xffff0000)>>16 );
  int tts         = int( (word64&0xf0)>>4);
  int slinkError  = int( (word64&0xf00)>>8);
  if(print) cout<<" Trailer "<<" len "<<slinkLength
		<<" tts "<<tts<<" error "<<slinkError<<" crc "<<hex<<crc<<dec<<endl;
  return slinkLength;
}
//
// Decode error FIFO
// Works for both, the error FIFO and the SLink error words. d.k. 25/04/07
int MyDecode::error(int word, int & fedChannel, int fed, bool print) {
  int status = -1;
  print = print || printErrors;

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
  unsigned int channel = 0;

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
     if(print) cout<<"Timeout Error- channel: ";
     for(int i=0;i<5;i++) {
       if( (index & 0x1) != 0) {
	 channel = offset + i + 1;
	 if(print) cout<<channel<<" ";
       }
       index = index >> 1;
     }
     //if(print) cout<<" for Fed "<<fed<<endl;
     status = -10;
     fedChannel = channel;
     //end of timeout  chip and channel decoding

  } else if( (word&errorMask) == eventNumError ) { // EVENT NUMBER ERROR
    channel =  (word & channelMask) >>26;
    unsigned int tbm_event   =  (word & tbmEventMask);
    
    if(print) cout<<"Event Number Error- channel: "<<channel<<" tbm event nr. "
		  <<tbm_event<<" ";
     status = -11;
     fedChannel = channel;
    
  } else if( ((word&errorMask) == trailError)) {  // TRAILER 
    channel =  (word & channelMask) >>26;
    unsigned int tbm_status   =  (word & tbmStatusMask);
    

    if(word & RocErrMask) {
      if(print) cout<<"Number of Rocs Error- "<<"channel: "<<channel<<" "; // <<endl;
     status = -12;
    }

    if(word & overflowMask) {
      if(print) cout<<"Overflow Error- "<<"channel: "<<channel<<" "; // <<endl;
     status = -14;
    }

    if(word & FsmErrMask) {
      if(print) cout<<"Finite State Machine Error- "<<"channel: "<<channel
			  <<" Error status:0x"<<hex<< ((word & FsmErrMask)>>9)<<dec<<" "; // <<endl;
     status = -13;
    }


    if(tbm_status!=0) {
      if(print) cout<<"Trailer Error- "<<"channel: "<<channel<<" TBM status:0x"
			  <<hex<<tbm_status<<dec<<" "; // <<endl;
     status = -15;
     // implement the resync/reset 17
    }
    fedChannel = channel;

  } else if((word&errorMask)==fifoError) {  // FIFO
    if(print) { 
      if(word & Fif2NFMask) cout<<"A fifo 2 is Nearly full- ";
      if(word & TrigNFMask) cout<<"The trigger fifo is nearly Full - ";
      if(word & ChnFifMask) cout<<"fifo-1 is nearly full for channel"<<(word & ChnFifMask);
      //cout<<endl;
      status = -16;
    }

  } else {
    cout<<" Unknown error?";
  }

  if(print && status <-1) cout<<" For FED "<<fed<<endl;
  return status;
}
///////////////////////////////////////////////////////////////////////////
int MyDecode::data(int word, int & fedChannel, int fed, bool print) {

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
    //if(print) cout<<"data "<<hex<<word<<dec;
    unsigned int channel = ((word&chnlmsk)>>26);
    if(channel>0 && channel<37) {  // valid channels 1-36
      //cout<<hex<<word<<dec;
      int dcol=(word&dclmsk)>>16;
      int pix=(word&pxlmsk)>>8;
      int adc=(word&plsmsk);
      fedChannel = channel;
      // print the roc number according to the online 0-15 scheme
      if(print) cout<<" Fed "<<fed<<" Channel- "<<channel<<" ROC- "<<(roc-1)<<" DCOL- "<<dcol<<" Pixel- "
          <<pix<<" ADC- "<<adc<<endl;
      status++;
      if(CHECK_PIXELS) {
        if(roc>ROCMAX) {
          cout<<" Fed "<<fed<<" wrong roc number chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -4;
	}
        if(dcol<0 || dcol>25) {
          cout<<" Fed "<<fed<<" wrong dcol number chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -3;
	}
        if(pix<2 || pix>181) {
          cout<<" Fed "<<fed<<" wrong pix number chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -3;
	}
        if(pix==0) {
          cout<<" Fed "<<fed<<" pix=0 chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -5;
	}
      }
    } else {
      cout<<"Wrong channel "<<channel<<endl;
      return -2;
    }
  } else {  // error word
    //cout<<"error word "<<hex<<word<<dec;
    status=error(word, fedChannel, fed, print);
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

  void beginJob();

  //void beginRun( const edm::EventSetup& ) {}

  // end of job 
  void endJob();

  /// get data, convert to digis attach againe to Event
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  edm::ParameterSet theConfig;
  int countEvents, countAllEvents;
  int countErrors;
  float sumPixels, sumFedSize, sumFedPixels[40];
  int fedErrors[40][36];
  int decodeErrors[40][36];
  int decodeErrors000[40][36];  // pix 0 problem
  int errorType[20];

  TH1F *hsize,*hsize0, *hsize1, *hsize2;
  TH1F *hsizeFeds[40];
  TH1F *hpixels, *hpixels0, *hpixels1, *hpixels2, *hpixels3, *hpixels4;
  TH1F *htotPixels,*htotPixels0, *htotPixels1;
  TH1F *herrors, *htotErrors, *herrorType, *herrorFed, *herrorChan;
  TH1F *herrorType0, *herrorFed0, *herrorChan0;
  TH2F *hfed2dErrors;

  TH1F *hevent, *hlumi, *horbit, *hbx, *hlumi1, *hbx1;

};

void SiPixelRawDumper::endJob() {
  if(countEvents>0) {
    sumPixels /= float(countEvents);
    sumFedSize /= float(countAllEvents);
    for(int i=0;i<40;++i) {
      sumFedPixels[i] /= float(countEvents);
      hpixels4->Fill(float(i),sumFedPixels[i]); //pixels only 
    }
  }
    
  cout<<" Total/non-empty events " <<countAllEvents<<" / "<<countEvents<<" average number of pixels "<<sumPixels<<endl;

  cout<<" Average Fed size for all events "<<sumFedSize<<endl;
  for(int i=0;i<40;++i) cout<<sumFedPixels[i]<<" ";
  cout<<endl;

  cout<<" Total number of errors "<<countErrors<<endl;
  cout<<" FED erros "<<endl<<"  Fed Channel Errors"<<endl;
  for(int i=0;i<40;++i) {
    for(int j=0;j<36;++j) if(fedErrors[i][j]>0) {
      cout<<i<<" "<<j<<" "<<fedErrors[i][j]<<endl;
    }
  }
  cout<<" Decode errors "<<endl<<"  Fed Channel Errors Pix_000"<<endl;
  for(int i=0;i<40;++i) {
    for(int j=0;j<36;++j) if(decodeErrors[i][j]>0) {
      cout<<i<<" "<<j<<" "<<decodeErrors[i][j]<<" "<<decodeErrors000[i][j]<<endl;
    }
  }

  cout<<" Total errors for all feds "<<endl<<" Type Errors"<<endl;
  for(int i=0;i<20;++i) {
    if( errorType[i]>0 ) cout<<"   "<<i<<" "<<errorType[i]<<endl;
  }

}

void SiPixelRawDumper::beginJob() {
  countEvents=0;
  countAllEvents=0;
  countErrors=0;
  sumPixels=0.;
  sumFedSize=0;
  for(int i=0;i<40;++i) {
    sumFedPixels[i]=0;
    for(int j=0;j<36;++j) fedErrors[i][j]=0;
    for(int j=0;j<36;++j) decodeErrors[i][j]=0;
    for(int j=0;j<36;++j) decodeErrors000[i][j]=0;
  }
  for(int i=0;i<20;++i) errorType[i]=0;

  edm::Service<TFileService> fs;
  hsize = fs->make<TH1F>( "hsize", "FED event size in words-4", 6000, -0.5, 5999.5);
  hsize0 = fs->make<TH1F>( "hsize0", "FED event size in words-4", 2000, -0.5, 19999.5);
  hsize1 = fs->make<TH1F>( "hsize1", "bpix FED event size in words-4", 6000, -0.5, 5999.5);
  hsize2 = fs->make<TH1F>( "hsize2", "fpix FED event size in words-4", 6000, -0.5, 5999.5);

  hpixels = fs->make<TH1F>( "hpixels", "pixels per FED", 2000, -0.5, 19999.5);
  hpixels0 = fs->make<TH1F>( "hpixels0", "pixels per FED", 6000, -0.5, 5999.5);
  hpixels1 = fs->make<TH1F>( "hpixels1", "pixels >0 per FED", 6000, -0.5, 5999.5);
  hpixels2 = fs->make<TH1F>( "hpixels2", "pixels >0 per BPix FED", 6000, -0.5, 5999.5);
  hpixels3 = fs->make<TH1F>( "hpixels3", "pixels >0 per Fpix FED", 6000, -0.5, 5999.5);
  hpixels4 = fs->make<TH1F>( "hpixels4", "pixels per each FED", 40, -0.5, 39.5);

  htotPixels = fs->make<TH1F>( "htotPixels", "pixels per event", 10000, -0.5, 99999.5);
  htotPixels0 = fs->make<TH1F>( "htotPixels0", "pixels per event", 20000, -0.5, 19999.5);
  htotPixels1 = fs->make<TH1F>( "htotPixels1", "pixels >0 per event", 20000, -0.5, 19999.5);

  herrors = fs->make<TH1F>( "herrors", "errors per FED", 100, -0.5, 99.5);
  htotErrors = fs->make<TH1F>( "htotErrors", "errors per event", 1000, -0.5, 999.5);
  herrorType0 = fs->make<TH1F>( "herrorType0", "all errors per type", 20, -0.5, 19.5);
  herrorFed0  = fs->make<TH1F>( "herrorFed0", "all errors per FED", 40, -0.5, 39.5);
  herrorChan0 = fs->make<TH1F>( "herrorChan0", "all errors per chan", 36, -0.5, 36.5);
  herrorType = fs->make<TH1F>( "herrorType", "readout errors per type", 20, -0.5, 19.5);
  herrorFed  = fs->make<TH1F>( "herrorFed", "readout errors per FED", 40, -0.5, 39.5);
  herrorChan = fs->make<TH1F>( "herrorChan", "readout errors per chan", 36, -0.5, 36.5);
  hfed2dErrors = fs->make<TH2F>( "hfed2DErrors", "errors per FED", 40,-0.5,39.5,
				 36, -0.5, 36.5);

  hsizeFeds[0] = fs->make<TH1F>( "hsizeFed0", "FED 0 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[1] = fs->make<TH1F>( "hsizeFed1", "FED 1 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[2] = fs->make<TH1F>( "hsizeFed2", "FED 2 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[3] = fs->make<TH1F>( "hsizeFed3", "FED 3 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[4] = fs->make<TH1F>( "hsizeFed4", "FED 4 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[5] = fs->make<TH1F>( "hsizeFed5", "FED 5 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[6] = fs->make<TH1F>( "hsizeFed6", "FED 6 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[7] = fs->make<TH1F>( "hsizeFed7", "FED 7 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[8] = fs->make<TH1F>( "hsizeFed8", "FED 8 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[9] = fs->make<TH1F>( "hsizeFed9", "FED 9 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[10] = fs->make<TH1F>( "hsizeFed10", "FED 10 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[11] = fs->make<TH1F>( "hsizeFed11", "FED 11 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[12] = fs->make<TH1F>( "hsizeFed12", "FED 12 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[13] = fs->make<TH1F>( "hsizeFed13", "FED 13 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[14] = fs->make<TH1F>( "hsizeFed14", "FED 14 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[15] = fs->make<TH1F>( "hsizeFed15", "FED 15 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[16] = fs->make<TH1F>( "hsizeFed16", "FED 16 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[17] = fs->make<TH1F>( "hsizeFed17", "FED 17 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[18] = fs->make<TH1F>( "hsizeFed18", "FED 18 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[19] = fs->make<TH1F>( "hsizeFed19", "FED 19 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[20] = fs->make<TH1F>( "hsizeFed20", "FED 20 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[21] = fs->make<TH1F>( "hsizeFed21", "FED 21 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[22] = fs->make<TH1F>( "hsizeFed22", "FED 22 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[23] = fs->make<TH1F>( "hsizeFed23", "FED 23 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[24] = fs->make<TH1F>( "hsizeFed24", "FED 24 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[25] = fs->make<TH1F>( "hsizeFed25", "FED 25 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[26] = fs->make<TH1F>( "hsizeFed26", "FED 26 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[27] = fs->make<TH1F>( "hsizeFed27", "FED 27 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[28] = fs->make<TH1F>( "hsizeFed28", "FED 28 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[29] = fs->make<TH1F>( "hsizeFed29", "FED 29 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[30] = fs->make<TH1F>( "hsizeFed30", "FED 30 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[31] = fs->make<TH1F>( "hsizeFed31", "FED 31 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[32] = fs->make<TH1F>( "hsizeFed32", "FED 32 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[33] = fs->make<TH1F>( "hsizeFed33", "FED 33 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[34] = fs->make<TH1F>( "hsizeFed34", "FED 34 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[35] = fs->make<TH1F>( "hsizeFed35", "FED 35 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[36] = fs->make<TH1F>( "hsizeFed36", "FED 36 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[37] = fs->make<TH1F>( "hsizeFed37", "FED 37 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[38] = fs->make<TH1F>( "hsizeFed38", "FED 38 event size ", 1000, -0.5, 3999.5);
  hsizeFeds[39] = fs->make<TH1F>( "hsizeFed39", "FED 39 event size ", 1000, -0.5, 3999.5);


  hevent = fs->make<TH1F>("hevent","event",1000,0,10000000.);
  horbit = fs->make<TH1F>("horbit","orbit",100, 0,100000000.);
  hlumi  = fs->make<TH1F>("hlumi", "lumi", 4000,0,4000.);
  hbx    = fs->make<TH1F>("hbx",   "bx",   4000,0,4000.);  
  hbx1    = fs->make<TH1F>("hbx1",   "bx",   4000,0,4000.);  
  hlumi1  = fs->make<TH1F>("hlumi1", "lumi", 4000,0,4000.);

  
}

void SiPixelRawDumper::analyze(const  edm::Event& ev, const edm::EventSetup& es) {

  // Access event information
  //int run       = ev.id().run();
  int event     = ev.id().event();
  int lumiBlock = ev.luminosityBlock();
  int bx        = ev.bunchCrossing();
  int orbit     = ev.orbitNumber();

  hevent->Fill(float(event));
  hlumi->Fill(float(lumiBlock));
  hbx->Fill(float(bx));
  horbit->Fill(float(orbit));

#ifdef L1
  // Get L1
  edm::Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  ev.getByLabel("gtDigis",L1GTRR);

  if (L1GTRR.isValid()) {
    bool l1a = L1GTRR->decision();
    cout<<" L1 status :"<<l1a<<endl;
  } else {
    cout<<" NO L1 status "<<endl;
  } // if l1a
#endif


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
  int countPixels=0;
  int eventId = -1;
  int countErrorsPerEvent=0;

  countAllEvents++;
  if(printHeaders) cout<<" Event = "<<countEvents<<endl;

  // Loop over FEDs
  for (int fedId = fedIds.first; fedId <= fedIds.second; fedId++) {
    LogDebug("SiPixelRawDumper")<< " GET DATA FOR FED: " <<  fedId ;
    if(printHeaders) cout<<" For FED = "<<fedId<<endl;

    PixelDataFormatter::Digis digis;
    PixelDataFormatter::Errors errors;

    //get event data for this fed
    const FEDRawData& rawData = buffers->FEDData( fedId );

    int nWords = rawData.size()/sizeof(Word64);
    //cout<<" size "<<nWords<<endl;
    sumFedSize += float(nWords);

    hsize->Fill(float(2*nWords)); // fed buffer size in words (32bit)
    hsize0->Fill(float(2*nWords)); // fed buffer size in words (32bit)
    if(fedId<32) hsize1->Fill(float(2*nWords)); // fed buffer size in words (32bit)
    else hsize2->Fill(float(2*nWords)); // fed buffer size in words (32bit)

    hsizeFeds[fedId]->Fill(float(2*nWords)); // size, includes errors and dummy words

    // check headers
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); 
    //cout<<hex<<*header<<dec<<endl;
    eventId = MyDecode::header(*header, fedId, printHeaders);
    //if(fedId = fedIds.first) 

    const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1);
    //cout<<hex<<*trailer<<dec<<endl;
    status = MyDecode::trailer(*trailer,fedId, printHeaders);

    int countPixelsInFed=0;
    int countErrorsInFed=0;
    int fedChannel = 0;
    // Loop over payload words
    for (const Word64* word = header+1; word != trailer; word++) {
      static const Word64 WORD32_mask  = 0xffffffff;

      Word32 w1 =  *word       & WORD32_mask;
      status = MyDecode::data(w1,fedChannel, fedId, printData);
      if(status>0) {
	countPixels++;
	countPixelsInFed++;
      } else if(status<0) {
	countErrorsInFed++;
	//cout<<" For FED "<<fedId<<" Event "<<eventId<<
	//" "<<fedChannel<<" "<<status<<endl;
	status=abs(status);
	if(status<20) errorType[status]++;
	herrorType0->Fill(float(status));
	herrorFed0->Fill(float(fedId));
	herrorChan0->Fill(float(fedChannel));
	if(status>=10) {
	  fedErrors[fedId][fedChannel]++;
	  herrorType->Fill(float(status));
	  herrorFed->Fill(float(fedId));
	  herrorChan->Fill(float(fedChannel));
	  hfed2dErrors->Fill(float(fedId),float(fedChannel));
	} else if(status>0) {
	  decodeErrors[fedId][fedChannel]++;
	  if(status==5) decodeErrors000[fedId][fedChannel]++;
	}
      }

      Word32 w2 =  *word >> 32 & WORD32_mask;
      status = MyDecode::data(w2,fedChannel, fedId,printData);
      if(status>0) {
	countPixels++;
	countPixelsInFed++;
      } else if(status<0) {
	countErrorsInFed++;
	//cout<<" For FED "<<fedId<<" Event "<<eventId<<
	//" "<<fedChannel<<" "<<status<<endl;
	//cout<<hex<<w1<<" "<<w2<<dec<<endl;
	status=abs(status);
	if(status<20) errorType[status]++;
	herrorType0->Fill(float(status));
	herrorFed0->Fill(float(fedId));
	herrorChan0->Fill(float(fedChannel));
	if(status>=10) {
	  fedErrors[fedId][fedChannel]++;
	  herrorType->Fill(float(status));
	  herrorFed->Fill(float(fedId));
	  herrorChan->Fill(float(fedChannel));
	  hfed2dErrors->Fill(float(fedId),float(fedChannel));
	} else if(status>0) {
	  decodeErrors[fedId][fedChannel]++;
	  if(status==5) decodeErrors000[fedId][fedChannel]++;
	}
      }
    } // loop over words

    countErrors += countErrorsInFed;
    countErrorsPerEvent += countErrorsInFed;

    //convert data to digi (dummy for the moment)
    formatter.interpretRawData( dummyErrorBool, fedId, rawData, digis, errors);
    //cout<<dummyErrorBool<<" "<<digis.size()<<" "<<errors.size()<<endl;

    if(countPixelsInFed>0)  {
      sumFedPixels[fedId] += countPixelsInFed;
    }

    hpixels->Fill(float(countPixelsInFed));
    hpixels0->Fill(float(countPixelsInFed));
    if(countPixelsInFed>0) hpixels1->Fill(float(countPixelsInFed));
    if(countPixelsInFed>0 && fedId<32) hpixels2->Fill(float(countPixelsInFed));
    if(countPixelsInFed>0 && fedId>=32) hpixels3->Fill(float(countPixelsInFed));
    herrors->Fill(float(countErrorsInFed));

    //    if(fedId == fedIds.first || countPixelsInFed>0 || countErrorsInFed>0 )  {
    //       eventId = MyDecode::header(*header, true);
    //       if(countPixelsInFed>0 || countErrorsInFed>0 ) cout<<"fed "<<fedId<<" pix "<<countPixelsInFed<<" err "<<countErrorsInFed<<endl;
    //       status = MyDecode::trailer(*trailer,true);
    //     }
    
  } // loop over feds
  
  htotPixels->Fill(float(countPixels));
  htotPixels0->Fill(float(countPixels));
  htotErrors->Fill(float(countErrorsPerEvent));

  if(countPixels>0) {
    hlumi1->Fill(float(lumiBlock));
    hbx1->Fill(float(bx));
    htotPixels1->Fill(float(countPixels));

    //cout<<"EVENT: "<<countEvents<<" "<<eventId<<" pixels "<<countPixels<<" errors "<<countErrors<<endl;
    sumPixels += countPixels;
    countEvents++;
    //int dummy=0;
    //cout<<" : ";
    //cin>>dummy;
  }
 
}
// 2 - wrong channel
// 3 - wrong roc
// 4 - wrong pix or dcol 
// 5 - pix=0

  // 10 - timeout ()
  // 11 - ene ()
  // 12 - mum pf rocs error ()
  // 13 - fsm ()
  // 14 - overflow ()
  // 15 - trailer ()
  // 16 - fifo  (30)
  // 17 - reset/resync NOT INCLUDED YET

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelRawDumper);
