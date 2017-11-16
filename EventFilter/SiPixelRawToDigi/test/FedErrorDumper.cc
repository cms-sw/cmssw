/** \class SiPixelRawDumper_H
 *  Plug-in module that dump raw data file 
 *  for pixel subdetector
 *  Added class to interpret the data d.k. 30/10/08
 *  Add histograms. Add pix 0 detection.
 * Adopt for SLC6, CMSSW620, 11.10.13, dk
 * Works for CMSSW7X (bytoken)d.k.
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

// for detids 
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/DetId/interface/DetId.h"


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
    

    if(tbm_status!=0) {
      if(print) cout<<"Trailer Error- "<<"channel: "<<channel<<" TBM status:0x"
			  <<hex<<tbm_status<<dec<<" "; // <<endl;
     status = -15;
     // implement the resync/reset 17
    }

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
    cout<<" Unknown error?, error word "<<hex<<word<<dec<<endl;
    status = -19;
  }

  if(print && status <-1) cout<<" For FED "<<fed<<endl;
  return status;
}
///////////////////////////////////////////////////////////////////////////
int MyDecode::data(int word, int & fedChannel, int fed, bool print) {
  const bool CHECK_PIXELS = true;
  //const int ROCMAX = 24;
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
      if(print) {
	int dcol1 = dcol/6;
	int dcol2 = dcol%6;
	int pix1  = pix/36;
	int pix2  = (pix%36)/6;
	int pix3  = pix%6;
	cout<<" Fed "<<fed<<" Channel- "<<channel<<" ROC- "<<(roc-1)<<" DCOL- "<<dcol<<" Pixel- "
	    <<pix<<" ADC- "<<adc<<" : "<<dcol1<<dcol2<<"-"<<pix1<<pix2<<pix3<<endl;
      }
      status++;
      if(CHECK_PIXELS) {

        if(dcol<0 || dcol>25) {
          if(print) cout<<" Fed "<<fed<<" wrong dcol number chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -3;
	}

        if(pix<2 || pix>181) {
          if(print) cout<<" Fed "<<fed<<" wrong pix number chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -3;
	}

	if( (fed>31 && roc>24) || (fed<=31 && roc>16)  ) {	  
          if(print) cout<<" Fed "<<fed<<" wrong roc number chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"
			<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -4;
	} else if( fed<=31 && channel<=24 && roc>8 ) {
	  // ptorotect for rerouted signals
	  if( !( (fed==13 && channel==17) ||(fed==15 && channel==5) ||(fed==31 && channel==10) 
		                          ||(fed==27 && channel==15) )  ) {
	    if(print) cout<<" Fed "<<fed<<" wrong roc number, chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc<<"/"
			  <<dcol<<"/"<<pix<<"/"<<adc<<endl;
	    status = -4;
	  }
	}

        if(pix==0) {
          if(print) cout<<" Fed "<<fed<<" pix=0 chan/roc/dcol/pix/adc = "<<channel<<"/"
                        <<roc<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -5;
	}
      }

    } else {
      cout<<"Wrong channel "<<channel<<endl;
      return -2;
    }

  } else if(roc==25) {  // 

    unsigned int channel = ((word&chnlmsk)>>26);
    cout<<"Wrong roc 25-"<<roc<<" in fed/chan "<<fed<<"/"<<channel<<endl;
    status=-4;

  } else {  // error word

    //cout<<"error word "<<hex<<word<<dec;
    status=error(word, fedChannel, fed, print);

  } // end if

  return status;
}



////////////////////////////////////////////////////////////////////////////

class FedErrorDumper : public edm::EDAnalyzer {
public:

  /// ctor
  explicit FedErrorDumper( const edm::ParameterSet& cfg); 

  /// dtor
  virtual ~FedErrorDumper() {}

  void beginJob();

  //void beginRun( const edm::EventSetup& ) {}

  // end of job 
  void endJob();

  /// get data, convert to digis attach againe to Event
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  edm::ParameterSet theConfig;
  edm::EDGetTokenT<edm::DetSetVector<SiPixelRawDataError> > fedErrorContainer;
  bool PRINT;
  //int countEvents, countAllEvents;
  int fedErrors, moduleErrors, spareCounts;
  //float sumPixels, sumFedSize, sumFedPixels[40];
  //int fedErrors[40][36];
  //int decodeErrors[40][36];
  //int decodeErrors000[40][36];  // pix 0 problem
  int countErrors[40], countErrors2[40];

  TH1F *hfeds, *hfedsF, *hfedsSlink, *hfedsCRC, *hfedsUnknown;
  TH1F *herrors, *herrorsF, *htype;
  TH1F *hmode, *hmodeF;
  TH1F *htbm, *htbmF;
  TH2F *hfedErrors0,*hfedErrors1,*hfedErrors2,*hfedErrors3,*hfedErrors4,*hfedErrors5,
    *hfedErrors6,*hfedErrors7,*hfedErrors8,*hfedErrors9;
  TH2F *hfedErrors0F, *hfed2d,*hfed2d0;

  TH1F *hlumi, *hbx;

};
//----------------------------------
FedErrorDumper::FedErrorDumper( const edm::ParameterSet& cfg) : theConfig(cfg) {

  //std::string src_ = theConfig.getUntrackedParameter<std::string>("InputLabel","source");
  std::string src = theConfig.getUntrackedParameter<std::string>("InputLabel","siPixelDigis");

  // For the ByToken method
  fedErrorContainer = consumes<edm::DetSetVector<SiPixelRawDataError> >(src);
} 

//---------------------------------
void FedErrorDumper::endJob() {
  cout<<" fed-errors "<<fedErrors<<" module-Errors "<<moduleErrors<<" "<<spareCounts<<endl;
  for(int i=0;i<40;++i) {
    if(countErrors[i]>0 || countErrors2[i]>0 ) cout<<i<<" "<<countErrors[i]<<" "<<countErrors2[i]<<endl;
  }

  //sumFedPixels[i] /= float(countEvents);
//       hpixels4->Fill(float(i),sumFedPixels[i]); //pixels only 
//     }
//   }

//   if(countEvents>0) {
//     sumPixels /= float(countEvents);
//     sumFedSize /= float(countAllEvents);
    
//   cout<<" Total/non-empty events " <<countAllEvents<<" / "<<countEvents<<" average number of pixels "<<sumPixels<<endl;

//   cout<<" Average Fed size for all events "<<sumFedSize<<endl;
//   for(int i=0;i<40;++i) cout<<sumFedPixels[i]<<" ";
//   cout<<endl;

//   cout<<" Total number of errors "<<countErrors<<endl;
//   cout<<" FED erros "<<endl<<"  Fed Channel Errors"<<endl;
//   for(int i=0;i<40;++i) {
//     for(int j=0;j<36;++j) if(fedErrors[i][j]>0) {
//       cout<<i<<" "<<j<<" "<<fedErrors[i][j]<<endl;
//     }
//   }
//   cout<<" Decode errors "<<endl<<"  Fed Channel Errors Pix_000"<<endl;
//   for(int i=0;i<40;++i) {
//     for(int j=0;j<36;++j) if(decodeErrors[i][j]>0) {
//       cout<<i<<" "<<j<<" "<<decodeErrors[i][j]<<" "<<decodeErrors000[i][j]<<endl;
//     }
//   }

//   cout<<" Total errors for all feds "<<endl<<" Type Errors"<<endl;
//   for(int i=0;i<20;++i) {
//     if( errorType[i]>0 ) cout<<"   "<<i<<" "<<errorType[i]<<endl;
//   }

}

void FedErrorDumper::beginJob() {
  //countEvents=0;
  //countAllEvents=0;
  //countErrors=0;
  fedErrors=0;
  moduleErrors=0;
  spareCounts=0.;
  //sumFedSize=0;  

  PRINT = theConfig.getUntrackedParameter<bool>("Verbosity",false);
  for(int i=0;i<40;++i) {
    countErrors[i]=0;
    countErrors2[i]=0;
  }

//   for(int i=0;i<40;++i) {
//     sumFedPixels[i]=0;
//     for(int j=0;j<36;++j) fedErrors[i][j]=0;
//     for(int j=0;j<36;++j) decodeErrors[i][j]=0;
//     for(int j=0;j<36;++j) decodeErrors000[i][j]=0;
//   }
//   for(int i=0;i<20;++i) errorType[i]=0;

  edm::Service<TFileService> fs;

  htbm  = fs->make<TH1F>("htbm", "tbm errors ",256, -0.5, 255.5);
  htbmF = fs->make<TH1F>("htbmF","tbm errors ",256, -0.5, 255.5);

  hmode  = fs->make<TH1F>( "hmode", "mode",16, -0.5, 15.5);
  hmodeF  = fs->make<TH1F>("hmodeF","mode",16, -0.5, 15.5);

  hfeds  = fs->make<TH1F>( "hfeds", "errors per FED",40, -0.5, 39.5);
  hfedsF = fs->make<TH1F>( "hfedsF", "errors per FED",40, -0.5, 39.5);
  hfedsSlink = fs->make<TH1F>( "hfedsSlink", "errors per FED",40, -0.5, 39.5);
  hfedsCRC = fs->make<TH1F>( "hfedsCRC", "errors per FED",40, -0.5, 39.5);
  hfedsUnknown = fs->make<TH1F>( "hfedsUnknown", "errors per FED",40, -0.5, 39.5);

  htype   = fs->make<TH1F>( "htype", "error type ",50, -0.5, 49.5);
  herrors = fs->make<TH1F>( "herrors", "error type ",50, -0.5, 49.5);
  herrorsF = fs->make<TH1F>( "herrorsF", "error type",50, -0.5, 49.5);

  hfedErrors0 = fs->make<TH2F>( "hfedErrors0", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // ALL
  hfedErrors0F = fs->make<TH2F>( "hfedErrors0F","errors",40,-0.5,39.5,37, -0.5, 36.5); // ALL
  hfedErrors1 = fs->make<TH2F>( "hfedErrors1", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // Timeout
  hfedErrors2 = fs->make<TH2F>( "hfedErrors2", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // Overflow
  hfedErrors3 = fs->make<TH2F>( "hfedErrors3", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // ENE
  hfedErrors4 = fs->make<TH2F>( "hfedErrors4", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // FIFO
  hfedErrors5 = fs->make<TH2F>( "hfedErrors5", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // NOR
  hfedErrors6 = fs->make<TH2F>( "hfedErrors6", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // TRAILER
  hfedErrors7 = fs->make<TH2F>( "hfedErrors7", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // FSM 
  hfedErrors8 = fs->make<TH2F>( "hfedErrors8", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // Invalid ROC
  hfedErrors9 = fs->make<TH2F>( "hfedErrors9", "errors", 40,-0.5,39.5,37, -0.5, 36.5); // Invalid DCOL-PIX

  hfed2d = fs->make<TH2F>( "hfed2d", "errors", 40,-0.5,39.5,20, 19.5, 39.5); // ALL
  hfed2d0 = fs->make<TH2F>("hfed2d0","errors", 40,-0.5,39.5,20, 19.5, 39.5); // ALL

  hlumi  = fs->make<TH1F>("hlumi", "lumi", 4000,0,4000.);
  hbx    = fs->make<TH1F>("hbx",   "bx",   4000,0,4000.);  
  
}
//--------------------------------------------------------------------------------
void FedErrorDumper::analyze(const  edm::Event& ev, const edm::EventSetup& es) {
  //const bool PRINT = false;

  // Access event information
  int run       = ev.id().run();
  int event     = ev.id().event();
  int lumiBlock = ev.luminosityBlock();
  int bx        = ev.bunchCrossing();
  //int orbit     = ev.orbitNumber(); // unused 

  hlumi->Fill(float(lumiBlock));
  hbx->Fill(float(bx));

//   edm::Handle<FEDRawDataCollection> buffers;
//   static std::string label = theConfig.getUntrackedParameter<std::string>("InputLabel","source");
//   static std::string instance = theConfig.getUntrackedParameter<std::string>("InputInstance","");  
//   ev.getByLabel( label, instance, buffers);

  edm::Handle< edm::DetSetVector<SiPixelRawDataError> >  input;
  //static std::string src_ = theConfig.getUntrackedParameter<std::string>("InputLabel","source");
  //static std::string src_ = theConfig.getUntrackedParameter<std::string>("InputLabel","siPixelDigis");
  //static std::string instance = theConfig.getUntrackedParameter<std::string>("InputInstance","");  
  //ev.getByLabel( src_, instance, input );
  ev.getByToken(fedErrorContainer , input);  // the new bytoken

  if (!input.isValid()) {cout<<" Container not found "<<endl; return;}

  if(PRINT) cout<<" Container found "<<run<<" "<<event<<" "<<lumiBlock<<" "<<bx<<endl;

  // Iterate on detector units
  edm::DetSetVector<SiPixelRawDataError>::const_iterator DSViter;
  for(DSViter = input->begin(); DSViter != input->end(); DSViter++) {
    //bool valid = false;
    unsigned int detid = DSViter->id; // = rawid
    //cout<<hex<<detid<<dec<<endl;

    if(detid==0xffffffff) { // whole fed

      if(PRINT) cout<<" FED errors "<<DSViter->data.size()<<endl;
      // Look at FED errors now	
      edm::DetSet<SiPixelRawDataError>::const_iterator  di;

      for(di = DSViter->data.begin(); di != DSViter->data.end(); di++) {
	int FedId = di->getFedId();                  // FED the error came from
	int errorType = di->getType();               // type of error
	uint32_t word32 = di->getWord32();
	uint64_t word64 = di->getWord64();

	fedErrors++;

 	herrors->Fill(float(errorType));
	herrorsF->Fill(float(errorType));
	hfedsF->Fill(float(FedId));
	hfeds->Fill(float(FedId));
	int errorTypeMod = errorType;

	if(PRINT) cout<<" fed " <<FedId<<" type "<<errorType<<" "<<hex<<word32<<" "<<word64<<dec<<" "<<di->getMessage()<<endl;

	int status=0;
	int fedChannel = -1;
	//const bool printData    = true;
	if(errorType>=26 && errorType<=31) {  // fed error	
	  status = MyDecode::error(word32, fedChannel, FedId, PRINT);
	  if(PRINT) cout<<" status "<<status<<endl;

	  if(fedChannel<1||fedChannel>36) {
	    cout<<" Cannot get a valid  fed channel number of a -fed- error "<<fedChannel<<endl;
	    continue; // skip this error
	  }
	    
	  hfedErrors0->Fill(float(FedId),float(fedChannel));
	  hfedErrors0F->Fill(float(FedId),float(fedChannel));

	  switch(errorType) {

	  case(28) : { // FIFO
	    hfedErrors4->Fill(float(FedId),float(fedChannel));
	    hfed2d0->Fill(float(FedId),float(errorTypeMod));
	    htype->Fill(float(errorTypeMod));
	    break;}

	  case(29) : {  // TIMEOUT 
	    hfedErrors1->Fill(float(FedId),float(fedChannel));
	    hfed2d0->Fill(float(FedId),float(errorTypeMod));
	    htype->Fill(float(errorTypeMod));
	    break;}

	  case(30) : {  // TRAILER

	    int tbm = (word32& 0xff);
	    int mode = (word32>>8) & 0xf;
	    hmodeF->Fill(float(mode));
	    hmode->Fill(float(mode));
	    //cout<<" error 30 "<<hex<<mode<<dec<<endl;
	    countErrors[30]++;

	    if( tbm != 0) { // trailer
	      hfedErrors6->Fill(float(FedId),float(fedChannel));
	      htbmF->Fill(float(tbm));
	      errorTypeMod=20;
	      hfed2d0->Fill(float(FedId),float(errorTypeMod));
	      htype->Fill(float(errorTypeMod));
	    }

	    if( (mode&0x8) != 0) { // nor
	      //cout<<" NOR "<<endl;
	      hfedErrors5->Fill(float(FedId),float(fedChannel));
	      errorTypeMod=21;
	      hfed2d0->Fill(float(FedId),float(errorTypeMod));
	      htype->Fill(float(errorTypeMod));
	    }


	    if( (mode&0x1) != 0) { // overflow
	      hfedErrors2->Fill(float(FedId),float(fedChannel));
	      errorTypeMod=22;
	      hfed2d0->Fill(float(FedId),float(errorTypeMod));
	      htype->Fill(float(errorTypeMod));
	    } 

	    if( (mode&0x6) != 0) { // fsm
	      hfedErrors7->Fill(float(FedId),float(fedChannel));
	      errorTypeMod=23;
	      hfed2d0->Fill(float(FedId),float(errorTypeMod));
	      htype->Fill(float(errorTypeMod));
	    } 

	    break;}

	  case(31) : {  // EVE
	    hfedErrors3->Fill(float(FedId),float(fedChannel));
	    hfed2d0->Fill(float(FedId),float(errorTypeMod));
	    htype->Fill(float(errorTypeMod));
	    break;}
	  } // end switch

	} else if (errorType>=32 &&errorType<=34) {
	  cout<<" Slink error "<<endl;

	  hfedsSlink->Fill(float(FedId));
	  hfed2d0->Fill(float(FedId),float(errorTypeMod));
	  htype->Fill(float(errorTypeMod));
	    
	} else if (errorType==25 || (errorType>=35 && errorType<=38) ) { // conversion error
	  cout<<" Should never happen, a -fed- conversion error?  "<<endl;
	  status = MyDecode::data(word32, fedChannel, FedId, PRINT);
	  if(PRINT) cout<<" status "<<status<<endl;

	  hfedErrors0->Fill(float(FedId),float(fedChannel));
	  hfed2d0->Fill(float(FedId),float(errorTypeMod));
	  htype->Fill(float(errorTypeMod));

	  switch(errorType) {
	  case(25) : { // inv ROC
	    hfedErrors8->Fill(float(FedId),float(fedChannel));
	    break;}
	  case(36) : { // inv ROC
	    hfedErrors8->Fill(float(FedId),float(fedChannel));
	    break;}
	  case(37) : { // inv DCOL&PIX
	    hfedErrors9->Fill(float(FedId),float(fedChannel));
	    //spareCounts++;
	    cout<<" does it ever happen "<<endl;
	    break;}
	  }

	} else if ( errorType==39) { // CRC 
	  cout<<" CRC error "<<endl;
	  hfedsCRC->Fill(float(FedId));
	  hfed2d0->Fill(float(FedId),float(errorTypeMod));
	  htype->Fill(float(errorTypeMod));

	} else {
	  cout<<" unknown error "<<errorType<<endl;
	  hfedsUnknown->Fill(float(FedId));
	  errorTypeMod=24;	  
	  hfed2d0->Fill(float(FedId),float(errorTypeMod));
	  htype->Fill(float(errorTypeMod));
	}

	hfed2d->Fill(float(FedId),float(errorTypeMod));

	if(errorTypeMod>=0 && errorTypeMod<40) countErrors[errorTypeMod]++;

      } // for errors

    } else { // module errors 

      DetId detId(detid);
      //const GeomDetUnit      * geoUnit = geom->idToDetUnit( detId );
      //const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
      unsigned int detType=detId.det(); // det type, tracker=1
      unsigned int subid=detId.subdetId(); //subdetector type, barrel=1
      

      if(PRINT) {
	cout<<" Module errors "<<DSViter->data.size()<<endl;
	cout<<"Det: "<<detId.rawId()<<":";
        //cout<<"Det: "<<detId.rawId()<<" "<<detId.null()<<" "<<detType<<" "<<subid<<endl;
      }

      int layer=0, ladder=0, module=0, disk=0, blade=0, zindex=0;
      if(subid==1) { // bpix
	PXBDetId pdetId = PXBDetId(detid);
	// Barell layer = 1,2,3
	int layerC=pdetId.layer(); // unused 
	// Barrel ladder id 1-20,32,44.
	int ladderC=pdetId.ladder(); // unused 
	// Barrel Z-index=1,8
	int zindex=pdetId.module();
	// Convert to online 
	PixelBarrelName pbn(pdetId);
	int sector = pbn.sectorName();
	ladder = pbn.ladderName();
	layer  = pbn.layerName();
	module = pbn.moduleName();
	int half  = pbn.isHalfModule();
	PixelBarrelName::Shell shell = pbn.shell();

	if(PRINT) 
	  cout<<" BPix: layer "<<layer<<" ladder "<<ladder<<" module "<<module<<" shell "<<shell<<" sector "<<sector<<" "
	      <<layerC<<" "<<ladderC<<" "<<zindex<<" "<<half<<" "<<detType<<endl;

      } else if(subid==2) {  // fpix

	PXFDetId pdetId = PXFDetId(detid);
	disk=pdetId.disk(); //1,2,3
	blade=pdetId.blade(); //1-24
	zindex=pdetId.module(); //
	int side=pdetId.side(); //size=1 for -z, 2 for +z
	int panel=pdetId.panel(); //panel=1
	
	if(PRINT) cout<<" FPix: disk "<<disk<<" "<<blade<<" "<<side<<" "<<panel<<endl; 

      }

      // Look at FED errors now	
      edm::DetSet<SiPixelRawDataError>::const_iterator  di;
      for(di = DSViter->data.begin(); di != DSViter->data.end(); di++) {
	int FedId = di->getFedId();                  // FED the error came from
	int errorType = di->getType();               // type of error
	uint32_t word32 = di->getWord32();
	//uint64_t word64 = di->getWord64();  // unused 
	
	herrors->Fill(float(errorType));
	moduleErrors++;
	int errorTypeMod = errorType;

	//cout<<" fed " <<FedId<<" type "<<errorType<<" "<<hex<<word32<<" "<<word64<<dec<<" "<<di->getMessage()<<endl;
	if(PRINT) cout<<" fed " <<FedId<<" type "<<errorType<<endl;

	int status=0;
	int fedChannel = 0;
	//const bool printData    = true;

	if(errorType>=26 && errorType<=31) {  // fed error	
	  status = MyDecode::error(word32, fedChannel, FedId, PRINT);
	  if(PRINT) cout<<" fed "<<FedId<<" "<<"Channel "<<fedChannel<<" "<<status<<endl;

	  hfedErrors0->Fill(float(FedId),float(fedChannel));

	  switch(errorType) {

	  case(28) : { // FIFO
	    hfedErrors4->Fill(float(FedId),float(fedChannel));
	    hfed2d0->Fill(float(FedId),float(errorTypeMod));
	    htype->Fill(float(errorTypeMod));
	    break;}

	  case(29) : {  // TIMEOUT 
	    hfedErrors1->Fill(float(FedId),float(fedChannel));
	    hfed2d0->Fill(float(FedId),float(errorTypeMod));
	    htype->Fill(float(errorTypeMod));
	    break;}

	  case(30) : {  // TRAILER

	    int tbm = (word32& 0xff);
	    int mode = (word32>>8) & 0xf;
	    hmode->Fill(float(mode));
	    //cout<<" error 30 "<<hex<<mode<<dec<<endl;

	    if( tbm!=0 ) { // trailer
	      htbm->Fill(float(tbm));
	      hfedErrors6->Fill(float(FedId),float(fedChannel));
	      errorTypeMod=20;
	      hfed2d0->Fill(float(FedId),float(errorTypeMod));
	      htype->Fill(float(errorTypeMod));
	    }

	    if( (mode&0x8) != 0) { // nor
	      //cout<<" NOR "<<endl;
	      hfedErrors5->Fill(float(FedId),float(fedChannel));
	      errorTypeMod=21;
	      hfed2d0->Fill(float(FedId),float(errorTypeMod));
	      htype->Fill(float(errorTypeMod));
	    }

	    if( (mode&0x1) != 0) { // overflow
	      hfedErrors2->Fill(float(FedId),float(fedChannel));
	      errorTypeMod=22;
	      hfed2d0->Fill(float(FedId),float(errorTypeMod));
	      htype->Fill(float(errorTypeMod));
	    } 

	    if( (mode&0x6) != 0) { // fsm
	      hfedErrors7->Fill(float(FedId),float(fedChannel));
	      errorTypeMod=23;
	      hfed2d0->Fill(float(FedId),float(errorTypeMod));
	      htype->Fill(float(errorTypeMod));
	    } 

	    break;}

	  case(31) : {  // EVE
	    hfedErrors3->Fill(float(FedId),float(fedChannel));
	    hfed2d0->Fill(float(FedId),float(errorTypeMod));
	    htype->Fill(float(errorTypeMod));
	    break;}
	  } // end switch

	} else if (errorType>=32 && errorType<=34) {
	  cout<<" Slink error "<<endl;
	  hfedsSlink->Fill(float(FedId));
	  hfed2d0->Fill(float(FedId),float(errorTypeMod));
	  htype->Fill(float(errorTypeMod));

	} else if (errorType==25 || (errorType>=35 &&errorType<=38) ) { // conversion error

	  status = MyDecode::data(word32, fedChannel, FedId, PRINT);
	  if(PRINT) cout<<" fed "<<FedId<<" "<<"Channel "<<fedChannel<<" "<<status<<endl;

	  hfedErrors0->Fill(float(FedId),float(fedChannel));
	  hfed2d0->Fill(float(FedId),float(errorTypeMod));
	  htype->Fill(float(errorTypeMod));

	  switch(errorType) {
	  case(25) : { // inv ROC
	    hfedErrors8->Fill(float(FedId),float(fedChannel));
	    break;}
	  case(36) : { // inv ROC
	    hfedErrors8->Fill(float(FedId),float(fedChannel));
	    break;}
	  case(37) : { // inv DCOL&PIX
	    hfedErrors9->Fill(float(FedId),float(fedChannel));
	    if(FedId==6 && fedChannel==35) {
	      spareCounts++;
	      cout<<errorType <<" "<<FedId<<" "<<fedChannel;
	      if(subid==1) cout<<" BPix: layer "<<layer<<" ladder "<<ladder<<" module "<<module<<endl;
	      else         cout<<" FPix: disk "<<disk<<" "<<blade<<" "<<zindex<<endl; 
	    }
	    break;}
	  }

	} else if ( errorType==39) { // CRC 
	  cout<<" CRC error "<<endl;
	  hfedsCRC->Fill(float(FedId));
	  hfed2d0->Fill(float(FedId),float(errorTypeMod));
	  htype->Fill(float(errorTypeMod));

	} else {
	  cout<<" unknown error "<<errorType<<endl;
	  hfedsUnknown->Fill(float(FedId));
	  errorTypeMod=24;
	  hfed2d0->Fill(float(FedId),float(errorTypeMod));
	  htype->Fill(float(errorTypeMod));
	}

	hfed2d->Fill(float(FedId),float(errorTypeMod));

	if(errorTypeMod>=0 && errorTypeMod<40) countErrors2[errorTypeMod]++;

      } // for

    } // if fed/module 

  } // end det loop 

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FedErrorDumper);
