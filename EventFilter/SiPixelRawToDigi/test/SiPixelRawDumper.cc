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


// For luminisoty NOT IN RAW
//#include "FWCore/Framework/interface/LuminosityBlock.h"
//#include "DataFormats/Luminosity/interface/LumiSummary.h"
//#include "DataFormats/Common/interface/ConditionsInEdm.h"


// To use root histos
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// For ROOT
#include <TROOT.h>
//#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1D.h>
#include <TProfile.h>
#include <TProfile2D.h>


#include <iostream>
#include <fstream>

using namespace std;

// #define L1  // L1 information not in RAW
//#define OUTFILE 

namespace {
  bool printErrors  = true;
  bool printData    = false;
  bool printHeaders = false;
  const bool CHECK_PIXELS = true;
  const bool PRINT_BASELINE = false;
  // to store the previous pixel 
  int fed0 = -1, chan0 = -1, roc0 = -1, dcol0 = -1, pix0 =-1, count0=-1;
}

// Include the helper decoding class
/////////////////////////////////////////////////////////////////////////////
class MyDecode {
public:
  MyDecode() {}
  ~MyDecode() {}
  static int error(int error,int & fedChannel, int fed, int & stat1, int & stat2, bool print=false);
  static int data(int error, int & fedChannel, int fed, int & stat1, int & stat2, bool print=false);
  static int header(unsigned long long word64, int fed, bool print, unsigned int & bx);
  static int trailer(unsigned long long word64, int fed, bool print);
  static int convertToCol(int dcol,int pix);
  static int convertToRow(int pix);
  static int checkLayerLink(int fed, int chan);
private:
};
/////////////////////////////////////////////////////////////////////////////
//Returns 1,2,3 for layer 1,2,3 full modules, 11,12,13 for 1/2 modules
// 0 for fpix
// needs fedid 0-31, and channel 1-36.
int MyDecode::checkLayerLink(int fed, int chan) {
  int layer = 0;
  if(fed<0 || fed>31) return layer;  // return 0 for invalid of fpix

  if( chan>24) {   // layer 3

    if(fed==0 || fed==8 )  {  // Type A

      if(chan==28 ||chan==34 ||chan==35 ||chan==36 ) layer=13;  // 1/2 module
      else layer = 3;

    } else if(fed==23 || fed==31 )  {  // Type A

      if(chan==27 ||chan==31 ||chan==32 ||chan==33 ) layer=13;  // 1/2 module
      else layer = 3;
  
    } else if(fed==7 || fed==15 || fed==16 || fed==24 )  { // Type D

      if(chan==25 ||chan==26 ||chan==29 ||chan==30 ) layer=13;
      else layer = 3;

    }  else { layer = 3;}

    return layer; // layer 3

  } else if( (chan>=13 && chan<=19) || chan==24 ) {

    return 2; //layer 2

  } else {

    if(fed==0 || fed==8 || fed==16 || fed==24 )  {  // Type A

      if(chan==5 ||chan==6 ||chan==22 ||chan==23 ) layer=12; // 1/2 module 
      else if(chan==4 ||chan==10 ||chan==11 ||chan==12 ) layer=11; // 1/2 module 
      else layer = 1;
  
    } else if(fed==7 || fed==15 || fed==23 || fed==31 )  { // Type D

      if(chan==1 ||chan==2 ||chan==20 ||chan==21 ) layer=12; // 1/2
      else if(chan==3 ||chan==7 ||chan==8 ||chan==9 ) layer=11; // 1/2 
      else layer = 1;

    } else if(
              fed==1  || fed==2  || fed==3  ||   
              fed==9  || fed==10 || fed==11 ||   
              fed==17 || fed==18 || fed==19 ||   
              fed==25 || fed==26 || fed==27  )  { // Type B

      if( (chan>=4 && chan<=6) || (chan>=10 && chan<=12) || (chan>=22 && chan<=23) ) layer=2;
      else layer = 1;

    } else if(
              fed==4  || fed==5  || fed==6  ||   
              fed==12 || fed==13 || fed==14 ||   
              fed==20 || fed==21 || fed==22 ||   
              fed==28 || fed==29 || fed==30  )  { // Type C

      if( (chan>=1 && chan<=3) || (chan>=7 && chan<=9) || (chan>=20 && chan<=21) ) layer=2;
      else layer = 1;

    } else {
      cout<<"unknown fed "<<fed<<endl;
    } // if fed 

    return layer;

  }  // if chan
}

int MyDecode::convertToCol(int dcol, int pix) {
  // First find if we are in the first or 2nd col of a dcol.
  int colEvenOdd = pix%2;  // module(2), 0-1st sol, 1-2nd col.
  // Transform
  return (dcol * 2 + colEvenOdd); // col address, starts from 0

}
int MyDecode::convertToRow(int pix) {
  return abs( int(pix/2) - 80); // row addres, starts from 0
}

int MyDecode::header(unsigned long long word64, int fed, bool print, unsigned int & bx) {
  int fed_id=(word64>>8)&0xfff;
  int event_id=(word64>>32)&0xffffff;
  bx =(word64>>20)&0xfff;
//   if(bx!=101) {
//     cout<<" Header "<<" for FED "
// 	<<fed_id<<" event "<<event_id<<" bx "<<bx<<endl;
//     int dummy=0;
//     cout<<" : ";
//     cin>>dummy;
//   }
  if(print) cout<<" Header "<<" for FED "
		<<fed_id<<" event "<<event_id<<" bx "<<bx<<endl;
  fed0=-1; // reset the previous hit fed id
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
int MyDecode::error(int word, int & fedChannel, int fed, int & stat1, int & stat2, bool print) {
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

    unsigned int bit20 =      (word & 0x100000)>>20; // works only for slink format

    if(bit20 == 0) { // 2nd word 

      unsigned int timeoutCnt = (word &  0x7f800)>>11; // only for slink
      // unsigned int timeoutCnt = ((word&0xfc000000)>>24) + ((word&0x1800)>>11); // only for fifo
      // More than 1 channel within a group can have a timeout error
      // More than 1 channel within a group can have a timeout error

      unsigned int index = (word & 0x1F);  // index within a group of 4/5
      unsigned int chip = (word& BlkNumMask)>>8;
      int offset = offsets[chip];
      if(print) cout<<"Timeout Error- channel: ";
      //cout<<"Timeout Error- channel: ";
      for(int i=0;i<5;i++) {
	if( (index & 0x1) != 0) {
	  channel = offset + i + 1;
	  if(print) cout<<channel<<" ";
	  //cout<<channel<<" ";
	}
	index = index >> 1;
      }

      if(print) cout << " TimeoutCount: " << timeoutCnt;
      //cout << " TimeoutCount: " << timeoutCnt<<endl;;
     
     //if(print) cout<<" for Fed "<<fed<<endl;
     status = -10;
     fedChannel = channel;
     //end of timeout  chip and channel decoding

    } else {  // this is the 1st timout word with the baseline correction 
   
      int baselineCorr = 0;
      if(word&0x200){
	baselineCorr = -(((~word)&0x1ff) + 1);
      } else {
	baselineCorr = (word&0x1ff);
      }

      if(PRINT_BASELINE && print) cout<<"Timeout BaselineCorr: "<<baselineCorr<<endl;
      //cout<<"Timeout BaselineCorr: "<<baselineCorr<<endl;
      status = 0;
    }


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
    cout<<" Unknown error?"<<hex<<word<<dec<<endl;
  }

  if(print && status <0) cout<<" For FED "<<fed<<endl;
  return status;
}
///////////////////////////////////////////////////////////////////////////
int MyDecode::data(int word, int & fedChannel, int fed, int & stat1, int & stat2, bool print) {

  //const int ROCMAX = 24;
  const unsigned int plsmsk = 0xff;   // pulse height
  const unsigned int pxlmsk = 0xff00; // pixel index
  const unsigned int dclmsk = 0x1f0000;
  const unsigned int rocmsk = 0x3e00000;
  const unsigned int chnlmsk = 0xfc000000;
  int status = 0;

  int roc = ((word&rocmsk)>>21); // rocs start from 1
  // Check for embeded special words
  if(roc>0 && roc<25) {  // valid ROCs go from 1-24
    //if(print) cout<<"data "<<hex<<word<<dec;
    int channel = ((word&chnlmsk)>>26);

    if(channel>0 && channel<37) {  // valid channels 1-36
      //cout<<hex<<word<<dec;
      int dcol=(word&dclmsk)>>16;
      int pix=(word&pxlmsk)>>8;
      int adc=(word&plsmsk);
      fedChannel = channel;

      int col = convertToCol(dcol,pix);
      int row = convertToRow(pix);

      // print the roc number according to the online 0-15 scheme
      if(print) cout<<" Fed "<<fed<<" Channel- "<<channel<<" ROC- "<<(roc-1)<<" DCOL- "<<dcol<<" Pixel- "
		    <<pix<<" ("<<col<<","<<row<<") ADC- "<<adc<<endl;
      status++;

      if(CHECK_PIXELS) {

	// invalid roc number
	if( (fed>31 && roc>24) || (fed<=31 && roc>16)  ) {  //inv ROC
          if(printErrors) cout<<" Fed "<<fed<<" wrong roc number chan/roc/dcol/pix/adc = "<<channel<<"/"
			      <<roc-1<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	  status = -4;
	} else if( fed<=31 && channel<=24 && roc>8 ) {
	  // ptorotect for rerouted signals
	  if( !( (fed==13 && channel==17) ||(fed==15 && channel==5) ||(fed==31 && channel==10) 
		                          ||(fed==27 && channel==15) )  ) {
	    if(printErrors) cout<<" Fed "<<fed<<" wrong roc number, chan/roc/dcol/pix/adc = "<<channel<<"/"
				<<roc-1<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<endl;
	    status = -4;
	  }
	}

        if(pix==0) {  // PIX=0
	  // Detect pixel 0 events
          if(printErrors) 
	    cout<<" Fed "<<fed
		<<" pix=0 chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc-1<<"/"<<dcol<<"/"
		<<pix<<"/"<<adc<<" ("<<col<<","<<row<<")"<<endl;
	  count0++;
	  stat1 = roc-1;
	  stat2 = count0;
	  status = -5;

	} else if( fed==fed0 && channel==chan0 && roc==roc0 && dcol==dcol0 && pix==pix0 ) {
	  // detect multiple pixels 

	  count0++;
          if(printErrors) cout<<" Fed "<<fed
	      <<" double pixel  chan/roc/dcol/pix/adc = "<<channel<<"/"<<roc-1<<"/"<<dcol<<"/"
	      <<pix<<"/"<<adc<<" ("<<col<<","<<row<<") "<<count0<<endl;
	  stat1 = roc-1;
	  stat2 = count0;
	  status = -6;

	} else {  // normal

	  count0=0;
	  fed0 = fed; chan0 =channel; roc0 =roc; dcol0 =dcol; pix0=pix;

	  // Decode errors
	  if(pix<2 || pix>181) {  // inv PIX
	    if(printErrors)cout<<" Fed "<<fed<<" wrong pix number chan/roc/dcol/pix/adc = "<<channel<<"/"
			       <<roc-1<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<" ("<<col<<","<<row<<")"<<endl;
	    status = -3;
	  }
	  
	  if(dcol<0 || dcol>25) {  // inv DCOL
	    if(printErrors) cout<<" Fed "<<fed<<" wrong dcol number chan/roc/dcol/pix/adc = "<<channel<<"/"
				<<roc-1<<"/"<<dcol<<"/"<<pix<<"/"<<adc<<" ("<<col<<","<<row<<")"<<endl;
	    status = -3;
	  }


	}

      }
    } else {
      cout<<"Wrong channel "<<channel<<endl;
      return -2;
    }

  } else if(roc==25) {  // 
    unsigned int channel = ((word&chnlmsk)>>26);
    cout<<"Wrong roc 25 "<<" in fed/chan "<<fed<<"/"<<channel<<endl;
    status=-4;

  } else {  // error word

    //cout<<"error word "<<hex<<word<<dec;
    status=error(word, fedChannel, fed, stat1, stat2, print);

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
  int printLocal;
  int countEvents, countAllEvents;
  int countErrors;
  float sumPixels, sumFedSize, sumFedPixels[40];
  int fedErrors[40][36];
  int fedErrorsENE[40][36];
  int decodeErrors[40][36];
  int decodeErrors000[40][36];  // pix 0 problem
  int decodeErrorsDouble[40][36];  // double pix  problem
  int errorType[20];

#ifdef OUTFILE
  ofstream outfile;
#endif

  TH1D *hsize,*hsize0, *hsize1, *hsize2, *hsize3;
  TH1D *hsizeFeds[40];
  TH1D *hpixels, *hpixels0, *hpixels1, *hpixels2, *hpixels3, *hpixels4;
  TH1D *htotPixels,*htotPixels0, *htotPixels1;
  TH1D *herrors, *htotErrors, *herrorType, *herrorFed, *herrorChan;
  TH1D *herrorType0, *herrorFed0, *herrorChan0, *hcountDouble, *hcount000, *hrocDouble, *hroc000;
  TH2F *hfed2DErrors,*hfed2DErrors0,*hfed2DErrors1,*hfed2DErrors2,*hfed2DErrors3,*hfed2DErrors4,*hfed2DErrors5,
    *hfed2DErrors6,*hfed2DErrors7,*hfed2DErrors8,*hfed2DErrors9,*hfed2DErrors10,*hfed2DErrors11,*hfed2DErrors12;
  TH2F *hfed2d, *hsize2d,*hfedError1ls,*hfedError2ls, *hcountDouble2, *hcount0002;

  TH1D *hevent, *hlumi, *horbit, *hbx, *hlumi0, *hbx0;
  TH1D *hbx1,*hbx2,*hbx3,*hbx4,*hbx5,*hbx6,*hbx7,*hbx8,*hbx9,*hbx10,*hbx11,*hbx12; 
  TProfile *htotPixelsls, *hsizels, *herror1ls, *herror2ls, *havsizels, *htotPixelsbx, *herror1bx,*herror2bx,*havsizebx,*hsizep;
  TProfile2D *hfedchannelsize;
  TH1D *herrorTimels, *herrorOverls, *herrorTimels1, *herrorOverls1, *herrorTimels2, *herrorOverls2, 
    *herrorTimels3, *herrorOverls3, *herrorTimels0, *herrorOverls0;
  TH1D *hfedchannelsizeb,*hfedchannelsizeb1,*hfedchannelsizeb2,*hfedchannelsizeb3,
    *hfedchannelsizef;


};

void SiPixelRawDumper::endJob() {
  string errorName[18] = {" "," ","wrong channel","wrong pix or dcol","wrong roc","pix=0",
			  " double-pix"," "," "," ","timeout","ENE","NOR","FSM","overflow",
			  "trailer","fifo","reset/resync"};

// 2 - wrong channel
// 3 - wrong pix or dcol 
// 4 - wrong roc
// 5 - pix=0
// 6 - double pixel 
  // 10 - timeout ()
  // 11 - ene ()
  // 12 - mum pf rocs error ()
  // 13 - fsm ()
  // 14 - overflow ()
  // 15 - trailer ()
  // 16 - fifo  (30)
  // 17 - reset/resync NOT INCLUDED YET

  if(countEvents>0) {
    sumPixels /= float(countEvents);
    sumFedSize /= float(countAllEvents);
    for(int i=0;i<40;++i) {
      sumFedPixels[i] /= float(countEvents);
      hpixels4->Fill(float(i),sumFedPixels[i]); //pixels only 
    }
  }
    
  cout<<" Total/non-empty events " <<countAllEvents<<" / "<<countEvents<<" average number of pixels "<<sumPixels<<endl;

  cout<<" Average Fed size per event for all events (in 4-words) "<< (sumFedSize*2./40.) 
      <<" total for all feds "<<(sumFedSize*2.) <<endl;

  cout<<" Size for ech FED per event in units of hit pixels:" <<endl;

  for(int i=0;i<40;++i) cout<< sumFedPixels[i]<<" ";
  cout<<endl;

  cout<<" Total number of errors "<<countErrors<<endl;
  cout<<" FED erros "<<endl<<"Fed Channel Errors ENE-Errors"<<endl;
  for(int i=0;i<40;++i) {
    for(int j=0;j<36;++j) if( (fedErrors[i][j]) > (countEvents/10000) || (fedErrorsENE[i][j] > 0) ) {
      cout<<" "<<i<<"     "<<(j+1)<<"    "<<fedErrors[i][j]<<" "<<fedErrorsENE[i][j]<<endl;
    }
  }
  cout<<" Decode errors "<<endl<<"Fed Channel Errors Pix_000 Double_Pix"<<endl;
  for(int i=0;i<40;++i) {
    for(int j=0;j<36;++j) {
      int tmp = decodeErrors[i][j] + decodeErrors000[i][j] + decodeErrorsDouble[i][j];
      if(tmp>10) 
	cout<<" "<<i<<" -  "<<(j+1)<<"   -  "
	    <<decodeErrors[i][j]<<"  -    "
	    <<decodeErrors000[i][j]<<"  -   "
	    <<decodeErrorsDouble[i][j]<<endl;
    }
  }

  cout<<" Total errors for all feds "<<endl<<" Type Name Num-Of-Errors"<<endl;
  for(int i=0;i<20;++i) {
    if( errorType[i]>0 ) cout<<"   "<<i<<" - "<<errorName[i]<<" - "<<errorType[i]<<endl;
  }


#ifdef OUTFILE
  outfile.close();
#endif



}

void SiPixelRawDumper::beginJob() {

  printLocal = theConfig.getUntrackedParameter<int>("Verbosity",1);
  cout<<" beginjob "<<printLocal<<endl;

  if(printLocal>0) printErrors  = true;
  else printErrors = false;
  if(printLocal>1) printData  = true;
  else printData = false;
  if(printLocal>2) printHeaders  = true;
  else printHeaders = false;

  countEvents=0;
  countAllEvents=0;
  countErrors=0;
  sumPixels=0.;
  sumFedSize=0;
  for(int i=0;i<40;++i) {
    sumFedPixels[i]=0;
    for(int j=0;j<36;++j) {fedErrors[i][j]=0; fedErrorsENE[i][j]=0;}
    for(int j=0;j<36;++j) decodeErrors[i][j]=0;
    for(int j=0;j<36;++j) decodeErrors000[i][j]=0;
    for(int j=0;j<36;++j) decodeErrorsDouble[i][j]=0;
  }
  for(int i=0;i<20;++i) errorType[i]=0;

  edm::Service<TFileService> fs;

  //const float pixMax = 5999.5;   // pp value 
  //const float totMax = 99999.5;  // pp value 
  //const float maxLink = 200.;    // pp value

  const float pixMax = 19999.5;  // hi value 
  const float totMax = 399999.5; // hi value 
  const float maxLink = 1000.;   // hi value


  hsize = fs->make<TH1D>( "hsize", "FED event size in words-4", 6000, -0.5, pixMax);
  hsize0 = fs->make<TH1D>( "hsize0", "FED event size in words-4", 2000, -0.5, 19999.5);
  hsize1 = fs->make<TH1D>( "hsize1", "bpix FED event size in words-4", 6000, -0.5, pixMax);
  hsize2 = fs->make<TH1D>( "hsize2", "fpix FED event size in words-4", 6000, -0.5, pixMax);
  hsize3 = fs->make<TH1D>( "hsize3", "ave bpix FED event size in words-4", 6000, -0.5, pixMax);

  hpixels = fs->make<TH1D>( "hpixels", "pixels per FED", 2000, -0.5, 19999.5);
  hpixels0 = fs->make<TH1D>( "hpixels0", "pixels per FED", 6000, -0.5, pixMax);
  hpixels1 = fs->make<TH1D>( "hpixels1", "pixels >0 per FED", 6000, -0.5, pixMax);
  hpixels2 = fs->make<TH1D>( "hpixels2", "pixels >0 per BPix FED", 6000, -0.5, pixMax);
  hpixels3 = fs->make<TH1D>( "hpixels3", "pixels >0 per Fpix FED", 6000, -0.5, pixMax);
  hpixels4 = fs->make<TH1D>( "hpixels4", "pixels per each FED", 40, -0.5, 39.5);

  htotPixels = fs->make<TH1D>( "htotPixels", "pixels per event", 10000, -0.5, totMax);
  htotPixels0 = fs->make<TH1D>( "htotPixels0", "pixels per event, zoom low region", 20000, -0.5, 19999.5);
  htotPixels1 = fs->make<TH1D>( "htotPixels1", "pixels >0 per event", 10000, -0.5, totMax);

  herrors = fs->make<TH1D>( "herrors", "errors per FED", 100, -0.5, 99.5);
  htotErrors = fs->make<TH1D>( "htotErrors", "errors per event", 1000, -0.5, 999.5);
  herrorType0 = fs->make<TH1D>( "herrorType0", "all errors per type", 20, -0.5, 19.5);
  herrorFed0  = fs->make<TH1D>( "herrorFed0", "all errors per FED", 40, -0.5, 39.5);
  herrorChan0 = fs->make<TH1D>( "herrorChan0", "all errors per chan", 37, -0.5, 36.5);
  herrorType = fs->make<TH1D>( "herrorType", "readout errors per type", 20, -0.5, 19.5);
  herrorFed  = fs->make<TH1D>( "herrorFed", "readout errors per FED", 40, -0.5, 39.5);
  herrorChan = fs->make<TH1D>( "herrorChan", "readout errors per chan", 37, -0.5, 36.5);
  hcountDouble = fs->make<TH1D>( "hcountDouble", "count double pixels", 100, -0.5, 99.5);
  hcountDouble2 = fs->make<TH2F>("hcountDouble2","count double pixels",40,-0.5,39.5, 10,0.,10.); 
  hcount000 = fs->make<TH1D>( "hcount000", "count 000 pixels", 100, -0.5, 99.5);
  hcount0002 = fs->make<TH2F>("hcount0002","count 000 pixels",40,-0.5,39.5, 10,0.,10.); 
  hrocDouble = fs->make<TH1D>( "hrocDouble", "double pixels rocs", 25,-0.5, 24.5);
  hroc000    = fs->make<TH1D>( "hroc000", "000 pixels rocs", 25,-0.5, 24.5);

  hfed2d = fs->make<TH2F>( "hfed2d", "errors", 40,-0.5,39.5,21, -0.5, 20.5); // ALL

  hsize2d = fs->make<TH2F>( "hsize2d", "size vs fed",40,-0.5,39.5, 50,0,500); // ALL
  hsizep  = fs->make<TProfile>( "hsizep", "size vs fed",40,-0.5,39.5,0,100000); // ALL
  //hsize2dls = fs->make<TH2F>( "hsize2dls", "size vs lumi",100,0,1000, 50,0.,500.); // ALL

  hfed2DErrors = fs->make<TH2F>("hfed2DErrors", "errors per FED",   40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors0 = fs->make<TH2F>("hfed2DErrors0", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors1 = fs->make<TH2F>("hfed2DErrors1", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors2 = fs->make<TH2F>("hfed2DErrors2", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors3 = fs->make<TH2F>("hfed2DErrors3", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors4 = fs->make<TH2F>("hfed2DErrors4", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors5 = fs->make<TH2F>("hfed2DErrors5", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors6 = fs->make<TH2F>("hfed2DErrors6", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors7 = fs->make<TH2F>("hfed2DErrors7", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors8 = fs->make<TH2F>("hfed2DErrors8", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors9 = fs->make<TH2F>("hfed2DErrors9", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors10 = fs->make<TH2F>("hfed2DErrors10", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors11 = fs->make<TH2F>("hfed2DErrors11", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);
  hfed2DErrors12 = fs->make<TH2F>("hfed2DErrors12", "errors per FED", 40,-0.5,39.5,37, -0.5, 36.5);

  hsizeFeds[0] = fs->make<TH1D>( "hsizeFed0", "FED 0 event size ", 1000, -0.5, pixMax);
  hsizeFeds[1] = fs->make<TH1D>( "hsizeFed1", "FED 1 event size ", 1000, -0.5, pixMax);
  hsizeFeds[2] = fs->make<TH1D>( "hsizeFed2", "FED 2 event size ", 1000, -0.5, pixMax);
  hsizeFeds[3] = fs->make<TH1D>( "hsizeFed3", "FED 3 event size ", 1000, -0.5, pixMax);
  hsizeFeds[4] = fs->make<TH1D>( "hsizeFed4", "FED 4 event size ", 1000, -0.5, pixMax);
  hsizeFeds[5] = fs->make<TH1D>( "hsizeFed5", "FED 5 event size ", 1000, -0.5, pixMax);
  hsizeFeds[6] = fs->make<TH1D>( "hsizeFed6", "FED 6 event size ", 1000, -0.5, pixMax);
  hsizeFeds[7] = fs->make<TH1D>( "hsizeFed7", "FED 7 event size ", 1000, -0.5, pixMax);
  hsizeFeds[8] = fs->make<TH1D>( "hsizeFed8", "FED 8 event size ", 1000, -0.5, pixMax);
  hsizeFeds[9] = fs->make<TH1D>( "hsizeFed9", "FED 9 event size ", 1000, -0.5, pixMax);
  hsizeFeds[10] = fs->make<TH1D>( "hsizeFed10", "FED 10 event size ", 1000, -0.5, pixMax);
  hsizeFeds[11] = fs->make<TH1D>( "hsizeFed11", "FED 11 event size ", 1000, -0.5, pixMax);
  hsizeFeds[12] = fs->make<TH1D>( "hsizeFed12", "FED 12 event size ", 1000, -0.5, pixMax);
  hsizeFeds[13] = fs->make<TH1D>( "hsizeFed13", "FED 13 event size ", 1000, -0.5, pixMax);
  hsizeFeds[14] = fs->make<TH1D>( "hsizeFed14", "FED 14 event size ", 1000, -0.5, pixMax);
  hsizeFeds[15] = fs->make<TH1D>( "hsizeFed15", "FED 15 event size ", 1000, -0.5, pixMax);
  hsizeFeds[16] = fs->make<TH1D>( "hsizeFed16", "FED 16 event size ", 1000, -0.5, pixMax);
  hsizeFeds[17] = fs->make<TH1D>( "hsizeFed17", "FED 17 event size ", 1000, -0.5, pixMax);
  hsizeFeds[18] = fs->make<TH1D>( "hsizeFed18", "FED 18 event size ", 1000, -0.5, pixMax);
  hsizeFeds[19] = fs->make<TH1D>( "hsizeFed19", "FED 19 event size ", 1000, -0.5, pixMax);
  hsizeFeds[20] = fs->make<TH1D>( "hsizeFed20", "FED 20 event size ", 1000, -0.5, pixMax);
  hsizeFeds[21] = fs->make<TH1D>( "hsizeFed21", "FED 21 event size ", 1000, -0.5, pixMax);
  hsizeFeds[22] = fs->make<TH1D>( "hsizeFed22", "FED 22 event size ", 1000, -0.5, pixMax);
  hsizeFeds[23] = fs->make<TH1D>( "hsizeFed23", "FED 23 event size ", 1000, -0.5, pixMax);
  hsizeFeds[24] = fs->make<TH1D>( "hsizeFed24", "FED 24 event size ", 1000, -0.5, pixMax);
  hsizeFeds[25] = fs->make<TH1D>( "hsizeFed25", "FED 25 event size ", 1000, -0.5, pixMax);
  hsizeFeds[26] = fs->make<TH1D>( "hsizeFed26", "FED 26 event size ", 1000, -0.5, pixMax);
  hsizeFeds[27] = fs->make<TH1D>( "hsizeFed27", "FED 27 event size ", 1000, -0.5, pixMax);
  hsizeFeds[28] = fs->make<TH1D>( "hsizeFed28", "FED 28 event size ", 1000, -0.5, pixMax);
  hsizeFeds[29] = fs->make<TH1D>( "hsizeFed29", "FED 29 event size ", 1000, -0.5, pixMax);
  hsizeFeds[30] = fs->make<TH1D>( "hsizeFed30", "FED 30 event size ", 1000, -0.5, pixMax);
  hsizeFeds[31] = fs->make<TH1D>( "hsizeFed31", "FED 31 event size ", 1000, -0.5, pixMax);
  hsizeFeds[32] = fs->make<TH1D>( "hsizeFed32", "FED 32 event size ", 1000, -0.5, pixMax);
  hsizeFeds[33] = fs->make<TH1D>( "hsizeFed33", "FED 33 event size ", 1000, -0.5, pixMax);
  hsizeFeds[34] = fs->make<TH1D>( "hsizeFed34", "FED 34 event size ", 1000, -0.5, pixMax);
  hsizeFeds[35] = fs->make<TH1D>( "hsizeFed35", "FED 35 event size ", 1000, -0.5, pixMax);
  hsizeFeds[36] = fs->make<TH1D>( "hsizeFed36", "FED 36 event size ", 1000, -0.5, pixMax);
  hsizeFeds[37] = fs->make<TH1D>( "hsizeFed37", "FED 37 event size ", 1000, -0.5, pixMax);
  hsizeFeds[38] = fs->make<TH1D>( "hsizeFed38", "FED 38 event size ", 1000, -0.5, pixMax);
  hsizeFeds[39] = fs->make<TH1D>( "hsizeFed39", "FED 39 event size ", 1000, -0.5, pixMax);

  hfedError1ls = fs->make<TH2F>( "hfedError1ls", "errors vs lumi",300,0,3000, 40,-0.5,39.5); // 
  hfedError2ls = fs->make<TH2F>( "hfedError2ls", "errors vs lumi",300,0,3000, 40,-0.5,39.5); //

  hevent = fs->make<TH1D>("hevent","event",1000,0,10000000.);
  //horbit = fs->make<TH1D>("horbit","orbit",100, 0,100000000.);
  hlumi  = fs->make<TH1D>("hlumi", "lumi", 3000,0,3000.);
  hlumi0  = fs->make<TH1D>("hlumi0", "lumi", 3000,0,3000.);
 
  hbx    = fs->make<TH1D>("hbx",   "bx",   4000,0,4000.);  
  hbx0    = fs->make<TH1D>("hbx0",   "bx",   4000,0,4000.);  
  hbx1    = fs->make<TH1D>("hbx1",   "bx",   4000,0,4000.);  
  hbx2    = fs->make<TH1D>("hbx2",   "bx",   4000,0,4000.);  
  hbx3    = fs->make<TH1D>("hbx3",   "bx",   4000,0,4000.);  
  hbx4    = fs->make<TH1D>("hbx4",   "bx",   4000,0,4000.);  
  hbx5    = fs->make<TH1D>("hbx5",   "bx",   4000,0,4000.);  
  hbx6    = fs->make<TH1D>("hbx6",   "bx",   4000,0,4000.);  
  hbx7    = fs->make<TH1D>("hbx7",   "bx",   4000,0,4000.);  
  hbx8    = fs->make<TH1D>("hbx8",   "bx",   4000,0,4000.);  
  hbx9    = fs->make<TH1D>("hbx9",   "bx",   4000,0,4000.);  
  hbx10    = fs->make<TH1D>("hbx10",   "bx",   4000,0,4000.);  
  hbx11    = fs->make<TH1D>("hbx11",   "bx",   4000,0,4000.);  
  hbx12    = fs->make<TH1D>("hbx12",   "bx",   4000,0,4000.);  
  herrorTimels = fs->make<TH1D>("herrorTimels","timeouts vs ls",3000,0,3000);
  herrorOverls = fs->make<TH1D>("herrorOverls","overflows vs ls",3000,0,3000);
  herrorTimels1 = fs->make<TH1D>("herrorTimels1","timeouts vs ls",3000,0,3000);
  herrorOverls1 = fs->make<TH1D>("herrorOverls1","overflows vs ls",3000,0,3000);
  herrorTimels2 = fs->make<TH1D>("herrorTimels2","timeouts vs ls",3000,0,3000);
  herrorOverls2 = fs->make<TH1D>("herrorOverls2","overflows vs ls",3000,0,3000);
  herrorTimels3 = fs->make<TH1D>("herrorTimels3","timeouts vs ls",3000,0,3000);
  herrorOverls3 = fs->make<TH1D>("herrorOverls3","overflows vs ls",3000,0,3000);
  herrorTimels0 = fs->make<TH1D>("herrorTimels0","timeouts vs ls",3000,0,3000);
  herrorOverls0 = fs->make<TH1D>("herrorOverls0","overflows vs ls",3000,0,3000);


  hsizels = fs->make<TProfile>("hsizels"," fed size vs ls",300,0,3000,0,200000.);
  htotPixelsls = fs->make<TProfile>("htotPixelsls"," tot pixels vs ls",300,0,3000,0,300000.);
  havsizels = fs->make<TProfile>("havsizels","av. pixels vs ls",300,0,3000,0,300000.);
  herror1ls = fs->make<TProfile>("herror1ls","error type 1 vs ls",300,0,3000,0,1000.);
  herror2ls = fs->make<TProfile>("herror2ls","error type 2 vs ls",300,0,3000,0,1000.);
  
  htotPixelsbx = fs->make<TProfile>("htotPixelsbx"," tot pixels vs bx",4000,-0.5,3999.5,0,300000.);
  havsizebx = fs->make<TProfile>("havsizebx"," ave pixels vs bx",4000,-0.5,3999.5,0,300000.);
  herror1bx = fs->make<TProfile>("herror1bx"," tot pixels vs bx",4000,-0.5,3999.5,0,300000.);
  herror2bx = fs->make<TProfile>("herror2bx"," tot pixels vs bx",4000,-0.5,3999.5,0,300000.);

  //hintgl  = fs->make<TProfile>("hintgl", "inst lumi vs ls ",1000,0.,3000.,0.0,1000.);
  //hinstl  = fs->make<TProfile>("hinstl", "intg lumi vs ls ",1000,0.,3000.,0.0,10.);

  hfedchannelsize  = fs->make<TProfile2D>("hfedchannelsize", "pixels per fed/channel",40,-0.5,39.5,37,-0.5,36.5, 0.0,10000.);
  hfedchannelsizeb   = fs->make<TH1D>("hfedchannelsizeb", "pixels per bpix channel",200,0.0,maxLink);
  hfedchannelsizeb1  = fs->make<TH1D>("hfedchannelsizeb1", "pixels per bpix1 channel",200,0.0,maxLink);
  hfedchannelsizeb2  = fs->make<TH1D>("hfedchannelsizeb2", "pixels per bpix2 channel",200,0.0,maxLink);
  hfedchannelsizeb3  = fs->make<TH1D>("hfedchannelsizeb3", "pixels per bpix3 channel",200,0.0,maxLink);
  hfedchannelsizef   = fs->make<TH1D>("hfedchannelsizef", "pixels per fpix channel",200,0.0,maxLink);


#ifdef OUTFILE
  outfile.open("pixfed.csv");
  for(int i=0;i<40;++i) {if(i<39) outfile<<i<<","; else outfile<<i<<endl;}
#endif

}
//-----------------------------------------------------------------------
void SiPixelRawDumper::analyze(const  edm::Event& ev, const edm::EventSetup& es) {

  // Access event information
  //int run       = ev.id().run();
  int event     = ev.id().event();
  int lumiBlock = ev.luminosityBlock();
  int bx        = ev.bunchCrossing();
  //int orbit     = ev.orbitNumber();

  hevent->Fill(float(event));
  hlumi0->Fill(float(lumiBlock));
  hbx0->Fill(float(bx));
  //horbit->Fill(float(orbit));

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


  // Get lumi info (does not work for raw)
//  edm::LuminosityBlock const& iLumi = ev.getLuminosityBlock();
//   edm::Handle<LumiSummary> lumi;
//   iLumi.getByLabel("lumiProducer", lumi);
//   edm::Handle<edm::ConditionsInLumiBlock> cond;
//   float intlumi = 0, instlumi=0;
//   int beamint1=0, beamint2=0;
//   iLumi.getByLabel("conditionsInEdm", cond);
//   // This will only work when running on RECO until (if) they fix it in the FW
//   // When running on RAW and reconstructing, the LumiSummary will not appear
//   // in the event before reaching endLuminosityBlock(). Therefore, it is not
//   // possible to get this info in the event
//   if (lumi.isValid()) {
//     intlumi =(lumi->intgRecLumi())/1000.; // integrated lumi per LS in -pb
//     instlumi=(lumi->avgInsDelLumi())/1000.; //ave. inst lumi per LS in -pb
//     beamint1=(cond->totalIntensityBeam1)/1000;
//     beamint2=(cond->totalIntensityBeam2)/1000;
//   } else {
//     std::cout << "** ERROR: Event does not get lumi info\n";
//   }
//   cout<<instlumi<<" "<<intlumi<<" "<<lumiBlock<<endl;

//   hinstl->Fill(float(lumiBlock),float(instlumi));
//   hintgl->Fill(float(lumiBlock),float(intlumi));


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
  int countErrorsPerEvent1=0;
  int countErrorsPerEvent2=0;
  double aveFedSize = 0.;
  int stat1=-1, stat2=-1;
  int fedchannelsize[36];

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

    for(int i=0;i<36;++i) fedchannelsize[i]=0;

    int nWords = rawData.size()/sizeof(Word64);
    //cout<<" size "<<nWords<<endl;
    sumFedSize += float(nWords);    
    if(fedId<32) aveFedSize += double(2.*nWords);

    hsize->Fill(float(2*nWords)); // fed buffer size in words (32bit)
    hsize0->Fill(float(2*nWords)); // fed buffer size in words (32bit)
    if(fedId<32) hsize1->Fill(float(2*nWords)); // fed buffer size in words (32bit)
    else hsize2->Fill(float(2*nWords)); // fed buffer size in words (32bit)

    hsizeFeds[fedId]->Fill(float(2*nWords)); // size, includes errors and dummy words
    hsize2d->Fill(float(fedId),float(2*nWords));
    hsizep->Fill(float(fedId),float(2*nWords));
    if(fedId<32) hsizels->Fill(float(lumiBlock),float(2*nWords));


    // check headers
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); 
    //cout<<hex<<*header<<dec<<endl;
    unsigned int bxid = 0;
    eventId = MyDecode::header(*header, fedId, printHeaders, bxid);
    //if(fedId = fedIds.first) 
    if(bx != int(bxid) ) cout<<" Inconsistent BX: from event "<<bx<<" from FED "<<bxid<<endl;

    const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1);
    //cout<<hex<<*trailer<<dec<<endl;
    status = MyDecode::trailer(*trailer,fedId, printHeaders);

    int countPixelsInFed=0;
    int countErrorsInFed=0;
    int countErrorsInFed1=0;
    int countErrorsInFed2=0;
    int fedChannel = 0;

    // Loop over payload words
    for (const Word64* word = header+1; word != trailer; word++) {
      static const Word64 WORD32_mask  = 0xffffffff;

      for(int ipart=0;ipart<2;++ipart) {
	Word32 w = 0;
	if(ipart==0) {
	  w =  *word       & WORD32_mask;  // 1st word
	} else if(ipart==1) {
	  w =  *word >> 32 & WORD32_mask;  // 2nd word
	}

	status = MyDecode::data(w,fedChannel, fedId, stat1, stat2, printData);
	int layer = MyDecode::checkLayerLink(fedId, fedChannel); // get bpix layer 
	if(layer>10) layer = layer-10; // ignore 1/2 modules 
	if(status>0) {  // data
	  countPixels++;
	  countPixelsInFed++;
	  fedchannelsize[fedChannel-1]++;

	} else if(status<0) {  // error word
	  countErrorsInFed++;
	  //if( status == -6 || status == -5) 
	  cout<<" For FED "<<fedId<<" Event "<<eventId<<"/"
	  <<countAllEvents<<" chan "<<fedChannel<<" status "<<status<<endl;
	  status=abs(status);
	  // 2 - wrong channel
	  // 3 - wrong pix or dcol 
	  // 4 - wrong roc
	  // 5 - pix=0
	  // 6 - double pixel
	  // 10 - timeout ()
	  // 11 - ene ()
	  // 12 - mum pf rocs error ()
	  // 13 - fsm ()
	  // 14 - overflow ()
	  // 15 - trailer ()
	  // 16 - fifo  (30)
	  // 17 - reset/resync NOT INCLUDED YET
	  
	  switch(status) {
	  case(10) : { // Timeout
	    hfed2DErrors1->Fill(float(fedId),float(fedChannel));
	    herrorTimels->Fill(float(lumiBlock));
	    if(layer==1)      herrorTimels1->Fill(float(lumiBlock));
	    else if(layer==2) herrorTimels2->Fill(float(lumiBlock));
	    else if(layer==3) herrorTimels3->Fill(float(lumiBlock));
	    else if(layer==0) herrorTimels0->Fill(float(lumiBlock));

	    hbx1->Fill(float(bx));
	    break; } 
	  case(14) : {  // OVER
	    hfed2DErrors2->Fill(float(fedId),float(fedChannel));
	    herrorOverls->Fill(float(lumiBlock));
	    if(layer==1)      herrorOverls1->Fill(float(lumiBlock));
	    else if(layer==2) herrorOverls2->Fill(float(lumiBlock));
	    else if(layer==3) herrorOverls3->Fill(float(lumiBlock));
	    else if(layer==0) herrorOverls0->Fill(float(lumiBlock));
	    hbx2->Fill(float(bx));
	    break; }
	  case(11) : {  // ENE
	    hfed2DErrors3->Fill(float(fedId),float(fedChannel));
	    hbx3->Fill(float(bx));
	    fedErrorsENE[fedId][(fedChannel-1)]++;
	    break; }
	  case(16) : { //FIFO
	    hfed2DErrors4->Fill(float(fedId),float(fedChannel));
	    break; }
	  case(12) : {  // NOR
	    hfed2DErrors5->Fill(float(fedId),float(fedChannel));
	    hbx5->Fill(float(bx));
	    break; }
	  case(15) : {  // TBM Trailer
	    hfed2DErrors6->Fill(float(fedId),float(fedChannel));
	    break; }
	  case(13) : {  // FSM
	    hfed2DErrors7->Fill(float(fedId),float(fedChannel));
	    break; }
	  case(3) : {  //  inv. pix-dcol
	    hfed2DErrors8->Fill(float(fedId),float(fedChannel));
	    hbx8->Fill(float(bx));
	    break; }
	  case(4) : {  // inv roc
	    hfed2DErrors9->Fill(float(fedId),float(fedChannel));
	    hbx9->Fill(float(bx));
	    break; }
	  case(5) : {  // pix=0
	    hfed2DErrors10->Fill(float(fedId),float(fedChannel));
	    hbx10->Fill(float(bx));
	    hroc000->Fill(float(stat1)); // count rocs
	    hcount000->Fill(float(stat2));
	    hcount0002->Fill(float(fedId),float(stat2));
	    break; }
	  case(6) : {  // double pix
	    hfed2DErrors12->Fill(float(fedId),float(fedChannel));
	    hbx12->Fill(float(bx));
	    hrocDouble->Fill(float(stat1)); // count rocs
	    hcountDouble->Fill(float(stat2));
	    hcountDouble2->Fill(float(fedId),float(stat2));
	    break; }
	  case(1) : {  // unknown
	    hfed2DErrors11->Fill(float(fedId),float(fedChannel));
	    break; }
	  }
	  
	  if(status<20) errorType[status]++;
	  herrorType0->Fill(float(status));
	  herrorFed0->Fill(float(fedId));
	  herrorChan0->Fill(float(fedChannel));
	  hfed2DErrors0->Fill(float(fedId),float(fedChannel));
	  hfed2d->Fill(float(fedId),float(status));
	  
	  if(status>=10) {  // hard errors
	    countErrorsInFed1++;
	    fedErrors[fedId][(fedChannel-1)]++;
	    herrorType->Fill(float(status));
	    herrorFed->Fill(float(fedId));
	    herrorChan->Fill(float(fedChannel));
	    hfed2DErrors->Fill(float(fedId),float(fedChannel));
	    hfedError1ls->Fill(float(lumiBlock),float(fedId)); // hard errors

	  } else if(status>0) {  // decode errors
	    countErrorsInFed2++;
	    hfedError2ls->Fill(float(lumiBlock),float(fedId)); // decode errors
	    if(status==5)      decodeErrors000[fedId][(fedChannel-1)]++;
	    else if(status==6) decodeErrorsDouble[fedId][(fedChannel-1)]++;
	    else               decodeErrors[fedId][(fedChannel-1)]++;
	  }

	}
      } // for  1/2 word

    } // loop over longlong  words

    countErrors += countErrorsInFed;
    countErrorsPerEvent += countErrorsInFed;
    countErrorsPerEvent1 += countErrorsInFed1;
    countErrorsPerEvent2 += countErrorsInFed2;

    //convert data to digi (dummy for the moment)
    formatter.interpretRawData( dummyErrorBool, fedId, rawData, digis, errors);
    //cout<<dummyErrorBool<<" "<<digis.size()<<" "<<errors.size()<<endl;

    if(countPixelsInFed>0)  {
      sumFedPixels[fedId] += countPixelsInFed;
    }

    hpixels->Fill(float(countPixelsInFed));
    hpixels0->Fill(float(countPixelsInFed));
    if(countPixelsInFed>0) hpixels1->Fill(float(countPixelsInFed));
    if(countPixelsInFed>0 && fedId<32)  hpixels2->Fill(float(countPixelsInFed));
    if(countPixelsInFed>0 && fedId>=32) hpixels3->Fill(float(countPixelsInFed));
    herrors->Fill(float(countErrorsInFed));

    for(int i=0;i<36;++i) { 
      hfedchannelsize->Fill( float(fedId), float(i+1), float(fedchannelsize[i]) );
      if(fedId<32) {
	hfedchannelsizeb->Fill( float(fedchannelsize[i]) );
	int layer = MyDecode::checkLayerLink(fedId, i); // get bpix layer 
	if(layer>10) layer = layer-10; // ignore 1/2 modules 
	if(layer==3)      hfedchannelsizeb3->Fill( float(fedchannelsize[i]) );  // layer 3
	else if(layer==2) hfedchannelsizeb2->Fill( float(fedchannelsize[i]) );  // layer 2
	else if(layer==1) hfedchannelsizeb1->Fill( float(fedchannelsize[i]) );  // layer 1
	else cout<<" Cannot be "<<layer<<" "<<fedId<<" "<<i<<endl;
      } else         hfedchannelsizef->Fill( float(fedchannelsize[i]) );  // fpix
    }
    //    if(fedId == fedIds.first || countPixelsInFed>0 || countErrorsInFed>0 )  {
    //       eventId = MyDecode::header(*header, true);
    //       if(countPixelsInFed>0 || countErrorsInFed>0 ) cout<<"fed "<<fedId<<" pix "<<countPixelsInFed<<" err "<<countErrorsInFed<<endl;
    //       status = MyDecode::trailer(*trailer,true);
    //     }
    

#ifdef OUTFILE
    // print number of bytes per fed, CSV
    if(fedId == fedIds.second) outfile<<(nWords*8)<<endl;
    else                       outfile<<(nWords*8)<<",";  
#endif

  } // loop over feds

  htotPixels->Fill(float(countPixels));
  htotPixels0->Fill(float(countPixels));
  htotErrors->Fill(float(countErrorsPerEvent));

  htotPixelsls->Fill(float(lumiBlock),float(countPixels));
  htotPixelsbx->Fill(float(bx),float(countPixels));

  herror1ls->Fill(float(lumiBlock),float(countErrorsPerEvent1));
  herror2ls->Fill(float(lumiBlock),float(countErrorsPerEvent2));

  herror1bx->Fill(float(bx),float(countErrorsPerEvent1));
  herror2bx->Fill(float(bx),float(countErrorsPerEvent2));


  aveFedSize /= 32.;
  hsize3->Fill(aveFedSize);
  //hsize2dls->Fill(float(lumiBlock),aveFedSize);

  havsizels->Fill(float(lumiBlock),aveFedSize);
  havsizebx->Fill(float(bx),aveFedSize);

  if(countPixels>0) {
    hlumi->Fill(float(lumiBlock));
    hbx->Fill(float(bx));
    htotPixels1->Fill(float(countPixels));

    //cout<<"EVENT: "<<countEvents<<" "<<eventId<<" pixels "<<countPixels<<" errors "<<countErrors<<endl;
    sumPixels += countPixels;
    countEvents++;
    //int dummy=0;
    //cout<<" : ";
    //cin>>dummy;
  }  // end if

 
} // end analyze

// 2 - wrong channel
// 4 - wrong roc
// 3 - wrong pix or dcol 
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
