//
// Hex ascii reading works only with compiled code 
// call by .L dumpBinaryFile.C+
// and then   
// dumpBinaryFiles("PhysicsData1_366.dmp")
// Corrected for SLC4 7/07

#ifndef __CINT__
#include <iostream>
#include <fstream>
using namespace std;
#endif

// Include the helper decoding class 
/////////////////////////////////////////////////////////////////////////////
class MyDecode {
 public:
  MyDecode() {}
  ~MyDecode() {}
  static int error(int error);
  static int data(int error);
 private:
};
/////////////////////////////////////////////////////////////////////////////
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
  const unsigned int  eventNumMask = 0x1fe000; // event number mask
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
     //end of timeout  chip and channel decoding
      
   } else if( (word&errorMask) == eventNumError ) { // EVENT NUMBER ERROR
     unsigned int channel =  (word & channelMask) >>26;
     unsigned int tbm_event   =  (word & tbmEventMask);
      
     cout<<"Event Number Error- channel: "<<channel<<" tbm event nr. "
         <<tbm_event;
                                                                                            
   } else if( ((word&errorMask) == trailError)) {
     unsigned int channel =  (word & channelMask) >>26;
     unsigned int tbm_status   =  (word & tbmStatusMask);
     if(word & RocErrMask)
       cout<<"Number of Rocs Error- "<<"channel: "<<channel<<" ";
     if(word & FsmErrMask)
       cout<<"Finite State Machine Error- "<<"channel: "<<channel
           <<" Error status:0x"<<hex<< ((word & FsmErrMask)>>9)<<dec<<" ";;
     if(word & overflowMask)
       cout<<"Overflow Error- "<<"channel: "<<channel<<" ";
     if(!((word & RocErrMask)|(word & FsmErrMask)|(word & overflowMask)))
       cout<<"Trailer Error- ";
  
     cout<<"channel: "<<channel<<" TBM status:0x"<<hex<<tbm_status<<dec<<" ";
      
   } else if((word&errorMask)==fifoError) {
     if(word & Fif2NFMask) cout<<"A fifo 2 is Nearly full- ";
     if(word & TrigNFMask) cout<<"The trigger fifo is nearly Full - ";
     if(word & ChnFifMask) cout<<"fifo-1 is nearly full for channel"<<(word & ChnFifMask);
      
   } else {
     cout<<" Unknown error?";
   }
 
   unsigned int event   =  (word & eventNumMask) >>13;
   //unsigned int tbm_status   =  (word & tbmStatusMask);
    
   if(event>0) cout<<":event: "<<event;
   cout<<endl;

  return status;
}
///////////////////////////////////////////////////////////////////////////
int MyDecode::data(int word) {
  const unsigned int plsmsk = 0xff;   // pulse height
  const unsigned int pxlmsk = 0xff00; // pixel index
  const unsigned int dclmsk = 0x1f0000;
  const unsigned int rocmsk = 0x3e00000;
  const unsigned int chnlmsk = 0xfc000000;
  int status = -1;

  int roc = ((word&rocmsk)>>21);
  // Check for embeded special words
  if(roc>0 && roc<25) {  // valid ROCs go from 1-24
    cout<<"data "<<hex<<word<<dec;
    unsigned int channel = ((word&chnlmsk)>>26);
    if(channel>0 && channel<37) {  // valid channels 1-36
      //cout<<hex<<word<<dec;
      int dcol=(word&dclmsk)>>16;
      int pix=(word&pxlmsk)>>8;
      int adc=(word&plsmsk);
      cout<<" Channel- "<<channel<<" ROC- "<<roc<<" DCOL- "<<dcol<<" Pixel- "
	  <<pix<<" ADC- "<<adc<<endl;
      status=0;
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
int dumpBinaryFiles(char * filename) {    
  //char filename[80] = "../../../SCRATCH/PhysicsData1_366.dmp";  
  ifstream in_file;  // data file pointer
  in_file.open(filename, ios::binary|ios::in ); // 
  //cout<<in_file.bad()<<" "<<in_file.good()<<" "<<in_file<<endl;
  if (!in_file) {
    cout << " File not found " << endl;
    return(1); // signal error
  }


  const int numMax=1000000;
  int numEvent=0; 
  int status=0;
  unsigned long long word64=0;

  // Event loop 
  while (numEvent < numMax) {  // 

    in_file.read((char*)&word64,8);
    if (in_file.eof()) {
      std::cout << "End of input file" <<  std::endl;
      return false;
    }
   
    cout<<hex<<word64<<dec<<endl;

    if ((word64 >> 60) != 0x5){
      std::cout << "DATA CORRUPTION!" <<  std::endl;
      std::cout << "Expected to find header, but read: 0x"
		<<std::hex<<word64<<std::dec <<  std::endl;
      return false;
    }
 
    unsigned int fed_id=(word64>>8)&0xfff;
    unsigned int event_id=(word64>>32)&0xffffff;
    unsigned int bx_id=(word64>>20)&0xfff;
    cout<<" Header "<<hex<<word64<<dec<<" for FED "
	<<fed_id<<" "<<event_id<<" "<<bx_id<<endl;
    if(event_id==1) cout<<" EVENT 1"<<endl;

    numEvent++;
    int i=0;
    do {
      in_file.read((char*)&word64,8);
      //buffer.push_back(data);
      if( (word64 >> 60) != 0xa ) {
	int data1 = int(  word64     &0x00000000ffffffffL );
	int data2 = int( (word64>>32)&0x00000000ffffffffL );
	//cout<<i<<" "<<hex<<word64<<" "<<data1<<" "<<data2<<dec<<endl;
	status = MyDecode::data(data1);
	status = MyDecode::data(data2);	
	i++;
      }
    } while((word64 >> 60) != 0xa);

    // Slink Trailer
    int slinkLength = int( (word64>>32) & 0xffffff );
    int tts         = int( (word64&0xf0)>>4);
    int slinkError  = int( (word64&0xf00)>>8);
    cout<<" Trailer "<<hex<<word64<<dec<<" "<<slinkLength<<"/"<<(i+2)<<" "<<tts<<" "
	<<slinkError<<endl;
    if(slinkLength != (i+2) ) cout <<"Error in the event length = slink = "<<slinkLength
				   <<" count = "<<(i+2)<<endl; 
    // 
    //if(event_id==1) {
    int dummy=0;
    cout<<" Enter 0 to continue, -1 to stop "; 
    cin>>dummy;
    if(dummy==-1) break;
    //}

  } // end event loop

  in_file.close();
  //cout << " Close input file " << endl;
  return(0);
}




