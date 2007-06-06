//
// Hex ascii reading works only with compiled code 
// call by .L dumpBinaryFile.C+
// and then   
// dumpBinaryFiles("PhysicsData1_366.dmp")
// 

#ifndef __CINT__
#include <iostream>
#include <fstream>
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
  const unsigned long  errorMask      = 0x3e00000;
  const unsigned long  dummyMask      = 0x03600000;
  const unsigned long  gapMask        = 0x03400000;
  const unsigned long  timeOut        = 0x3a00000;
  const unsigned long  eventNumError  = 0x3e00000;
  const unsigned long  trailError     = 0x3c00000;
  const unsigned long  fifoError      = 0x3800000;
 
//  const unsigned long  timeOutChannelMask = 0x1f;  // channel mask for timeouts
  const unsigned long  eventNumMask = 0x1fe000; // event number mask
  const unsigned long  channelMask = 0xfc000000; // channel num mask
  const unsigned long  tbmEventMask = 0xff;    // tbm event num mask
  const unsigned long  overflowMask = 0x100;   // data overflow
  const unsigned long  tbmStatusMask = 0xff;   //TBM trailer info
  const unsigned long  BlkNumMask = 0x700;   //pointer to error fifo #
  const unsigned long  FsmErrMask = 0x600;   //pointer to FSM errors
  const unsigned long  RocErrMask = 0x800;   //pointer to #Roc errors
  const unsigned long  ChnFifMask = 0x1f;   //channel mask for fifo error
  const unsigned long  Fif2NFMask = 0x40;   //mask for fifo2 NF
  const unsigned long  TrigNFMask = 0x80;   //mask for trigger fifo NF
 
 const int offsets[8] = {0,4,9,13,18,22,27,31};
 
 //cout<<"error word "<<hex<<word<<dec<<endl;                                                                                  
  if( (word&errorMask) == dummyMask ) { // DUMMY WORD
   cout<<" Dummy word";
  } else if( (word&errorMask) == gapMask ) { // GAP WORD
    cout<<" Gap word";
  } else if( (word&errorMask)==timeOut ) { // TIMEOUT
     // More than 1 channel within a group can have a timeout error
     unsigned long index = (word & 0x1F);  // index within a group of 4/5
     unsigned long chip = (word& BlkNumMask)>>8;
     int offset = offsets[chip];
     cout<<"Timeout Error- channels: ";
     for(int i=0;i<5;i++) {
       if( (index & 0x1) != 0) {
         int channel = offset + i + 1;
         cout<<channel<<" ";
       }
       index = index >> 1;
     }
     //end of timeout  chip and channel decoding
      
   } else if( (word&errorMask) == eventNumError ) { // EVENT NUMBER ERROR
     unsigned long channel =  (word & channelMask) >>26;
     unsigned long tbm_event   =  (word & tbmEventMask);
      
     cout<<"Event Number Error- channel: "<<channel<<" tbm event nr. "
         <<tbm_event;
                                                                                            
   } else if( ((word&errorMask) == trailError)) {
     unsigned long channel =  (word & channelMask) >>26;
     unsigned long tbm_status   =  (word & tbmStatusMask);
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
 
   unsigned long event   =  (word & eventNumMask) >>13;
   //unsigned long tbm_status   =  (word & tbmStatusMask);
    
   if(event>0) cout<<":event: "<<event;
   cout<<endl;

  return status;
}
///////////////////////////////////////////////////////////////////////////
int MyDecode::data(int word) {
  const unsigned long int plsmsk = 0xff;   // pulse height
  const unsigned long int pxlmsk = 0xff00; // pixel index
  const unsigned long int dclmsk = 0x1f0000;
  const unsigned long int rocmsk = 0x3e00000;
  const unsigned long int chnlmsk = 0xfc000000;
  int status = -1;

  int roc = ((word&rocmsk)>>21);
  // Check for embeded special words
  if(roc>0 && roc<25) {  // valid ROCs go from 1-24
    cout<<"data "<<hex<<word<<dec;
    int channel = ((word&chnlmsk)>>26);
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
    cout<<"error word "<<hex<<word<<dec;
    status=error(word);
  }

  return status;
}
////////////////////////////////////////////////////////////////////////////
int dumpBinaryFiles(char * filename) {    
  //char filename[80] = "../../../SCRATCH/PhysicsData1_366.dmp";  
  ifstream in_file;  // data file pointer
  in_file.open(filename, ios::binary|ios::in ); // 
  if (in_file.bad()) {
    cout << " File not found " << endl;
    return(1); // signal error
  }

  const int numMax=100;
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
   
    if ((word64 >> 60) != 0x5){
      std::cout << "DATA CORRUPTION!" <<  std::endl;
      std::cout << "Expected to find header, but read: 0x"
		<<std::hex<<word64<<std::dec <<  std::endl;
      return false;
    }
 
    unsigned int fed_id=(word64>>8)&0xfff;
    cout<<" Header "<<hex<<word64<<dec<<" for FED "<<fed_id<<endl;

    numEvent++;
    int i=0;
    do {
      in_file.read((char*)&word64,8);
      //buffer.push_back(data);
      if( (word64 >> 60) != 0xa ) {
	int data1 = (word64&0x00000000ffffffff);
	int data2 = (word64&0xffffffff00000000)>>32;
	//cout<<i<<" "<<hex<<word64<<" "<<data1<<" "<<data2<<dec<<endl;
	status = MyDecode::data(data1);
	status = MyDecode::data(data2);
	
	i++;
      }
    } while((word64 >> 60) != 0xa);
    cout<<" Trailer "<<hex<<word64<<dec<<endl;
    
    // 
    int dummy=0;
    cout<<" Enter 0 to continue, -1 to stop "; 
    cin>>dummy;
    if(dummy==-1) break;

  } // end event loop

  in_file.close();
  //cout << " Close input file " << endl;
  return(0);
}




