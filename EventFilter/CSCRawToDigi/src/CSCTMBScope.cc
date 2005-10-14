//_________________________________________________________
//
//  CSCTMBScope 9/11/03  B.Mohr                             
//  Unpacks TMB Logic Analyzer and stores in CSCTMBScope.h  
//_________________________________________________________
//
#include "EventFilter/CSCRawToDigi/interface/CSCTMBScope.h"
#include <iostream>

bool CSCTMBScope::debug = false;

CSCTMBScope::CSCTMBScope(unsigned short *buf,int b05Line,int e05Line) {

  size_ = UnpackScope(buf,b05Line,e05Line);

} //CSCTMBScope


int CSCTMBScope::UnpackScope(unsigned short *buf,int b05Line,int e05Line) {

  int pretrig_chan[4]={0,0,0,0};
  unsigned int tbin_strt,tbin_stop;
  unsigned int ibit,jbit,itbin,ich,iram,iline,iadr;
  unsigned int lct_bxn;


  if(debug) {
    std::cout << " ................................................" << std::endl;
    std::cout << " .....TMBHeader -- unpacking Logic Analyzer......" << std::endl;
  }

  if((e05Line-b05Line) == 1537) {

    if(debug) std::cout << "Scope data found" << std::endl;

    //load scope_ram from raw-hits format readout
    iline = b05Line + 1;
    for(iram=0;iram<6;iram++){
      for(iadr=0;iadr<256;iadr++) {
	itbin = iadr;   //ram_sel*256
	scope_ram[itbin][iram] = buf[iline];
	iline++;
      }
    }

    for(ich=0;ich<51;ich++) data[ich]=0;  //clear all data


    //----- find pretrig chan offset -----------------
    pretrig_chan[0] = GetPretrig(0);
    pretrig_chan[1] = GetPretrig(32);
    pretrig_chan[2] = GetPretrig(48);
    pretrig_chan[3] = GetPretrig(64);

    //----- ram 1 ------------------------------------
    tbin_strt = pretrig_chan[0]-7;
    tbin_stop = pretrig_chan[0]+24;

    for(ich=0;ich<=14;ich++) {
      iram=ich/16;
      jbit=0;
      for(itbin=tbin_strt;itbin<=tbin_stop;itbin++) {
	ibit = (scope_ram[itbin][iram] >> (ich%16) ) & 1;
	data[ich] = (ibit << jbit) | data[ich];
	jbit++;
      }
    }

    //----- ram 2 ------------------------------------
    for(ich=16;ich<=30;ich++) {
      iram=ich/16;
      jbit=0;
      for(itbin=tbin_strt;itbin<=tbin_stop;itbin++) {
	ibit = (scope_ram[itbin][iram] >> (ich%16) ) & 1;
	data[ich-1] = (ibit << jbit) | data[ich-1];
	jbit++;
      }
    }

    //----- ram 3 ------------------------------------
    tbin_strt = pretrig_chan[1]-7;
    tbin_stop = pretrig_chan[1]+24;

    for(ich=32;ich<=36;ich++) {
      iram=ich/16;
      jbit=0;
      for(itbin=tbin_strt;itbin<=tbin_stop;itbin++) {
	ibit = (scope_ram[itbin][iram] >> (ich%16) ) & 1;
	data[ich-2] = (ibit << jbit) | data[ich-2];
	jbit++;
      }
    }

    tbin_strt = pretrig_chan[1]-7+120;
    tbin_stop = pretrig_chan[1]+24+120;

    for(ich=37;ich<=40;ich++) {
      iram=ich/16;
      jbit=0;
      for(itbin=tbin_strt;itbin<=tbin_stop;itbin++) {
	ibit = (scope_ram[itbin][iram] >> (ich%16) ) & 1;
	data[ich-2] = (ibit << jbit) | data[ich-2];
	jbit++;
      }
    }

    tbin_strt = pretrig_chan[1]-7;
    tbin_stop = pretrig_chan[1]+24;

    for(ich=41;ich<=46;ich++) {
      iram=ich/16;
      jbit=0;
      for(itbin=tbin_strt;itbin<=tbin_stop;itbin++) {
	ibit = (scope_ram[itbin][iram] >> (ich%16) ) & 1;
	data[ich-2] = (ibit << jbit) | data[ich-2];
	jbit++;
      }
    }

    //----- ram 4 ------------------------------------
    tbin_strt = pretrig_chan[2]-7;
    tbin_stop = pretrig_chan[2]+24;

    for(ich=48;ich<=53;ich++) {
      iram=ich/16;
      jbit=0;
      for(itbin=tbin_strt;itbin<=tbin_stop;itbin++) {
	ibit = (scope_ram[itbin][iram] >> (ich%16) ) & 1;
	data[ich-3] = (ibit << jbit) | data[ich-3];
	jbit++;
      }
    }

    //----- ram 5 - bxn ------------------------------
    lct_bxn = 0;
    jbit=0;

    for(ich=65;ich<=76;ich++) {
      iram=ich/16;
      itbin = pretrig_chan[3];
      ibit = (scope_ram[itbin][iram] >> (ich%16) ) & 1;
      lct_bxn = (ibit << jbit) | lct_bxn;
      jbit++;
    }
    data[51]=lct_bxn;

    if(debug) std::cout << "Scope bxn at LCT (seq_pretrig): " 
			<< lct_bxn << std::endl;

    //----- now read back decoded scope data ---------
    if(debug) {
      std::cout << "\n" << std::endl;
      for(ich=0;ich<=50;ich++) {
	for(itbin=0;itbin<32;itbin++) {
	  ibit = (data[ich] >> itbin ) & 1;
	  if(ibit == 0) std::cout << "_";            //display symbol for logic 0
	  if(ibit == 1) std::cout << "-";            //display symbol for logic 1
	}
	std::cout << std::endl;
      }
      std::cout << "\n" << std::endl;
    }

  } //end if(b05-e05)


  //-------------- if no scope data: fill everything with 0 --------------------
  else {
    for(ich=0;ich<51;ich++) data[ich]=0;
    lct_bxn  = 0xff0000;    //value not possible for real data (short)
    data[51] = lct_bxn;

    if(debug) std::cout << "No scope data found: wrdcnt: " 
			<< (e05Line-b05Line) << std::endl;
  }


  if(debug) {
    std::cout << " .....END -- unpacking Logic Analyzer..........." << std::endl;
    std::cout << " ..............................................." << std::endl;
  }

  return (e05Line - b05Line + 1);

} //UnpackScope


int CSCTMBScope::GetPretrig(int ich) {

  unsigned int ibit,itbin,iram;
  int value = 0;

  ibit=0;
  itbin=0;
  iram=ich/16;
  while(!ibit) {
    ibit = (scope_ram[itbin][iram] >> (ich%16) ) & 1;
    value = itbin;
    itbin++;
  }

  if(debug) std::cout << "TMB SCOPE: ------- Pretrig value: " << value << std::endl;
  return value;

} //GetPretrig
