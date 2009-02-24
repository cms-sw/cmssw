#include "L1Trigger/TextToDigi/src/SourceCardRouting.h"
#include <sstream>
#include <iostream>
#include <cstdlib>
using namespace std;

int main(void){

  unsigned short logicalCardID=0;
  unsigned short eventNumber=0;
  string tempString;

  unsigned long VHDCI[2][2]={{0}};
  SourceCardRouting::SourceCardRouting temp;

  unsigned short RC[18][7][2]={{{0}}};
  unsigned short RCof[18][7][2]={{{0}}};
  unsigned short RCtau[18][7][2]={{{0}}};
  unsigned short HF[18][4][2]={{{0}}};
  unsigned short HFQ[18][4][2]={{{0}}};

  unsigned short testRC[18][7][2]={{{0}}};
  unsigned short testRCof[18][7][2]={{{0}}};
  unsigned short testRCtau[18][7][2]={{{0}}};
  unsigned short testHF[18][4][2]={{{0}}};
  unsigned short testHFQ[18][4][2]={{{0}}};

  cout <<"fill with junk..."<<endl;
  for(int RCTcrate=0; RCTcrate<18; RCTcrate++){			
    for(int RCcard=0; RCcard<7; RCcard++){
      for(int Region=0; Region<2; Region++){
	RC[RCTcrate][RCcard][Region]=rand()&0x3ff;  //Fill with junk for the time being			
	RCof[RCTcrate][RCcard][Region]=rand()&0x1;  //Fill with junk for the time being			
	RCtau[RCTcrate][RCcard][Region]=rand()&0x1; //Fill with junk for the time being			
      }
    }
  }

  for(int RCTcrate=0; RCTcrate<18; RCTcrate++){			
    for(int HFeta=0; HFeta<4; HFeta++){
      for(int HFRegion=0; HFRegion<2; HFRegion++){
	HF[RCTcrate][HFeta][HFRegion]=rand()&0xff; //Fill with junk for the time being			
	HFQ[RCTcrate][HFeta][HFRegion]=rand()&0x1; //Fill with junk for the time being			
      }
    }
  }

  cout<<"now do the proper stuff..."<<endl;

  int RoutingMode=1;
  for(int RCTcrate=0; RCTcrate<18; RCTcrate++){			
    temp.RoutingModetoLogicalCardID(logicalCardID, RoutingMode, RCTcrate);
    temp.RC56HFtoSTRING(logicalCardID, eventNumber, RC[RCTcrate], RCof[RCTcrate], RCtau[RCTcrate], HF[RCTcrate], HFQ[RCTcrate], tempString);

		
    //now check its working ok
    //cout<<"MID: "<<tempString<<endl;
    temp.STRINGtoVHDCI(logicalCardID,eventNumber,tempString,VHDCI);
    temp.VHDCItoRC56HF(testRC[RCTcrate], testRCof[RCTcrate], testRCtau[RCTcrate], testHF[RCTcrate], testHFQ[RCTcrate], VHDCI);
    /*cout<<"------------------------------------"<<endl;
      for(int RCcard=5; RCcard<7; RCcard++){
      for(int Region=0; Region<2; Region++){
      cout << RC[RCTcrate][RCcard][Region] << '\t' << testRC[RCTcrate][RCcard][Region] << '\t';		
      cout <<	RCof[RCTcrate][RCcard][Region] << '\t' << testRCof[RCTcrate][RCcard][Region] << '\t';
      cout <<	RCtau[RCTcrate][RCcard][Region] << '\t' << testRCtau[RCTcrate][RCcard][Region] << endl;
      }
      }
      for(int HFeta=0; HFeta<4; HFeta++){
      for(int HFRegion=0; HFRegion<2; HFRegion++){
      cout << HF[RCTcrate][HFeta][HFRegion] << '\t' << testHF[RCTcrate][HFeta][HFRegion] << '\t';		
      cout <<	HFQ[RCTcrate][HFeta][HFRegion] << '\t' << testHFQ[RCTcrate][HFeta][HFRegion] << endl;
      }
      }*/ 
  }
  		

  RoutingMode=2;
  for(int RCTcrate=0; RCTcrate<18; RCTcrate++){			
    temp.RoutingModetoLogicalCardID(logicalCardID, RoutingMode, RCTcrate);
    temp.RC012toSTRING(logicalCardID, eventNumber, RC[RCTcrate], RCof[RCTcrate], RCtau[RCTcrate], tempString);

		
    //now check its working ok
    //cout<<"MID: "<<tempString<<endl;
    temp.STRINGtoVHDCI(logicalCardID,eventNumber,tempString,VHDCI);
    temp.VHDCItoRC012(testRC[RCTcrate], testRCof[RCTcrate], testRCtau[RCTcrate], VHDCI);
    /*cout<<"------------------------------------"<<endl;
      for(int RCcard=0; RCcard<7; RCcard++){
      for(int Region=0; Region<2; Region++){
      cout << RC[RCTcrate][RCcard][Region] << '\t' << testRC[RCTcrate][RCcard][Region] << '\t';		
      cout <<	RCof[RCTcrate][RCcard][Region] << '\t' << testRCof[RCTcrate][RCcard][Region] << '\t';
      cout <<	RCtau[RCTcrate][RCcard][Region] << '\t' << testRCtau[RCTcrate][RCcard][Region] << endl;
      }
      }*/

  }


  RoutingMode=3;
  for(int RCTcrate=0; RCTcrate<9; RCTcrate++){			
    temp.RoutingModetoLogicalCardID(logicalCardID, RoutingMode, RCTcrate);
    temp.RC234toSTRING(logicalCardID, eventNumber, RC[RCTcrate], RCof[RCTcrate], RCtau[RCTcrate], RC[RCTcrate+9], RCof[RCTcrate+9], RCtau[RCTcrate+9], tempString);

		
    //now check its working ok
    //cout<<"MID: "<<tempString<<endl;
    temp.STRINGtoVHDCI(logicalCardID,eventNumber,tempString,VHDCI);
    temp.VHDCItoRC234(testRC[RCTcrate], testRCof[RCTcrate], testRCtau[RCTcrate], testRC[RCTcrate+9], testRCof[RCTcrate+9], testRCtau[RCTcrate+9], VHDCI);
    /*cout<<"------------------------------------"<<endl;
      for(int RCcard=2; RCcard<5; RCcard++){
      for(int Region=0; Region<2; Region++){
      cout << RC[RCTcrate][RCcard][Region] << '\t' << testRC[RCTcrate][RCcard][Region] << '\t';		
      cout <<	RCof[RCTcrate][RCcard][Region] << '\t' << testRCof[RCTcrate][RCcard][Region] << '\t';
      cout <<	RCtau[RCTcrate][RCcard][Region] << '\t' << testRCtau[RCTcrate][RCcard][Region] << endl;
      cout << RC[RCTcrate+9][RCcard][Region] << '\t' << testRC[RCTcrate+9][RCcard][Region] << '\t';		
      cout <<	RCof[RCTcrate+9][RCcard][Region] << '\t' << testRCof[RCTcrate+9][RCcard][Region] << '\t';
      cout <<	RCtau[RCTcrate+9][RCcard][Region] << '\t' << testRCtau[RCTcrate+9][RCcard][Region] << endl;
      }
      }*/

  }

  cout<<"and now to check..."<<endl;
  for(int RCTcrate=0; RCTcrate<18; RCTcrate++){			
    cout<<"------------------ "<<RCTcrate<<" ------------------"<<endl;
    for(int RCcard=0; RCcard<7; RCcard++){
      for(int Region=0; Region<2; Region++){
	if ( (RC[RCTcrate][RCcard][Region]!=testRC[RCTcrate][RCcard][Region])||(RCof[RCTcrate][RCcard][Region]!=testRCof[RCTcrate][RCcard][Region])||(RCtau[RCTcrate][RCcard][Region]!=testRCtau[RCTcrate][RCcard][Region]) ){
	  cout << RC[RCTcrate][RCcard][Region] << '\t' << testRC[RCTcrate][RCcard][Region] << '\t';		
	  cout <<	RCof[RCTcrate][RCcard][Region] << '\t' << testRCof[RCTcrate][RCcard][Region] << '\t';
	  cout <<	RCtau[RCTcrate][RCcard][Region] << '\t' << testRCtau[RCTcrate][RCcard][Region] << endl;
	}
      }
    }
    for(int HFeta=0; HFeta<4; HFeta++){
      for(int HFRegion=0; HFRegion<2; HFRegion++){
	if ( (HF[RCTcrate][HFeta][HFRegion]!=testHF[RCTcrate][HFeta][HFRegion])||( HFQ[RCTcrate][HFeta][HFRegion]!=testHFQ[RCTcrate][HFeta][HFRegion]) ){
	  cout << HF[RCTcrate][HFeta][HFRegion] << '\t' << testHF[RCTcrate][HFeta][HFRegion] << '\t';		
	  cout <<	HFQ[RCTcrate][HFeta][HFRegion] << '\t' << testHFQ[RCTcrate][HFeta][HFRegion] << endl;
	} 
      }
    }


  }


}
