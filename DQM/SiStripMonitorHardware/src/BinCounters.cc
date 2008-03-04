#include "DQM/SiStripMonitorHardware/interface/BinCounters.h"

BinCounters::BinCounters()
	:f(32),
	number_(0),
	counter0(0),
        counter1(0),
        counter2(0),
        counter3(0),
        counter4(0),
        counter5(0),
        counter6(0),
        counter7(0),
        counter8(0),
        counter9(0),
        counter10(0),
        counter11(0),
        counter12(0),
        counter13(0),
        counter14(0),
        counter15(0),
        counter16(0),
        counter17(0),
        counter18(0),
        counter19(0),
        counter20(0),
        counter21(0),
        counter22(0),
        counter23(0),
        counter24(0),
        counter25(0),
        counter26(0),
        counter27(0),
        counter28(0),
        counter29(0),
        counter30(0),
        counter31(0)

{
//nada
}

BinCounters::~BinCounters(){
}


void BinCounters::setBinCounters(unsigned int number_){

        for(int i = (f-1); i>=0; i--){
                if( convert(number_, i) == 0 ){
	          if(i == 31)counter31++;
	          else if(i == 30)counter30++;
	          else if(i == 29)counter29++;
	          else if(i == 28)counter28++;
	          else if(i == 27)counter27++;
	          else if(i == 26)counter26++;
	          else if(i == 25)counter25++;
	          else if(i == 24)counter24++;
	          else if(i == 23)counter23++;
	          else if(i == 22)counter22++;
	          else if(i == 21)counter21++;
	          else if(i == 20)counter20++;
	          else if(i == 19)counter19++;
	          else if(i == 18)counter18++;
	          else if(i == 17)counter17++;
	          else if(i == 16)counter16++;
	          else if(i == 15)counter15++;
                  else if(i == 14)counter14++;
                  else if(i == 13)counter13++;
                  else if(i == 12)counter12++;
                  else if(i == 11)counter11++;
                  else if(i == 10)counter10++;
                  else if(i == 9)counter9++;
                  else if(i == 8)counter8++;
                  else if(i == 7)counter7++;
                  else if(i == 6)counter6++;
                  else if(i == 5)counter5++;
                  else if(i == 4)counter4++;
                  else if(i == 3)counter3++;
                  else if(i == 2)counter2++;
                  else if(i == 1)counter1++;
                  else if(i == 0)counter0++;
                }
        }

}

int BinCounters::getBinCounters(int counterNumber){

  if(counterNumber == 0)return counter0;
  else if(counterNumber == 1)return counter1;
  else if(counterNumber == 2)return counter2;
  else if(counterNumber == 3)return counter3;
  else if(counterNumber == 4)return counter4;
  else if(counterNumber == 5)return counter5;
  else if(counterNumber == 6)return counter6;
  else if(counterNumber == 7)return counter7;
  else if(counterNumber == 8)return counter8;
  else if(counterNumber == 9)return counter9;
  else if(counterNumber == 10)return counter10;
  else if(counterNumber == 11)return counter11;
  else if(counterNumber == 12)return counter12;
  else if(counterNumber == 13)return counter13;
  else if(counterNumber == 14)return counter14;
  else if(counterNumber == 15)return counter15;
  else if(counterNumber == 16)return counter16;
  else if(counterNumber == 17)return counter17;
  else if(counterNumber == 18)return counter18;
  else if(counterNumber == 19)return counter19;
  else if(counterNumber == 20)return counter20;
  else if(counterNumber == 21)return counter21;
  else if(counterNumber == 22)return counter22;
  else if(counterNumber == 23)return counter23;
  else if(counterNumber == 24)return counter24;
  else if(counterNumber == 25)return counter25;
  else if(counterNumber == 26)return counter26;
  else if(counterNumber == 27)return counter27;
  else if(counterNumber == 28)return counter28;
  else if(counterNumber == 29)return counter29;
  else if(counterNumber == 30)return counter30;
  else if(counterNumber == 31)return counter31;
  else return 0;
}

bool BinCounters::convert(unsigned int input, int bitval){

      return ( 0x1 & (input >> bitval));
}

int BinCounters::invert8(int input){

	int inverted = ( ( ((0x1 & (input >> 0))&0xFF)<<7) |
                	 ( ((0x1 & (input >> 1))&0xFF)<<6) |
                	 ( ((0x1 & (input >> 2))&0xFF)<<5) |
                	 ( ((0x1 & (input >> 3))&0xFF)<<4) |
                	 ( ((0x1 & (input >> 4))&0xFF)<<3) |
                	 ( ((0x1 & (input >> 5))&0xFF)<<2) |
               	 	 ( ((0x1 & (input >> 6))&0xFF)<<1) |
                	 ( ((0x1 & (input >> 7))&0xFF)<<0)   );

	return inverted;
}
