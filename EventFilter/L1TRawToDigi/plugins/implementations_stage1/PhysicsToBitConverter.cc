#include "PhysicsToBitConverter.h"


namespace l1t{
  PhysicsToBitConverter::PhysicsToBitConverter() {
    for (int m=0;m<2;m++){
      for (int n=0;n<6;n++){
        words32bitLink[m][n]=0;
      }
    }
    for (int m=0;m<2;m++){
      for (int n=0;n<192;n++){
        bitsLink[m][n]=0;
      }
    }
  }

  void PhysicsToBitConverter::Convert() {

    for (int iword=0;iword<6;iword++){
      for (int ibit=0;ibit<32;ibit++){
	//bitsLink[0].push_back(ReadBitInInt(ibit,words32bitLink[0][iword]));
	//bitsLink[1].push_back(ReadBitInInt(ibit,words32bitLink[1][iword]));
	bitsLink[0][ibit+iword*32]=ReadBitInInt(ibit,words32bitLink[0][iword]);
	bitsLink[1][ibit+iword*32]=ReadBitInInt(ibit,words32bitLink[1][iword]);

      }
    }
  }

  void PhysicsToBitConverter::Extract32bitwords() {
    //link,words
    
    for (int ilink=0;ilink<2;ilink++){
      for (int iword=0;iword<6;iword++){
        words32bitLink[ilink][iword]=BuildDecimalValue(iword*32,32,ilink);
      }
    }
  }


  int PhysicsToBitConverter::GetObject(rctDataBase::rctObjectType t, int firstindex, int secondindex)
  {
    int mystart = databaseobject.GetIndices(t,firstindex,secondindex);
    int mylength = databaseobject.GetLength(t);
    int mylink = databaseobject.GetLink(t);

    return BuildDecimalValue(mystart, mylength, mylink);
  } 
  
  void PhysicsToBitConverter::SetObject(rctDataBase::rctObjectType t, int value, int firstindex, int secondindex)
  { 
    int mystart = databaseobject.GetIndices(t,firstindex,secondindex);
    int mylength = databaseobject.GetLength(t);
    int mylink = databaseobject.GetLink(t);
    
    if(value>(pow(2,mylength)-1)) std::cout<<"The value you are trying to set has more bins than expected "<<std::endl;
    for (int i=0;i<mylength;i++) bitsLink[mylink][i+mystart]=(value>>i)&0x1;

  }


  int PhysicsToBitConverter::ReadBitInInt(int bit,int value){

    std::bitset<32> foo(value);
    return foo[bit];

  }

  int PhysicsToBitConverter::BuildDecimalValue(int firstbit,int bitlength,int linkid){

    int myvalue=0;
    int counter=0;

    for (int m=firstbit;m<firstbit+bitlength;m++){
      myvalue|=(bitsLink[linkid][m]&(0x1)) << counter;
      counter++;
    }
    return myvalue;
  }
}
