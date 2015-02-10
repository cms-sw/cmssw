#include "PhysicsToBitConverter.h"


namespace l1t{
  PhysicsToBitConverter::PhysicsToBitConverter() {

  }

  void PhysicsToBitConverter::Convert() {

    for (int iword=0;iword<6;iword++){
      for (int ibit=0;ibit<32;ibit++){
	bitsLink[0].push_back(ReadBitInInt(ibit,words32bitLink[0][iword]));
	bitsLink[1].push_back(ReadBitInInt(ibit,words32bitLink[1][iword]));
      }
    }
  }

  int PhysicsToBitConverter::GetObject(rctDataBase::rctObjectType t, int firstindex, int secondindex)
  {
    int mystart = databaseobject.GetIndices(t,firstindex,secondindex);
    int mylength = databaseobject.GetLength(t);
    int mylink = databaseobject.GetLink(t);

    return BuildPhysicsValue(mystart, mylength, mylink);
  }

  int PhysicsToBitConverter::ReadBitInInt(int bit,int value){

    std::bitset<32> foo(value);
    return foo[bit];

  }

  int PhysicsToBitConverter::BuildPhysicsValue(int firstbit,int bitlength,int linkid){

    int myvalue=0;
    int counter=0;

    for (int m=firstbit;m<firstbit+bitlength;m++){
      myvalue|=(bitsLink[linkid][m]&(0x1)) << counter;
      counter++;
    }
    return myvalue;
  }
}
