#include "CondCore/ORA/interface/OId.h"
#include "CondCore/ORA/interface/Exception.h"
#include <iostream>

int main(){
  try {

    ora::OId oid0( 2540, 1234567890 );
    std::string soid = oid0.toString();
    ora::OId oid1;
    if( !oid1.isInvalid() ){
      ora::throwException("Empty OID has been found valid...","testOId"); 
    }
    oid1.fromString( soid );
    if( oid1!=oid0){
      ora::throwException("OID parsed is different from source.","testOId"); 
    } else {
      std::cout <<" OID source="<<oid0<<" parsed="<<oid1<<std::endl;
    }
    oid1.reset();
    if( !oid1.isInvalid() ){
      ora::throwException("Empty OID has been found valid after reset...","testOId"); 
    }
    if( !oid1.fromString("0001-00000000") ){
      std::cout <<"#### case (0) cannot be parsed..."<<std::endl;
    } else std::cout <<"case (0) read="<<oid1<<std::endl;
    oid1.reset();
    if( !oid1.fromString("EFGH-00000000") ){
      std::cout <<"#### case (1) cannot be parsed..."<<std::endl;
    } else std::cout <<"case (1) read="<<oid1<<std::endl;
    oid1.reset();
    if( !oid1.fromString("1234567890") ){
      std::cout <<"#### case (2) cannot be parsed..."<<std::endl;
    } else std::cout <<"case (2) read="<<oid1<<std::endl;
    oid1.reset();
    if( !oid1.fromString("-00000000") ){
      std::cout <<"#### case (3) cannot be parsed..."<<std::endl;
    } else std::cout <<"case (3) read="<<oid1<<std::endl;
    oid1.reset();
    if( !oid1.fromString("003-00000000") ){
      std::cout <<"#### case (4) cannot be parsed..."<<std::endl;
    } else std::cout <<"case (4) read="<<oid1<<std::endl;
    oid1.reset();
    if( !oid1.fromString("0000_1000000") ){
      std::cout <<"#### case (5) cannot be parsed..."<<std::endl;
    } else std::cout <<"case (5) read="<<oid1<<std::endl;
    oid1.reset();
    if( !oid1.fromString("0000-") ){
      std::cout <<"#### case (6) cannot be parsed..."<<std::endl;
    } else std::cout <<"case (6) read="<<oid1<<std::endl;
    if( !oid1.fromString("0005-000000250") ){
      std::cout <<"#### case (7) cannot be parsed..."<<std::endl;
    } else std::cout <<"case (7) read="<<oid1<<std::endl;
    oid1 = ora::OId( 50000, 1000000000 );
    std::cout <<"#### case (8)="<<oid1<<std::endl;
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
  }
}

