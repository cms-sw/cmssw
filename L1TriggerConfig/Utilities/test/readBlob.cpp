#include "CondFormats/Serialization/interface/Test.h"


int main(int argc, char *argv[]){

    if( argc != 3 ){
        std::cout<<"provide filename with the blob"<<std::endl;
        return 0;
    }

//    L1TriggerKeyStage1 deserializedObject;
    l1t::CaloParams deserializedObject1, deserializedObject2;

    std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);
    cond::serialization::InputArchive ia(ifs);
    std::cout << "Deserializing " << typeid(l1t::CaloParams).name() << " ..." << std::endl;
    ia >> deserializedObject1;


    std::ifstream ifs2(argv[2], std::ios::in | std::ios::binary);
    cond::serialization::InputArchive ia2(ifs2);
    std::cout << "Deserializing " << typeid(l1t::CaloParams).name() << " ..." << std::endl;
    ia2 >> deserializedObject2;

    return 0;
}
