//---- Example Program for class CondCachedIter() ----
//----  Using a blob in the database ----

//-----------------------------------------------
//name of the database:
//sqlite_file:/afs/cern.ch/user/x/xiezhen/public/strip.db
//tag:
//noise_tag
//name Blob:
//DefaultBlobStreamingService
//-----------------------------------------------


void testCondIter_Blob(){


    std::cout <<std::endl<< "---- Test Program for CondCachedIter ----"<<std::endl;

    std::string NameDB;
    NameDB ="sqlite_file:/afs/cern.ch/user/x/xiezhen/public/strip.db";

    std::string TagData;
    TagData = "noise_tag";

    std::string nameBlob;
    nameBlob = "COND/Services/DefaultBlobStreamingService";
    
//---- I create the CondCachedIter<>
    CondCachedIter <mySiStripNoises> *Iterator = new CondCachedIter<mySiStripNoises>;
//---- CondCachedIter<> now works
    Iterator->create (NameDB,TagData,"","",nameBlob);
    
    const mySiStripNoises* reference;
    
    while(reference = Iterator->next()){
        std::cout << "Address of reference = " << reference << std::endl;  
    } 

}

