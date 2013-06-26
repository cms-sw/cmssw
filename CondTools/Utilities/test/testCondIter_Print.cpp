//---- Example Program for class CondCachedIter() ----
//----  how to dump a database ----
//simple program to create a dump of the content of a database
//the object "SiStripFedCampling" has a method called "print"

//-----------------------------------------------
//name of the database:
//sqlite_file:///afs/cern.ch/cms/data/CMSSW/HLTrigger/Configuration/SiStrip/152/sistripfedcabling.db
//tag:
//SiStripFedCabling_v1
//-----------------------------------------------

//-----------------------------------------------
//name of the database:
//frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_STRIP
//tag:
//CSA07_SiStripFedCabling
//-----------------------------------------------



void testCondIter_Print(){


    std::cout <<std::endl<< "---- Test Program for CondCachedIter ----"<<std::endl;

    std::string NameDB;
    NameDB ="sqlite_file:///afs/cern.ch/cms/data/CMSSW/HLTrigger/Configuration/SiStrip/152/sistripfedcabling.db";

    std::string TagData;
    TagData = "SiStripFedCabling_v1";

//---- I create the CondCachedIter<>
    CondCachedIter <SiStripFedCabling> *Iterator = new CondCachedIter<SiStripFedCabling>;
//---- CondCachedIter<> now works
    Iterator->create (NameDB,TagData);
    
    
    //----Second set of Data ----
    
    std::string NameDB_2;
    NameDB_2 ="frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_STRIP";

    std::string TagData_2;
    TagData_2 = "CSA07_SiStripFedCabling";

//---- I create the CondCachedIter<>
    CondCachedIter <SiStripFedCabling> *Iterator_2 = new CondCachedIter<SiStripFedCabling>;
//---- CondCachedIter<> now works
    Iterator_2->create (NameDB_2,TagData_2);

    
    //---- Dumping
    
    
    std::string NameFile = "DataA.txt";
    std::string NameFile2 = "DataB.txt";

    
    ofstream myfile;
    myfile.open (NameFile.c_str());
    const SiStripFedCabling* reference;
    reference = Iterator->next();
    std::stringstream Inside;
    reference->print(Inside);
//     std::cout << Inside.str();
    myfile<<Inside.str();  
    myfile.close();      
        
    
    myfile.open (NameFile2.c_str());
    const SiStripFedCabling* reference2;
    reference2 = Iterator_2->next();
    std::stringstream Inside2;
    reference2->print(Inside2);
    myfile<<Inside2.str();  
    myfile.close();      
    
    
      
}

