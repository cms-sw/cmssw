//---- Example Program for class CondCachedIter() ----
//----  how to access database ----
//simple program to create a dump of the content of a database
//the object "SiStripFedCampling" has a method called "print"

//-----------------------------------------------
//name of the database:
//sqlite_file:///afs/cern.ch/cms/data/CMSSW/HLTrigger/Configuration/SiStrip/152/sistripfedcabling.db
//tag:
//SiStripFedCabling_v1
//-----------------------------------------------


void testCondIter_PrintA(){


    std::cout <<std::endl<< "---- Test Program for CondCachedIter ----"<<std::endl;

    std::string NameDB;
    NameDB ="sqlite_file:///afs/cern.ch/cms/data/CMSSW/HLTrigger/Configuration/SiStrip/152/sistripfedcabling.db";

    std::string TagData;
    TagData = "SiStripFedCabling_v1";

//---- I create the CondCachedIter<>
    CondCachedIter <SiStripFedCabling> *Iterator = new CondCachedIter<SiStripFedCabling>;
//---- CondCachedIter<> now works
    Iterator->create (NameDB,TagData);
    
 
    std::string NameFile = "DataA.txt";
    ofstream myfile;
    myfile.open (NameFile.c_str());
    const SiStripFedCabling* reference;
    reference = Iterator->next();
    std::stringstream Inside;
    reference->print(Inside);
    myfile<<Inside.str();  
    myfile.close();      
              
}

