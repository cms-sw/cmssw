//---- Example Program for class CondCachedIter() ----
//----  how to dump a database ----
//simple program to create a dump of the content of a database
//the object "SiStripFedCampling" has a method called "print"

//-----------------------------------------------
//name of the database:
//frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_STRIP
//tag:
//CSA07_SiStripFedCabling
//-----------------------------------------------


void testCondIter_PrintB(){


    std::cout <<std::endl<< "---- Test Program for CondCachedIter ----"<<std::endl;

    std::string NameDB;
    NameDB ="frontier://cmsfrontier.cern.ch:8000/FrontierDev/CMS_COND_STRIP";

    std::string TagData;
    TagData = "CSA07_SiStripFedCabling";

//---- I create the CondCachedIter<>
    CondCachedIter <SiStripFedCabling> *Iterator = new CondCachedIter<SiStripFedCabling>;
//---- CondCachedIter<> now works
    Iterator->create (NameDB,TagData);

   std::string NameFile = "DataB.txt";
   
    ofstream myfile;
    myfile.open (NameFile.c_str());
    const SiStripFedCabling* reference;
    reference = Iterator->next();
    std::stringstream Inside;
    reference->print(Inside);
    myfile<<Inside.str();  
    myfile.close();      

    
      
}

