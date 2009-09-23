#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/Utilities/interface/CondBasicIter.h"
#include <iostream>
CondBasicIter::CondBasicIter():ioviterator(0),mysession(),myconnection(){}

CondBasicIter::~CondBasicIter(){
    if (mysession.isOpen()) {
        mysession.transaction().commit();
    }
    if (ioviterator) delete ioviterator;
    if(myconnection) delete myconnection;
}

void CondBasicIter::setStartTime(unsigned int start){
    m_startTime = start;
}

void CondBasicIter::setStopTime(unsigned int stop){
    m_stopTime = stop;
}

void CondBasicIter::setTime(unsigned int time){
    m_time = time;
}


void CondBasicIter::create(const std::string & NameDB,const std::string & File,const std::string & User,const std::string & Pass,const std::string & nameBlob){


    
  std::string Command1;
    Command1 = NameDB;
    //You need to write all the sintax like "oracle://cms_orcoff_int2r/SOMETHING"
    std::string Command4 = " -t ";
    std::string Command5 = File;
    std::string Command6 = " -u " + User;
    std::string Command7 = " -p " + Pass;


    std::cout << "Instructions " << Command1 << Command4 << Command5 << Command6 << Command7 << std::endl;   
    std::cout << "Blob name = " << nameBlob << std::endl; 
    
    std::string tag;
    std::string connect;
    std::string user = User;
    std::string pass = Pass;
  
    connect=Command1;
  
    tag=Command5;

    if(myconnection) delete myconnection;
    myconnection = new cond::DbConnection();
    myconnection->configuration().setMessageLevel( coral::Error );
    //session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut( 600 );
    //session->configuration().connectionConfiguration()->enableConnectionSharing();
    //session->configuration().connectionConfiguration()->enableReadOnlySessionOnUpdateConnections();
    std::string userenv(std::string("CORAL_AUTH_USER=")+user);
    std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);

    if (!mysession.isOpen()) {
        putenv(const_cast<char*>(userenv.c_str()));
        putenv(const_cast<char*>(passenv.c_str()));
    }
    myconnection->configure();
    mysession = myconnection->createSession();
    if (nameBlob.size()) {
      mysession.setBlobStreamingService(nameBlob.c_str());
    }

    mysession.open( connect );
    
    cond::MetaData metadata_svc(mysession);
    std::string token;
    mysession.transaction().start(true);
    
    token=metadata_svc.getToken(tag);
    mysession.transaction().commit();
    //int test = 0;
    //if (!p) {
    //  pooldb = &(myconnection->poolTransaction());
    //test = 1;
    //}
    
    // timetype irrelevant
    cond::IOVService iovservice(mysession);
    //-----------------------------------------
    if (ioviterator) {
      delete ioviterator;
      ioviterator = 0;
    }
    //-----------------------------------------
    
    ioviterator=iovservice.newIOVIterator(token);
    
      
    //if (test==1){
    //  pooldb->start(true);
    //}

    // printing out the iteratot size 
    std::cout<< " ioviterator->size() == " << ioviterator->size() << std::endl; 
   payloadContainer=iovservice.payloadContainerName(token); 



    
    //---- inizializer
    iter_Min = 0;
    iter_Max = 0; 
       
}

void CondBasicIter::setRange(unsigned int min,unsigned int max){
  if (min<max) {
    iter_Min = (unsigned int) min;
    iter_Max = (unsigned int) max;
    std::cout << "min = " << iter_Min << " and max = " << iter_Max << std::endl;
  }
  else std::cout << "Not possible: Minimum > Maximum" <<std::endl;
}

void CondBasicIter::setMin(unsigned int min){
  if (((unsigned int) min)>iter_Max) std::cout << "Not possible: Minimum > Maximum";
  else iter_Min = (unsigned int) min;
  std::cout << "min = " << iter_Min << " and max = " << iter_Max<< std::endl;
  
}

void CondBasicIter::setMax(unsigned int max){
  if (((unsigned int) max)>=iter_Min) iter_Max = (unsigned int) max;
  else std::cout << "Not possible: Maximum < Minimum";

  std::cout << "min = " << iter_Min << " and max = " << iter_Max<< std::endl;
  
}

void CondBasicIter::setRange(int min,int max){
  try{ 
    if (min<max) {
      iter_Min = (unsigned int) min;
      iter_Max = (unsigned int) max;
      std::cout << "min = " << iter_Min << " and max = " << iter_Max << std::endl;
    }
    else throw 1;
  }catch(int) {
    throw cond::Exception("Not possible: Minimum > Maximum");
  }
}

void CondBasicIter::setMin(int min){
  if (((unsigned int) min)>iter_Max) std::cout << "Not possible: Minimum > Maximum";
  else iter_Min = (unsigned int) min;
  std::cout << "min = " << iter_Min << " and max = " << iter_Max<< std::endl;  
}

void CondBasicIter::setMax(int max){
  if (((unsigned int) max)<iter_Min) std::cout << "Not possible: Maximum < Minimum";
  else iter_Max = (unsigned int) max;
  std::cout << "min = " << iter_Min << " and max = " << iter_Max<< std::endl;
}

unsigned int CondBasicIter::getMin(){return iter_Min;}

unsigned int CondBasicIter::getMax(){return iter_Max;}

void CondBasicIter::getRange(unsigned int * Min_out,unsigned int * Max_out){
  *Min_out = iter_Min;
  *Max_out = iter_Max;
}


unsigned int CondBasicIter::getTime(){return m_time;}

unsigned int CondBasicIter::getStartTime(){return m_startTime;}

unsigned int CondBasicIter::getStopTime(){return m_stopTime;}


   
