#include "CondTools/RPC/interface/RPCRunIOV.h"


RPCRunIOV::RPCRunIOV(unsigned long long since, unsigned long long till)
{}

RPCRunIOV::~RPCRunIOV(){}

std::vector<RPCObImon::I_Item> 
RPCRunIOV::getData()
{
  
  std::cout << std::endl << "=============================================" << std::endl;
  std::cout << std::endl << "============  RUN IOV ASSOCIATOR  ===========" << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;
  
  
  std::cout << ">> RUN start: " << since << std::endl;
  std::cout << ">> RUN  stop: " << till << std::endl;
  std::cout << std::endl << "=============================================" << std::endl << std::endl;

  std::vector<unsigned long long> iov_vect;
  RPCIOVReader iov_list ("sqlite_file:dati.db", "CMS_COND_GENERAL_R", "rd0548in");
  iov_vect = iov_list.listIOV();
  
  

   unsigned long long iov;
   std::vector<unsigned long long> final_vect;
   std::vector<unsigned long long>::iterator it, it_fin;

   if (iov_vect.front() < since) {
     for (it = iov_vect.begin(); it != iov_vect.end(); it++) {
       iov = *(it);
       //std::cout << iov << std::endl;
       if (since < iov && iov < till) {
	 if (final_vect.size() == 0) {
	   *it--;
	   final_vect.push_back(*it);
	   *it++;
	 } 
	 final_vect.push_back(iov); 
       } 
     }
     std::cout << std::endl << "=============================================" << std::endl;
     std::cout <<              "        Accessing the following IOVs\n        "<< std::endl; 
     for (it_fin = final_vect.begin(); it_fin != final_vect.end(); it_fin++) {
       iov = *(it_fin);
       std::cout << iov << "\n";
     }
   } else {
     std::cout << "   WARNING: run not included in data range\n";
   }

   std::vector<RPCObImon::I_Item> IMON;
   IMON = iov_list.getIMON(final_vect.front(), final_vect.back());
   std::cout << "\n>> Imon vector created --> size: " << IMON.size() << std::endl;
   
   // PRINT
//    RPCObImon::I_Item temp;
//    std::string day,time;
//    for (std::vector<RPCObImon::I_Item>::iterator ii = IMON.begin(); ii != IMON.end(); ii++) {
//      temp = *(ii);
//      day  = iov_list.toDay(temp.day);
//      time = iov_list.toTime(temp.time);
//      std::cout << "ID: " << temp.dpid << " - Val: " << temp.value << " - Day: " << day << " - Time: " << time << std::endl;
//    }

   return IMON;

}

//define this as a plug-in
//DEFINE_FWK_MODULE(RPCRunIOV);
