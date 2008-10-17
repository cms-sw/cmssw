#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondTools/SiPixel/test/SiPixelBadModuleReader.h"
#include "DataFormats/DetId/interface/DetId.h"


#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

SiPixelBadModuleReader::SiPixelBadModuleReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

SiPixelBadModuleReader::~SiPixelBadModuleReader(){}

void SiPixelBadModuleReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){
  
  edm::ESHandle<SiPixelQuality> SiPixelBadModule_;
  iSetup.get<SiPixelQualityRcd>().get(SiPixelBadModule_);
  edm::LogInfo("SiPixelBadModuleReader") << "[SiPixelBadModuleReader::analyze] End Reading SiPixelBadModule" << std::endl;
  
  std::vector<SiPixelQuality::disabledModuleType>disabledModules = SiPixelBadModule_->getBadComponentList();
  
  if (printdebug_)
    for (size_t id=0;id<disabledModules.size();id++)
      {
	SiPixelQuality::disabledModuleType badmodule = disabledModules[id];

      //////////////////////////////////////
      //  errortype "whole" = int 0 in DB //
      //  errortype "tbmA" = int 1 in DB  //
      //  errortype "tbmB" = int 2 in DB  //
      //////////////////////////////////////

	std::string errorstring;

        if(badmodule.errorType == 0)
          errorstring = "whole";
        else if(badmodule.errorType == 1)
          errorstring = "tbmA";
        else if(badmodule.errorType == 2)
          errorstring = "tbmB";
        else if(badmodule.errorType == 3)
          errorstring = "none";

//          edm::LogInfo("SiPixelBadModuleReader")  << "Values stored in DB:  DetID of disabled module: " << badmodule.DetID << " \t"
//                                                  << "Error type is: " << badmodule.errorType << " \t"
//                                                  << "Bad ROCs are: "<< badmodule.BadRocs << " \t"
//      	                                         << std::endl; 
	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;  //to make the reading easier 
	std::cout<< "Values stored in DB:  DetID of disabled module: " << badmodule.DetID << " \t"
                 << "Error type is: " << badmodule.errorType << " \t"
                 << "Bad ROCs are: "<< badmodule.BadRocs << " \t"
                 <<std::endl;
   
	 std::cout << "In Human Readable Form, "<< std::endl;
         std::cout << "DetID is: " << badmodule.DetID << " and this has an error type of '"<<errorstring<<"'"<<std::endl;
	 std::cout << "And the bad ROCs are: "<< std::endl;
         for (unsigned short n = 0; n < 16; n++){
	  unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
	  if (badmodule.BadRocs & mask)
 	    std::cout << n <<", ";
	 }
	 std::cout << std::endl;
	 std::cout <<" "<<std::endl;
	 std::cout <<" "<<std::endl;  //to make teh reading easier
             
      }

  std::cout<<"IsModuleBad should be true, 302123296: "<<  SiPixelBadModule_->IsModuleBad(302123296)<<std::endl;
  std::cout<<"IsModuleBad should be false, 302195555: "<<  SiPixelBadModule_->IsModuleBad(302195555)<<std::endl;
  std::cout<<"IsModuleBad should be false, 302125076: "<<  SiPixelBadModule_->IsModuleBad(302125076)<<std::endl;
//   std::cout<<"IsModuleUsable should be false: "<<  SiPixelBadModule_->IsModuleUsable(302123296)<<std::endl;
//   std::cout<<"IsModuleUsable should be true: "<<  SiPixelBadModule_->IsModuleUsable(302195555)<<std::endl;
//   std::cout<<"IsRocBad should be true: "<<  SiPixelBadModule_->IsRocBad(302126364, 8)<<std::endl;
//   std::cout<<"IsRocBad should be false: "<<  SiPixelBadModule_->IsRocBad(302126364, 5)<<std::endl;
//   std::cout<<"IsRocBad should be true: "<<  SiPixelBadModule_->IsRocBad(302121992, 5)<<std::endl;
//   std::cout<<"IsRocBad should be false: "<<  SiPixelBadModule_->IsRocBad(302121992, 8)<<std::endl;


   std::sort(disabledModules.begin(),disabledModules.end(),SiPixelQuality::BadComponentStrictWeakOrdering());

    for (size_t id=0;id<disabledModules.size();id++)
      {
	SiPixelQuality::disabledModuleType badmodule = disabledModules[id];

      //////////////////////////////////////
      //  errortype "whole" = int 0 in DB //
      //  errortype "tbmA" = int 1 in DB  //
      //  errortype "tbmB" = int 2 in DB  //
      //////////////////////////////////////

	std::string errorstring;

        if(badmodule.errorType == 0)
          errorstring = "whole";
        else if(badmodule.errorType == 1)
          errorstring = "tbmA";
        else if(badmodule.errorType == 2)
          errorstring = "tbmB";
        else if(badmodule.errorType == 3)
          errorstring = "none";

//          edm::LogInfo("SiPixelBadModuleReader")  << "Values stored in DB:  DetID of disabled module: " << badmodule.DetID << " \t"
//                                                  << "Error type is: " << badmodule.errorType << " \t"
//                                                  << "Bad ROCs are: "<< badmodule.BadRocs << " \t"
//      	                                         << std::endl; 
	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;  //to make the reading easier 
	std::cout<< "Values stored in DB:  DetID of disabled module: " << badmodule.DetID << " \t"
                 << "Error type is: " << badmodule.errorType << " \t"
                 << "Bad ROCs are: "<< badmodule.BadRocs << " \t"
                 <<std::endl;
   
	 std::cout << "In Human Readable Form, "<< std::endl;
         std::cout << "DetID is: " << badmodule.DetID << " and this has an error type of '"<<errorstring<<"'"<<std::endl;
	 std::cout << "And the bad ROCs are: "<< std::endl;
         for (unsigned short n = 0; n < 16; n++){
	  unsigned short mask = 1 << n;  // 1 << n = 2^{n} using bitwise shift
	  if (badmodule.BadRocs & mask)
 	    std::cout << n <<", ";
	 }
	 std::cout << std::endl;
	 std::cout <<" "<<std::endl;
	 std::cout <<" "<<std::endl;  //to make the reading easier
             
      }


}
