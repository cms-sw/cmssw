
#include "GeneratorInterface/Pythia6Interface/interface/PtYDistributor.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace edm;

PtYDistributor::PtYDistributor(std::string inputfile, CLHEP::HepRandomEngine& fRandomEngine){
  
   theProbSize1 = 0;
   theProbSize2 = 0;

   std::string file = "GeneratorInterface/Pythia6Interface/data/"+ inputfile +".txt";
   edm::FileInPath f1(file);
   std::string fDataFile = f1.fullPath();
   
   std::cout<<" File from "<<fDataFile <<std::endl;
   std::ifstream in( fDataFile.c_str() );
   std::string line;
  
   std::cout << " Start to read file "<<fDataFile<<" "<<std::getline( in, line)<<std::endl;
   
   while( std::getline( in, line)){
      
      if(!line.size() || line[0]=='#') {
	 std::cout<<" continue "<<std::endl;
	 continue;
      }
      std::istringstream linestream(line);
      double par,par1;
      int type;
      linestream>>type>>par1>>par;
      std::cout<<" type= "<<type<<" par= "<<par<<std::endl;
      if(type == 1)
	 {
	    // y
	    aProbFunc1[theProbSize1] = par;
	    theProbSize1++;
	 }
      if(type == 2)
	 {
	    // pt
	    aProbFunc2[theProbSize2] = par;
	    theProbSize2++;
	 }
   }  // while
   
  fYGenerator = new RandGeneral(fRandomEngine,aProbFunc1,theProbSize1);
  fPtGenerator = new RandGeneral(fRandomEngine,aProbFunc2,theProbSize2);
  
} // from file

double
PtYDistributor::fireY(double ymin, double ymax){

   double y = -999;

   if(fYGenerator){
      while(y < ymin || y > ymax)
	 y = 20*fYGenerator->fire()-10;
   }else{
      throw edm::Exception(edm::errors::NullPointerError,"PtYDistributor")
	 <<"Random y requested but Random Number Generator for y not Initialized!";
   }
   return y;
}

double 
PtYDistributor::firePt(double ptmin, double ptmax){

   double pt = -999;

   if(fPtGenerator){
      while(pt < ptmin || pt > ptmax)
         pt = 100*fPtGenerator->fire();
   }else{
      throw edm::Exception(edm::errors::NullPointerError,"PtYDistributor")
         <<"Random pt requested but Random Number Generator for pt not Initialized!";
   }
   return pt;
}
