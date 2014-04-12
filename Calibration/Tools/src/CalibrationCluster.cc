#include "Calibration/Tools/interface/CalibrationCluster.h"
#include <iostream>
#include <string>

using namespace std;

CalibrationCluster::CalibrationCluster(){

}

CalibrationCluster::~CalibrationCluster(){
}

///////////////////////////////////////////////////////////////////////

std::vector<EBDetId> CalibrationCluster::get5x5Id(EBDetId const & maxHitId){


   Xtals5x5.clear();
   
   
//   std::cout << "get5x5Id: max Containment crystal " << maxHitId.ic() << " eta " << maxHitId.ieta() << " phi " << maxHitId.iphi() << std::endl;

   for (unsigned int icry=0;icry<25;icry++)
     {
       unsigned int row = icry / 5;
       unsigned int column= icry %5;
//       std::cout << "CalibrationCluster::icry = " << icry << std::endl;
       
       int curr_eta=maxHitId.ieta() + column - (5/2);
       int curr_phi=maxHitId.iphi() + row - (5/2);
	      
       if (curr_eta * maxHitId.ieta() <= 0) {if (maxHitId.ieta() > 0) curr_eta--; else curr_eta++; }  // JUMP over 0
       if (curr_phi < 1) curr_phi += 360;
       if (curr_phi > 360) curr_phi -= 360;
      
       try
	 {
//         Xtals5x5.push_back(EBDetId(maxHitId.ieta()+column-2,maxHitId.iphi()+row-2,EBDetId::ETAPHIMODE));
         Xtals5x5.push_back(EBDetId(curr_eta,curr_phi,EBDetId::ETAPHIMODE));
	 }
       catch ( ... )
	 {
	   std::cout << "Cannot construct 5x5 matrix around EBDetId " << maxHitId << std::endl;
	 }
     }

   return Xtals5x5;
}

///////////////////////////////////////////////////////////////////////

std::vector<EBDetId> CalibrationCluster::get3x3Id(EBDetId const & maxHitId){

    Xtals3x3.clear();

    for (unsigned int icry=0;icry<9;icry++)
     {
       unsigned int row = icry / 3;
       unsigned int column= icry %3;
       
       
       try
	 {
         Xtals3x3.push_back(EBDetId(maxHitId.ieta()+column-1,maxHitId.iphi()+row-1,EBDetId::ETAPHIMODE));
	 }
       catch ( ... )
	 {
	   std::cout << "Cannot construct 3x3 matrix around EBDetId " << maxHitId << std::endl;
	 }
     }

   return Xtals3x3;
}

///////////////////////////////////////////////////////////////////////


CalibrationCluster::CalibMap CalibrationCluster::getMap(int minEta, int maxEta, int minPhi, int maxPhi){
   
   calibRegion.clear();
   int rowSize=maxEta-minEta+1;
   int columnSize=maxPhi-minPhi+1;
   int reducedSize=rowSize*columnSize;
   

   for (int icry=0;icry<reducedSize;icry++)
     {
       unsigned int eta = minEta + icry/columnSize;
       unsigned int phi = minPhi + icry%columnSize;

       
       try
	 {
         calibRegion.insert(pippo(EBDetId(eta,phi,EBDetId::ETAPHIMODE),icry));
	 }
       catch ( ... )
	 {
	   std::cout << "Cannot construct full matrix !!! " << std::endl;
	 }
     }

   return calibRegion;

}

///////////////////////////////////////////////////////////////////////


std::vector<float> CalibrationCluster::getEnergyVector(const EBRecHitCollection*
hits, CalibMap & ReducedMap, std::vector<EBDetId> & XstalsNxN, float & outBoundEnergy, int & nXtalsOut){

 energyVector.clear();
 std::vector<EBDetId>::iterator it;

// std::cout << "Reduced Map Size =" << ReducedMap.size() << std::endl;
// std::cout << "XstalsNxN Size =" << XstalsNxN.size() << std::endl;
 energyVector.resize(ReducedMap.size(),0.);

   outBoundEnergy=0.;   
   nXtalsOut=0;
   for(it=XstalsNxN.begin();it!=XstalsNxN.end();++it)
   {
       if(ReducedMap.find(*it) != ReducedMap.end()){
       CalibMap::iterator  it2 = ReducedMap.find(*it);
   
       int icry = it2->second;
   
       energyVector[icry]=(hits->find(*it))->energy();
       
       } else {
       
//       std::cout << " Cell out of Reduced map: did you subtracted the cell energy from P ???" << std::endl;
       outBoundEnergy+=(hits->find(*it))->energy();
       nXtalsOut++;
       }
    }



return energyVector;

}

