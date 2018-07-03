#include "L1Trigger/L1TNtuples/interface/L1AnalysisBMTFInputs.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <sstream>
#include <string>

using namespace std;

L1Analysis::L1AnalysisBMTFInputs::L1AnalysisBMTFInputs()
{
}


L1Analysis::L1AnalysisBMTFInputs::~L1AnalysisBMTFInputs()
{

}

void L1Analysis::L1AnalysisBMTFInputs::SetBMPH(const edm::Handle<L1MuDTChambPhContainer > L1MuDTChambPhContainer, unsigned int maxDTPH)
{

   L1MuDTChambPhContainer::Phi_Container const *PhContainer = L1MuDTChambPhContainer->getContainer();

   bmtf_.phSize = PhContainer->size();
   int iphtr=0;
   for( L1MuDTChambPhContainer::Phi_Container::const_iterator
       DTPhDigiItr =  PhContainer->begin() ;
       DTPhDigiItr != PhContainer->end() ;
       ++DTPhDigiItr )
    {
      if((unsigned int) iphtr>maxDTPH-1) continue;
      bmtf_.phBx.push_back     (  DTPhDigiItr->bxNum() );
      bmtf_.phTs2Tag.push_back     ( DTPhDigiItr->Ts2Tag() );
      bmtf_.phWh.push_back     (  DTPhDigiItr->whNum() );
      bmtf_.phSe.push_back     (  DTPhDigiItr->scNum() );
      bmtf_.phSt.push_back     (  DTPhDigiItr->stNum() );
      bmtf_.phAng.push_back    (  DTPhDigiItr->phi()   );
      bmtf_.phBandAng.push_back(  DTPhDigiItr->phiB()  );
      bmtf_.phCode.push_back   (  DTPhDigiItr->code()  );

      iphtr++;
    }

}


void L1Analysis::L1AnalysisBMTFInputs::SetBMTH(const edm::Handle<L1MuDTChambThContainer > L1MuDTChambThContainer, unsigned int maxDTTH)
{

   L1MuDTChambThContainer::The_Container const *ThContainer = L1MuDTChambThContainer->getContainer();

   int ithtr=0;
   bmtf_.thSize = ThContainer->size();

   for( L1MuDTChambThContainer::The_Container::const_iterator
	 DTThDigiItr =  ThContainer->begin() ;
       DTThDigiItr != ThContainer->end() ;
       ++DTThDigiItr )
     {

      if((unsigned int) ithtr>maxDTTH-1) continue;
      bmtf_.thBx.push_back( DTThDigiItr->bxNum()  );
      bmtf_.thWh.push_back( DTThDigiItr->whNum() );
      bmtf_.thSe.push_back( DTThDigiItr->scNum() );
      bmtf_.thSt.push_back( DTThDigiItr->stNum() );

      ostringstream  ss1, ss2; 
      ss1.clear(); ss2.clear();
      ss1<<"9"; ss2<<"9";

      for(int j=0; j<7; j++){
        ss1<<DTThDigiItr->position(j);
        ss2<<DTThDigiItr->code(j) ;
      }
      bmtf_.thTheta.push_back(stoi(ss1.str())) ;
      bmtf_.thCode.push_back(stoi(ss2.str()));

      ithtr++;

    }
}





