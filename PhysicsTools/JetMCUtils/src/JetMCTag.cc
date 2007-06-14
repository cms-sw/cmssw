#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace JetMCTagUtils;
using namespace CandMCTagUtils;

///////////////////////////////////////////////////////////////////////

double JetMCTagUtils::EnergyRatioFromBHadrons(const Candidate & c)
{
   double ratioForBjet=0;
   double ratio = 0;
   for( Candidate::const_iterator itC  = c.begin();
                                  itC != c.end();
                                  itC ++) {

      const Candidate* theMasterClone;
      bool isFromB=false;
      if (itC->hasMasterClone ()) {
        theMasterClone = itC->masterClone().get();
        isFromB = decayFromBHadron(*theMasterClone);
      }
      ratio = itC->energy() / c.energy() ;
      if( isFromB ) ratioForBjet += ratio;
   }   
   return ratioForBjet;
}

double JetMCTagUtils::EnergyRatioFromCHadrons(const Candidate & c)
{
   double ratioForCjet=0;
   double ratio = 0;
   for( Candidate::const_iterator itC  = c.begin();
                                  itC != c.end();
                                  itC ++) {

      const Candidate* theMasterClone;
      bool isFromC=false;
      if (itC->hasMasterClone ()) {
        theMasterClone = itC->masterClone().get();
        isFromC = decayFromCHadron(*theMasterClone);
      }
      ratio = itC->energy() / c.energy() ;
      if( isFromC ) ratioForCjet += ratio;
   }
   return ratioForCjet;
}

bool JetMCTagUtils::decayFromBHadron(const Candidate & c)
{
   bool isFromB = false;
   vector<const Candidate *> allParents = getAncestors( c );
   for( vector<const Candidate *>::const_iterator aParent  = allParents.begin();
                                                  aParent != allParents.end(); 
                                                  aParent ++ ) 
     {
         if( hasBottom(**aParent) ) isFromB = true;
/*
         cout << "     particle Parent is " << (*aParent)->status()
              << " type " << (*aParent)->pdgId()
              << " pt=" << (*aParent)->pt()
              << " isB = " << isFromB
              << endl;
*/
     }
   return isFromB;

}

bool JetMCTagUtils::decayFromCHadron(const Candidate & c)
{
   bool isFromC = false;
   vector<const Candidate *> allParents = getAncestors( c );
   for( vector<const Candidate *>::const_iterator aParent  = allParents.begin();
                                                  aParent != allParents.end();
                                                  aParent ++ )
     {
         if( hasCharm(**aParent) ) isFromC = true;
/*
         cout << "     particle Parent is " << (*aParent)->status()
              << " type " << (*aParent)->pdgId()
              << " pt=" << (*aParent)->pt()
              << " isC = " << isFromC
              << endl;
*/
     }
   return isFromC;
}
