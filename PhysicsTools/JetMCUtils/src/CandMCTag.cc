#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"
#include <iostream>

using namespace std;
using namespace reco;
using namespace CandMCTagUtils;

///////////////////////////////////////////////////////////////////////

std::vector<const reco::Candidate *> CandMCTagUtils::getAncestors(const reco::Candidate &c)
{
  vector<const reco::Candidate *> moms;
  if( c.numberOfMothers() == 1 ) {
    const Candidate * dau = &c;
    const Candidate * mom = c.mother();
    while ( dau->numberOfMothers() == 1) {
      moms.push_back( dau );
      dau = mom ;
      mom = dau->mother();
    } 
  } 
  return moms;
}


bool CandMCTagUtils::hasBottom(const reco::Candidate &c) 
{
  int code1;
  int code2;
  bool tmpHasBottom = false;
  code1 = (int)( ( abs(c.pdgId() ) / 100)%10 );
  code2 = (int)( ( abs(c.pdgId() ) /1000)%10 );
  if ( code1 == 5 || code2 == 5) tmpHasBottom = true;
  return tmpHasBottom;
 }

bool CandMCTagUtils::hasCharm(const reco::Candidate &c) 
{
  int code1;
  int code2;
  bool tmpHasCharm = false;
  code1 = (int)( ( abs(c.pdgId() ) / 100)%10 );
  code2 = (int)( ( abs(c.pdgId() ) /1000)%10 );
  if ( code1 == 4 || code2 == 4) tmpHasCharm = true;
  return tmpHasCharm;
}

bool CandMCTagUtils::isParton(const reco::Candidate &c)
{
   int id = abs(c.pdgId());

   if( id == 1 ||
       id == 2 ||
       id == 3 ||
       id == 4 ||
       id == 5 ||
       id == 21 )
     return true;
   else
     return false;
}

bool CandMCTagUtils::isLightParton(const reco::Candidate &c)
{
   int id = abs(c.pdgId());

   if( id == 1 ||
       id == 2 ||
       id == 3 ||
       id == 21 )
     return true;
   else
     return false;
}
