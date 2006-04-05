#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"


L1GctJet::L1GctJet(ULong rank, ULong eta, ULong phi, bool tauVeto) :
  myRank(rank),
  myEta(eta),
  myPhi(phi),
  myTauVeto(tauVeto)
{

}

L1GctJet::~L1GctJet()
{
}

