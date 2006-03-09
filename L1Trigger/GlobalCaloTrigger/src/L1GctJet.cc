#include "../interface/L1GctJet.h"

L1GctJet::L1GctJet()
{
}

L1GctJet::L1GctJet(ULong rank, ULong eta, ULong phi)
{
    SixBit tempRank(rank);
    FiveBit tempEta(eta);
    FourBit tempPhi(phi);
    
    m_rank = tempRank;
    m_eta = tempEta;
    m_phi = tempPhi;
}

L1GctJet::~L1GctJet()
{
}
