#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounterLut.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctMap.h"

#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>

using namespace std;

//DEFINE STATICS
const unsigned int L1GctJetCounterLut::MAX_CUT_TYPE = nullCutType;

L1GctJetCounterLut::L1GctJetCounterLut(vector<validCutType> cutType, vector<unsigned> cutValue1, vector<unsigned> cutValue2) :
  m_nCuts(cutType.size()),
  m_cutType(cutType),
  m_cutValue1(cutValue1),
  m_cutValue2(cutValue2)
{
  if (cutValue1.size() != m_nCuts) {
    throw cms::Exception("L1GctSetupError")
      << "Error in constructing L1GctJetCounterLut; size of first vector of cut values is " << cutValue1.size()
      << " but size of cutType vector is " << cutType.size() << endl;
  }
  if (cutValue2.size() != m_nCuts) {
    throw cms::Exception("L1GctSetupError")
      << "Error in constructing L1GctJetCounterLut; size of second vector of cut values is " << cutValue2.size()
      << " but size of cutType vector is " << cutType.size() << endl;
  }
  for (unsigned i=0; i<m_nCuts; i++) {
    checkCut(cutType.at(i), cutValue1.at(i), cutValue2.at(i));
  }
}

L1GctJetCounterLut::L1GctJetCounterLut(validCutType cutType, unsigned cutValue1, unsigned cutValue2) :
  m_nCuts(1),
  m_cutType(1, cutType),
  m_cutValue1(1, cutValue1),
  m_cutValue2(1, cutValue2)
{
  checkCut(cutType, cutValue1, cutValue2);
}

L1GctJetCounterLut::L1GctJetCounterLut() :
  m_nCuts(1),
  m_cutType(1, nullCutType),
  m_cutValue1(1, 0),
  m_cutValue2(1, 0)
{
}

L1GctJetCounterLut::~L1GctJetCounterLut()
{
}

ostream& operator << (ostream& os, const L1GctJetCounterLut& lut)
{
  os << "===L1GctJetCounterLut===" << endl;
  os << "Number of cuts for this Lut is " << lut.m_nCuts;
  for (unsigned i=0; i<lut.m_nCuts; i++) {
    if (i==0) {
      os << "Jet passes ";
    } else {
      os << "and " ;
    }
    switch (lut.m_cutType.at(i))
      {
      case L1GctJetCounterLut::minRank:
	os << "if its rank is at least " << lut.m_cutValue1.at(i) << endl ;
	break;

      case L1GctJetCounterLut::maxRank:
	os << "if its rank is at most " << lut.m_cutValue1.at(i) << endl ;
	break;

      case L1GctJetCounterLut::centralEta:
	os << "if it is in the central eta range with max eta " << lut.m_cutValue1.at(i) << endl ;
	break;

      case L1GctJetCounterLut::forwardEta:
	os << "if it is in the forward eta range with min eta " << lut.m_cutValue1.at(i) << endl ;
	break;

      case L1GctJetCounterLut::phiWindow:
	if (lut.m_cutValue2.at(i)>lut.m_cutValue1.at(i)) {
	  os << "if its phi is between " << lut.m_cutValue1.at(i) << " and " << lut.m_cutValue2.at(i) << endl;
	} else {
	  os << "if its phi is larger than " << lut.m_cutValue1.at(i) << " or smaller than " << lut.m_cutValue2.at(i) << endl;
	}
	break;

      case L1GctJetCounterLut::nullCutType:
	os << "if Hell freezes over " << endl;
	break;

      default: // do nothing, shouldn't get here!
	break;
      }
  }
  os << endl;
  return os;
}

bool L1GctJetCounterLut::passesCut(const L1GctJetCand jet) const
{
  bool result = !jet.isNullJet();
  for (unsigned i=0; i<m_nCuts; i++) {
    if (!result) { break; } // for efficiency
//     result &= jetPassesThisCut(jet, i);
    result = result & jetPassesThisCut(jet, i);
  }
  return result;
}
  
void L1GctJetCounterLut::checkCut(const validCutType cutType, const unsigned cutValue1, const unsigned cutValue2) const
{
  switch (cutType)
    {
    case minRank:
      if (cutValue1>=(1<<6)) {
	throw cms::Exception("L1GctSetupError")
	  << "Error in constructing L1GctJetCounterLut; rank value " << cutValue1
	  << " should be less than 64 " << endl;
      }
      return;

    case maxRank:
      if (cutValue1>=(1<<6)) {
	throw cms::Exception("L1GctSetupError")
	  << "Error in constructing L1GctJetCounterLut; rank value " << cutValue1
	  << " should be less than 64 " << endl;
      }
      return;

    case centralEta:
      if (cutValue1>=L1GctMap::N_RGN_ETA/2) {
	throw cms::Exception("L1GctSetupError")
	  << "Error in constructing L1GctJetCounterLut; eta value " << cutValue1
	  << " should be less than " << L1GctMap::N_RGN_ETA/2 << endl;
      }
      return;

    case forwardEta:
      if (cutValue1>=L1GctMap::N_RGN_ETA/2) {
	throw cms::Exception("L1GctSetupError")
	  << "Error in constructing L1GctJetCounterLut; eta value " << cutValue1
	  << " should be less than " << L1GctMap::N_RGN_ETA/2 << endl;
      }
      return;

    case phiWindow:
      if (cutValue1>=L1GctMap::N_RGN_PHI) {
	throw cms::Exception("L1GctSetupError")
	  << "Error in constructing L1GctJetCounterLut; phi value1 " << cutValue1
	  << " should be less than " << L1GctMap::N_RGN_PHI << endl;
      }
      if (cutValue2>=L1GctMap::N_RGN_PHI) {
	throw cms::Exception("L1GctSetupError")
	  << "Error in constructing L1GctJetCounterLut; phi value2 " << cutValue2
	  << " should be less than " << L1GctMap::N_RGN_PHI << endl;
      }
      return;

    case nullCutType:
      return;

    default:
    throw cms::Exception("L1GctSetupError")
      << "Error in constructing L1GctJetCounterLut; unrecognised cut type " << cutType << endl;
    }

}
  
bool L1GctJetCounterLut::jetPassesThisCut(const L1GctJetCand jet, const unsigned i) const
{
  switch (m_cutType.at(i))
    {
    case minRank:
      return (jet.rank()>=m_cutValue1.at(i));

    case maxRank:
      return (jet.rank()<=m_cutValue1.at(i));

    case centralEta:
      return (jet.eta()<=m_cutValue1.at(i));

    case forwardEta:
      return (jet.eta()>=m_cutValue1.at(i));

    case phiWindow:
      return (m_cutValue2.at(i)>m_cutValue1.at(i) ?
	      ((jet.phi() >= m_cutValue1.at(i)) && (jet.phi() <= m_cutValue2.at(i))) :
	      ((jet.phi() >= m_cutValue1.at(i)) || (jet.phi() <= m_cutValue2.at(i))));

    case nullCutType:
      return false;

    default:
    throw cms::Exception("L1GctProcessingError")
      << "Error in L1GctJetCounterLut; encountered unrecognised cut type " << m_cutType.at(i) << endl;
    }

}
