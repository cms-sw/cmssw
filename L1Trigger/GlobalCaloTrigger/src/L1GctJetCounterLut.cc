#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounterLut.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


//DEFINE STATICS
const int L1GctJetCounterLut::NAddress=JET_COUNTER_LUT_ADD_BITS;

L1GctJetCounterLut::L1GctJetCounterLut(const L1GctJetCounterSetup::cutsListForJetCounter& cuts) :
  L1GctLut<NAddress,1>(),
  m_cutList(cuts)
{
  m_setupOk = true;
  for (unsigned i=0; i<cuts.size(); i++) {
    m_setupOk &= checkCut(cuts.at(i));
  }
}

L1GctJetCounterLut::L1GctJetCounterLut(const L1GctJetCounterSetup::cutDescription& cut) :
  L1GctLut<NAddress,1>(),
  m_cutList(1, cut)
{
  m_setupOk = checkCut(cut);
}

L1GctJetCounterLut::L1GctJetCounterLut() :
  L1GctLut<NAddress,1>(),
  m_cutList(1)
{
  m_setupOk = true;
}

L1GctJetCounterLut::L1GctJetCounterLut(const L1GctJetCounterLut& lut) :
  L1GctLut<NAddress,1>(),
  m_cutList(lut.cutList())
{
  m_setupOk = true;
}

L1GctJetCounterLut::~L1GctJetCounterLut()
{
}

L1GctJetCounterLut L1GctJetCounterLut::operator= (const L1GctJetCounterLut& lut)
{
  L1GctJetCounterLut temp(lut);
  return temp;
}

std::ostream& operator << (std::ostream& os, const L1GctJetCounterLut& lut)
{
//   os << "===L1GctJetCounterLut===" << endl;
//   os << "Number of cuts for this Lut is " << lut.nCuts() << endl;
  for (unsigned i=0; i<lut.nCuts(); i++) {
    if (i==0) {
      os << "Jet passes ";
    } else {
      os << "\nand " ;
    }
    switch (lut.m_cutList.at(i).cutType)
      {
      case L1GctJetCounterSetup::minRank:
	os << "if its rank is at least " << lut.m_cutList.at(i).cutValue1 ;
	break;

      case L1GctJetCounterSetup::maxRank:
	os << "if its rank is at most " << lut.m_cutList.at(i).cutValue1 ;
	break;

      case L1GctJetCounterSetup::centralEta:
	os << "if it is in the central eta range with max (local) eta " << lut.m_cutList.at(i).cutValue1 ;
	break;

      case L1GctJetCounterSetup::forwardEta:
	os << "if it is in the forward eta range with min (local) eta " << lut.m_cutList.at(i).cutValue1 ;
	break;

      case L1GctJetCounterSetup::phiWindow:
	if (lut.m_cutList.at(i).cutValue2>lut.m_cutList.at(i).cutValue1) {
	  os << "if its phi is between " << lut.m_cutList.at(i).cutValue1
                             << " and " << lut.m_cutList.at(i).cutValue2 ;
	} else {
	  os << "if its phi is larger than " << lut.m_cutList.at(i).cutValue1
                     << " or smaller than " << lut.m_cutList.at(i).cutValue2 ;
	}
	break;

      case L1GctJetCounterSetup::nullCutType:
	os << "if Hell freezes over " ;
	break;

      default: // do nothing, shouldn't get here!
	break;
      }
  }
  os << "\n===Lookup table contents===\n" << std::endl;
  const L1GctLut<L1GctJetCounterLut::NAddress,1>* temp=&lut;
  os << *temp;
  return os;
}

template class L1GctLut<L1GctJetCounterLut::NAddress,1>;

bool L1GctJetCounterLut::passesCut(const L1GctJetCand jet) const
{
  bool result = !jet.empty();
  for (unsigned i=0; i<nCuts() && result; i++) {
    result &= jetPassesThisCut(jet, i);
  }
  return result;
}

bool L1GctJetCounterLut::passesCut(const uint16_t lutAddress) const
{
  // Make a jet out of a 16-bit lut address. 
  // Note this is intentionally not the same as the raw representation of the jet
  // We ignore the sign of eta; and we need to encode the type of jet (forward, central or tau)
  unsigned rank = (lutAddress & 0x3f);
  unsigned eta  = (lutAddress >> 6) & 0x7;  
  unsigned phi  = (lutAddress >> 9) & 0x1f;
  bool isFor    = (lutAddress & 0x4000) != 0;
  bool isTau    = (lutAddress & 0x8000) != 0;

  // Check the address corresponds to a physical jet
  if (phi<L1CaloRegionDetId::N_PHI && !(isFor && isTau)) {
    // Make a jet and see if it passes the cuts
    L1GctJetCand jc(rank, phi, eta, isTau, isFor);
    return passesCut(jc);
  } else {
    return false;
  }
}
  
bool L1GctJetCounterLut::checkCut(const L1GctJetCounterSetup::cutDescription cut) const
{
  bool result = true;

  switch (cut.cutType)
    {
    case L1GctJetCounterSetup::minRank:
      if (cut.cutValue1>=(1<<6)) {
	if (m_verbose) {
	  edm::LogWarning("L1GctSetupError")
	    << "Invalid cut supplied to L1GctJetCounterLut; rank value " << cut.cutValue1
	    << " should be less than 64 " << std::endl;
	}
	result = false;
      }
      break;

    case L1GctJetCounterSetup::maxRank:
      if (cut.cutValue1>=(1<<6)) {
	if (m_verbose) {
	  edm::LogWarning("L1GctSetupError")
	    << "Invalid cut supplied to L1GctJetCounterLut; rank value " << cut.cutValue1
	    << " should be less than 64 " << std::endl;
	}
	result = false;
      }
      break;

    case L1GctJetCounterSetup::centralEta:
      if (cut.cutValue1>=L1CaloRegionDetId::N_ETA/2) {
	if (m_verbose) {
	  edm::LogWarning("L1GctSetupError")
	    << "Invalid cut supplied to L1GctJetCounterLut; eta value " << cut.cutValue1
	    << " should be less than " << L1CaloRegionDetId::N_ETA/2 << std::endl;
	}
	result = false;
      }
      break;

    case L1GctJetCounterSetup::forwardEta:
      if (cut.cutValue1>=L1CaloRegionDetId::N_ETA/2) {
	if (m_verbose) {
	  edm::LogWarning("L1GctSetupError")
	    << "Invalid cut supplied to L1GctJetCounterLut; eta value " << cut.cutValue1
	    << " should be less than " << L1CaloRegionDetId::N_ETA/2 << std::endl;
	}
	result = false;
      }
      break;

    case L1GctJetCounterSetup::phiWindow:
      if (cut.cutValue1>=L1CaloRegionDetId::N_PHI) {
	if (m_verbose) {
	  edm::LogWarning("L1GctSetupError")
	    << "Invalid cut supplied to L1GctJetCounterLut; phi value1 " << cut.cutValue1
	    << " should be less than " << L1CaloRegionDetId::N_PHI << std::endl;
	}
	result = false;
      }
      if (cut.cutValue2>=L1CaloRegionDetId::N_PHI) {
	if (m_verbose) {
	  edm::LogWarning("L1GctSetupError")
	    << "Invalid cut supplied to L1GctJetCounterLut; phi value2 " << cut.cutValue2
	    << " should be less than " << L1CaloRegionDetId::N_PHI << std::endl;
	}
	result = false;
      }
      break;

    case L1GctJetCounterSetup::nullCutType:
      break;

    default:
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "Invalid cut supplied to L1GctJetCounterLut; unrecognised cut type "
	  << cut.cutType << std::endl;
      }
      result = false;
    }

  return result;

}
  
bool L1GctJetCounterLut::jetPassesThisCut(const L1GctJetCand jet, const unsigned i) const
{
  switch (m_cutList.at(i).cutType)
    {
    case L1GctJetCounterSetup::minRank:
      return (jet.rank()>=m_cutList.at(i).cutValue1);

    case L1GctJetCounterSetup::maxRank:
      return (jet.rank()<=m_cutList.at(i).cutValue1);

    case L1GctJetCounterSetup::centralEta:
      return (rctEta(jet)<=m_cutList.at(i).cutValue1);

    case L1GctJetCounterSetup::forwardEta:
      return (rctEta(jet)>=m_cutList.at(i).cutValue1);

    case L1GctJetCounterSetup::phiWindow:
      return (m_cutList.at(i).cutValue2>m_cutList.at(i).cutValue1 ?
	      ((globalPhi(jet) >= m_cutList.at(i).cutValue1) && (globalPhi(jet) <= m_cutList.at(i).cutValue2)) :
	      ((globalPhi(jet) >= m_cutList.at(i).cutValue1) || (globalPhi(jet) <= m_cutList.at(i).cutValue2)));

    case L1GctJetCounterSetup::nullCutType:
      return false;

    default:
      // It's an error but should have been picked up in the setup checking
      return false;
    }
}

uint16_t L1GctJetCounterLut::value (const uint16_t lutAddress) const
{
  return (passesCut(lutAddress) ? 0 : 1) ;
}

unsigned L1GctJetCounterLut::rctEta(const L1GctJetCand jet) const
{
  return (jet.etaIndex() & 0x7) + (jet.isForward() ? 7 : 0);
}

unsigned L1GctJetCounterLut::globalPhi(const L1GctJetCand jet) const
{
  return jet.phiIndex();
}

