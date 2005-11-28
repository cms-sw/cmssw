// CaloJet.cc
// $Id: CaloJet.cc,v 1.1 2005/11/28 14:45:33 llista Exp $
// Initial Version From Fernando Varela Rodriguez
// Revisions: R. Harris, 19-Oct-2005, modified to work with real 
//            CaloTowers from Jeremy Mans.  Commented out energy
//            fractions until we can figure out how to determine 
//            composition of total energy, and the underlying HB, HE, 
//            HF, HO and Ecal.

//Own header file
#include "DataFormats/JetReco/interface/CaloJet.h"
using namespace std;
using namespace reco;

CaloJet::CaloJet() {
}

CaloJet::CaloJet( double px, double py, double pz, double e, 
	     double maxEInEmTowers, double maxEInHadTowers, 
	     double energyFractionInHCAL, double energyFractionInECAL,
	     int n90 ) :
  p4_( px, py, pz, e ), 
  maxEInEmTowers_( maxEInEmTowers ), maxEInHadTowers_( maxEInHadTowers ),
  energyFractionInHCAL_( energyFractionInHCAL ), energyFractionInECAL_( energyFractionInECAL ),
  n90_( n90 ) {
}

CaloJet::~CaloJet() {
}
