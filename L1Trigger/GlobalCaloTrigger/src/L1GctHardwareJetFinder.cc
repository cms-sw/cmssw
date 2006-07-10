#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHardwareJetFinder.h"
 
#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>
using namespace std;

//DEFINE STATICS
const unsigned int L1GctHardwareJetFinder::MAX_REGIONS_IN = (((L1CaloRegionDetId::N_ETA)/2)+1)*L1GctHardwareJetFinder::N_COLS;

const int L1GctHardwareJetFinder::N_COLS = 2;
const unsigned int L1GctHardwareJetFinder::CENTRAL_COL0 = 0;

const unsigned int L1GctHardwareJetFinder::JET_THRESHOLD = 1;

L1GctHardwareJetFinder::L1GctHardwareJetFinder(int id, vector<L1GctSourceCard*> sourceCards,
					       L1GctJetEtCalibrationLut* jetEtCalLut):
  L1GctJetFinderBase(id, sourceCards, jetEtCalLut),
  m_protoJetRegions(MAX_REGIONS_IN)
{
  // Setup the position info in protoJetRegions.
  // Note the transformation to global phi is not the same as that
  // contained in L1CaloRegionDetId, because the passing of protoJets
  // to neighbour jetFinders results in a shift in phi
  for (unsigned column=0; column<2; ++column) {
    for (unsigned row=0; row<COL_OFFSET; ++row) {
      unsigned ieta;
      unsigned iphi;
      if (id<static_cast<int>(L1CaloRegionDetId::N_PHI/2)) {
	ieta = (L1CaloRegionDetId::N_ETA/2-row);
	iphi = (L1CaloRegionDetId::N_PHI - (m_id)*2 + 2)%L1CaloRegionDetId::N_PHI + column;
      } else {
	ieta = (L1CaloRegionDetId::N_ETA/2-1+row);
	iphi = ((L1CaloRegionDetId::N_PHI - m_id)*2 + 2)%L1CaloRegionDetId::N_PHI + column;
      }
      L1CaloRegion temp(0, ieta, iphi);
      m_protoJetRegions.at(column*COL_OFFSET+row) = temp;
    }
  }
  this->reset();
}

L1GctHardwareJetFinder::~L1GctHardwareJetFinder()
{
}

ostream& operator << (ostream& os, const L1GctHardwareJetFinder& algo)
{
  os << "===L1GctHardwareJetFinder===" << endl;
  const L1GctJetFinderBase* temp = &algo;
  os << *temp;
  return os;
}

void L1GctHardwareJetFinder::reset()
{
  L1GctJetFinderBase::reset();
  // Reset m_protoJetRegions without disturbing the position information
  for (unsigned j=0; j<m_protoJetRegions.size(); ++j) {
    m_protoJetRegions.at(j).reset();
  }
}

void L1GctHardwareJetFinder::fetchInput()
{
  fetchCentreStripsInput();
  findProtoJets();
}

void L1GctHardwareJetFinder::process() 
{
  fetchProtoJetsFromNeighbour();
  findJets();
  sortJets();
  doEnergySums();
}

/// HERE IS THE JETFINDER CODE

/// The first stage of clustering, called by fetchInput()
void L1GctHardwareJetFinder::findProtoJets()
{
  findLocalMaxima(m_inputRegions);
  findClusters(m_inputRegions, true);
  convertClustersToProtoJets();
}

/// The second stage of clustering, called by process()
void L1GctHardwareJetFinder::findJets()
{
  fillRegionsFromProtoJets();
  findLocalMaxima(m_protoJetRegions);
  findClusters(m_protoJetRegions, false);
  convertClustersToOutputJets();
}

/// Both clustering stages need to find local maxima in the search array
//  Find the local et maxima in the 2x11 array of regions
void L1GctHardwareJetFinder::findLocalMaxima(const RegionsVector rgv)
{
  m_localMaxima.clear();
  m_localMaxima.resize(MAX_JETS_OUT);
  UShort jetNum = 0; //holds the number of jets currently found
  UShort centreIndex = COL_OFFSET*this->centralCol0();
  for(UShort column = 0; column <2; ++column)  //Find jets in the central search region
  {
    //don't include row zero as it is not in the search region
    ++centreIndex;
    for (UShort row = 1; row < COL_OFFSET; ++row)  
    {
      // Here's the array of greater-than and greater-or-equal tests
      // to ensure each localMaximum appears once and only once in the list
      bool localMax = (rgv.at(centreIndex).et()>=JET_THRESHOLD);
      localMax     &= (rgv.at(centreIndex).et() >  rgv.at(centreIndex-1).et());
      if (row < (COL_OFFSET-1)) {
	localMax   &= (rgv.at(centreIndex).et() >= rgv.at(centreIndex+1).et());
      }
      if (column==0) {
	localMax   &= (rgv.at(centreIndex).et() >= rgv.at(centreIndex+COL_OFFSET).et());
	localMax   &= (rgv.at(centreIndex).et() >  rgv.at(centreIndex+COL_OFFSET-1).et());
	if (row < (COL_OFFSET-1)) {
	  localMax &= (rgv.at(centreIndex).et() >  rgv.at(centreIndex+COL_OFFSET-1).et());
	}
      } else {
	localMax   &= (rgv.at(centreIndex).et() >  rgv.at(centreIndex-COL_OFFSET).et());
	localMax   &= (rgv.at(centreIndex).et() >= rgv.at(centreIndex-COL_OFFSET-1).et());
	if (row < (COL_OFFSET-1)) {
	  localMax &= (rgv.at(centreIndex).et() >= rgv.at(centreIndex-COL_OFFSET-1).et());
	}
      }
      if (localMax) {
        assert(jetNum < MAX_JETS_OUT);
                
        m_localMaxima.at(jetNum) = rgv.at(centreIndex);
        ++jetNum;
      }
      ++centreIndex;
    }
  }

  // Sort the maxima by et at this point
  sort(m_localMaxima.begin(), m_localMaxima.end(), etGreaterThan());

  m_numberOfClusters = jetNum;

}

/// Both clustering stages need to convert local maxima to clusters
//  For each local maximum, find the cluster et in a 2x3 region.
//  The logic ensures that a given region et cannot be used in more than one cluster.
//  The sorting of the local maxima ensures the highest et maximum has priority.
void L1GctHardwareJetFinder::findClusters(const RegionsVector rgv, const bool preClusterLogic)
{
  m_clusters.clear();
  m_clusters.resize(MAX_JETS_OUT);

  // Use each row in the array once only; remember which ones we have used
  vector<bool> usedThisRow(COL_OFFSET, false);

  // Loop over local maxima
  for (unsigned j=0; j<m_numberOfClusters; ++j) {
    unsigned localEta = m_localMaxima.at(j).rctEta();
    unsigned localPhi = m_localMaxima.at(j).rctPhi();

    unsigned etCluster = 0;
    bool tauVetoOr = false;
    bool ovrFlowOr = false;

    for (unsigned row=localEta; ((row<(localEta+3)) && (row<COL_OFFSET)); ++row) {
      if (!usedThisRow.at(row)) {
	for (unsigned column=0; column<2; ++column) {
	  if ((preClusterLogic) || (column != localPhi) || (row == (localEta+1))) {
	    unsigned index = column*COL_OFFSET + row;
	    etCluster += rgv.at(index).et();
	    tauVetoOr |= rgv.at(index).tauVeto();
	    ovrFlowOr |= rgv.at(index).overFlow();
	  }
	}
	usedThisRow.at(row) = true;
      }
    }
    unsigned eta = m_localMaxima.at(j).gctEta();
    unsigned phi = m_localMaxima.at(j).gctPhi();

    L1CaloRegion temp(etCluster, ovrFlowOr, tauVetoOr, false, false, eta, phi);
    m_clusters.at(j) = temp;
  }
}

/// Fill search array for the second stage of clustering based on the pre-clustered jets
void L1GctHardwareJetFinder::fillRegionsFromProtoJets()
{
  for (unsigned j=0; j<MAX_JETS_OUT; ++j) {
    if (m_rcvdProtoJets.at(j).et()>=JET_THRESHOLD) {
      unsigned eta0 = m_rcvdProtoJets.at(j).rctEta();
      m_protoJetRegions.at(eta0+1) = m_rcvdProtoJets.at(j);
    }
    if (m_keptProtoJets.at(j).et()>=JET_THRESHOLD) {
      unsigned eta1 = m_keptProtoJets.at(j).rctEta();
      m_protoJetRegions.at(eta1+1+COL_OFFSET) = m_keptProtoJets.at(j);
    }
  }
}

/// Organise the pre-clustered jets into the ones we keep and those we send to the neighbour
void L1GctHardwareJetFinder::convertClustersToProtoJets()
{
  unsigned numberOfSentJets = 0;
  unsigned numberOfKeptJets = 0;
  for (unsigned j=0; j<MAX_JETS_OUT; ++j) {
    if (m_clusters.at(j).et()>=JET_THRESHOLD) {
      if (m_clusters.at(j).rctPhi()==0) {
	m_sentProtoJets.at(numberOfKeptJets++) = m_clusters.at(j);
      } else {
	m_keptProtoJets.at(numberOfSentJets++) = m_clusters.at(j);
      }
    }
  }
}

/// Organise the final clustered jets into L1GctJets
void L1GctHardwareJetFinder::convertClustersToOutputJets()
{
  for (unsigned j=0; j<MAX_JETS_OUT; ++j) {
    if (m_clusters.at(j).et()>=JET_THRESHOLD) {
      unsigned rawsum = m_clusters.at(j).et();
      if (m_clusters.at(j).overFlow()) { rawsum = rawsum | (1<<L1GctJet::RAWSUM_BITWIDTH); }
      L1GctJet temp(rawsum, m_clusters.at(j).gctEta(), m_clusters.at(j).gctPhi(), m_clusters.at(j).tauVeto(), m_jetEtCalLut);
      m_outputJets.at(j) = temp;
    }
  }
}

