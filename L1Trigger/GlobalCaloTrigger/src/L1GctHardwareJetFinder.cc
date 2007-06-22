#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHardwareJetFinder.h"
 
#include "FWCore/Utilities/interface/Exception.h"  

#include <iostream>
using namespace std;

//DEFINE STATICS
const unsigned int L1GctHardwareJetFinder::MAX_REGIONS_IN = (((L1CaloRegionDetId::N_ETA)/2)+1)*L1GctHardwareJetFinder::N_COLS;

const unsigned int L1GctHardwareJetFinder::N_COLS = 2;
const unsigned int L1GctHardwareJetFinder::CENTRAL_COL0 = 0;

const unsigned int L1GctHardwareJetFinder::JET_THRESHOLD = 1;

L1GctHardwareJetFinder::L1GctHardwareJetFinder(int id):
  L1GctJetFinderBase(id),
  m_protoJetRegions(MAX_REGIONS_IN)
{
  // Setup the position info in protoJetRegions.
  for (unsigned column=0; column<2; ++column) {
    for (unsigned row=0; row<COL_OFFSET; ++row) {
      unsigned ieta;
      unsigned iphi;
      if (id<static_cast<int>(L1CaloRegionDetId::N_PHI/2)) {
	ieta = (L1CaloRegionDetId::N_ETA/2-row);
	iphi = (L1CaloRegionDetId::N_PHI - (m_id)*2 + 4)%L1CaloRegionDetId::N_PHI + (1-column);
      } else {
	ieta = (L1CaloRegionDetId::N_ETA/2-1+row);
	iphi = ((L1CaloRegionDetId::N_PHI - m_id)*2 + 4)%L1CaloRegionDetId::N_PHI + (1-column);
      }
      L1CaloRegion temp(0, ieta, iphi, 0);
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
  findProtoJets();
}

void L1GctHardwareJetFinder::process() 
{
  fetchProtoJetsFromNeighbour(TOPBOT);
  findJets();
  sortJets();
  doEnergySums();
}

/// HERE IS THE JETFINDER CODE

/// The first stage of clustering, called by fetchInput()
void L1GctHardwareJetFinder::findProtoJets()
{
  findLocalMaxima();
  findProtoClusters();
  convertClustersToProtoJets();
}

/// The second stage of clustering, called by process()
void L1GctHardwareJetFinder::findJets()
{
  fillRegionsFromProtoJets();
  findFinalClusters();
  convertClustersToOutputJets();
}

/// Both clustering stages need to find local maxima in the search array
//  Find the local et maxima in the 2x11 array of regions
void L1GctHardwareJetFinder::findLocalMaxima()
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
      bool localMax = (m_inputRegions.at(centreIndex).et()>=JET_THRESHOLD);
      localMax     &= (m_inputRegions.at(centreIndex).et() >  m_inputRegions.at(centreIndex-1).et());
      if (row < (COL_OFFSET-1)) {
	localMax   &= (m_inputRegions.at(centreIndex).et() >= m_inputRegions.at(centreIndex+1).et());
      }
      if (column==0) {
	localMax   &= (m_inputRegions.at(centreIndex).et() >= m_inputRegions.at(centreIndex+COL_OFFSET).et());
	localMax   &= (m_inputRegions.at(centreIndex).et() >= m_inputRegions.at(centreIndex+COL_OFFSET-1).et());
	if (row < (COL_OFFSET-1)) {
	  localMax &= (m_inputRegions.at(centreIndex).et() >= m_inputRegions.at(centreIndex+COL_OFFSET+1).et());
	}
      } else {
	localMax   &= (m_inputRegions.at(centreIndex).et() >  m_inputRegions.at(centreIndex-COL_OFFSET).et());
	localMax   &= (m_inputRegions.at(centreIndex).et() >  m_inputRegions.at(centreIndex-COL_OFFSET-1).et());
	if (row < (COL_OFFSET-1)) {
	  localMax &= (m_inputRegions.at(centreIndex).et() >  m_inputRegions.at(centreIndex-COL_OFFSET+1).et());
	}
      }
      if (localMax) {
        assert(jetNum < MAX_JETS_OUT);
                
        m_localMaxima.at(jetNum) = m_inputRegions.at(centreIndex);
        ++jetNum;
      }
      ++centreIndex;
    }
  }

  m_numberOfClusters = jetNum;

}

/// Both clustering stages need to convert local maxima to clusters
//  For each local maximum, find the cluster et in a 2x3 region.
//  The logic ensures that a given region et cannot be used in more than one cluster.
//  The sorting of the local maxima ensures the highest et maximum has priority.
void L1GctHardwareJetFinder::findProtoClusters()
{
  m_clusters.clear();
  m_clusters.resize(MAX_JETS_OUT);

  RegionsVector    topJets(MAX_JETS_OUT),         botJets(MAX_JETS_OUT);
  vector<unsigned> topJetsPosition(MAX_JETS_OUT), botJetsPosition(MAX_JETS_OUT);
  unsigned         numberOfTopJets=0,             numberOfBotJets=0;


  // Loop over local maxima
  for (unsigned j=0; j<m_numberOfClusters; ++j) {
    unsigned localEta = m_localMaxima.at(j).rctEta();
    unsigned localPhi = m_localMaxima.at(j).rctPhi();

    unsigned etCluster = 0;
    bool tauVetoOr = false;
    bool ovrFlowOr = false;

    for (unsigned row=localEta; ((row<(localEta+3)) && (row<COL_OFFSET)); ++row) {
	for (unsigned column=0; column<2; ++column) {
	    unsigned index = column*COL_OFFSET + row;
	    etCluster += m_inputRegions.at(index).et();
	    tauVetoOr |= m_inputRegions.at(index).tauVeto();
	    ovrFlowOr |= m_inputRegions.at(index).overFlow();
	}
    }
    unsigned eta = m_localMaxima.at(j).gctEta();
    unsigned phi = m_localMaxima.at(j).gctPhi();

    L1CaloRegion temp(etCluster, ovrFlowOr, tauVetoOr, false, false, eta, phi);
    if (localPhi==0) {
    // Store "top edge" jets
      topJets.at(numberOfTopJets) = temp;
      topJetsPosition.at(numberOfTopJets) = 0;
      for (unsigned k=0; k<numberOfTopJets; ++k) {
        if (topJets.at(numberOfTopJets).et() >= topJets.at(k).et()) { ++topJetsPosition.at(k); }
        if (topJets.at(numberOfTopJets).et() <= topJets.at(k).et()) { ++topJetsPosition.at(numberOfTopJets); }
      }
      ++numberOfTopJets;
    } else {
    // Store "bottom edge" jets
      botJets.at(numberOfBotJets) = temp;
      botJetsPosition.at(numberOfBotJets) = 0;
      for (unsigned k=0; k<numberOfBotJets; ++k) {
        if (botJets.at(numberOfBotJets).et() >= botJets.at(k).et()) { ++botJetsPosition.at(k); }
        if (botJets.at(numberOfBotJets).et() <= botJets.at(k).et()) { ++botJetsPosition.at(numberOfBotJets); }
      }
      ++numberOfBotJets;
    }
  }
  // Now we've found all the proto-jets, copy the best ones to the output array
  //
  // We fill the first half of the array with "bottom jets"
  // and the remainder with "top jets". For cases where
  // we have found too many jets in one phi column,
  // we keep those with the highest Et.
  static const unsigned int MAX_TOPBOT_JETS = MAX_JETS_OUT/2;
  unsigned pos=0;
  for (unsigned j=0; j<numberOfBotJets; ++j) {
    if (botJetsPosition.at(j)<MAX_TOPBOT_JETS) {
      m_clusters.at(pos++) = botJets.at(j);
    }
  }
  pos=MAX_TOPBOT_JETS;
  for (unsigned j=0; j<numberOfTopJets; ++j) {
    if (topJetsPosition.at(j)<MAX_TOPBOT_JETS) {
      m_clusters.at(pos++) = topJets.at(j);
    }
  }
}

/// Convert protojets to final jets
void L1GctHardwareJetFinder::findFinalClusters()
{
  m_clusters.clear();
  m_clusters.resize(MAX_JETS_OUT);

  // Loop over proto-jets received from neighbours.
  // Form a jet to send to the output if there is no proto-jet nearby in the
  // list of jets found locally. If local jets are found nearby, form a jet
  // if the received jet has higher Et than any one of the local ones.
  for (unsigned j=0; j<MAX_JETS_OUT; ++j) {
    unsigned et0       = m_rcvdProtoJets.at(j).et();
    unsigned localEta0 = m_rcvdProtoJets.at(j).rctEta();
    unsigned localPhi0 = m_rcvdProtoJets.at(j).rctPhi();
	if (et0>=JET_THRESHOLD) {
		bool storeJet=false;
		bool isolated=true;
		for (unsigned k=0; k<MAX_JETS_OUT; ++k) {
			unsigned et1       = m_keptProtoJets.at(k).et();
			unsigned localEta1 = m_keptProtoJets.at(k).rctEta();
			unsigned localPhi1 = m_keptProtoJets.at(k).rctPhi();
    //---------------------------------------------------------------------------------
    // The following emulates a VHDL "feature" that accepts a small number
    // of low-energy jets at large eta that shouldn't really get through
			if (et1==0)
				{
					localEta1=L1CaloRegionDetId::N_ETA/2;
					localPhi1=(k<(MAX_JETS_OUT/2) ? 1 : 0);
				}
    //---------------------------------------------------------------------------------
			bool distantJet = ((localPhi0==localPhi1) ||
						       (localEta1 > localEta0+1) || (localEta0 > localEta1+1));

		    isolated &=  distantJet;
			storeJet |= !distantJet && ((et0 > et1) || ((et0 == et1) && localPhi0==1));

		}

		storeJet |= isolated;

		if (storeJet) {

			// Start with the et sum, tau veto and overflow flags of the protoJet (2x3 regions)
			unsigned etCluster = et0;
			bool tauVetoOr = m_rcvdProtoJets.at(j).tauVeto();
			bool ovrFlowOr = m_rcvdProtoJets.at(j).overFlow();

			// Combine with the corresponding regions from
			// the local array to make a 3x3 jet cluster 
			unsigned column=1-localPhi0;
			unsigned index = COL_OFFSET*(this->centralCol0()+column)+localEta0;
			for (unsigned row=localEta0; ((row<(localEta0+3)) && (row<COL_OFFSET)); ++row) {
				etCluster += m_inputRegions.at(index).et();
				tauVetoOr |= m_inputRegions.at(index).tauVeto();
				ovrFlowOr |= m_inputRegions.at(index).overFlow();
				++index;
			}

			// Store the new jet
			unsigned eta = m_rcvdProtoJets.at(j).gctEta();
			unsigned phi = m_rcvdProtoJets.at(j).gctPhi();

			L1CaloRegion temp(etCluster, ovrFlowOr, tauVetoOr, false, false, eta, phi);
			m_clusters.at(j) = temp;

		}
	}
  }
}

/// Fill search array for the second stage of clustering based on the pre-clustered jets
void L1GctHardwareJetFinder::fillRegionsFromProtoJets()
{
  static const unsigned int MAX_TOPBOT_JETS = MAX_JETS_OUT/2;
  for (unsigned j=0; j<MAX_JETS_OUT; ++j) {
    if (m_rcvdProtoJets.at(j).et()>=JET_THRESHOLD) {
      unsigned eta0 = m_rcvdProtoJets.at(j).rctEta();
      if (j>=MAX_TOPBOT_JETS) { eta0 += COL_OFFSET; }
      m_protoJetRegions.at(eta0+1) = m_rcvdProtoJets.at(j);
    }
  }
}

/// Organise the pre-clustered jets into the ones we keep and those we send to the neighbour
void L1GctHardwareJetFinder::convertClustersToProtoJets()
{
  for (unsigned j=0; j<MAX_JETS_OUT; ++j) {
    if (m_clusters.at(j).et()>=JET_THRESHOLD) {
	m_keptProtoJets.at(j) = m_clusters.at(j);
	m_sentProtoJets.at(j) = m_clusters.at(j);
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
      L1GctJet temp(rawsum, m_clusters.at(j).gctEta(), m_clusters.at(j).gctPhi(), m_clusters.at(j).tauVeto());
      m_outputJets.at(j) = temp;
    }
  }
}

