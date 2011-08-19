#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHardwareJetFinder.h"

//DEFINE STATICS
const unsigned int L1GctHardwareJetFinder::N_COLS = 2;
const unsigned int L1GctHardwareJetFinder::CENTRAL_COL0 = 0;
const unsigned int L1GctHardwareJetFinder::MAX_REGIONS_IN = (((L1CaloRegionDetId::N_ETA)/2)+N_EXTRA_REGIONS_ETA00)*L1GctHardwareJetFinder::N_COLS;

L1GctHardwareJetFinder::L1GctHardwareJetFinder(int id):
  L1GctJetFinderBase(id),
  m_positiveEtaWheel(id >= (int) (L1CaloRegionDetId::N_PHI/2)),
  m_localMaxima     (MAX_JETS_OUT),
  m_clusters        (MAX_JETS_OUT),
  m_numberOfClusters(0),
  m_localMax00(2),
  m_cluster00 (2)
{
  this->reset();
  // Initialise parameters for Region input calculations in the 
  // derived class so we get the right values of constants.
  static const unsigned NPHI = L1CaloRegionDetId::N_PHI;
  m_minColThisJf = (NPHI + m_id*2 - CENTRAL_COL0) % NPHI;
}

L1GctHardwareJetFinder::~L1GctHardwareJetFinder()
{
}

std::ostream& operator << (std::ostream& os, const L1GctHardwareJetFinder& algo)
{
  os << "===L1GctHardwareJetFinder===" << std::endl;
  const L1GctJetFinderBase* temp = &algo;
  os << *temp;
  return os;
}

void L1GctHardwareJetFinder::reset()
{
  L1GctJetFinderBase::reset();
}

void L1GctHardwareJetFinder::fetchInput()
{
  if (setupOk()) {
    findProtoJets();
  }
}

void L1GctHardwareJetFinder::process() 
{
  if (setupOk()) {
    fetchProtoJetsFromNeighbour(TOPBOT);
    findJets();
    sortJets();
    doEnergySums();
  }
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
  findFinalClusters();
  convertClustersToOutputJets();
}

/// Both clustering stages need to find local maxima in the search array
//  Find the local et maxima in the 2x11 array of regions
void L1GctHardwareJetFinder::findLocalMaxima()
{
  m_localMaxima.clear();
  m_localMaxima.resize(MAX_JETS_OUT);
  m_localMax00.clear();
  m_localMax00.resize(2);

  UShort jetNum = 0; //holds the number of jets currently found
  UShort centreIndex = COL_OFFSET*this->centralCol0();
  for(UShort column = 0; column <2; ++column)  //Find jets in the central search region
  {
    // The input regions include two extra bins on the other side of eta=0. This allows "seamless" 
    // jetfinding across the eta=0 boundary. We skip the first input region in each row. We perform 
    // the full pre-clustering on the next region but store the resulting clusters separately
    // from the main list of output pre-clusters - they will be used in the final cluster stage to
    // make sure we do not produce jets in adjacent regions on opposite sides of eta=0. 
    ++centreIndex;
    for (UShort row = 1; row < COL_OFFSET; ++row)  
    {
      // Here's the array of greater-than and greater-or-equal tests
      // to ensure each localMaximum appears once and only once in the list
      // It is different for forward and backward eta.
      unsigned JET_THRESHOLD = ( (row > m_EtaBoundry) ? m_FwdJetSeed : m_CenJetSeed);
      bool localMax = !m_inputRegions.at(centreIndex).empty() && (m_inputRegions.at(centreIndex).et()>=JET_THRESHOLD);
      if (m_positiveEtaWheel) {      // Forward eta
	localMax     &= (m_inputRegions.at(centreIndex).et() >= m_inputRegions.at(centreIndex-1).et());
        if (row < (COL_OFFSET-1)) {
	  localMax   &= (m_inputRegions.at(centreIndex).et() >  m_inputRegions.at(centreIndex+1).et());
        }
        if (column==0) {
	  localMax   &= (m_inputRegions.at(centreIndex).et() >  m_inputRegions.at(centreIndex+COL_OFFSET).et());
	  localMax   &= (m_inputRegions.at(centreIndex).et() >  m_inputRegions.at(centreIndex+COL_OFFSET-1).et());
	  if (row < (COL_OFFSET-1)) {
	    localMax &= (m_inputRegions.at(centreIndex).et() >  m_inputRegions.at(centreIndex+COL_OFFSET+1).et());
	  }
        } else {
	  localMax   &= (m_inputRegions.at(centreIndex).et() >= m_inputRegions.at(centreIndex-COL_OFFSET).et());
	  localMax   &= (m_inputRegions.at(centreIndex).et() >= m_inputRegions.at(centreIndex-COL_OFFSET-1).et());
	  if (row < (COL_OFFSET-1)) { 
	    localMax &= (m_inputRegions.at(centreIndex).et() >= m_inputRegions.at(centreIndex-COL_OFFSET+1).et());
	  }
        }
      } else {      // Backward eta
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
      }
      if (localMax) {
	if (row>1) {
	  if (jetNum < MAX_JETS_OUT) {
	    m_localMaxima.at(jetNum) = m_inputRegions.at(centreIndex);
	    ++jetNum;
	  }
	} 
	// Treat row 1 as a separate case. It's not required for jetfinding but
	// is used for vetoing of jets double counted across the eta=0 boundary
	else {
	  unsigned phi = m_inputRegions.at(centreIndex).rctPhi();
	  m_localMax00.at(phi) = m_inputRegions.at(centreIndex);
	}
      }
      ++centreIndex;
    }
  }

  m_numberOfClusters = jetNum;
}

//  For each local maximum, find the cluster et in a 2x3 region.
//  The logic ensures that a given region et cannot be used in more than one cluster.
//  The sorting of the local maxima ensures the highest et maximum has priority.
void L1GctHardwareJetFinder::findProtoClusters()
{
  m_clusters.clear();
  m_clusters.resize(MAX_JETS_OUT);
  m_cluster00.clear();
  m_cluster00.resize(2);

  RegionsVector         topJets(MAX_JETS_OUT),         botJets(MAX_JETS_OUT);
  std::vector<unsigned> topJetsPosition(MAX_JETS_OUT), botJetsPosition(MAX_JETS_OUT);
  unsigned              numberOfTopJets=0,             numberOfBotJets=0;

  // Loop over local maxima
  for (unsigned j=0; j<m_numberOfClusters; ++j) {
    // Make a proto-jet cluster
    L1GctRegion temp = makeProtoJet(m_localMaxima.at(j));

    if (m_localMaxima.at(j).rctPhi()==0) {
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
  // Finally, deal with eta00 maxima
  if (!m_localMax00.at(0).empty()) m_cluster00.at(0) = makeProtoJet(m_localMax00.at(0));  
  if (!m_localMax00.at(1).empty()) m_cluster00.at(1) = makeProtoJet(m_localMax00.at(1));
}


/// Method to make a single proto-jet
L1GctRegion L1GctHardwareJetFinder::makeProtoJet(L1GctRegion localMax) {
  unsigned eta = localMax.gctEta();
  unsigned phi = localMax.gctPhi();
  int16_t  bx  = localMax.bx();

  unsigned localEta = localMax.rctEta();
  unsigned localPhi = localMax.rctPhi();

  unsigned etCluster = 0;
  bool ovrFlowOr = false;
  bool tauVetoOr = false;
  unsigned rgnsAboveIsoThreshold = 0;

  // check for row00
  const unsigned midEta=(L1CaloRegionDetId::N_ETA)/2;
  bool wrongEtaWheel = ( (!m_positiveEtaWheel) && (eta>=midEta) ) || ( (m_positiveEtaWheel) && (eta<midEta) );

  // Which rows are we looking over?
  unsigned rowStart, rowEnd, rowMid;
  static const unsigned row0 = N_EXTRA_REGIONS_ETA00 - 1;
  if (wrongEtaWheel) {
    if (localEta > row0 - 1) {
      rowStart = 0;
      rowMid = 0;
    } else {
      rowStart = row0 - 1 - localEta;
      rowMid = rowStart + 1;
    }
    if (localEta > row0 + 2) { // Shouldn't happen, but big problems if it does
      rowEnd = 0;
    } else { 
      rowEnd   = row0 + 2 - localEta;
    }
  } else {
    rowStart = row0 + localEta;
    rowMid = rowStart + 1;
    if (localEta < COL_OFFSET - row0 - 2) {
      rowEnd = rowStart + 3;
    } else {
      rowEnd = COL_OFFSET;
    }
  }

  for (unsigned row=rowStart; row<rowEnd; ++row) {
    for (unsigned column=0; column<2; ++column) {
      unsigned index = column*COL_OFFSET + row;
      etCluster += m_inputRegions.at(index).et();
      ovrFlowOr |= m_inputRegions.at(index).overFlow();
      // Distinguish between central and tau-flagged jets. Two versions of the algorithm.
      if (m_useImprovedTauAlgo) {

//===========================================================================================
// "Old" version of improved tau algorithm tests the tau veto for the central region always
// 	if ((row==(localEta+N_EXTRA_REGIONS_ETA00)) && (column==localPhi)) {
// 	  // central region - check the tau veto
// 	  tauVetoOr |= m_inputRegions.at(index).tauVeto();
// 	} else {
// 	  // other regions - check the tau veto if required
// 	  if (!m_ignoreTauVetoBitsForIsolation) {
// 	    tauVetoOr |= m_inputRegions.at(index).tauVeto();
// 	  }
// 	  // check the region energy against the isolation threshold
// 	  if (m_inputRegions.at(index).et() >= m_tauIsolationThreshold) {
// 	    rgnsAboveIsoThreshold++;
// 	  }
// 	}
//===========================================================================================

        // In the hardware, the ignoreTauVetoBitsForIsolation switch ignores all the veto bits,
        // including the one for the central region.
	if (!((row==rowMid) && (column==localPhi))) {
	  // non-central region - check the region energy against the isolation threshold
	  if (m_inputRegions.at(index).et() >= m_tauIsolationThreshold) {
	    rgnsAboveIsoThreshold++;
	  }
	}
	// all regions - check the tau veto if required
	if (!m_ignoreTauVetoBitsForIsolation) {
	  tauVetoOr |= m_inputRegions.at(index).tauVeto();
	}
	// End of improved tau algorithm
      } else {
	// Original tau algorithm
	tauVetoOr |= m_inputRegions.at(index).tauVeto();
      }
    }
  }
  // Encode the number of towers over threshold for the isolated tau algorithm
  bool tauFeatureBit = false;
  if (m_useImprovedTauAlgo) {
    tauVetoOr     |= (rgnsAboveIsoThreshold  > 1);
    tauFeatureBit |= (rgnsAboveIsoThreshold == 1);
  }

  L1GctRegion temp(L1GctRegion::makeProtoJetRegion(etCluster, ovrFlowOr, tauVetoOr, tauFeatureBit, eta, phi, bx));
  return temp;
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
    unsigned JET_THRESHOLD = ( (localEta0 >= m_EtaBoundry) ? m_FwdJetSeed : m_CenJetSeed);
	if (et0>=JET_THRESHOLD) {
		bool storeJet=false;
		bool isolated=true;
		// eta00 boundary check/veto
		if (localEta0==0) {
		  unsigned neighbourEt=m_cluster00.at(1-localPhi0).et();
		  isolated &= et0 >= neighbourEt;
		}
		// If the jet is NOT vetoed, look at the jets found locally (m_keptProtoJets).
		// We accept the jet if there are no local jets nearby, or if the local jet
		// (there should be no more than one) has lower Et.
		if (isolated) {
		  for (unsigned k=0; k<MAX_JETS_OUT; ++k) {
			unsigned et1       = m_keptProtoJets.at(k).et();
			unsigned localEta1 = m_keptProtoJets.at(k).rctEta();
			unsigned localPhi1 = m_keptProtoJets.at(k).rctPhi();
			if (et1>0) {
			  bool distantJet = ((localPhi0==localPhi1) ||
						       (localEta1 > localEta0+1) || (localEta0 > localEta1+1));

			  isolated &=  distantJet;
			  storeJet |= !distantJet && ((et0 > et1) || ((et0 == et1) && localPhi0==1));
			}
		  }
		}

		storeJet |= isolated;

		if (storeJet) { 
			// Start with the et sum, tau veto and overflow flags of the protoJet (2x3 regions)
			unsigned etCluster = et0;
			bool ovrFlowOr = m_rcvdProtoJets.at(j).overFlow();
			bool tauVetoOr = m_rcvdProtoJets.at(j).tauVeto();
			unsigned rgnsAboveIsoThreshold = ( m_rcvdProtoJets.at(j).featureBit0() ? 1 : 0);

			// Combine with the corresponding regions from
			// the local array to make a 3x3 jet cluster 
			unsigned column=1-localPhi0;
			// Which rows are we looking over?
			unsigned rowStart, rowEnd;
			static const unsigned row0 = N_EXTRA_REGIONS_ETA00 - 1;
			rowStart = row0 + localEta0;
			if (localEta0 < COL_OFFSET - row0 - 2) {
			  rowEnd = rowStart + 3;
			} else {
			  rowEnd = COL_OFFSET;
			}
			unsigned index = COL_OFFSET*(this->centralCol0()+column) + rowStart;
			  for (unsigned row=rowStart; row<rowEnd; ++row) {
			    etCluster += m_inputRegions.at(index).et();
			    ovrFlowOr |= m_inputRegions.at(index).overFlow();
				if (m_useImprovedTauAlgo) {
				  if (!m_ignoreTauVetoBitsForIsolation) {
				    tauVetoOr |= m_inputRegions.at(index).tauVeto();
				  }
				  // check the region energy against the isolation threshold
				  if (m_inputRegions.at(index).et() >= m_tauIsolationThreshold) {
				    rgnsAboveIsoThreshold++;
				  }
				} else {
				  tauVetoOr |= m_inputRegions.at(index).tauVeto();
				}

				++index;
			  }

			// Store the new jet
			unsigned eta = m_rcvdProtoJets.at(j).gctEta();
			unsigned phi = m_rcvdProtoJets.at(j).gctPhi();
			int16_t  bx  = m_rcvdProtoJets.at(j).bx();

			// Use the number of towers over threshold for the isolated tau algorithm
			if (m_useImprovedTauAlgo) {
			  tauVetoOr     |= (rgnsAboveIsoThreshold  > 1);
			}

			L1GctRegion temp(L1GctRegion::makeFinalJetRegion(etCluster, ovrFlowOr, tauVetoOr, eta, phi, bx));
			m_clusters.at(j) = temp;

		}
	}
  }
}

/// Organise the pre-clustered jets into the ones we keep and those we send to the neighbour
void L1GctHardwareJetFinder::convertClustersToProtoJets()
{
  for (unsigned j=0; j<MAX_JETS_OUT; ++j) {
    bool isForward = (m_clusters.at(j).rctEta()>=m_EtaBoundry);
    unsigned JET_THRESHOLD = ( isForward ? m_FwdJetSeed : m_CenJetSeed);
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
    bool isForward = (m_clusters.at(j).rctEta()>=m_EtaBoundry);
    unsigned JET_THRESHOLD = ( isForward ? m_FwdJetSeed : m_CenJetSeed);
    if (m_clusters.at(j).et()>=JET_THRESHOLD) {
      L1GctJet temp(m_clusters.at(j).et(), m_clusters.at(j).gctEta(), m_clusters.at(j).gctPhi(), 
                    m_clusters.at(j).overFlow(), isForward, m_clusters.at(j).tauVeto(), m_clusters.at(j).bx());
      m_outputJets.at(j) = temp;
    }
  }
}

