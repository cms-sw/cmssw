// Original author: Brock Tweedie (JHU)
// Ported to CMSSW by: Sal Rappoccio (JHU)
// $Id: CATopJetAlgorithm.cc,v 1.2 2008/11/14 18:56:33 srappocc Exp $

#include "RecoJets/JetAlgorithms/interface/CATopJetAlgorithm.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"


using namespace std;
using namespace reco;
using namespace JetReco;
using namespace edm;


//  Run the algorithm
//  ------------------
void CATopJetAlgorithm::run( const vector<fastjet::PseudoJet> & cell_particles, 
			     vector<CompoundPseudoJet> & hardjetsOutput,
			     edm::EventSetup const & c )  
{

  bool verbose = false;

  if ( verbose ) cout << "Welcome to CATopSubJetAlgorithm::run" << endl;

  // Sum Et of the event
  double sumEt = 0.;


  //make a list of input objects ordered by ET and calculate sum et


  // list of fastjet pseudojet constituents
  for (unsigned i = 0; i < cell_particles.size(); ++i) {
    sumEt += cell_particles[i].perp();
  }

  // Determine which bin we are in for et clustering
  int iPt = -1;
  for ( unsigned int i = 0; i < ptBins_.size(); ++i ) {
    if ( sumEt / 2.0 > ptBins_[i] ) iPt = i;
  }

  if ( verbose ) cout << "Using sumEt = " << sumEt << ", bin = " << iPt << endl;

  // If the sum et is too low, exit
  if ( iPt < 0 ) {    
    return;
  }

  // Get the calo geometry
  edm::ESHandle<CaloGeometry> geometry;
  c.get<CaloGeometryRecord>().get(geometry);
  const CaloSubdetectorGeometry* towerGeometry = 
    geometry->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);  

  // empty 4-vector
  fastjet::PseudoJet blankJetA(0,0,0,0);
  blankJetA.set_user_index(-1);
  const fastjet::PseudoJet blankJet = blankJetA;


  int nCellMin = nCellBins_[iPt];

  if ( verbose ) cout << "Using nCellMin = " << nCellMin << endl;


  // Define strategy, recombination scheme, and jet definition
  fastjet::Strategy strategy = fastjet::Best;
  fastjet::RecombinationScheme recombScheme = fastjet::E_scheme;

  // pick algorithm
  fastjet::JetAlgorithm algorithm = static_cast<fastjet::JetAlgorithm>( algorithm_ );

  fastjet::JetDefinition jetDef( algorithm, 
				 rBins_[iPt], recombScheme, strategy);

  if ( verbose ) cout << "About to do jet clustering in CA" << endl;
  // run the jet clustering
  fastjet::ClusterSequence clusterSeq(cell_particles, jetDef);

  if ( verbose ) cout << "Getting inclusive jets" << endl;
  // Get the transient inclusive jets
  vector<fastjet::PseudoJet> inclusiveJets = clusterSeq.inclusive_jets(ptMin_);

  if ( verbose ) cout << "Getting central jets" << endl;
  // Find the transient central jets
  vector<fastjet::PseudoJet> centralJets;
  for (unsigned int i = 0; i < inclusiveJets.size(); i++) {
    if (inclusiveJets[i].perp() > ptMin_ && fabs(inclusiveJets[i].rapidity()) < centralEtaCut_) {
      centralJets.push_back(inclusiveJets[i]);
    }
  }
  // Sort the transient central jets in Et
  GreaterByEtPseudoJet compEt;
  sort( centralJets.begin(), centralJets.end(), compEt );


  // These will store the 4-vectors of each hard jet
  vector<math::XYZTLorentzVector> p4_hardJets;

  // These will store the indices of each subjet that 
  // are present in each jet
  vector<vector<int> > indices( centralJets.size() );

  // Loop over central jets, attempt to find substructure
  vector<fastjet::PseudoJet>::iterator jetIt = centralJets.begin(),
    centralJetsBegin = centralJets.begin(),
    centralJetsEnd = centralJets.end();
  for ( ; jetIt != centralJetsEnd; ++jetIt ) {

    fastjet::PseudoJet localJet = *jetIt;

    // Get the 4-vector for this jet
    p4_hardJets.push_back( math::XYZTLorentzVector(localJet.px(), localJet.py(), localJet.pz(), localJet.e() ));

    // jet decomposition.  try to find 3 or 4 hard, well-localized subjets, characteristic of a boosted top.
    double ptHard = ptFracBins_[iPt]*localJet.perp();
    vector<fastjet::PseudoJet> leftoversAll;

    // stage 1:  primary decomposition.  look for when the jet declusters into two hard subjets
    if ( verbose ) cout << "Doing decomposition 1" << endl;
    fastjet::PseudoJet ja, jb;
    vector<fastjet::PseudoJet> leftovers1;
    bool hardBreak1 = decomposeJet(localJet,clusterSeq,cell_particles,towerGeometry,ptHard,nCellMin,ja,jb,leftovers1);
    leftoversAll.insert(leftoversAll.end(),leftovers1.begin(),leftovers1.end());
	
    // stage 2:  secondary decomposition.  look for when the hard subjets found above further decluster into two hard sub-subjets
    //
    // ja -> jaa+jab ?
    if ( verbose ) cout << "Doing decomposition 2" << endl;
    fastjet::PseudoJet jaa, jab;
    vector<fastjet::PseudoJet> leftovers2a;
    bool hardBreak2a = false;
    if (hardBreak1)  hardBreak2a = decomposeJet(ja,clusterSeq,cell_particles,towerGeometry,ptHard,nCellMin,jaa,jab,leftovers2a);
    leftoversAll.insert(leftoversAll.end(),leftovers2a.begin(),leftovers2a.end());
    // jb -> jba+jbb ?
    fastjet::PseudoJet jba, jbb;
    vector<fastjet::PseudoJet> leftovers2b;
    bool hardBreak2b = false;
    if (hardBreak1)  hardBreak2b = decomposeJet(jb,clusterSeq,cell_particles,towerGeometry,ptHard,nCellMin,jba,jbb,leftovers2b);
    leftoversAll.insert(leftoversAll.end(),leftovers2b.begin(),leftovers2b.end());

    // NOTE:  it might be good to consider some checks for whether these subjets can be further decomposed.  e.g., the above procedure leaves
    //        open the possibility of "subjets" that actually consist of two or more distinct hard clusters.  however, this kind of thing
    //        is a rarity for the simulations so far considered.

    // proceed if one or both of the above hard subjets successfully decomposed
    if ( verbose ) cout << "Done with decomposition" << endl;

    int nBreak2 = 0;
    fastjet::PseudoJet hardA = blankJet, hardB = blankJet, hardC = blankJet, hardD = blankJet;
    if (!hardBreak2a && !hardBreak2b) { nBreak2 = 0; hardA = ja;  hardB = jb;  hardC = blankJet; hardD = blankJet; }
    if ( hardBreak2a && !hardBreak2b) { nBreak2 = 1; hardA = jaa; hardB = jab; hardC = jb;       hardD = blankJet; }
    if (!hardBreak2a &&  hardBreak2b) { nBreak2 = 1; hardA = jba; hardB = jbb; hardC = ja;       hardD = blankJet;}
    if ( hardBreak2a &&  hardBreak2b) { nBreak2 = 2; hardA = jaa; hardB = jab; hardC = jba;      hardD = jbb; }

    // check if we are left with >= 3 hard subjets
    fastjet::PseudoJet subjet1 = blankJet;
    fastjet::PseudoJet subjet2 = blankJet;
    fastjet::PseudoJet subjet3 = blankJet;
    fastjet::PseudoJet subjet4 = blankJet;
    subjet1 = hardA; subjet2 = hardB; subjet3 = hardC; subjet4 = hardD;

    // record the hard subjets
    vector<fastjet::PseudoJet> hardSubjets;
    
    if ( subjet1.user_index() >= 0 )
      hardSubjets.push_back(subjet1);
    if ( subjet2.user_index() >= 0 )
      hardSubjets.push_back(subjet2);
    if ( subjet3.user_index() >= 0 )
      hardSubjets.push_back(subjet3);
    if ( subjet4.user_index() >= 0 )
      hardSubjets.push_back(subjet4);
    sort(hardSubjets.begin(), hardSubjets.end(), compEt );

    // create the subjets objects to put into the "output" objects
    vector<CompoundPseudoSubJet>  subjetsOutput;
    std::vector<fastjet::PseudoJet>::const_iterator itSubJetBegin = hardSubjets.begin(),
      itSubJet = itSubJetBegin, itSubJetEnd = hardSubjets.end();
    for (; itSubJet != itSubJetEnd; ++itSubJet ){
      //       if ( verbose ) cout << "Adding input collection element " << (*itSubJet).user_index() << endl;
      //       if ( (*itSubJet).user_index() >= 0 && (*itSubJet).user_index() < cell_particles.size() )

      // Get the transient subjet constituents from fastjet
      vector<fastjet::PseudoJet> subjetFastjetConstituents = clusterSeq.constituents( *itSubJet );

      // Get the indices of the constituents
      vector<int> constituents;

      // Loop over the constituents and get the indices
      vector<fastjet::PseudoJet>::const_iterator fastSubIt = subjetFastjetConstituents.begin(),
	transConstBegin = subjetFastjetConstituents.begin(),
	transConstEnd = subjetFastjetConstituents.end();
      for ( ; fastSubIt != transConstEnd; ++fastSubIt ) {
	if ( fastSubIt->user_index() >= 0 && static_cast<unsigned int>(fastSubIt->user_index()) < cell_particles.size() ) {
	  constituents.push_back( fastSubIt->user_index() );
	}
      }

      // Make a CompoundPseudoSubJet object to hold this subjet and the indices of its constituents
      subjetsOutput.push_back( CompoundPseudoSubJet( *itSubJet, constituents ) );
    }
    
    
    // Make a CompoundPseudoJet object to hold this hard jet, and the subjets that make it up
    hardjetsOutput.push_back( CompoundPseudoJet( *jetIt, subjetsOutput ) );
    
  }
}




//-----------------------------------------------------------------------
// determine whether two clusters (made of calorimeter towers) are living on "adjacent" cells.  if they are, then
// we probably shouldn't consider them to be independent objects!
//
// From Sal: Ignoring genjet case
//
bool CATopJetAlgorithm::adjacentCells(const fastjet::PseudoJet & jet1, const fastjet::PseudoJet & jet2, 
				      const vector<fastjet::PseudoJet> & cell_particles,
				      const CaloSubdetectorGeometry  * fTowerGeometry,
				      const fastjet::ClusterSequence & theClusterSequence,
				      int nCellMin ) const {

  // Get each tower (depending on user input, can be either max pt tower, or centroid).
  CaloTowerDetId tower1 = getCaloTower( jet1, cell_particles, fTowerGeometry, theClusterSequence );
  CaloTowerDetId tower2 = getCaloTower( jet2, cell_particles, fTowerGeometry, theClusterSequence );

  // Get the number of calo towers away that the two towers are.
  // Can be non-integral fraction if "centroid" case is chosen.
  int distance = getDistance( tower1, tower2, fTowerGeometry );

  if ( distance <= nCellMin ) return true;
  else return false;
}

//-------------------------------------------------------------------------
// Find the highest pt tower inside the jet
fastjet::PseudoJet CATopJetAlgorithm::getMaxTower( const fastjet::PseudoJet & jet,
						   const vector<fastjet::PseudoJet> & cell_particles,
						   const fastjet::ClusterSequence & theClusterSequence
						   ) const
{
  // If this jet is a calo tower itself, return it
  if ( jet.user_index() > 0 ) return jet;
  // Check for the bug in fastjet where it sets the user_index to 0 instead of -1
  // in the clustering, since it might actually BE zero. 
  else if (  jet.user_index() == 0 && jet.perp() > 0 && jet.perp() == cell_particles[0].perp() ) {
    return jet;
  }
  // Otherwise, search through the constituents and find the highest pt tower, return it.
  else {
    vector<fastjet::PseudoJet> constituents = theClusterSequence.constituents( jet );
    GreaterByEtPseudoJet compEt;
    sort( constituents.begin(), constituents.end(), compEt );
    return constituents[0];
  }
}


//-------------------------------------------------------------------------
// Find the calo tower associated with the jet
CaloTowerDetId CATopJetAlgorithm::getCaloTower( const fastjet::PseudoJet & jet,
						const vector<fastjet::PseudoJet> & cell_particles,
						const CaloSubdetectorGeometry  * fTowerGeometry,
						const fastjet::ClusterSequence & theClusterSequence ) const
{

  // If the jet is just a single calo tower, this is trivial
  bool isTower = jet.user_index() > 0;
  // This is where it really IS index 0.
  // There's a bug in fastjet. They set the user_index to 0 instead of -1 for the previous output. 
  // Need to check if it's "really" index 0 in input, or it's a reconstructed jet.
  if ( jet.user_index() == 0 && jet.perp() > 0 && jet.perp() == cell_particles[0].perp() ) isTower = true;

//   if ( isTower ) {
//     return cell_particles[jet.user_index()].id();
//   }

//   // If the user requested to get the max tower, return the max tower
//   if ( useMaxTower_ ) {
//     return cell_particles[getMaxTower( jet, cell_particles, theClusterSequence ).user_index()].id();
//   }


  // Otherwise, we find the closest calorimetery tower to the jet centroid
  reco::Particle::LorentzVector v( jet.px(), jet.py(), jet.pz(), jet.e() );
  GlobalPoint centroid( v.x(), v.y(), v.z() );

  // Find the closest calo det id
  CaloTowerDetId detId ( fTowerGeometry->getClosestCell( centroid ) );

  // Return closest calo det id
  return detId;
}

//-------------------------------------------------------------------------
// Get number of calo towers away that the two calo towers are
int CATopJetAlgorithm::getDistance ( CaloTowerDetId const & t1, CaloTowerDetId const & t2, 
				     const CaloSubdetectorGeometry  * fTowerGeometry ) const
{

  int ieta1 = t1.ieta();
  int iphi1 = t1.iphi();
  int ieta2 = t2.ieta();
  int iphi2 = t2.iphi();

  int deta = abs(ieta2 - ieta1);
  int dphi = abs(iphi2 - iphi1);
  
  while ( dphi >= CaloTowerDetId::kMaxIEta ) dphi -= CaloTowerDetId::kMaxIEta;
  while ( dphi <= 0 ) dphi += CaloTowerDetId::kMaxIEta;

  return deta + dphi;
}

//-------------------------------------------------------------------------
// attempt to decompose a jet into "hard" subjets, where hardness is set by ptHard
//
bool CATopJetAlgorithm::decomposeJet(const fastjet::PseudoJet & theJet, 
				     const fastjet::ClusterSequence & theClusterSequence, 
				     const vector<fastjet::PseudoJet> & cell_particles,
				     const CaloSubdetectorGeometry  * fTowerGeometry,
				     double ptHard, int nCellMin,
				     fastjet::PseudoJet & ja, fastjet::PseudoJet & jb, 
				     vector<fastjet::PseudoJet> & leftovers) const {

  bool goodBreak;
  fastjet::PseudoJet j = theJet;
  leftovers.clear();
  
  while (1) {                                                      // watch out for infinite loop!
    goodBreak = theClusterSequence.has_parents(j,ja,jb);
    if (!goodBreak)                                 break;         // this is one cell, can't decluster anymore

    if ( useAdjacency_ &&
	 adjacentCells(ja,jb,cell_particles,
		       fTowerGeometry,
		       theClusterSequence,
		       nCellMin) )                  break;         // the clusters are "adjacent" in the calorimeter => shouldn't have decomposed
    if (ja.perp() < ptHard && jb.perp() < ptHard)   break;         // broke into two soft clusters, dead end
    if (ja.perp() > ptHard && jb.perp() > ptHard)   return true;   // broke into two hard clusters, we're done!
    else if (ja.perp() > jb.perp()) {                              // broke into one hard and one soft, ditch the soft one and try again
      j = ja;
      vector<fastjet::PseudoJet> particles = theClusterSequence.constituents(jb);
      leftovers.insert(leftovers.end(),particles.begin(),particles.end());
    }
    else {
      j = jb;
      vector<fastjet::PseudoJet> particles = theClusterSequence.constituents(ja);
      leftovers.insert(leftovers.end(),particles.begin(),particles.end());
    }
  }

  // did not decluster into hard subjets
  ja.reset(0,0,0,0);
  jb.reset(0,0,0,0);
  leftovers.clear();
  return false;
}
