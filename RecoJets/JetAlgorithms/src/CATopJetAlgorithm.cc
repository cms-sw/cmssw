// Original author: Brock Tweedie (JHU)
// Ported to CMSSW by: Sal Rappoccio (JHU)
// $Id: CATopJetAlgorithm.cc,v 1.14 2012/09/03 18:15:21 jpilot Exp $

#include "RecoJets/JetAlgorithms/interface/CATopJetAlgorithm.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/angle.h"

using namespace std;
using namespace reco;
using namespace edm;


//  Run the algorithm
//  ------------------
void CATopJetAlgorithm::run( const vector<fastjet::PseudoJet> & cell_particles, 
			     vector<fastjet::PseudoJet> & hardjetsOutput,
			     boost::shared_ptr<fastjet::ClusterSequence> & fjClusterSeq
			     )  
{
	if ( verbose_ ) cout << "Welcome to CATopSubJetAlgorithm::run" << endl;
	
	// Sum Et of the event
	double sumEt = 0.;
	
	//make a list of input objects ordered by ET and calculate sum et
	// list of fastjet pseudojet constituents
	for (unsigned i = 0; i < cell_particles.size(); ++i) {
		sumEt += cell_particles[i].perp();
	}
	
	// Determine which bin we are in for et clustering
	int sumEtBinId = -1;
	for ( unsigned int i = 0; i < sumEtBins_.size(); ++i ) {
		if ( sumEt > sumEtBins_[i] ) sumEtBinId = i;
	}
	if ( verbose_ ) cout << "Using sumEt = " << sumEt << ", bin = " << sumEtBinId << endl;
	
	// If the sum et is too low, exit
	if ( sumEtBinId < 0 ) {    
		return;
	}
	
	// empty 4-vector
	fastjet::PseudoJet blankJetA(0,0,0,0);
	blankJetA.set_user_index(-1);
	const fastjet::PseudoJet blankJet = blankJetA;
	
	// Define adjacency variables which depend on which sumEtBin we are in
	double deltarcut = deltarBins_[sumEtBinId];
	double nCellMin = nCellBins_[sumEtBinId];
	
	if ( verbose_ )cout<<"useAdjacency_ = "<<useAdjacency_<<endl;
	if ( verbose_ && useAdjacency_==0)cout<<"No Adjacency"<<endl;
	if ( verbose_ && useAdjacency_==1)cout<<"using deltar adjacency"<<endl;
	if ( verbose_ && useAdjacency_==2)cout<<"using modified adjacency"<<endl;
	if ( verbose_ && useAdjacency_==3)cout<<"using calorimeter nearest neighbor based adjacency"<<endl;
	if ( verbose_ && useAdjacency_==1)cout << "Using deltarcut = " << deltarcut << endl;
	if ( verbose_ && useAdjacency_==3)cout << "Using nCellMin = " << nCellMin << endl;
	
	if ( verbose_ ) cout << "About to do jet clustering in CA" << endl;
	// run the jet clustering

	//cluster the jets with the jet definition jetDef:
	// run algorithm
	// boost::shared_ptr<fastjet::ClusterSequence> fjClusterSeq;
	// if ( !doAreaFastjet_ ) {
	//   fjClusterSeq = boost::shared_ptr<fastjet::ClusterSequence>( new fastjet::ClusterSequence( cell_particles, jetDef ) );
	// } else if (voronoiRfact_ <= 0) {
	//   fjClusterSeq = boost::shared_ptr<fastjet::ClusterSequence>( new fastjet::ClusterSequenceArea( cell_particles, jetDef , *fjActiveArea_ ) );
	// } else {
	//   fjClusterSeq = boost::shared_ptr<fastjet::ClusterSequence>( new fastjet::ClusterSequenceVoronoiArea( cell_particles, jetDef , fastjet::VoronoiAreaSpec(voronoiRfact_) ) );
	// }
	
	if ( verbose_ ) cout << "Getting inclusive jets" << endl;
	// Get the transient inclusive jets
	vector<fastjet::PseudoJet> inclusiveJets = fjClusterSeq->inclusive_jets(ptMin_);
	
	if ( verbose_ ) cout << "Getting central jets" << endl;
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
	  centralJetsEnd = centralJets.end();
	if ( verbose_ )cout<<"Loop over jets"<<endl;
	int i=0;
	for ( ; jetIt != centralJetsEnd; ++jetIt ) {
		if ( verbose_ )cout<<"\nJet "<<i<<endl;
		i++;
		fastjet::PseudoJet localJet = *jetIt;
		
		// Get the 4-vector for this jet
		p4_hardJets.push_back( math::XYZTLorentzVector(localJet.px(), localJet.py(), localJet.pz(), localJet.e() ));
		
		// jet decomposition.  try to find 3 or 4 hard, well-localized subjets, characteristic of a boosted top.
		if ( verbose_ )cout<<"local jet pt = "<<localJet.perp()<<endl;
		if ( verbose_ )cout<<"deltap = "<<ptFracBins_[sumEtBinId]<<endl;
		
		double ptHard = ptFracBins_[sumEtBinId]*localJet.perp();
		vector<fastjet::PseudoJet> leftoversAll;
		
		// stage 1:  primary decomposition.  look for when the jet declusters into two hard subjets
		if ( verbose_ ) cout << "Doing decomposition 1" << endl;
		fastjet::PseudoJet ja, jb;
		vector<fastjet::PseudoJet> leftovers1;
		bool hardBreak1 = decomposeJet(localJet,*fjClusterSeq,cell_particles,ptHard,nCellMin,deltarcut,ja,jb,leftovers1);
		leftoversAll.insert(leftoversAll.end(),leftovers1.begin(),leftovers1.end());
		
		// stage 2:  secondary decomposition.  look for when the hard subjets found above further decluster into two hard sub-subjets
		//
		// ja -> jaa+jab ?
		if ( verbose_ ) cout << "Doing decomposition 2. ja->jaa+jab?" << endl;
		fastjet::PseudoJet jaa, jab;
		vector<fastjet::PseudoJet> leftovers2a;
		bool hardBreak2a = false;
		if (hardBreak1)  hardBreak2a = decomposeJet(ja,*fjClusterSeq,cell_particles,ptHard,nCellMin,deltarcut,jaa,jab,leftovers2a);
		leftoversAll.insert(leftoversAll.end(),leftovers2a.begin(),leftovers2a.end());
		// jb -> jba+jbb ?
		if ( verbose_ ) cout << "Doing decomposition 2. ja->jba+jbb?" << endl;
		fastjet::PseudoJet jba, jbb;
		vector<fastjet::PseudoJet> leftovers2b;
		bool hardBreak2b = false;
		if (hardBreak1)  hardBreak2b = decomposeJet(jb,*fjClusterSeq,cell_particles,ptHard,nCellMin,deltarcut,jba,jbb,leftovers2b);
		leftoversAll.insert(leftoversAll.end(),leftovers2b.begin(),leftovers2b.end());
		
		// NOTE:  it might be good to consider some checks for whether these subjets can be further decomposed.  e.g., the above procedure leaves
		//        open the possibility of "subjets" that actually consist of two or more distinct hard clusters.  however, this kind of thing
		//        is a rarity for the simulations so far considered.
		
		// proceed if one or both of the above hard subjets successfully decomposed
		if ( verbose_ ) cout << "Done with decomposition" << endl;

 		if ( verbose_ ) cout<<"hardBreak1 = "<<hardBreak1<<endl;
		if ( verbose_ ) cout<<"hardBreak2a = "<<hardBreak2a<<endl;
		if ( verbose_ ) cout<<"hardBreak2b = "<<hardBreak2b<<endl;
            
		fastjet::PseudoJet hardA = blankJet, hardB = blankJet, hardC = blankJet, hardD = blankJet;
		if (!hardBreak1) { 
		  hardA = localJet;  
		  hardB = blankJet;  
		  hardC = blankJet; 
		  hardD = blankJet; 
		  if(verbose_)cout<<"Hardbreak failed. Save subjet1=localJet"<<endl;
		} 
		if (hardBreak1 && !hardBreak2a && !hardBreak2b) { 
		  hardA = ja;  
		  hardB = jb;  
		  hardC = blankJet; 
		  hardD = blankJet; 
		  if(verbose_)cout<<"First decomposition succeeded, both second decompositions failed. Save subjet1=ja subjet2=jb"<<endl;
		}
		if (hardBreak1 && hardBreak2a && !hardBreak2b) { 
		  hardA = jaa; 
		  hardB = jab; 
		  hardC = jb; 
		  hardD = blankJet; 
		  if(verbose_)cout<<"First decomposition succeeded, ja split succesfully, jb did not split. Save subjet1=jaa subjet2=jab subjet3=jb"<<endl;
		}
		if (hardBreak1 && !hardBreak2a &&  hardBreak2b) { 
		  hardA = jba; 
		  hardB = jbb; 
		  hardC = ja; 
		  hardD = blankJet; 
		  if(verbose_)cout<<"First decomposition succeeded, jb split succesfully, ja did not split. Save subjet1=jba subjet2=jbb subjet3=ja"<<endl;
		}
		if (hardBreak1 && hardBreak2a &&  hardBreak2b) { 
		  hardA = jaa; 
		  hardB = jab; 
		  hardC = jba; 
		  hardD = jbb; 
		  if(verbose_)cout<<"First decomposition and both secondary decompositions succeeded. Save subjet1=jaa subjet2=jab subjet3=jba subjet4=jbb"<<endl;
		}
		
		// check if we are left with >= 3 hard subjets
		fastjet::PseudoJet subjet1 = blankJet;
		fastjet::PseudoJet subjet2 = blankJet;
		fastjet::PseudoJet subjet3 = blankJet;
		fastjet::PseudoJet subjet4 = blankJet;
		subjet1 = hardA; subjet2 = hardB; subjet3 = hardC; subjet4 = hardD;
		
		// record the hard subjets
		vector<fastjet::PseudoJet> hardSubjets;

		if ( verbose_ ) {
		  std::cout << "HardA : user_index = " << hardA.user_index() << ", (Pt,Y,Phi,M) = (" 
			    << hardA.pt() << ", " << hardA.rapidity() << ", " 
			    << hardA.phi() << ", " << hardA.m() << ")" << std::endl;

		  std::cout << "HardB : user_index = " << hardB.user_index() << ", (Pt,Y,Phi,M) = (" 
			    << hardB.pt() << ", " << hardB.rapidity() << ", " 
			    << hardB.phi() << ", " << hardB.m() << ")" << std::endl;

		  std::cout << "HardC : user_index = " << hardC.user_index() << ", (Pt,Y,Phi,M) = (" 
			    << hardC.pt() << ", " << hardC.rapidity() << ", " 
			    << hardC.phi() << ", " << hardC.m() << ")" << std::endl;

		  std::cout << "HardD : user_index = " << hardD.user_index() << ", (Pt,Y,Phi,M) = (" 
			    << hardD.pt() << ", " << hardD.rapidity() << ", " 
			    << hardD.phi() << ", " << hardD.m() << ")" << std::endl;
		}

		// Check to see if any subjects are counted amongst the "hard" subjets from previous
		// lines. NOTE: In Fastjet 3.0, the default "user_index" changed from 0 to -1, so
		// this can no longer be used as a designator for the veto of "blankJet" subjets,
		// and now switch to pt > some small value. 
		if ( subjet1.pt() > 0.0001 )
			hardSubjets.push_back(subjet1);
		if ( subjet2.pt() > 0.0001 )
			hardSubjets.push_back(subjet2);
		if ( subjet3.pt() > 0.0001 )
			hardSubjets.push_back(subjet3);
		if ( subjet4.pt() > 0.0001 )
			hardSubjets.push_back(subjet4);
		sort(hardSubjets.begin(), hardSubjets.end(), compEt );

		// Use new fastjet functionality to create a Pseudojet from constituents
		fastjet::PseudoJet candidate = join(hardSubjets);
		// Reset the jet's 4-vector to the "ungroomed" value
		candidate.reset_momentum( jetIt->px(), jetIt->py(), jetIt->pz(), jetIt->e() );

		if ( verbose_ ) {
		  std::cout << "Final top-jet candidate: (Pt,Y,Phi,M) = (" 
			    << candidate.pt() << ", " << candidate.rapidity() << ", " 
			    << candidate.phi() << ", " << candidate.m() << ")" << std::endl;
		  std::vector<fastjet::PseudoJet> pieces = candidate.pieces();
		  std::cout << "Number of pieces = " << pieces.size() << std::endl;
		  for ( std::vector<fastjet::PseudoJet>::const_iterator ibegin = pieces.begin(), iend = pieces.end(), i = ibegin;
			i != iend; ++i ) {
		    std::cout << "   Piece : " << i - ibegin << ", (Pt,Y,Phi,M) = (" 
			      << i->pt() << ", " << i->rapidity() << ", " 
			      << i->phi() << ", " << i->m() << ")" << std::endl;
		  }
		}
		// Add to the list
		hardjetsOutput.push_back( candidate );		
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
									  const fastjet::ClusterSequence & theClusterSequence,
									  double nCellMin ) const {
	
	
	double eta1 = jet1.rapidity();
	double phi1 = jet1.phi();
	double eta2 = jet2.rapidity();
	double phi2 = jet2.phi();
	
	double deta = abs(eta2 - eta1) / 0.087;
	double dphi = fabs( reco::deltaPhi(phi2,phi1) ) / 0.087;
	
	return ( ( deta + dphi ) <= nCellMin );
}



//-------------------------------------------------------------------------
// attempt to decompose a jet into "hard" subjets, where hardness is set by ptHard
//
bool CATopJetAlgorithm::decomposeJet(const fastjet::PseudoJet & theJet, 
									 const fastjet::ClusterSequence & theClusterSequence, 
									 const vector<fastjet::PseudoJet> & cell_particles,
									 double ptHard, double nCellMin, double deltarcut,
									 fastjet::PseudoJet & ja, fastjet::PseudoJet & jb, 
									 vector<fastjet::PseudoJet> & leftovers) const {
	
	bool goodBreak;
	fastjet::PseudoJet j = theJet;
	double InputObjectPt = j.perp();
	if ( verbose_ )cout<<"Input Object Pt = "<<InputObjectPt<<endl;
	if ( verbose_ )cout<<"ptHard = "<<ptHard<<endl;
	leftovers.clear();
	if ( verbose_ )cout<<"start while loop"<<endl;
	
	while (1) {                                                      // watch out for infinite loop!
		goodBreak = theClusterSequence.has_parents(j,ja,jb);
		if (!goodBreak){
			if ( verbose_ )cout<<"bad break. this is one cell. can't decluster anymore."<<endl;
			break;         // this is one cell, can't decluster anymore
		}
		
		if ( verbose_ )cout<<"good break. ja Pt = "<<ja.perp()<<" jb Pt = "<<jb.perp()<<endl;
		
		/// Adjacency Requirement ///
		
		// check if clusters are adjacent using a constant deltar adjacency.
		double clusters_deltar=fabs(ja.eta()-jb.eta())+fabs(deltaPhi(ja.phi(),jb.phi()));
		
		if ( verbose_  && useAdjacency_ ==1)cout<<"clusters_deltar = "<<clusters_deltar<<endl;
		if ( verbose_  && useAdjacency_ ==1)cout<<"deltar cut = "<<deltarcut<<endl;
		
		if ( useAdjacency_==1 && clusters_deltar < deltarcut){
			if ( verbose_ )cout<<"clusters too close. consant adj. break."<<endl;
			break;
		} 
		
		// Check if clusters are adjacent using a DeltaR adjacency which is a function of pT.
		double clusters_deltaR=deltaR( ja.rapidity(), ja.phi(), jb.rapidity(), jb.phi() );
		
		if ( verbose_  && useAdjacency_ ==2)cout<<"clusters_deltaR = "<<clusters_deltaR<<endl;
		if ( verbose_  && useAdjacency_ ==2)cout<<"0.4-0.0004*InputObjectPt = "<<0.4-0.0004*InputObjectPt<<endl;
		
		if ( useAdjacency_==2 && clusters_deltaR < 0.4-0.0004*InputObjectPt)
		{
			if ( verbose_ )cout<<"clusters too close. modified adj. break."<<endl;
			break;
		} 

		// Check if clusters are adjacent in the calorimeter. 
		if ( useAdjacency_==3 &&  adjacentCells(ja,jb,cell_particles,theClusterSequence,nCellMin) ){                  
			if ( verbose_ )cout<<"clusters too close in the calorimeter. calorimeter adj. break."<<endl;
			break;         // the clusters are "adjacent" in the calorimeter => shouldn't have decomposed
		}
				
		if ( verbose_ )cout<<"clusters pass distance cut"<<endl;
		
		/// Pt Fraction Requirement ///
		
		if ( verbose_ )cout<<"ptHard = "<<ptHard<<endl;
		
		if (ja.perp() < ptHard && jb.perp() < ptHard){
			if ( verbose_ )cout<<"two soft clusters. dead end"<<endl;
			break;         // broke into two soft clusters, dead end
		}
		
		if (ja.perp() > ptHard && jb.perp() > ptHard){
			if ( verbose_ )cout<<"two hard clusters. done"<<endl;
			return true;   // broke into two hard clusters, we're done!
		}
		
		else if (ja.perp() > jb.perp()) {                              // broke into one hard and one soft, ditch the soft one and try again
			if ( verbose_ )cout<<"ja hard jb soft. try to split hard. j = ja"<<endl; 
			j = ja;
			vector<fastjet::PseudoJet> particles = theClusterSequence.constituents(jb);
			leftovers.insert(leftovers.end(),particles.begin(),particles.end());
		}
		else {
			if ( verbose_ )cout<<"ja hard jb soft. try to split hard. j = jb"<<endl; 
			j = jb;
			vector<fastjet::PseudoJet> particles = theClusterSequence.constituents(ja);
			leftovers.insert(leftovers.end(),particles.begin(),particles.end());
		}
	}
	
	if ( verbose_ )cout<<"did not decluster."<<endl;  // did not decluster into hard subjets
	
	ja.reset(0,0,0,0);
	jb.reset(0,0,0,0);
	leftovers.clear();
	return false;
}
