// File: CMSMidpointAlgorithm.cc
// Description:  An algorithm for CMS jet reconstruction.
// Author:  R. M. Harris
// Creation Date:  RMH Feb. 4, 2005   Inital version of the CMS Midpoint Algorithm starting 
//                                    from the CDF Midpoint Algorithm code by Matthias Toennesmann
// Revisions       RMH Oct 19, 2005   Modified to work with real CaloTowers from Jeremy Mans.
//
// Pseudo Code for the CMS Midpoint Algorithm
//--------------------------------------------
// Loop over all Towers in ET order 
//    If (Tower ET > Seed ET Threshold) Add tower to Seed list
// Loop over all Seeds
//    Add all towers with dR(Tower,Seed) < R_Cone to SeedCluster Tower List.
//    Find eta0 and phi0 of seed cluster's Lorentz Vector
//    Recenter cone on eta0 and phi0 and recalculate SeedCluster Tower List
//    Iterate until eta and phi stable or max iterations
//    Add unique SeedClusters to list of Protojets
// Loop over all Protojets
//    Add all protojets within 2 R_Cone (of each other) to a protojet group list.
// Loop over all Protojet Groups
//    Find the midpoint of each group: sum of Lorentz Vectors.
//    As for seeds, make midpoint clusters of towers within R_Cone and iterate.
//    Add unique midpoint clusters to list of ProtoJets
// Loop over Protojets in Pt order
//   If protojet overlaps with another proto jet.
//     Get protojet energies.
//     Get energy in overlap region and calculate overlap fraction.
//     If (overlap fraction > overlap threshold)Merge ProtoJets into one jet
//     If (overlap fraction < overlap threshold)Split ProtoJets into two jets
//   Else If protojet does not overlap with any other protojet
//     Promote protojet to jet without any changes
//

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoParticleFlow/PFRootEvent/interface/ProtoJet.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetAlgoHelper.h"
#include "RecoParticleFlow/PFRootEvent/interface/CMSMidpointAlgorithm.h"



using namespace std;
using namespace reco;
using namespace JetReco;

// helping stuff
namespace {

  bool sameTower (InputItem c1, InputItem c2) {
    return c1 == c2;
  }

  std::vector<InputItem> towersWithinCone(const InputCollection& fInput, double coneEta, double conePhi, double coneRadius){
    std::vector<InputItem> result;
    double r2 = coneRadius*coneRadius;
    InputCollection::const_iterator towerIter = fInput.begin();
    InputCollection::const_iterator towerIterEnd = fInput.end();
    for (;towerIter != towerIterEnd; ++towerIter) {
      InputItem caloTowerPointer = *towerIter;
      double dR2 = deltaR2 (coneEta, conePhi, caloTowerPointer->eta(), caloTowerPointer->phi());
      if(dR2 < r2){
	result.push_back(caloTowerPointer);
      }
    }
    return result;
  }
  
  // etOrderedCaloTowers returns an Et order list of pointers to CaloTowers with Et>etTreshold
  std::vector<InputItem> etOrderedCaloTowers(const InputCollection& fInput, double etThreshold) {
    std::vector<InputItem> result;
    InputCollection::const_iterator towerIter = fInput.begin();
    InputCollection::const_iterator towerIterEnd = fInput.end();
    for (;towerIter != towerIterEnd; ++towerIter) {
      InputItem caloTowerPointer = *towerIter;
      if(caloTowerPointer->et() > etThreshold){
	result.push_back(caloTowerPointer);
      }
    }   
    GreaterByEtRef <InputItem> compCandidate;
    sort (result.begin(), result.end(), compCandidate);
    return result;
  }
  
  
  
  
}


//  Run the algorithm
//  ------------------
void CMSMidpointAlgorithm::run(const InputCollection& fInput, OutputCollection* fOutput)
{
  if (!fOutput) {
    std::cerr << "CMSMidpointAlgorithm::run-> ERROR: no output collection" << std::endl;
  }
  if(theDebugLevel>=1)cout << "[CMSMidpointAlgorithm] Num of input constituents = " << fInput.size() << endl;
  if(theDebugLevel>=2) {
    unsigned index = 0;
    for (; index < fInput.size (); index++) {
      cout << index << " consituent p/eta/phi: " 
	   << fInput[index]->p() << '/'
	   << fInput[index]->eta() << '/'
	   << fInput[index]->phi() << std::endl;
    }
  }
  // Find proto-jets from the seeds.
  InternalCollection protoJets;         // Initialize working container
  // vector<ProtoJet> finalJets;         // Final proto-jets container
  findStableConesFromSeeds(fInput, &protoJets);   //Find the proto-jets from the seeds
  if(theDebugLevel>=1)cout << "[CMSMidpointAlgorithm] Num proto-jets from Seeds = " << protoJets.size() << endl;

  // Find proto-jets from the midpoints, and add them to the list
  if(protoJets.size()>0)findStableConesFromMidPoints(fInput, &protoJets);// Add midpoints
  if(theDebugLevel>=1)cout << "[CMSMidpointAlgorithm] Num proto-jets from Seeds and Midpoints = " << protoJets.size() << endl;

  // Split and merge the proto-jets, assigning each tower in the protojets to one and only one final jet.
  if(protoJets.size()>0)splitAndMerge(fInput, &protoJets, fOutput);    // Split and merge 
  if(theDebugLevel>=1)cout << "[CMSMidpointAlgorithm] Num final jets = " << fOutput->size() << endl;

  // Make the CaloJets from the final protojets
  //  MakeCaloJet(*theCtcp, finalJets, caloJets);

}



// Find the proto-jets from the seed towers.
// ----------------------------------------
void CMSMidpointAlgorithm::findStableConesFromSeeds(const InputCollection& fInput, InternalCollection* fOutput) {
  // This dictates that the cone size will be reduced in the iterations procedure,
  // to prevent excessive cone movement, and then will be enlarged at the end of
  // the iteration procedure (ala CDF).
  bool reduceConeSize = true;  
  
  // Get the Seed Towers sorted by Et.  
  vector<InputItem> seedTowers = etOrderedCaloTowers(fInput, theSeedThreshold);  //This gets towers

  // Loop over all Seeds
  for(vector<InputItem>::const_iterator i = seedTowers.begin(); i != seedTowers.end(); ++i) {
    double seedEta = (*i)->eta ();
    double seedPhi = (*i)->phi (); 

    // Find stable cone from seed.  
    // This iterates the cone centroid, makes a proto-jet, and adds it to the list.
    if(theDebugLevel>=2) cout << endl << "[CMSMidpointAlgorithm] seed " << i-seedTowers.begin() << ": eta=" << seedEta << ", phi=" << seedPhi << endl;
    iterateCone(fInput, seedEta, seedPhi, 0, reduceConeSize, fOutput);
    
  }

}

// Iterate the proto-jet center until it is stable
// -----------------------------------------------
void CMSMidpointAlgorithm::iterateCone(const InputCollection& fInput,
				       double startRapidity, 
				       double startPhi, 
				       double startE, 
				       bool reduceConeSize, 
				       InternalCollection* stableCones) {
  //  The workhorse of the algorithm.
  //  Throws a cone around a position and iterates the cone until the centroid and energy of the clsuter is stable.
  //  Uses a reduced cone size to prevent excessive cluster drift, ala CDF.
  //  Adds unique clusters to the list of proto-jets.
  //
  int nIterations = 0;
  bool keepJet = true;
  ProtoJet* trialCone = new ProtoJet;
  double iterationtheConeRadius = theConeRadius;
  if(reduceConeSize)iterationtheConeRadius *= sqrt(theConeAreaFraction);
  while(++nIterations <= theMaxIterations + 1 && keepJet){
    
    // Last iteration uses the full cone size.  Others use reduced cone size.
    if(nIterations == theMaxIterations + 1)iterationtheConeRadius = theConeRadius;
    
    //Add all towers in cone and over threshold to the cluster
    vector<InputItem> towersInSeedCluster 
      = towersWithinCone(fInput, startRapidity, startPhi, iterationtheConeRadius);
    if(theDebugLevel>=2)cout << "[CMSMidpointAlgorithm] iter=" << nIterations << ", towers=" <<towersInSeedCluster.size();    
    
    if(towersInSeedCluster.size()<1) {  // Just in case there is an empty cone.
      keepJet = false;
      if(theDebugLevel>=2)cout << endl;
    } else{
      //Put seed cluster into trial cone 
      trialCone->putTowers(towersInSeedCluster);
      double endRapidity = trialCone->y();
      double endPhi      = trialCone->phi();
      double endPt       = trialCone->pt();
      double endE        = trialCone->e();
      if(theDebugLevel>=2)cout << ", y=" << endRapidity << ", phi=" << endPhi << ", PT=" << endPt << ", constituents=" << trialCone->getTowerList().size () << endl;
      if(nIterations <= theMaxIterations){
	// Do we have a stable cone?
	if(std::abs(endRapidity-startRapidity)<.001 && 
	   std::abs(endPhi-startPhi)<.001 && 
	   std::abs(endE - startE) < .001) {
	  nIterations = theMaxIterations;       // If cone size is reduced, then still one more iteration.
	  if(!reduceConeSize) ++nIterations; // Otherwise, this is the last iteration.
	}
	else{
	  // Another iteration.
	  startRapidity = endRapidity;
	  startPhi      = endPhi;
	  startE        = endE;
	}
      }	
      if(nIterations==theMaxIterations+1) {
	for(vector<InputItem>::const_iterator i = towersInSeedCluster.begin(); i != towersInSeedCluster.end(); ++i) {
	  InputItem t = *i;
	  if(theDebugLevel>=2) cout << "[CMSMidpointAlgorithm] Tower " <<
				 i-towersInSeedCluster.begin() << ": eta=" << t->eta() << 
				 ", phi=" << t->phi() << ", ET="  << t->et() << endl;
	}
      }         
    }
  }
  
  if(keepJet){  // Our trial cone is now stable
    bool identical = false;
    // Loop over proto-jets and check that our trial cone is a unique proto-jet
    for (unsigned icone = 0; icone < stableCones->size(); icone++) {
      if (trialCone->p4() == (*stableCones) [icone]->p4()) identical = true;  // This proto-jet is not unique.
    }
    if(!identical){
      stableCones->push_back(trialCone);  // Save the unique proto-jets
      trialCone = 0;
      if(theDebugLevel>=2)cout << "[CMSMidpointAlgorithm] Unique Proto-Jet Saved" << endl;
    }    
  }
  delete trialCone;
}


// Find proto-jets from the midpoints
// ----------------------------------
void CMSMidpointAlgorithm::findStableConesFromMidPoints(const InputCollection& fInput, InternalCollection* stableCones){
  // We take the previous list of stable protojets from seeds as input and add to it
  // those from the midpoints between proto-jet pairs, triplets, etc.
  // distanceOK[i-1][j] = Is distance between stableCones i and j (i>j) less than 2*_theConeRadius?
  vector< vector<bool> > distanceOK;         // A vector of vectors
  distanceOK.resize(stableCones->size() - 1); // Set the outer vector size, num protojets - 1
  for(unsigned int nCluster1 = 1; nCluster1 < stableCones->size(); ++nCluster1){  // Loop over the protojets
    distanceOK[nCluster1 - 1].resize(nCluster1);               //Set inner vector size: monotonically increasing.
    const ProtoJet* cluster1 = (*stableCones)[nCluster1];
    for(unsigned int nCluster2 = 0; nCluster2 < nCluster1; ++nCluster2){         // Loop over the other proto-jets
      const ProtoJet* cluster2 = (*stableCones)[nCluster2];
      double dR2 = deltaR2 (cluster1->y(), cluster1->phi(), cluster2->y(), cluster2->phi());
      distanceOK[nCluster1 - 1][nCluster2] = dR2 < 4*theConeRadius*theConeRadius;
    }
  }

  // Find all pairs (triplets, ...) of stableCones which are less than 2*theConeRadius apart from each other.
  vector< vector<int> > pairs(0);  
  vector<int> testPair(0);
  int maxClustersInPair = theMaxPairSize;  // Set maximum proto-jets to a pair or a triplet, etc.
  if(!maxClustersInPair)maxClustersInPair = stableCones->size();  // If zero, then skys the limit!
  addClustersToPairs(fInput, testPair,pairs,distanceOK,maxClustersInPair);  // Make the pairs, triplets, etc.
  
  // Loop over all combinations. Calculate MidPoint. Make midPointClusters.
  bool reduceConeSize = false;  // Note that here we keep the iteration cone size fixed.
  for(vector<vector<int> >::const_iterator iPair = pairs.begin(); iPair != pairs.end(); ++iPair) {
    const vector<int> & Pair = *iPair;
    // Calculate rapidity, phi and energy of MidPoint.
    reco::Particle::LorentzVector midPoint (0,0,0,0);
    for(vector<int>::const_iterator iPairMember = Pair.begin(); iPairMember != Pair.end(); ++iPairMember) {
      midPoint += (*stableCones)[*iPairMember]->p4();
    }
    if(theDebugLevel>=2)cout << endl << "[CMSMidpointAlgorithm] midpoint " << iPair-pairs.begin() << ": y = " << midPoint.Rapidity() << ", phi=" << midPoint.Phi() <<
			  ", size=" << Pair.size() << endl;
    iterateCone(fInput, midPoint.Rapidity(),midPoint.Phi(),midPoint.e(),reduceConeSize,stableCones);
  }
  GreaterByPtPtr<ProtoJet> compJets;
  sort (stableCones->begin(), stableCones->end(), compJets);
}

// Add proto-jets to pairs from which we will find the midpoint
// ------------------------------------------------------------
void CMSMidpointAlgorithm::addClustersToPairs(const InputCollection& fInput,
					      vector<int>& testPair, vector< vector<int> >& pairs,
					      vector< vector<bool> >& distanceOK, int maxClustersInPair)
{
  // Recursively adds clusters to pairs, triplets, ... whose mid-points are then calculated.
  
  // Find StableCone number to start with (either 0 at the beginning or last element of testPair + 1).
  int nextClusterStart = 0;
  if(testPair.size())
    nextClusterStart = testPair.back() + 1;
  for(unsigned int nextCluster = nextClusterStart; nextCluster <= distanceOK.size(); ++nextCluster){
    // Is new SeedCone less than 2*_theConeRadius apart from all clusters in testPair?
    bool addCluster = true;
    for(unsigned int iCluster = 0; iCluster < testPair.size() && addCluster; ++iCluster)
      if(!distanceOK[nextCluster - 1][testPair[iCluster]])
	addCluster = false;
    if(addCluster){
      // Add it to the testPair.
      testPair.push_back(nextCluster);
      // If testPair is a pair, add it to pairs.
      if(testPair.size() > 1)
	pairs.push_back(testPair);
      // If not bigger than allowed, find more clusters within 2*theConeRadius.
      if(testPair.size() < unsigned(maxClustersInPair))
	addClustersToPairs(fInput, testPair,pairs,distanceOK,maxClustersInPair);
      // All combinations containing testPair found. Remove last element.
      testPair.pop_back();
    }
  }
}

// Split and merge the proto-jets, assigning each tower in the protojets to one and only one final jet.
// ----------------------------------------------------------------------------------------------------
void CMSMidpointAlgorithm::splitAndMerge(const InputCollection& fInput,
					 InternalCollection* stableCones, OutputCollection* fFinalJets) {
  //
  // This can use quite a bit of optimization and simplification.
  //
  
  // Debugging
  if(theDebugLevel>=2){
    // Take a look at the proto-jets we were given
    int numProtojets = stableCones->size();
    for(int i = 0; i < numProtojets ; ++i){
      const ProtoJet* icone = (*stableCones)[i];
      int numTowers = icone->getTowerList().size();
      cout << endl << "[CMSMidpointAlgorithm] ProtoJet " << i << ": PT=" << (*stableCones)[i]->pt()
	   << ", y="<< icone->y()
	   << ", phi="<< icone->phi()  
	   << ", ntow="<< numTowers << endl;
      ProtoJet::Constituents protojetTowers = icone->getTowerList(); 
      for(int j = 0; j < numTowers; ++j){
	cout << "[CMSMidpointAlgorithm] Tower " << j << ": ET=" << protojetTowers[j]->et() 
	     << ", eta="<< protojetTowers[j]->eta()
	     << ", phi="<<   protojetTowers[j]->phi() << endl;
      }
    }      
  }
  
  // Start of split and merge algorithm
  bool mergingNotFinished = true;
  while(mergingNotFinished){
    
    // Sort the stable cones (highest pt first).
    GreaterByPtPtr<ProtoJet> compJets;
    sort(stableCones->begin(),stableCones->end(),compJets);
    // clean removed clusters
    //InternalCollection::const_iterator i = find (stableCones->begin(), stableCones->end(), (ProtoJet*)0);
    //    for (; i != stableCones->end(); ++i) std::cout << "CMSMidpointAlgorithm::splitAndMerge-> removing pointer " << *i << std::endl; 
    stableCones->erase (find (stableCones->begin(), stableCones->end(), (ProtoJet*)0), stableCones->end()); 
    
    // Start with the highest pt cone left in list. List changes in loop,
    // getting smaller with each iteration.
    InternalCollection::iterator stableConeIter1 = stableCones->begin();
    
    if(stableConeIter1 == stableCones->end()) {  // Stable cone list empty?
      mergingNotFinished = false;
    }
    else {
      ProtoJet* stableCone1 = *stableConeIter1;
      if (!stableCone1) {
	std::cerr << "CMSMidpointAlgorithm::splitAndMerge-> Error: stableCone1 should never be 0" << std::endl;
	continue;
      }
      bool coneNotModified = true;
      
      // Determine whether highest pt cone has an overlap with other stable cones.
      InternalCollection::iterator stableConeIter2 = stableConeIter1; 
      ++stableConeIter2;   // Iterator for 2nd highest pt cone, just past 1st cone.
      
      while(coneNotModified && stableConeIter2 != stableCones->end()){
	ProtoJet* stableCone2 = *stableConeIter2;
	if (!stableCone2) {
	  std::cerr << "CMSMidpointAlgorithm::splitAndMerge-> Error: stableCone2 should never be 0" << std::endl;
	  continue;
	}
	
	// Calculate overlap of the two cones.
	bool overlap = false;
	vector<InputItem> overlapTowers;  // Make a list to hold the overlap towers
	//cout << "1st cone num towers=" << stableCone1->getTowerList().size() << endl;
	//int numTowers1=0;
	
	//Loop over towers in higher Pt cone
	for(ProtoJet::Constituents::const_iterator towerIter1 = stableCone1->getTowerList().begin();
	    towerIter1 != stableCone1->getTowerList().end();
	    ++towerIter1){
	  //cout << "1st cone tower " << numTowers1 << endl;
	  //++numTowers1;
	  bool isInCone2 = false;
	  //cout << "2nd cone num towers=" << stableCone2->getTowerList().size() << endl;
	  //int numTowers2=0;
	  
	  // Loop over towers in lower Pt cone
	  for(ProtoJet::Constituents::const_iterator towerIter2 = stableCone2->getTowerList().begin();
	      towerIter2 != stableCone2->getTowerList().end();
	      ++towerIter2) {
	    //cout << "2st cone tower " << numTowers2 << endl;
	    //++numTowers2;
	    
	    // Check if towers are the same by checking for unique eta, phi and energy values.
	    // Will want to replace this with checking the tower index when available.
	    if(sameTower(*towerIter1, *towerIter2)) {
	      isInCone2 = true;   //The tower is in both cones
	      //cout << "Merging found overlap tower: eta=" << (*towerIter1)->eta << ", phi=" << (*towerIter1)->phi << " in 1st protojet "  << 
	      //                             " and eta=" << (*towerIter2)->eta << ", phi=" << (*towerIter2)->phi << " in 2nd protojet "  << endl;
	    }
	  }
	  if(isInCone2){
	    overlapTowers.push_back(*towerIter1);  //Add tower in both cones to the overlap list
	    overlap = true;
	  }          
	}
	if(overlap){   
	  // non-empty overlap.  Decide on splitting or merging.
	  
	  // Make a proto-jet with the overlap towers so we can calculate things for the overlap
	  ProtoJet overlap;
	  overlap.putTowers(overlapTowers);
	  coneNotModified = false;
	  
	  // Compare the overlap pt with the overlap fractcion threshold times the lower jet pt.
	  if(overlap.pt() >= theOverlapThreshold*stableCone2->pt()){
	    
	    // Merge the two cones.
	    // Get a copy of the list of towers in higher Pt proto-jet 
	    ProtoJet::Constituents stableCone1Towers = stableCone1->getTowerList(); 
	    
	    //Loop over the list of towers lower Pt jet
	    for(ProtoJet::Constituents::const_iterator towerIter2 = stableCone2->getTowerList().begin();
		towerIter2 != stableCone2->getTowerList().end();
		++towerIter2){
	      bool isInOverlap = false;
	      
	      //Check if that tower is in the overlap region
	      for(vector<InputItem>::iterator overlapTowerIter = overlapTowers.begin();
		  overlapTowerIter != overlapTowers.end();
		  ++overlapTowerIter){
		// Check if towers are the same by checking for unique eta, phi and energy values.
		// Will want to replace this with checking the tower index when available.
		if(sameTower(*overlapTowerIter, *towerIter2))   isInOverlap = true;
	      }
	      //Add  non-overlap tower from proto-jet 2 into the proto-jet 1 tower list.
	      if(!isInOverlap)stableCone1Towers.push_back(*towerIter2);  
	    }
	    
	    if(theDebugLevel>=2)cout << endl << "[CMSMidpointAlgorithm] Merging: 1st Proto-jet grows: "
				  " y=" << stableCone1->y() << 
				  ", phi=" << stableCone1->phi() << 
				  " increases from " << stableCone1->getTowerList().size() << 
				  " to "  << stableCone1Towers.size() << " towers." << endl;
	    
	    // Put the new expanded list of towers into the first proto-jet
	    stableCone1->putTowers(stableCone1Towers);
	    
	    if(theDebugLevel>=2)cout << "[CMSMidpointAlgorithm] Merging: 1st protojet now at y=" << stableCone1->y() <<
				  ", phi=" << stableCone1->phi() << endl;
	    
	    if(theDebugLevel>=2)cout << "[CMSMidpointAlgorithm] Merging: 2nd Proto-jet removed:" 
				  " y=" << stableCone2->y() << 
				  ", phi=" << stableCone2->phi() << endl;
	    
	    // Remove the second proto-jet.
	    delete *stableConeIter2;
	    *stableConeIter2 = 0;
	    
	  }
	  else{
	    // Split the two proto-jets.
	    
	    // Create lists of towers to remove from each proto-jet
	    vector<InputItem> removeFromCone1,removeFromCone2;
	    
	    // Which tower goes where?
	    // Loop over the overlap towers
	    for(vector<InputItem>::iterator towerIter = overlapTowers.begin();
		towerIter != overlapTowers.end();
		++towerIter){
	      double dR2Jet1 = deltaR2 ((*towerIter)->p4().Rapidity(), (*towerIter)->phi(), 
				      stableCone1->y(), stableCone1->phi()); 
	      // Calculate distance from proto-jet 2.
	      double dR2Jet2 = deltaR2 ((*towerIter)->p4().Rapidity(), (*towerIter)->phi(),
				      stableCone2->y(), stableCone2->phi()); 
	      
	      if(dR2Jet1 < dR2Jet2){
		// Tower is closer to proto-jet 1. To be removed from proto-jet 2.
		removeFromCone2.push_back(*towerIter);
	      }
	      else {
		// Tower is closer to proto-jet 2. To be removed from proto-jet 1.
		removeFromCone1.push_back(*towerIter);
	      }
	    }
	    // Remove towers in the overlap region from the cones to which they have the larger distance.
	    
	    // Remove towers from proto-jet 1.
	    vector<InputItem> towerList1 (stableCone1->getTowerList().begin(), stableCone1->getTowerList().end()); 

	    // Loop over towers in remove list
	    for(vector<InputItem>::iterator towerIter = removeFromCone1.begin();
		towerIter != removeFromCone1.end();
		++towerIter) {
	      // Loop over towers in protojet
	      for(vector<InputItem>::iterator towerIter1 = towerList1.begin(); towerIter1 != towerList1.end(); ++towerIter1) {
		// Check if they are equal
		if(sameTower(*towerIter, *towerIter1)) {
		  // Remove the tower
		  towerList1.erase(towerIter1);
		  break;
		}
	      }
	    }
	    
	    if(theDebugLevel>=2)cout << endl << "[CMSMidpointAlgorithm] Splitting: 1st Proto-jet  shrinks: y=" << 
				  stableCone1->y() << 
				  ", phi=" << stableCone1->phi() << 
				  " decreases from" << stableCone1->getTowerList().size() << 
				  " to "  << towerList1.size() << " towers." << endl;
	    
	    //Put the new reduced list of towers into proto-jet 1.
	    stableCone1->putTowers(towerList1); 
	    // Remove towers from cone 2.
	    vector<InputItem> towerList2 (stableCone2->getTowerList().begin(), stableCone2->getTowerList().end()); 
	    
	    // Loop over towers in remove list
	    for(vector<InputItem>::iterator towerIter = removeFromCone2.begin();
		towerIter != removeFromCone2.end();
		++towerIter) {
	      // Loop over towers in protojet
	      for(vector<InputItem>::iterator towerIter2 = towerList2.begin(); towerIter2 != towerList2.end(); ++towerIter2){
		// Check if they are equal
		if(sameTower(*towerIter, *towerIter2)) {
		  // Remove the tower
		  towerList2.erase(towerIter2);
		  break;
		}
	      }
	    }
	    
	    if(theDebugLevel>=2)cout << "[CMSMidpointAlgorithm] Splitting: 2nd Proto-jet shrinks: y=" <<
				  stableCone2->y() << 
				  ", phi=" << stableCone2->phi() << 
				  " decreases from" << stableCone2->getTowerList().size() << 
				  " to "  << towerList2.size() << " towers." << endl;
	    
	    //Put the new reduced list of towers into proto-jet 2.
	    stableCone2->putTowers(towerList2); 
	  }
	}
	else {
	  if(theDebugLevel>=2)cout << endl << 
				"[CMSMidpointAlgorithm] no overlap between 1st protojet at  y=" << stableCone1->y() << 
				", phi=" << stableCone1->phi() << 
				" and 2nd protojet at  y=" << stableCone2->y() << 
				", phi=" << stableCone2->phi() <<endl;
	}
	
	++stableConeIter2;  //Increment iterator to the next highest Pt protojet
      }
      if(coneNotModified){
	
	if(theDebugLevel>=2)cout << 
			      "[CMSMidpointAlgorithm] Saving: Proto-jet  at y=" << stableCone1->y() << 
			      ", phi=" << stableCone1->phi() <<  " has no overlap" << endl;
	
	fFinalJets->push_back(ProtoJet (stableCone1->p4(), stableCone1->getTowerList()));
	delete *stableConeIter1;
	*stableConeIter1 = 0;
      }
    }
  }
  
  GreaterByPt<ProtoJet> compJets;
  sort(fFinalJets->begin(),fFinalJets->end(),compJets);
}
