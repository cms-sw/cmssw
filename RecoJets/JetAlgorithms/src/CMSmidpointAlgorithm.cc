// File: CMSmidpointAlgorithm.cc
// Description:  An algorithm for CMS jet reconstruction.
// Author:  R. M. Harris
// Creation Date:  RMH Feb. 4, 2005   Inital version of the CMS Midpoint Algorithm starting 
//                                    from the CDF Midpoint Algorithm code by Matthias Toennesmann
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

#include "RecoJets/JetAlgorithms/interface/CMSmidpointAlgorithm.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJetComparisons.h"
#include "RecoJets/JetAlgorithms/interface/MakeCaloJet.h"

using namespace std;

//  Run the algorithm
//  ------------------
void CMSmidpointAlgorithm::run(const CaloTowerCollection* theCtcp,
			       CaloJetCollection & caloJets)
{
  //Create a CaloTowerHelper for this collection
    CaloTowerHelper theHelper(theCtcp); 

  // Find proto-jets from the seeds.
  vector<ProtoJet> protoJets;         // Initialize working container
  vector<ProtoJet> finalJets;         // Final proto-jets container
  findStableConesFromSeeds(theHelper, theCtcp, protoJets);   //Find the proto-jets from the seeds
  if(theDebugLevel>=1)cout << "[CMSmidpointAlgorithm] Num proto-jets from Seeds = " << protoJets.size() << endl;

  // Find proto-jets from the midpoints, and add them to the list
  if(protoJets.size()>0)findStableConesFromMidPoints(theHelper, theCtcp, protoJets);// Add midpoints
  if(theDebugLevel>=1)cout << "[CMSmidpointAlgorithm] Num proto-jets from Seeds and Midpoints = " << protoJets.size() << endl;

  // Split and merge the proto-jets, assigning each tower in the protojets to one and only one final jet.
  if(protoJets.size()>0)splitAndMerge(theHelper, theCtcp, protoJets,finalJets);    // Split and merge 
  if(theDebugLevel>=1)cout << "[CMSmidpointAlgorithm] Num final jets = " << finalJets.size() << endl;

  // Make the CaloJets from the final protojets
  MakeCaloJet(*theCtcp, finalJets, caloJets);

};



// Find the proto-jets from the seed towers.
// ----------------------------------------
void CMSmidpointAlgorithm::findStableConesFromSeeds(CaloTowerHelper& theHelper,
						    const CaloTowerCollection* theCtcp,
						    vector<ProtoJet> & stableCones) {
  // This dictates that the cone size will be reduced in the iterations procedure,
  // to prevent excessive cone movement, and then will be enlarged at the end of
  // the iteration procedure (ala CDF).
  bool reduceConeSize = true;  
  
  // Get the Seed Towers sorted by Et.  
  vector<const CaloTower*> seedTowers = theHelper.etOrderedCaloTowers(theSeedThreshold);  //This gets towers

  // Loop over all Seeds
  for(vector<const CaloTower *>::const_iterator i = seedTowers.begin(); i != seedTowers.end(); ++i) {
    double seedEta = (*i)->getEta();
    double seedPhi = (*i)->getPhi(); 

    // Find stable cone from seed.  
    // This iterates the cone centroid, makes a proto-jet, and adds it to the list.
    if(theDebugLevel>=2) cout << endl << "[CMSmidpointAlgorithm] seed " << i-seedTowers.begin() << ": eta=" << seedEta << ", phi=" << seedPhi << endl;
    iterateCone(theHelper, theCtcp, seedEta, seedPhi, 0, reduceConeSize, stableCones);
    
  }

};

// Iterate the proto-jet center until it is stable
// -----------------------------------------------
 void CMSmidpointAlgorithm::iterateCone(CaloTowerHelper& theHelper,
					const CaloTowerCollection* /* theCtcp */,
					double startRapidity, double startPhi, 
					double startE, bool reduceConeSize, vector<ProtoJet> & stableCones){
//  The workhorse of the algorithm.
//  Throws a cone around a position and iterates the cone until the centroid and energy of the clsuter is stable.
//  Uses a reduced cone size to prevent excessive cluster drift, ala CDF.
//  Adds unique clusters to the list of proto-jets.
//
    int nIterations = 0;
    bool keepJet = true;
    ProtoJet trialCone;
    double iterationtheConeRadius = theConeRadius;
    if(reduceConeSize)iterationtheConeRadius *= sqrt(theConeAreaFraction);
    while(++nIterations <= theMaxIterations + 1 && keepJet){
      
      // Last iteration uses the full cone size.  Others use reduced cone size.
      if(nIterations == theMaxIterations + 1)iterationtheConeRadius = theConeRadius;

      //Add all towers in cone and over threshold to the cluster
      vector<const CaloTower*> towersInSeedCluster 
          = theHelper.towersWithinCone(startRapidity, startPhi, iterationtheConeRadius, theTowerThreshold);
      if(theDebugLevel>=2)cout << "[CMSmidpointAlgorithm] iter=" << nIterations << ", towers=" <<towersInSeedCluster.size();    

      if(towersInSeedCluster.size()<1) {  // Just in case there is an empty cone.
        keepJet = false;
	if(theDebugLevel>=2)cout << endl;
      } else{
        //Put seed cluster into trial cone 
        trialCone.putTowers(towersInSeedCluster);
        HepLorentzVector endLorentzVector =  trialCone.getLorentzVector();
	double endRapidity = endLorentzVector.rapidity();
	double endPhi      = endLorentzVector.phi();
	double endPt       =  endLorentzVector.perp();
	double endE        = endLorentzVector.e();
        if(theDebugLevel>=2)cout << ", y=" << endRapidity << ", phi=" << endPhi << ", PT=" << endPt << endl;
        if(nIterations <= theMaxIterations){
	  // Do we have a stable cone?
	  if(abs(endRapidity-startRapidity)<.001 && 
	      abs(endPhi-startPhi)<.001 && 
	      abs(endE - startE) < .001) {
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
	  for(vector<const CaloTower *>::const_iterator i = towersInSeedCluster.begin(); i != towersInSeedCluster.end(); ++i) {
	    const CaloTower &t = **i;
	    if(theDebugLevel>=2) cout << "[CMSmidpointAlgorithm] Tower " <<
              i-towersInSeedCluster.begin() << ": eta=" << t.getEta() << 
	      ", phi=" << t.getPhi() << ", ET="  << t.getEt() << endl;
          }
        }         
      }
    }

  if(keepJet){  // Our trial cone is now stable
    bool identical = false;
    // Loop over proto-jets and check that our trial cone is a unique proto-jet
    for(vector<ProtoJet>::iterator stableConeIter = stableCones.begin(); stableConeIter != stableCones.end(); ++stableConeIter)
    {
      HepLorentzVector trialConeVector =  trialCone.getLorentzVector();
      HepLorentzVector stableConeVector = stableConeIter->getLorentzVector();
      if(trialConeVector == stableConeVector) identical = true;  // This proto-jet is not unique.
    }
    if(!identical){
      stableCones.push_back(trialCone);  // Save the unique proto-jets
      if(theDebugLevel>=2)cout << "[CMSmidpointAlgorithm] Unique Proto-Jet Saved" << endl;
    }    

  }

};


// Find proto-jets from the midpoints
// ----------------------------------
void CMSmidpointAlgorithm::findStableConesFromMidPoints(CaloTowerHelper& theHelper,
							 const CaloTowerCollection* theCtcp,
							 vector<ProtoJet>& stableCones){
// We take the previous list of stable protojets from seeds as input and add to it
// those from the midpoints between proto-jet pairs, triplets, etc.
  // distanceOK[i-1][j] = Is distance between stableCones i and j (i>j) less than 2*_theConeRadius?
  vector< vector<bool> > distanceOK;         // A vector of vectors
  distanceOK.resize(stableCones.size() - 1); // Set the outer vector size, num protojets - 1
  for(unsigned int nCluster1 = 1; nCluster1 < stableCones.size(); ++nCluster1){  // Loop over the protojets
    distanceOK[nCluster1 - 1].resize(nCluster1);                      //Set inner vector size: monotonically increasing.
    double cluster1Rapidity = stableCones[nCluster1].getLorentzVector().rapidity();  // Jet 1 y
    double cluster1Phi      = stableCones[nCluster1].getLorentzVector().phi();       // Jet 1 phi
    for(unsigned int nCluster2 = 0; nCluster2 < nCluster1; ++nCluster2){         // Loop over the other proto-jets
      double cluster2Rapidity = stableCones[nCluster2].getLorentzVector().rapidity(); // Jet 2 y
      double cluster2Phi      = stableCones[nCluster2].getLorentzVector().phi();      // Jet 2 phi
      double dRapidity = fabs(cluster1Rapidity - cluster2Rapidity);   
      double dPhi      = fabs(cluster1Phi      - cluster2Phi);
      if(dPhi > M_PI)
        dPhi = 2*M_PI - dPhi;
      double dR = sqrt(dRapidity*dRapidity + dPhi*dPhi);
      distanceOK[nCluster1 - 1][nCluster2] = dR < 2*theConeRadius;
    }
  }

  // Find all pairs (triplets, ...) of stableCones which are less than 2*theConeRadius apart from each other.
  vector< vector<int> > pairs(0);  
  vector<int> testPair(0);
  int maxClustersInPair = theMaxPairSize;  // Set maximum proto-jets to a pair or a triplet, etc.
  if(!maxClustersInPair)maxClustersInPair = stableCones.size();  // If zero, then skys the limit!
  addClustersToPairs(theHelper, theCtcp, testPair,pairs,distanceOK,maxClustersInPair);  // Make the pairs, triplets, etc.

  // Loop over all combinations. Calculate MidPoint. Make midPointClusters.
  bool reduceConeSize = false;  // Note that here we keep the iteration cone size fixed.
  for(vector<vector<int> >::const_iterator iPair = pairs.begin(); iPair != pairs.end(); ++iPair) {
    const vector<int> & Pair = *iPair;
    // Calculate rapidity, phi and energy of MidPoint.
    HepLorentzVector midPoint(0,0,0,0);
    for(vector<int>::const_iterator iPairMember = Pair.begin(); iPairMember != Pair.end(); ++iPairMember) {
      midPoint += stableCones[*iPairMember].getLorentzVector();
    }
    if(theDebugLevel>=2)cout << endl << "[CMSmidpointAlgorithm] midpoint " << iPair-pairs.begin() << ": y = " << midPoint.rapidity() << ", phi=" << midPoint.phi() <<
      ", size=" << Pair.size() << endl;
    iterateCone(theHelper, theCtcp, midPoint.rapidity(),midPoint.phi(),midPoint.e(),reduceConeSize,stableCones);
  }
  sort(stableCones.begin(),stableCones.end(),ProtoJetPtGreater());
};

// Add proto-jets to pairs from which we will find the midpoint
// ------------------------------------------------------------
void CMSmidpointAlgorithm::addClustersToPairs(CaloTowerHelper& theHelper,
					      const CaloTowerCollection* theCtcp,
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
	addClustersToPairs(theHelper, theCtcp, testPair,pairs,distanceOK,maxClustersInPair);
      // All combinations containing testPair found. Remove last element.
      testPair.pop_back();
    }
  }
};

// Split and merge the proto-jets, assigning each tower in the protojets to one and only one final jet.
// ----------------------------------------------------------------------------------------------------
void CMSmidpointAlgorithm::splitAndMerge(CaloTowerHelper& /* theHelper */,
					 const CaloTowerCollection* /* theCtcp */,
					 vector<ProtoJet>& stableCones, vector<ProtoJet>& finalJets)
{
//
// This can use quite a bit of optimization and simplification.
//

  // Debugging
  if(theDebugLevel>=2){
    // Take a look at the proto-jets we were given
    int numProtojets = stableCones.size();
    for(int i = 0; i < numProtojets ; ++i){
      int numTowers = stableCones[i].getTowerList().size();
      cout << endl << "[CMSmidpointAlgorithm] ProtoJet " << i << ": PT=" << stableCones[i].getLorentzVector().perp()
                              << ", y="<< stableCones[i].getLorentzVector().rapidity()
                              << ", phi="<< stableCones[i].getLorentzVector().phi()  
			      << ", ntow="<< numTowers << endl;     
      vector<const CaloTower*> protojetTowers = stableCones[i].getTowerList(); 
      for(int j = 0; j < numTowers; ++j){
        cout << "[CMSmidpointAlgorithm] Tower " << j << ": ET=" << protojetTowers[j]->getEt() 
                                << ", eta="<< protojetTowers[j]->getEta()
                                << ", phi="<<   protojetTowers[j]->getPhi() << endl;
      }
    }      
  }

  // Start of split and merge algorithm
  bool mergingNotFinished = true;
  while(mergingNotFinished){

    // Sort the stable cones (highest pt first).
   sort(stableCones.begin(),stableCones.end(),ProtoJetPtGreater());

    // Start with the highest pt cone left in list. List changes in loop,
    // getting smaller with each iteration.
    vector<ProtoJet>::iterator stableConeIter1 = stableCones.begin();
    
    if(stableConeIter1 == stableCones.end())   // Stable cone list empty?
    {
      mergingNotFinished = false;
    }
    else
    {
      bool coneNotModified = true;

      // Determine whether highest pt cone has an overlap with other stable cones.
      vector<ProtoJet>::iterator stableConeIter2 = stableConeIter1; 
      ++stableConeIter2;   // Iterator for 2nd highest pt cone, just past 1st cone.

      while(coneNotModified && stableConeIter2 != stableCones.end()){

	// Calculate overlap of the two cones.
        bool overlap = false;
        vector<const CaloTower*> overlapTowers;  // Make a list to hold the overlap towers
        //cout << "1st cone num towers=" << stableConeIter1->getTowerList().size() << endl;
        //int numTowers1=0;

        //Loop over towers in higher Pt cone
 	for(vector<const CaloTower*>::const_iterator towerIter1 = stableConeIter1->getTowerList().begin();
	    towerIter1 != stableConeIter1->getTowerList().end();
	    ++towerIter1){
	  //cout << "1st cone tower " << numTowers1 << endl;
          //++numTowers1;
	  bool isInCone2 = false;
          //cout << "2nd cone num towers=" << stableConeIter2->getTowerList().size() << endl;
          //int numTowers2=0;

          // Loop over towers in lower Pt cone
	  for(vector<const CaloTower*>::const_iterator towerIter2 = stableConeIter2->getTowerList().begin();
	      towerIter2 != stableConeIter2->getTowerList().end();
	      ++towerIter2)
          {
	    //cout << "2st cone tower " << numTowers2 << endl;
            //++numTowers2;

            // Check if towers are the same by checking for unique eta, phi and energy values.
            // Will want to replace this with checking the tower index when available.
            if((abs((*towerIter1)->getEta()-(*towerIter2)->getEta())<.001) && 
	        (abs((*towerIter1)->getPhi()-(*towerIter2)->getPhi())<.001) &&
	        (abs((*towerIter1)->getE()-(*towerIter2)->getE())<.001)){
	        isInCone2 = true;   //The tower is in both cones
                //cout << "Merging found overlap tower: eta=" << (*towerIter1)->getEta() << ", phi=" << (*towerIter1)->getPhi() << " in 1st protojet "  << 
	        //                             " and eta=" << (*towerIter2)->getEta() << ", phi=" << (*towerIter2)->getPhi() << " in 2nd protojet "  << endl;
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
	  if(overlap.getLorentzVector().perp() >= theOverlapThreshold*stableConeIter2->getLorentzVector().perp()){

	    // Merge the two cones.
            // Get a copy of the list of towers in higher Pt proto-jet 
            vector<const CaloTower*> stableCone1Towers = stableConeIter1->getTowerList(); 

             //Loop over the list of towers lower Pt jet
	    for(vector<const CaloTower*>::const_iterator towerIter2 = stableConeIter2->getTowerList().begin();
		towerIter2 != stableConeIter2->getTowerList().end();
		++towerIter2){
	      bool isInOverlap = false;

              //Check if that tower is in the overlap region
	      for(vector<const CaloTower*>::iterator overlapTowerIter = overlapTowers.begin();
		  overlapTowerIter != overlapTowers.end();
		  ++overlapTowerIter){
                // Check if towers are the same by checking for unique eta, phi and energy values.
                // Will want to replace this with checking the tower index when available.
                if((abs((*overlapTowerIter)->getEta()-(*towerIter2)->getEta())<.001) && 
	            (abs((*overlapTowerIter)->getPhi()-(*towerIter2)->getPhi())<.001) &&
	            (abs((*overlapTowerIter)->getE()-(*towerIter2)->getE())<.001))
		    isInOverlap = true;
              }
              //Add  non-overlap tower from proto-jet 2 into the proto-jet 1 tower list.
	      if(!isInOverlap)stableCone1Towers.push_back(*towerIter2);  
	    }
 
            if(theDebugLevel>=2)cout << endl << "[CMSmidpointAlgorithm] Merging: 1st Proto-jet grows: "
	    " y=" << stableConeIter1->getLorentzVector().rapidity() << 
            ", phi=" << stableConeIter1->getLorentzVector().phi() << 
	    " increases from " << stableConeIter1->getTowerList().size() << 
	    " to "  << stableCone1Towers.size() << " towers." << endl;

            // Put the new expanded list of towers into the first proto-jet
            stableConeIter1->putTowers(stableCone1Towers);

	    if(theDebugLevel>=2)cout << "[CMSmidpointAlgorithm] Merging: 1st protojet now at y=" << stableConeIter1->getLorentzVector().rapidity() <<
	                            ", phi=" << stableConeIter1->getLorentzVector().phi() << endl;
	    
            if(theDebugLevel>=2)cout << "[CMSmidpointAlgorithm] Merging: 2nd Proto-jet removed:" 
	    " y=" << stableConeIter2->getLorentzVector().rapidity() << 
            ", phi=" << stableConeIter2->getLorentzVector().phi() << endl;

	    // Remove the second proto-jet.
	    stableCones.erase(stableConeIter2);

	  }
	  else{
	    // Split the two proto-jets.

            // Create lists of towers to remove from each proto-jet
	    vector<const CaloTower*> removeFromCone1,removeFromCone2;

	    // Which tower goes where?
            // Loop over the overlap towers
	    for(vector<const CaloTower*>::iterator towerIter = overlapTowers.begin();
		towerIter != overlapTowers.end();
		++towerIter){
	      double towerRapidity = (*towerIter)->getEta();
	      double towerPhi      = (*towerIter)->getPhi();

	      // Calculate distance from proto-jet 1.
	      double dRapidity1 = fabs(towerRapidity - stableConeIter1->getLorentzVector().rapidity());
	      double dPhi1      = fabs(towerPhi      - stableConeIter1->getLorentzVector().phi());
	      if(dPhi1 > M_PI)
		dPhi1 = 2*M_PI - dPhi1;
	      double dRJet1 = sqrt(dRapidity1*dRapidity1 + dPhi1*dPhi1);

	      // Calculate distance from proto-jet 2.
	      double dRapidity2 = fabs(towerRapidity - stableConeIter2->getLorentzVector().rapidity());
	      double dPhi2      = fabs(towerPhi      - stableConeIter2->getLorentzVector().phi());
	      if(dPhi2 > M_PI)
		dPhi2 = 2*M_PI - dPhi2;
	      double dRJet2 = sqrt(dRapidity2*dRapidity2 + dPhi2*dPhi2);

	      if(dRJet1 < dRJet2){
		// Tower is closer to proto-jet 1. To be removed from proto-jet 2.
		removeFromCone2.push_back(*towerIter);
              }
	      else
		// Tower is closer to proto-jet 2. To be removed from proto-jet 1.
		removeFromCone1.push_back(*towerIter);
	    }
	    // Remove towers in the overlap region from the cones to which they have the larger distance.
	    
	    // Remove towers from proto-jet 1.
            vector<const CaloTower*> towerList1 = stableConeIter1->getTowerList(); 

            // Loop over towers in remove list
	    for(vector<const CaloTower*>::iterator towerIter = removeFromCone1.begin();
		towerIter != removeFromCone1.end();
		++towerIter)
	    {

                // Loop over towers in protojet
                for(vector<const CaloTower*>::iterator towerIter1 = towerList1.begin(); towerIter1 != towerList1.end(); ++towerIter1)
		{

                   // Check if they are equal
                   if((abs((*towerIter)->getEta()-(*towerIter1)->getEta())<.001) && 
	              (abs((*towerIter)->getPhi()-(*towerIter1)->getPhi())<.001) &&
	              (abs((*towerIter)->getE()-(*towerIter1)->getE())<.001))
                  {

                     // Remove the tower
	             towerList1.erase(towerIter1);
	             break;
                  }
                }
            }

            if(theDebugLevel>=2)cout << endl << "[CMSmidpointAlgorithm] Splitting: 1st Proto-jet  shrinks: y=" << 
	                stableConeIter1->getLorentzVector().rapidity() << 
            ", phi=" << stableConeIter1->getLorentzVector().phi() << 
	    " decreases from" << stableConeIter1->getTowerList().size() << 
	    " to "  << towerList1.size() << " towers." << endl;

	    //Put the new reduced list of towers into proto-jet 1.
            stableConeIter1->putTowers(towerList1); 

	    // Remove towers from cone 2.
            vector<const CaloTower*> towerList2 = stableConeIter2->getTowerList(); 

            // Loop over towers in remove list
	    for(vector<const CaloTower*>::iterator towerIter = removeFromCone2.begin();
		towerIter != removeFromCone2.end();
		++towerIter)
	    {
               // Loop over towers in protojet
               for(vector<const CaloTower*>::iterator towerIter2 = towerList2.begin(); towerIter2 != towerList2.end(); ++towerIter2)
		{
                  // Check if they are equal
                   if((abs((*towerIter)->getEta()-(*towerIter2)->getEta())<.001) && 
	              (abs((*towerIter)->getPhi()-(*towerIter2)->getPhi())<.001) &&
	              (abs((*towerIter)->getE()-(*towerIter2)->getE())<.001))
                  {
                    // Remove the tower
	             towerList2.erase(towerIter2);
	             break;
                  }
                }
            }

             if(theDebugLevel>=2)cout << "[CMSmidpointAlgorithm] Splitting: 2nd Proto-jet shrinks: y=" <<
	                stableConeIter2->getLorentzVector().rapidity() << 
            ", phi=" << stableConeIter2->getLorentzVector().phi() << 
	    " decreases from" << stableConeIter2->getTowerList().size() << 
	    " to "  << towerList2.size() << " towers." << endl;

	    //Put the new reduced list of towers into proto-jet 2.
            stableConeIter2->putTowers(towerList2); 

 	  }
	}
	else
	{
	if(theDebugLevel>=2)cout << endl << 
	  "[CMSmidpointAlgorithm] no overlap between 1st protojet at  y=" << stableConeIter1->getLorentzVector().rapidity() << 
                                          ", phi=" << stableConeIter1->getLorentzVector().phi() << 
                        " and 2nd protojet at  y=" << stableConeIter2->getLorentzVector().rapidity() << 
                                          ", phi=" << stableConeIter2->getLorentzVector().phi() <<endl;
	}

	++stableConeIter2;  //Increment iterator to the next highest Pt protojet
      }
      if(coneNotModified){

        if(theDebugLevel>=2)cout << 
	  "[CMSmidpointAlgorithm] Saving: Proto-jet  at y=" << stableConeIter1->getLorentzVector().rapidity() << 
                            ", phi=" << stableConeIter1->getLorentzVector().phi() <<  " has no overlap" << endl;

	// Cone 1 has no overlap with any of the other cones and can become a jet.
 	finalJets.push_back(*stableConeIter1);
	stableCones.erase(stableConeIter1);

      }
    }
  }

  sort(finalJets.begin(),finalJets.end(),ProtoJetPtGreater());
};


