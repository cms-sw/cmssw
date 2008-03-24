#include "RecoMuon/TrackerSeedGenerator/plugins/CombinedTSG.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <vector>

//constructor
CombinedTSG::CombinedTSG(const edm::ParameterSet & par) : CompositeTSG(par) {
  theCategory = "CombinedTSG";
}

//destructor
CombinedTSG::~CombinedTSG(){
 //
}

void CombinedTSG::trackerSeeds(const TrackCand & muonTrackCand, const TrackingRegion& region, std::vector<TrajectorySeed> & result){
  //run all the seed generators registered
  firstTime = true;

  std::vector<TrajectorySeed>  triplets;
  std::vector<TrajectorySeed>  st_pairs;

  for (uint iTSG=0; iTSG!=theTSGs.size();iTSG++){
    if(theTSGs[iTSG]) {

      std::vector<TrajectorySeed>  tmpResult;

      theTSGs[iTSG]->trackerSeeds(muonTrackCand,region,tmpResult);

     // Fill triplets and pairs vectors 
     if(theNames[iTSG] == "firstTSG:TSGFromOrderedHits")  triplets.assign(tmpResult.begin(),tmpResult.end());
     if(theNames[iTSG] == "secondTSG:TSGFromOrderedHits") st_pairs.assign(tmpResult.begin(),tmpResult.end());

      if( !triplets.empty() && !st_pairs.empty() && firstTime ){
         firstTime = false;
         // clean
         LogDebug(theCategory) << st_pairs.size()<< " Pair Collection before triplets shared hits cleaning.";
         std::vector<TrajectorySeed> pairsCl = cleanBySharedInput(triplets,st_pairs);
         LogDebug(theCategory) << pairsCl.size()<< " Pair Collection afther triplets shared hits cleaning.";
         // security check
         tmpResult.swap(pairsCl);     
         pairsCl.clear();
      }
      
      result.insert(result.end(),tmpResult.begin(),tmpResult.end());
      if(theHistos[iTSG]) theHistos[iTSG]->Fill(tmpResult.size());
    }
  }
}



// clean pairs seeds if in triplets
std::vector<TrajectorySeed> CombinedTSG::cleanBySharedInput(const std::vector<TrajectorySeed> & seedTr,const std::vector<TrajectorySeed> & seedPair)
{

  // loop over triplets
  std::vector<TrajectorySeed> result;

  for(TrajectorySeedCollection::const_iterator s1 = seedTr.begin(); s1 != seedTr.end(); ++s1){
     //rechits from seed

     TrajectorySeed::range r1 = s1->recHits();
     if(s1->nHits()==0) continue ;
     for(TrajectorySeedCollection::const_iterator s2 = seedPair.begin(); s2 != seedPair.end(); ++s2){
        //empty
        if(s2->nHits()==0) continue ;

        TrajectorySeed::range r2 = s2->recHits();
        TrajectorySeed::const_iterator h2 = r2.first;

        //number of shared hits;   
        int shared = 0;

        for(;h2 < r2.second;h2++){
          for(TrajectorySeed::const_iterator h1 = r1.first; h1 < r1.second;h1++){
             if(h2->sharesInput(&(*h1),TrackingRecHit::all)) shared++;
             LogDebug(theCategory)<< shared<< " shared hits counter if 2 erease the seed.";
          }
        }
        if(shared!=2) result.push_back(*s2) ;
          
     }//end pairs loop
    
   }// end triplets loop

   return result;

}

