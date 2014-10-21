#include "RecoTracker/NuclearSeedGenerator/interface/NuclearTester.h"
#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//----------------------------------------------------------------------
NuclearTester::NuclearTester(unsigned int max_hits, const MeasurementEstimator* est, const TrackerGeometry* track_geom) :
    maxHits(max_hits), theEstimator(est), trackerGeom(track_geom) { NuclearIndex=0; }

//----------------------------------------------------------------------
bool NuclearTester::isNuclearInteraction( ) {
// TODO : if energy of primary track is below a threshold don't use checkWithMultiplicity but only checkwith compatible_hits.front

        // 1. if momentum of the primary track is below 5 GeV and if the number of compatible hits >0
        //    assume that a nuclear interaction occured at the end of the track
        if( allTM.front().first.updatedState().globalMomentum().mag() < 5.0 && compatible_hits.front()>0) { NuclearIndex=1; return true; } 

        // 2. else to use multiplicity we require at least 3 TM vectors to check if nuclear interactions occurs
        if(nHitsChecked()<3) return false;

        // 2. check with multiplicity :
	if( checkWithMultiplicity() == true ) return true;
    	else  {
               // 3. last case : uncompleted track with at least 1 compatible hits in the last layer
               if( nHitsChecked() >= maxHits && compatible_hits.front()>0) {NuclearIndex=1; return true; }
        }

	return false;
}
//----------------------------------------------------------------------
bool NuclearTester::checkWithMultiplicity() {
    //RQ: assume that the input vector of compatible hits has been filled from Outside to Inside the tracker !

    // find the first min nb of compatible TM :
    std::vector<int>::iterator min_it = min_element(compatible_hits.begin(), compatible_hits.end());

    // if the outermost hit has no compatible TM, min_it has to be recalculated :
    if(min_it == compatible_hits.begin() && *min_it!=0) return false;
    if(min_it == compatible_hits.begin() && *min_it==0) min_it=min_element(compatible_hits.begin()+1, compatible_hits.end());

    // this first min cannot be the innermost TM :
    if(min_it == compatible_hits.end()-1) return false;

    // if the previous nb of compatible TM is > min+2 and if the next compatible TM is min+-1 -> NUCLEAR
    // example : Nhits = 5, 8, 2, 2, ...
    if((*(min_it-1) - *min_it) > 2 && (*(min_it+1) - *min_it) < 2 ) {
            NuclearIndex = min_it - compatible_hits.begin();
            return true;
    }

    // case of : Nhits = 5, 8, 3, 2, 2, ...
    if(min_it-1 != compatible_hits.begin())  //because min_it must be at least at the third position
    {
      if(min_it-1 != compatible_hits.begin() && (*(min_it-1) - *min_it) < 2 && (*(min_it-2) - *(min_it-1)) > 2 ) {
           NuclearIndex = min_it - 1 - compatible_hits.begin();
           return true;
      }
    }

    return false;
}

//----------------------------------------------------------------------
double NuclearTester::meanHitDistance(const std::vector<TrajectoryMeasurement>& vecTM) const {
    std::vector<GlobalPoint> vgp = this->HitPositions(vecTM);
    double mean_dist=0;
    int ncomb=0;
    if(vgp.size()<2) return 0;
    for(std::vector<GlobalPoint>::iterator itp = vgp.begin(); itp != vgp.end()-1; itp++) {
       for(std::vector<GlobalPoint>::iterator itq = itp+1; itq != vgp.end(); itq++) {
          double dist = ((*itp) - (*itq)).mag();
          // to calculate mean distance between particles and not hits (to not take into account twice stereo hits)
          if(dist > 1E-12) {
                mean_dist += dist;
                ncomb++;
          }
        }
    }
    return mean_dist/ncomb;
}
//----------------------------------------------------------------------
std::vector<GlobalPoint> NuclearTester::HitPositions(const std::vector<TrajectoryMeasurement>& vecTM) const {
   std::vector<GlobalPoint> gp;

   std::vector<TM>::const_iterator last = this->lastValidTM(vecTM);

   for(std::vector<TrajectoryMeasurement>::const_iterator itm = vecTM.begin(); itm!=last; itm++) {
               ConstRecHitPointer trh = itm->recHit();
               if(trh->isValid()) gp.push_back(trackerGeom->idToDet(trh->geographicalId())->surface().toGlobal(trh->localPosition()));
   }
   return gp;
}
//----------------------------------------------------------------------
double NuclearTester::fwdEstimate(const std::vector<TrajectoryMeasurement>& vecTM) const {
       if(vecTM.empty()) return 0;

       auto hit = vecTM.front().recHit().get();
       if( hit->isValid() )
          return theEstimator->estimate( vecTM.front().forwardPredictedState(), *hit ).second;
       else return -1;
/*
       double meanEst=0;
       int    goodTM=0;
       std::vector<TM>::const_iterator last;
       //std::vector<TM>::const_iterator last = this->lastValidTM(vecTM);
       if(vecTM.size() > 2) last = vecTM.begin()+2;
       else last = vecTM.end();

       for(std::vector<TrajectoryMeasurement>::const_iterator itm = vecTM.begin(); itm!=last; itm++) {
             meanEst += itm->estimate();
             goodTM++;
       }
       return meanEst/goodTM;
*/
}
//----------------------------------------------------------------------
std::vector<TrajectoryMeasurement>::const_iterator NuclearTester::lastValidTM(const std::vector<TM>& vecTM) const {
   if (vecTM.empty()) return vecTM.end();
   if (vecTM.front().recHit()->isValid())
            return std::find_if( vecTM.begin(), vecTM.end(), RecHitIsInvalid());
   else return vecTM.end();
}
//----------------------------------------------------------------------
