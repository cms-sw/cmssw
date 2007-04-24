#include "RecoNuclear/NuclearSeedGenerator/interface/NuclearTester.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoTracker/CkfPattern/src/RecHitIsInvalid.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <class T>
int index(T it_begin, T it_end, T it_)
{
           int indx=0;
           for(T the_it=it_begin; the_it!=it_end; the_it++, indx++)
                   if(the_it == it_) return indx;
           return indx;
}

NuclearTester::NuclearTester(const edm::EventSetup& es, const edm::ParameterSet& iConfig) :
checkCompletedTrack(iConfig.getParameter<bool>("checkCompletedTrack"))
{
  es.get<TrackerDigiGeometryRecord> ().get (trackerGeom);
  NuclearIndex=0;
}

//----------------------------------------------------------------------
bool NuclearTester::isNuclearInteraction( ) {
    //RQ: assume that the input vector a has been filled from Outside to Inside the tracker !
    // require at least 3 TM vectors to check if nuclear interactions occurs
    if(compatible_hits.size()<4) return false;
    for(std::vector<int>::const_iterator it=compatible_hits.begin(); it!=compatible_hits.end(); it++) LogDebug("NuclearInteractionFinder") << "compatible_hits : " << *it << "\n";

    // find the first min nb of compatible TM :
    std::vector<int>::iterator min_it = min_element(compatible_hits.begin(), compatible_hits.end());

    // this first min cannot be the outermost TM :
    if(min_it == compatible_hits.begin() && checkCompletedTrack==false) return false;

    // if the outermost hit has no compatible TM min_it has to be recalculated (if required by checkCompletedTrack) :
    if(min_it == compatible_hits.begin() && checkCompletedTrack==true && *min_it!=0) return false;
    if(min_it == compatible_hits.begin() && checkCompletedTrack==true && *min_it==0) min_it=min_element(compatible_hits.begin()+1, compatible_hits.end());

    // this first min cannot be the innermost TM :
    if(min_it == compatible_hits.end()-1) return false;

    // if the previous nb of compatible TM is > min+2 and if the next compatible TM is min+-1 -> NUCLEAR
    // example : Nhits = 5, 8, 2, 2, ...
    if((*(min_it-1) - *min_it) > 2 && (*(min_it+1) - *min_it) < 2 ) { 
            NuclearIndex = index<std::vector<int>::iterator>(compatible_hits.begin(),compatible_hits.end(),min_it);
            return true;
    }
  
    // case of : Nhits = 5, 8, 3, 2, 2, ...
    if(min_it-1 == compatible_hits.begin()) return false; //because min_it must be at least at the third position
    if((*(min_it-1) - *min_it) < 2 && (*(min_it-2) - *(min_it-1)) > 2 ) { 
           NuclearIndex = index<std::vector<int>::const_iterator>(compatible_hits.begin(),compatible_hits.end(),min_it-1);
           return true;
    }

    //if the nb of current comp. TM is larger than 2 and larger than the next one+1 return true
    //if(*max_it > 2 && *max_it > *(max_it+1)+1) return true;   
    // TODO: check that position of max meanEstimate = position found with ncompatible_hits (=min_it-1 or min_it-2) !
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
double NuclearTester::meanEstimate(const std::vector<TrajectoryMeasurement>& vecTM) const {
       double meanEst=0;
       int    goodTM=0;
       if(vecTM.empty()) return 0;
       std::vector<TM>::const_iterator last;
       //std::vector<TM>::const_iterator last = this->lastValidTM(vecTM);
       if(vecTM.size() > 2) last = vecTM.begin()+2;
       else last = vecTM.end();

       for(std::vector<TrajectoryMeasurement>::const_iterator itm = vecTM.begin(); itm!=last; itm++) {
             meanEst += itm->estimate();
             goodTM++;
       }
       return meanEst/goodTM;
}
//----------------------------------------------------------------------
std::vector<TrajectoryMeasurement>::const_iterator NuclearTester::lastValidTM(const std::vector<TM>& vecTM) const {
   if (vecTM.empty()) return vecTM.end();
   if (vecTM.front().recHit()->isValid())
            return std::find_if( vecTM.begin(), vecTM.end(), RecHitIsInvalid());
   else return vecTM.end();
}
//----------------------------------------------------------------------
