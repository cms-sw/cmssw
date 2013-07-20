/** \file
 *
 * $Date: 2008/12/03 12:52:22 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTCombinatorialExtendedPatternReco.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCleaner.h"
#include "RecoLocalMuon/DTSegment/src/DTHitPairForFit.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentCand.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentExtendedCand.h"


/* C++ Headers */
#include <iterator>
using namespace std;
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/* ====================================================================== */

/// Constructor
DTCombinatorialExtendedPatternReco::DTCombinatorialExtendedPatternReco(const edm::ParameterSet& pset) : 
DTRecSegment2DBaseAlgo(pset), theAlgoName("DTCombinatorialExtendedPatternReco")
{
  theMaxAllowedHits = pset.getParameter<unsigned int>("MaxAllowedHits"); // 100
  theAlphaMaxTheta = pset.getParameter<double>("AlphaMaxTheta");// 0.1 ;
  theAlphaMaxPhi = pset.getParameter<double>("AlphaMaxPhi");// 1.0 ;
  debug = pset.getUntrackedParameter<bool>("debug"); //true;
  theUpdator = new DTSegmentUpdator(pset);
  theCleaner = new DTSegmentCleaner(pset);
  string theHitAlgoName = pset.getParameter<string>("recAlgo");
  usePairs = !(theHitAlgoName=="DTNoDriftAlgo");
}

/// Destructor
DTCombinatorialExtendedPatternReco::~DTCombinatorialExtendedPatternReco() {
}

/* Operations */ 
edm::OwnVector<DTSLRecSegment2D>
DTCombinatorialExtendedPatternReco::reconstruct(const DTSuperLayer* sl,
                                                const std::vector<DTRecHit1DPair>& pairs){

  if(debug) cout << "DTCombinatorialExtendedPatternReco::reconstruct" << endl;
  theTriedPattern.clear();
  edm::OwnVector<DTSLRecSegment2D> result;
  vector<DTHitPairForFit*> hitsForFit = initHits(sl, pairs);

  vector<DTSegmentCand*> candidates = buildSegments(sl, hitsForFit);

  vector<DTSegmentCand*>::const_iterator cand=candidates.begin();
  while (cand<candidates.end()) {
    
    DTSLRecSegment2D *segment = (**cand);

    theUpdator->update(segment);

    result.push_back(segment);

    if (debug) {
      cout<<"Reconstructed 2D extended segments "<< result.back() <<endl;
    }

    delete *(cand++); // delete the candidate!
  }

  for (vector<DTHitPairForFit*>::iterator it = hitsForFit.begin(), ed = hitsForFit.end(); 
        it != ed; ++it) delete *it;

  return result;
}

void DTCombinatorialExtendedPatternReco::setES(const edm::EventSetup& setup){
  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  theUpdator->setES(setup);
}

void DTCombinatorialExtendedPatternReco::setClusters(vector<DTSLRecCluster> clusters) {
  theClusters = clusters;
}

vector<DTHitPairForFit*>
DTCombinatorialExtendedPatternReco::initHits(const DTSuperLayer* sl,
                                     const std::vector<DTRecHit1DPair>& hits){  
  
  vector<DTHitPairForFit*> result;
  for (vector<DTRecHit1DPair>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {
    result.push_back(new DTHitPairForFit(*hit, *sl, theDTGeometry));
  }
  return result;
}

vector<DTSegmentCand*>
DTCombinatorialExtendedPatternReco::buildSegments(const DTSuperLayer* sl,
                                          const std::vector<DTHitPairForFit*>& hits){

  typedef vector<DTHitPairForFit*> hitCont;
  typedef hitCont::const_iterator  hitIter;
  vector<DTSegmentCand*> result;
  
  if(debug) {
    cout << "DTCombinatorialExtendedPatternReco::buildSegments: " << sl->id() << " nHits " << hits.size() << endl;
    for (vector<DTHitPairForFit*>::const_iterator hit=hits.begin();
         hit!=hits.end(); ++hit) cout << **hit<< endl;
  }

  // 10-Mar-2004 SL
  // put a protection against heavily populated chambers, for which the segment
  // building could lead to infinite memory usage...
  if (hits.size() > theMaxAllowedHits ) {
    if(debug) {
      cout << "Warning: this SuperLayer " << sl->id() << " has too many hits : "
        << hits.size() << " max allowed is " << theMaxAllowedHits << endl;
      cout << "Skipping segment reconstruction... " << endl;
    }
    return result;
  }

  /// get two hits in different layers and see if there are other / hits
  //  compatible with them
  for (hitCont::const_iterator firstHit=hits.begin(); firstHit!=hits.end();
       ++firstHit) {
    for (hitCont::const_reverse_iterator lastHit=hits.rbegin(); 
         (*lastHit)!=(*firstHit); ++lastHit) {
      // hits must nor in the same nor in adiacent layers
      if ( fabs((*lastHit)->id().layerId()-(*firstHit)->id().layerId())<=1 ) continue;
      if(debug) {
        cout << "Selected these two hits pair " << endl;
        cout << "First " << *(*firstHit) << " Layer Id: " << (*firstHit)->id().layerId() << endl;
        cout << "Last "  << *(*lastHit)  << " Layer Id: " << (*lastHit)->id().layerId()  << endl;
      }

      GlobalPoint IP;
      float DAlphaMax;
      if ((sl->id()).superlayer()==2)  // Theta SL
        DAlphaMax=theAlphaMaxTheta;
      else // Phi SL
        DAlphaMax=theAlphaMaxPhi;

      DTEnums::DTCellSide codes[2]={DTEnums::Right, DTEnums::Left};
      for (int firstLR=0; firstLR<2; ++firstLR) {
        for (int lastLR=0; lastLR<2; ++lastLR) {
          // TODO move the global transformation in the DTHitPairForFit class
          // when it will be moved I will able to remove the sl from the input parameter
          GlobalPoint gposFirst=sl->toGlobal( (*firstHit)->localPosition(codes[firstLR]) );
          GlobalPoint gposLast= sl->toGlobal( (*lastHit)->localPosition(codes[lastLR]) );

          GlobalVector gvec=gposLast-gposFirst;
          GlobalVector gvecIP=gposLast-IP;

          // difference in angle measured
          float DAlpha=fabs(gvec.theta()-gvecIP.theta());

          // cout << "DAlpha " << DAlpha << endl;
          if (DAlpha<DAlphaMax) {

            // create a segment hypotesis
            // I don't need a true segment, just direction and position
            LocalPoint posIni = (*firstHit)->localPosition(codes[firstLR]);
            LocalVector dirIni = 
              ((*lastHit)->localPosition(codes[lastLR])-posIni).unit();

            // search for other compatible hits, with or without the L/R solved
            vector<AssPoint> assHits = findCompatibleHits(posIni, dirIni, hits);
            if(debug) 
              cout << "compatible hits " << assHits.size() << endl;

            // here return an extended candidate (which _has_ the original
            // segment)
            DTSegmentExtendedCand* seg = buildBestSegment(assHits, sl);

            if (seg) {
              if(debug) 
                cout << "segment " << *seg<< endl;

              // check if the chi2 and #hits are ok
              if (!seg->good()) { // good is reimplmented in extended segment
                delete seg;
              } else { 

                // remove duplicated segments 
                if (checkDoubleCandidates(result,seg)) {
                  // add to the vector of hypotesis
                  // still work with extended segments
                  result.push_back(seg);
                  if(debug) 
                    cout << "result is now " << result.size() << endl;
                } else { // delete it!
                  delete seg;
                  if(debug) 
                    cout << "already existing" << endl;
                }
              }
            }
          }
        }
      }
    }
  }
  if (debug) {
    for (vector<DTSegmentCand*>::const_iterator seg=result.begin();
         seg!=result.end(); ++seg) 
      cout << *(*seg) << endl;
  }

  // now I have a couple of segment hypotesis, should check for ghost
  // still with extended candidates
  result = theCleaner->clean(result);
  if (debug) {
    cout << "result no ghost  " << result.size() << endl;
    for (vector<DTSegmentCand*>::const_iterator seg=result.begin();
         seg!=result.end(); ++seg) 
      cout << *(*seg) << endl;
  }

  // here, finally, I have to return the set of _original_ segments, not the
  // extended ones.
  return result;
}


vector<DTCombinatorialExtendedPatternReco::AssPoint>
DTCombinatorialExtendedPatternReco::findCompatibleHits(const LocalPoint& posIni,
                                               const LocalVector& dirIni,
                                               const vector<DTHitPairForFit*>& hits) {
  if (debug) cout << "Pos: " << posIni << " Dir: "<< dirIni << endl;
  vector<AssPoint> result;

  // counter to early-avoid double counting in hits pattern
  vector<int> tried;
  int nCompatibleHits=0;

  typedef vector<DTHitPairForFit*> hitCont;
  typedef hitCont::const_iterator  hitIter;
  for (hitIter hit=hits.begin(); hit!=hits.end(); ++hit) {
    pair<bool,bool> isCompatible = (*hit)->isCompatible(posIni, dirIni);
    if (debug) 
      cout << "isCompatible " << isCompatible.first << " " <<
        isCompatible.second << endl;

    // if only one of the two is compatible, then the LR is assigned,
    // otherwise is undefined

    DTEnums::DTCellSide lrcode;
    if (isCompatible.first && isCompatible.second) {
      usePairs ? lrcode=DTEnums::undefLR : lrcode=DTEnums::Left ; // if not usePairs then only use single side 
      tried.push_back(3);
      nCompatibleHits++;
    }
    else if (isCompatible.first) {
      lrcode=DTEnums::Left;
      tried.push_back(2);
      nCompatibleHits++;
    }
    else if (isCompatible.second) {
      lrcode=DTEnums::Right;
      tried.push_back(1);
      nCompatibleHits++;
    }
    else {
      tried.push_back(0);
      continue; // neither is compatible
    }
    result.push_back(AssPoint(*hit, lrcode));
  }
  

  // check if too few associated hits or pattern already tried
  if ( nCompatibleHits < 3 || find(theTriedPattern.begin(), theTriedPattern.end(),tried) == theTriedPattern.end()) {
    theTriedPattern.push_back(tried);
  } else {
    if (debug) {
      vector<vector<int> >::const_iterator t=find(theTriedPattern.begin(),
                                                  theTriedPattern.end(),
                                                  tried);
      cout << "Already tried";
      copy((*t).begin(), (*t).end(), ostream_iterator<int>(std::cout));
      cout << endl;
    }
    // empty the result vector
    result.clear();
  }
  return result;
}

DTSegmentExtendedCand*
DTCombinatorialExtendedPatternReco::buildBestSegment(std::vector<AssPoint>& hits,
                                                     const DTSuperLayer* sl) {
  if (debug) cout << "DTCombinatorialExtendedPatternReco::buildBestSegment " <<
    hits.size()  << endl;
  if (hits.size()<3) {
    //cout << "buildBestSegment: hits " << hits.size()<< endl;
    return 0; // a least 3 point
  }

  // hits with defined LR
  vector<AssPoint> points;

  // without: I store both L and R, a deque since I need front insertion and
  // deletion
  deque<DTHitPairForFit* > pointsNoLR; 

  // first add only the hits with LR assigned
  for (vector<AssPoint>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {
    if ((*hit).second != DTEnums::undefLR) {
      points.push_back(*hit);
    } else { // then also for the undef'd one
      pointsNoLR.push_back((*hit).first);
    }
  }

  if(debug) {
    cout << "points " << points.size() << endl;
    cout << "pointsNoLR " << pointsNoLR.size() << endl;
  }

  // build all possible candidates using L/R ambiguity
  vector<DTSegmentCand*> candidates ;

  buildPointsCollection(points, pointsNoLR, candidates, sl);

  // here I try to add the external clusters and build a set of "extended
  // segment candidate
  vector<DTSegmentExtendedCand*> extendedCands = extendCandidates(candidates,
                                                                  sl); 
  if (debug) cout << "extended candidates " << extendedCands.size() << endl;

  // so now I have build a given number of segments, I should find the best one,
  // by #hits and chi2.
  vector<DTSegmentExtendedCand*>::const_iterator bestCandIter = extendedCands.end();
  double minChi2=999999.;
  unsigned int maxNumHits=0;
  for (vector<DTSegmentExtendedCand*>::const_iterator iter=extendedCands.begin();
       iter!=extendedCands.end(); ++iter) {
    if ((*iter)->nHits()==maxNumHits && (*iter)->chi2()<minChi2) {
      minChi2=(*iter)->chi2();
      bestCandIter=iter;
    } else if ((*iter)->nHits()>maxNumHits) {
      maxNumHits=(*iter)->nHits();
      minChi2=(*iter)->chi2();
      bestCandIter=iter;
    }
  }

  // delete all candidates but the best one!
  for (vector<DTSegmentExtendedCand*>::iterator iter=extendedCands.begin();
       iter!=extendedCands.end(); ++iter) 
    if (iter!=bestCandIter) delete (*iter);

  // return the best candate if any
  if (bestCandIter != extendedCands.end()) {
    return (*bestCandIter);
  }
  return 0;
}

void
DTCombinatorialExtendedPatternReco::buildPointsCollection(vector<AssPoint>& points, 
                                                  deque<DTHitPairForFit*>& pointsNoLR, 
                                                  vector<DTSegmentCand*>& candidates,
                                                  const DTSuperLayer* sl) {

  if(debug) {
    cout << "DTCombinatorialExtendedPatternReco::buildPointsCollection " << endl;
    cout << "points: " << points.size() << " NOLR: " << pointsNoLR.size()<< endl;
  }
  if (pointsNoLR.size()>0) { // still unassociated points!
    DTHitPairForFit* unassHit = pointsNoLR.front();
    // try with the right
    if(debug)
      cout << "Right hit" << endl;
    points.push_back(AssPoint(unassHit, DTEnums::Right));
    pointsNoLR.pop_front();
    buildPointsCollection(points, pointsNoLR, candidates, sl);
    pointsNoLR.push_front((unassHit));
    points.pop_back();

    // try with the left
    if(debug)
      cout << "Left hit" << endl;
    points.push_back(AssPoint(unassHit, DTEnums::Left));
    pointsNoLR.pop_front();
    buildPointsCollection(points, pointsNoLR, candidates, sl);
    pointsNoLR.push_front((unassHit));
    points.pop_back();
  } else { // all associated

    if(debug) {
      cout << "The Hits were" << endl;
      copy(points.begin(), points.end(),
           ostream_iterator<DTSegmentCand::AssPoint>(std::cout));
      cout << "----" << endl;
      cout << "All associated " << endl;
    }
    DTSegmentCand::AssPointCont pointsSet;

    // for (vector<AssPoint>::const_iterator point=points.begin();
    //      point!=points.end(); ++point) 
    pointsSet.insert(points.begin(),points.end());

    if(debug) {
      cout << "The Hits are" << endl;
      copy(pointsSet.begin(), pointsSet.end(),
           ostream_iterator<DTSegmentCand::AssPoint>(std::cout));
      cout << "----" << endl;
    }

    DTSegmentCand* newCand = new DTSegmentCand(pointsSet,sl);
    if (theUpdator->fit(newCand)) candidates.push_back(newCand);
    else delete newCand; // bad seg, too few hits
  }
}

bool
DTCombinatorialExtendedPatternReco::checkDoubleCandidates(vector<DTSegmentCand*>& cands,
                                                  DTSegmentCand* seg) {
  for (vector<DTSegmentCand*>::iterator cand=cands.begin();
       cand!=cands.end(); ++cand) 
    if (*(*cand)==*seg) return false;
  return true;
}

vector<DTSegmentExtendedCand*>
DTCombinatorialExtendedPatternReco::extendCandidates(vector<DTSegmentCand*>& candidates,
                                                     const DTSuperLayer* sl) {
  if (debug) cout << "extendCandidates " << candidates.size() << endl;
  vector<DTSegmentExtendedCand*> result;

  // in case of phi SL just return
  if (sl->id().superLayer() != 2 ) {
    for (vector<DTSegmentCand*>:: const_iterator cand=candidates.begin();
       cand!=candidates.end(); ++cand) {
      DTSegmentExtendedCand* extendedCand = new DTSegmentExtendedCand(*cand);
      // and delete the original candidate
      delete *cand;
      result.push_back(extendedCand);
    }
    return result;
  }

  // first I have to select the cluster which are compatible with the actual
  // candidate, namely +/-1 sector/station/wheel 
  vector<DTSegmentExtendedCand::DTSLRecClusterForFit> clustersWithPos;
  if (debug) cout << "AllClustersWithPos " << theClusters.size() << endl;
  if(debug) cout << "SL:   " << sl->id() << endl;
  for (vector<DTSLRecCluster>::const_iterator clus=theClusters.begin();
       clus!=theClusters.end(); ++clus) {
    if(debug) cout << "CLUS: " << (*clus).superLayerId() << endl;
    if ((*clus).superLayerId().superLayer()==2 && closeSL(sl->id(),(*clus).superLayerId())) {
      // and then get their pos in the actual SL frame
      const DTSuperLayer* clusSl =
        theDTGeometry->superLayer((*clus).superLayerId());
      LocalPoint pos=sl->toLocal(clusSl->toGlobal((*clus).localPosition()));
      //LocalError err=sl->toLocal(clusSl->toGlobal((*clus).localPositionError()));
      LocalError err=(*clus).localPositionError();
      clustersWithPos.push_back(DTSegmentExtendedCand::DTSLRecClusterForFit(*clus, pos, err) );
    }
  }
  if (debug) cout << "closeClustersWithPos " << clustersWithPos.size() << endl;

  for (vector<DTSegmentCand*>:: const_iterator cand=candidates.begin();
       cand!=candidates.end(); ++cand) {
    // create an extended candidate
    DTSegmentExtendedCand* extendedCand = new DTSegmentExtendedCand(*cand);
    // and delete the original candidate
    delete *cand;
    // do this only for theta SL
    if (extendedCand->superLayer()->id().superLayer() == 2 ) {
      // first check compatibility between cand and clusForFit
      for (vector<DTSegmentExtendedCand::DTSLRecClusterForFit>::const_iterator
           exClus=clustersWithPos.begin(); exClus!=clustersWithPos.end(); ++exClus) {
        if (extendedCand->isCompatible(*exClus)) {
          if (debug) cout << "is compatible " << endl;
          // add compatible cluster
          extendedCand->addClus(*exClus);
        }
      }
      // fit the segment
      if (debug) cout << "extended cands nHits: " << extendedCand->nHits() <<endl;
      if (theUpdator->fit(extendedCand)) {
        // add to result
        result.push_back(extendedCand);
      } else {
        cout << "Bad fit" << endl;
        delete extendedCand;
      }
    } else { // Phi SuperLayer
      result.push_back(extendedCand);
    }
  }

  return result;
}

bool DTCombinatorialExtendedPatternReco::closeSL(const DTSuperLayerId& id1,
                                                 const DTSuperLayerId& id2) {
  if (id1==id2) return false;
  if (abs(id1.wheel()-id2.wheel())>1 ) return false;
  // take into account also sector 13 and 14
  int sec1 = ( id1.sector()==13 ) ? 4: id1.sector();
  sec1=(sec1==14)? 10: sec1;
  int sec2 = ( id2.sector()==13 ) ? 4: id2.sector();
  sec2=(sec2==14)? 10: sec2;
  // take into account also sector 1/12
  if (abs(sec1-sec2)>1 && abs(sec1-sec2)!=11 ) return false;
  //if (abs(id1.station()-id2.station())>1 ) return false;
  return true;
}
