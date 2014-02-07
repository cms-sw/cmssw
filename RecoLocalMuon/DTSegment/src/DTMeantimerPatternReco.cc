/** \file
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 * \author Piotr Traczyk - SINS Warsaw <ptraczyk@fuw.edu.pl>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTMeantimerPatternReco.h"

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

/* C++ Headers */
#include <iterator>
using namespace std;
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/* ====================================================================== */


/// Constructor
DTMeantimerPatternReco::DTMeantimerPatternReco(const edm::ParameterSet& pset) : 
DTRecSegment2DBaseAlgo(pset), theAlgoName("DTMeantimerPatternReco")
{

  theMaxAllowedHits = pset.getParameter<unsigned int>("MaxAllowedHits"); // 100
  theAlphaMaxTheta = pset.getParameter<double>("AlphaMaxTheta");// 0.1 ;
  theAlphaMaxPhi = pset.getParameter<double>("AlphaMaxPhi");// 1.0 ;
  theMaxT0 = pset.getParameter<double>("MaxT0");
  theMinT0 = pset.getParameter<double>("MinT0");
  theMaxChi2 = pset.getParameter<double>("MaxChi2");// 8.0 ;
  debug = pset.getUntrackedParameter<bool>("debug");
  theUpdator = new DTSegmentUpdator(pset);
  theCleaner = new DTSegmentCleaner(pset);
}

/// Destructor
DTMeantimerPatternReco::~DTMeantimerPatternReco() {
  delete theUpdator;
  delete theCleaner;
}

/* Operations */ 
edm::OwnVector<DTSLRecSegment2D>
DTMeantimerPatternReco::reconstruct(const DTSuperLayer* sl,
                                        const std::vector<DTRecHit1DPair>& pairs){

  edm::OwnVector<DTSLRecSegment2D> result;
  vector<DTHitPairForFit*> hitsForFit = initHits(sl, pairs);

  vector<DTSegmentCand*> candidates = buildSegments(sl, hitsForFit);

  vector<DTSegmentCand*>::const_iterator cand=candidates.begin();
  while (cand<candidates.end()) {
    
    DTSLRecSegment2D *segment = (**cand);

    theUpdator->update(segment);

    if (debug) 
      cout<<"Reconstructed 2D segments "<<*segment<<endl;
    result.push_back(segment);

    delete *(cand++); // delete the candidate!
  }

  for (vector<DTHitPairForFit*>::iterator it = hitsForFit.begin(), ed = hitsForFit.end(); 
        it != ed; ++it) delete *it;

  return result;
}

void DTMeantimerPatternReco::setES(const edm::EventSetup& setup){
  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(theDTGeometry);
  theUpdator->setES(setup);
}

vector<DTHitPairForFit*>
DTMeantimerPatternReco::initHits(const DTSuperLayer* sl,
                                     const std::vector<DTRecHit1DPair>& hits){  
  
  vector<DTHitPairForFit*> result;
  for (vector<DTRecHit1DPair>::const_iterator hit=hits.begin();
       hit!=hits.end(); ++hit) {
    result.push_back(new DTHitPairForFit(*hit, *sl, theDTGeometry));
  }
  return result;
}

vector<DTSegmentCand*>
DTMeantimerPatternReco::buildSegments(const DTSuperLayer* sl,
                                          const std::vector<DTHitPairForFit*>& hits){

  typedef vector<DTHitPairForFit*> hitCont;
  typedef hitCont::const_iterator  hitIter;
  vector<DTSegmentCand*> result;
  DTEnums::DTCellSide codes[2]={DTEnums::Left, DTEnums::Right};

  if(debug) {
    cout << "buildSegments: " << sl->id() << " nHits " << hits.size() << endl;
    for (hitIter hit=hits.begin(); hit!=hits.end(); ++hit) cout << **hit<< " wire: " << (*hit)->id() << " DigiTime: " << (*hit)->digiTime() << endl;
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

  GlobalPoint IP;
  float DAlphaMax;
  if ((sl->id()).superlayer()==2)  // Theta SL
    DAlphaMax=theAlphaMaxTheta;
  else // Phi SL
    DAlphaMax=theAlphaMaxPhi;
  
  vector<AssPoint> usedHits;

  // get two hits in different layers and see if there are other hits
  //  compatible with them
  for (hitCont::const_iterator firstHit=hits.begin(); firstHit!=hits.end();
       ++firstHit) {
    for (hitCont::const_reverse_iterator lastHit=hits.rbegin(); 
         (*lastHit)!=(*firstHit); ++lastHit) {
      
      // a geometrical sensibility cut for the two hits
      if (!geometryFilter((*firstHit)->id(),(*lastHit)->id())) continue;
	 
      // create a set of hits for the fit (only the hits between the two selected ones)
      hitCont hitsForFit;
      for (hitCont::const_iterator tmpHit=firstHit+1; (*tmpHit)!=(*lastHit); tmpHit++) 
        if ((geometryFilter((*tmpHit)->id(),(*lastHit)->id())) 
            && (geometryFilter((*tmpHit)->id(),(*firstHit)->id()))) hitsForFit.push_back(*tmpHit);

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
          if (DAlpha>DAlphaMax) continue;
	  
	  if(debug) {
              cout << "Selected hit pair:" << endl;
              cout << "  First " << *(*firstHit) << " Layer Id: " << (*firstHit)->id().layerId() << " Side: " << firstLR << " DigiTime: " << (*firstHit)->digiTime() << endl;
              cout << "  Last "  << *(*lastHit)  << " Layer Id: " << (*lastHit)->id().layerId()  << " Side: " << lastLR << " DigiTime: " << (*lastHit)->digiTime() <<  endl;
          }
        
          vector<AssPoint> assHits;
          // create a candidate hit list
          assHits.push_back(AssPoint(*firstHit,codes[firstLR]));
          assHits.push_back(AssPoint(*lastHit,codes[lastLR]));

          // run hit adding/segment building 
          maxfound = 3;
          addHits(sl,assHits,hitsForFit,result, usedHits);
        }
      }
    }
  }

  // now I have a couple of segment hypotheses, should check for ghost
  if (debug) {
    cout << "Result (before cleaning): " << result.size() << endl;
    for (vector<DTSegmentCand*>::const_iterator seg=result.begin(); seg!=result.end(); ++seg) 
      cout << *(*seg) << endl;
  }        
  result = theCleaner->clean(result);
  if (debug) {
    cout << "Result (after cleaning): " << result.size() << endl;
    for (vector<DTSegmentCand*>::const_iterator seg=result.begin();
         seg!=result.end(); ++seg) 
      cout << *(*seg) << endl;
  }

  return result;
}



void 
DTMeantimerPatternReco::addHits(const DTSuperLayer* sl, vector<AssPoint>& assHits, const vector<DTHitPairForFit*>& hits, 
                                vector<DTSegmentCand*> &result, vector<AssPoint>& usedHits) {

  typedef vector<DTHitPairForFit*> hitCont;
  double chi2l,chi2r,t0_corrl,t0_corrr;
  bool foundSomething = false;

  if (debug) 
    cout << "DTMeantimerPatternReco::addHit " << endl << "   Picked " << assHits.size() << " hits, " << hits.size() << " left." << endl;
  
  if (assHits.size()+hits.size()<maxfound) return;
          
  // loop over the remaining hits
  for (hitCont::const_iterator hit=hits.begin(); hit!=hits.end(); ++hit) {
    if (debug)
      cout << "     Trying B: " << **hit<< " wire: " << (*hit)->id() << endl;
            
    assHits.push_back(AssPoint(*hit, DTEnums::Left));
    bool left_ok=fitWithT0(assHits, chi2l, t0_corrl,0);
    assHits.pop_back();

    assHits.push_back(AssPoint(*hit, DTEnums::Right));
    bool right_ok=fitWithT0(assHits, chi2r, t0_corrr,0);
    assHits.pop_back();

    if (debug) {
      int nHits=assHits.size()+1;
      cout << "         Left:  t0_corr = " << t0_corrl << "ns  chi2/nHits = " << chi2l << "/" << nHits << "  ok: " << left_ok << endl;
      cout << "        Right:  t0_corr = " << t0_corrr << "ns  chi2/nHits = " << chi2r << "/" << nHits << "  ok: " << right_ok << endl;
    }

    if (!left_ok && !right_ok) continue;

    foundSomething = true;    

    // prepare the hit set for the next search, start from the other side                        
    hitCont hitsForFit;
    for (hitCont::const_iterator tmpHit=hit+1; tmpHit!=hits.end(); tmpHit++) 
      if (geometryFilter((*tmpHit)->id(),(*hit)->id())) hitsForFit.push_back(*tmpHit); 
      
    reverse(hitsForFit.begin(),hitsForFit.end());

    // choose only one - left or right
    if (assHits.size()>3 && left_ok && right_ok) {
      if (chi2l<chi2r-0.1) right_ok=false; else
        if (chi2r<chi2l-0.1) left_ok=false;
    }
    if (left_ok) { 
      assHits.push_back(AssPoint(*hit, DTEnums::Left));
      addHits(sl,assHits,hitsForFit,result,usedHits);
      assHits.pop_back();
    }
    if (right_ok) { 
      assHits.push_back(AssPoint(*hit, DTEnums::Right));
      addHits(sl,assHits,hitsForFit,result,usedHits);
      assHits.pop_back();
    }
  }

  if (foundSomething) return;

  // If we already have a segment with more hits from this hit pair - don't save this one.  
  if (assHits.size()<maxfound) return;

  // Check if semgent Ok, calculate chi2
  if (!fitWithT0(assHits, chi2l, t0_corrl,debug)) return;

  // If no more iterations - store the current segment

  DTSegmentCand::AssPointCont pointsSet;
  pointsSet.insert(assHits.begin(),assHits.end());
  DTSegmentCand* seg = new DTSegmentCand(pointsSet,sl);
  theUpdator->fit(seg);

  if (seg) {
    for (vector<AssPoint>::const_iterator hiti = assHits.begin()+1; hiti != assHits.end()-1; hiti++)
      usedHits.push_back(*hiti);

    if (assHits.size()>maxfound) maxfound = assHits.size();
    seg->setChi2(chi2l); // we need to set the chi^2 so that the cleaner can pick the best segments
    if (debug) cout << "Segment built: " << *seg<< endl;
    if (checkDoubleCandidates(result,seg)) {
      result.push_back(seg);
      if (debug) cout << "   Result is now " << result.size() << endl;
    } else {
      if (debug) cout << "   Exists - skipping" << endl;
      delete seg;
    }
  }
}


bool
DTMeantimerPatternReco::geometryFilter( const DTWireId first, const DTWireId second ) const
{
//  return true;

  const int layerLowerCut[4]={0,-1,-2,-2};
  const int layerUpperCut[4]={0, 2, 2, 3};
//  const int layerLowerCut[4]={0,-2,-4,-5};
//  const int layerUpperCut[4]={0, 3, 4, 6};

  // deal only with hits that are in the same SL
  if (first.layerId().superlayerId().superLayer()!=second.layerId().superlayerId().superLayer()) 
    return true;
    
  int deltaLayer=abs(first.layerId().layer()-second.layerId().layer());

  // drop hits in the same layer
  if (!deltaLayer) return false;

  // protection against unexpected layer numbering
  if (deltaLayer>3) { 
    cout << "*** WARNING! DT Layer numbers differ by more than 3! for hits: " << endl;
    cout << "             " << first << endl;
    cout << "             " << second << endl;
    return false;
  }

  // accept only hits in cells "not too far away"
  int deltaWire=first.wire()-second.wire();
  if (second.layerId().layer()%2==0) deltaWire=-deltaWire; // yet another trick to get it right...
  if ((deltaWire<=layerLowerCut[deltaLayer]) || (deltaWire>=layerUpperCut[deltaLayer])) return false;

  return true;
}


bool
DTMeantimerPatternReco::fitWithT0(const vector<AssPoint> &assHits, double &chi2, double &t0_corr, const bool fitdebug) 
{
  typedef vector < pair<double,double> > hitCoord;
  double a,b,coordError,x,y;
  double sx=0,sy=0,sxy=0,sxx=0,ssx=0,ssy=0,s=0,ss=0;
  int leftHits=0,rightHits=0;

  if (assHits.size()<3) return false;

  // I'm assuming the single hit error is the same for all hits...
  coordError=((*(assHits.begin())).first)->localPositionError().xx();

  for (vector<AssPoint>::const_iterator assHit=assHits.begin(); assHit!=assHits.end(); ++assHit) {
    if (coordError!=((*(assHits.begin())).first)->localPositionError().xx()) 
      cout << "   DTMeantimerPatternReco: Warning! Hit errors different within one segment!" << endl;

    x=((*assHit).first)->localPosition((*assHit).second).z();
    y=((*assHit).first)->localPosition((*assHit).second).x();

    sx+=x;
    sy+=y;
    sxy+=x*y;
    sxx+=x*x;
    s++;
    
    if ((*assHit).second==DTEnums::Left) {
      leftHits++;
      ssx+=x;
      ssy+=y;
      ss++;
    } else {
      rightHits++;
      ssx-=x;
      ssy-=y;
      ss--;
    }  
  }      
                    
  if (fitdebug && debug)
    cout  << "   DTMeantimerPatternReco::fitWithT0   Left hits: " << leftHits << "  Right hits: " << rightHits << endl;

  if (leftHits && rightHits) {

    double delta = ss*ss*sxx+s*sx*sx+s*ssx*ssx-s*s*sxx-2*ss*sx*ssx;
    t0_corr=0.;

    if (delta) {
      a=(ssy*s*ssx+sxy*ss*ss+sy*sx*s-sy*ss*ssx-ssy*sx*ss-sxy*s*s)/delta;
      b=(ssx*sy*ssx+sxx*ssy*ss+sx*sxy*s-sxx*sy*s-ssx*sxy*ss-sx*ssy*ssx)/delta;
      t0_corr=(ssx*s*sxy+sxx*ss*sy+sx*sx*ssy-sxx*s*ssy-sx*ss*sxy-ssx*sx*sy)/delta;
    } else return false;
  } else {
    double d = s*sxx - sx*sx;
    b = (sxx*sy- sx*sxy)/ d;
    a = (s*sxy - sx*sy) / d;
    t0_corr=0;
  }

// Calculate the chi^2 of the hits AFTER t0 correction

  double chi,chi2_not0;  
  chi2=0;
  chi2_not0=0;
  
  for (vector<AssPoint>::const_iterator assHit=assHits.begin(); assHit!=assHits.end(); ++assHit) {
    x=((*assHit).first)->localPosition((*assHit).second).z();
    y=((*assHit).first)->localPosition((*assHit).second).x();

    chi=y-a*x-b;
    chi2_not0+=chi*chi/coordError;

    if ((*assHit).second==DTEnums::Left) chi-=t0_corr;
      else chi+=t0_corr;
      
    chi2+=chi*chi/coordError;
  }

  // For 3-hit segments ignore timing information
  if (assHits.size()<4) {
    chi2=chi2_not0;
//    if (chi2<theMaxChi2) return true;
    if (chi2<200.) return true;
      else return false;
  }

  t0_corr/=-0.00543; // convert drift distance to time

  if (debug && fitdebug) 
  {
    cout << "      t0_corr = " << t0_corr << "ns  chi2/nHits = " << chi2 << "/" << assHits.size() << endl;
    
    if (((chi2/(assHits.size()-2)<theMaxChi2) && (t0_corr<theMaxT0) && (t0_corr>theMinT0)) || (assHits.size()==4)) {
      cout << "      a  = " << a  << "       b  = " << b  << endl;
      for (vector<AssPoint>::const_iterator assHit=assHits.begin(); assHit!=assHits.end(); ++assHit) {
        x=((*assHit).first)->localPosition((*assHit).second).z();
        y=((*assHit).first)->localPosition((*assHit).second).x();
        cout << "   z= " << x << "   x= " << y;
        if ((*assHit).second==DTEnums::Left) cout << "   x_corr= " << y+t0_corr*0.00543;
                                       else  cout << "   x_corr= " << y-t0_corr*0.00543;
        cout << "   seg= " << a*x+b << endl;
      }
    }
  }

  if ((chi2/(assHits.size()-2)<theMaxChi2) && (t0_corr<theMaxT0) && (t0_corr>theMinT0)) return true;
    else return false;
}

void 
DTMeantimerPatternReco::rawFit(double &a, double &b, const vector< pair<double,double> > &hits) {

  double s=0,sx=0,sy=0,x,y;
  double sxx=0,sxy=0;

  a=b=0;
  if (hits.size()==0) return;
    
  if (hits.size()==1) {
    b=(*(hits.begin())).second;
  } else {
    for (unsigned int i = 0; i != hits.size(); i++) {
      x=hits[i].first;
      y=hits[i].second;
      sy += y;
      sxy+= x*y;
      s += 1.;
      sx += x;
      sxx += x*x;
    }
    // protect against a vertical line???

    double d = s*sxx - sx*sx;
    b = (sxx*sy- sx*sxy)/ d;
    a = (s*sxy - sx*sy) / d;
  }
}

bool
DTMeantimerPatternReco::checkDoubleCandidates(vector<DTSegmentCand*>& cands,
                                                  DTSegmentCand* seg) {
  for (vector<DTSegmentCand*>::iterator cand=cands.begin();
       cand!=cands.end(); ++cand) {
    if (*(*cand)==*seg) return false;
    if (((*cand)->nHits()>=seg->nHits()) && ((*cand)->chi2ndof()<seg->chi2ndof()))
      if ((*cand)->nSharedHitPairs(*seg)>int(seg->nHits()-2)) return false;
  }    
  return true;
}
