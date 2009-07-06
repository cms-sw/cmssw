#include "RecoHIMuon/HiMuTracking/interface/HICSimpleNavigationSchool.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkNavigation/interface/SimpleBarrelNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleForwardNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"
#include "RecoTracker/TkNavigation/interface/DiskLessInnerRadius.h"
#include "RecoTracker/TkNavigation/interface/SymmetricLayerFinder.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include "Utilities/General/interface/CMSexception.h"

#include <functional>
#include <algorithm>
#include <map>
#include <cmath>

using namespace std;

HICSimpleNavigationSchool::HICSimpleNavigationSchool(const GeometricSearchTracker* theInputTracker,
					             const MagneticField* field) : 
  theBarrelLength(0),theField(field), theTracker(theInputTracker)
{

  theAllDetLayersInSystem=&theInputTracker->allLayers();
//  vector<DetLayer*> theDetLayers;
  // Get barrel layers
  vector<BarrelDetLayer*> blc = theTracker->barrelLayers(); 
//  int kk=-1;
  for ( vector<BarrelDetLayer*>::iterator i = blc.begin(); i != blc.end(); i++) {
    if((*i)->specificSurface().radius()<20. || (*i)->specificSurface().radius()>65.) {
//     cout<<"Barrel surface "<<(*i)->specificSurface().radius()<<endl;
//     kk++;
//     if(kk==excludedBarrelLayer) continue; 
     theBarrelLayers.push_back( (*i) );
     theDetLayers.push_back( (*i) );
    }
  }

  // get forward layers

  vector<ForwardDetLayer*> flc = theTracker->forwardLayers(); 
  for ( vector<ForwardDetLayer*>::iterator i = flc.begin(); i != flc.end(); i++) {
    if(((*i)->specificSurface().outerRadius()>60.&& (*i)->specificSurface().innerRadius()>20.)|| (*i)->specificSurface().innerRadius()<10.) {
 //     cout<<" Endcap surface "<<(*i)->specificSurface().outerRadius()<<" "<<(*i)->specificSurface().innerRadius()<<endl;
      theForwardLayers.push_back( (*i) );
      theDetLayers.push_back( (*i) );
    }
  }
 
//  cout<<" Number of detectors"<< theDetLayers.size()<<endl;
 
  FDLI middle = find_if( theForwardLayers.begin(), theForwardLayers.end(),
			 not1(DetBelowZ(0)));
 // cout<<" Point 0 "<<endl;
  theLeftLayers  = FDLC( theForwardLayers.begin(), middle);
 //  cout<<" Point 1 "<<endl;
  theRightLayers = FDLC( middle, theForwardLayers.end());
 //  cout<<" Point 2 "<<endl; 
  SymmetricLayerFinder symFinder( theForwardLayers);
 //  cout<<" Point 3 "<<endl;
  // only work on positive Z side; negative by mirror symmetry later
  linkBarrelLayers( symFinder);
  //  cout<<" Point 4 "<<endl;
  linkForwardLayers( symFinder);
   // cout<<" Point 5 "<<endl;
  establishInverseRelations();
  // cout<<" Point 6 "<<endl;
}

HICSimpleNavigationSchool::HICSimpleNavigationSchool(const GeometricSearchTracker* theInputTracker,
					             const MagneticField* field, int ll, int nn) : 
  theBarrelLength(0),theField(field), theTracker(theInputTracker)
{

  excludedBarrelLayer = ll;
  excludedForwardLayer = nn;

  theAllDetLayersInSystem=&theInputTracker->allLayers();
//  vector<DetLayer*> theDetLayers;
  // Get barrel layers
  vector<BarrelDetLayer*> blc = theTracker->barrelLayers(); 
  int kk = 0;

  cout<<" kk= "<<kk<<" excludedBarrel "<<excludedBarrelLayer<<" excludedForward "<<excludedForwardLayer<<endl;

  for ( vector<BarrelDetLayer*>::iterator i = blc.begin(); i != blc.end(); i++) {
    if((*i)->specificSurface().radius()<20. || (*i)->specificSurface().radius()>65.) {
//     cout<<"Barrel surface "<<kk<<" "<<(*i)->specificSurface().radius()<<" "<<excludedBarrelLayer<<endl;
     if(kk != excludedBarrelLayer ) {
      //cout<<"Included layer "<<endl;  
      theBarrelLayers.push_back( (*i) );
      theDetLayers.push_back( (*i) );
     }
       else {
  //      cout<<"Excluded layer "<<endl;
       }
    }
    kk++; 
  }

  // get forward layers
  if( excludedForwardLayer > -1 ) {
  kk=0; 
  vector<ForwardDetLayer*> flc = theTracker->forwardLayers(); 
  for ( vector<ForwardDetLayer*>::iterator i = flc.begin(); i != flc.end(); i++) 
  {
    if(((*i)->specificSurface().outerRadius()>60.&& 
                  (*i)->specificSurface().innerRadius()>20.)|| (*i)->specificSurface().innerRadius()<10.) 
    {
//      cout<<" Endcap surface "<<kk<<" "<<(*i)->specificSurface().position().z()<<endl;
       if(kk != excludedForwardLayer && kk != excludedForwardLayer + 14) 
       {
        //cout<<"Excluded layer "<<endl;
        theForwardLayers.push_back( (*i) );
        theDetLayers.push_back( (*i) );
       } else {
  //       cout<<"Excluded layer "<<endl; 
       }
     } // radius
      kk++;
   } // forward layers
   } else {
  vector<ForwardDetLayer*> flc = theTracker->forwardLayers(); 
  for ( vector<ForwardDetLayer*>::iterator i = flc.begin(); i != flc.end(); i++) 
  {
    if(((*i)->specificSurface().outerRadius()>60.&& 
                  (*i)->specificSurface().innerRadius()>20.)|| (*i)->specificSurface().innerRadius()<10.) 
    {
//      cout<<" Endcap surface "<<kk<<" "<<(*i)->specificSurface().position().z()<<endl;
        //cout<<"Excluded layer "<<endl;
        theForwardLayers.push_back( (*i) );
        theDetLayers.push_back( (*i) );
     } // radius
      kk++;
   } // forward layers
   } // No Excluded layer
 
//  cout<<" Number of detectors"<< theDetLayers.size()<<endl;
 
  FDLI middle = find_if( theForwardLayers.begin(), theForwardLayers.end(),
			 not1(DetBelowZ(0)));
  cout<<" Point 0 "<<endl;
  theLeftLayers  = FDLC( theForwardLayers.begin(), middle);
   cout<<" Point 1 "<<endl;
  theRightLayers = FDLC( middle, theForwardLayers.end());
   cout<<" Point 2 "<<endl; 
  SymmetricLayerFinder symFinder( theForwardLayers);
   cout<<" Point 3 "<<endl;
  // only work on positive Z side; negative by mirror symmetry later
  linkBarrelLayers( symFinder);
    cout<<" Point 4 "<<endl;
  linkForwardLayers( symFinder);
    cout<<" Point 5 "<<endl;
  establishInverseRelations();
   cout<<" Point 6 "<<endl;
}

HICSimpleNavigationSchool::StateType 
HICSimpleNavigationSchool::navigableLayers() const
{
  StateType result;
  for ( vector< SimpleBarrelNavigableLayer*>::const_iterator 
	  ib = theBarrelNLC.begin(); ib != theBarrelNLC.end(); ib++) {
    result.push_back( *ib);
  }
  for ( vector< SimpleForwardNavigableLayer*>::const_iterator 
	  ifl = theForwardNLC.begin(); ifl != theForwardNLC.end(); ifl++) {
    result.push_back( *ifl);
  }
  return result;
}

void HICSimpleNavigationSchool::
linkBarrelLayers( SymmetricLayerFinder& symFinder)
{
  // Link barrel layers outwards
  for ( BDLI i = theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
    BDLC reachableBL;
    FDLC leftFL;
    FDLC rightFL;

    // always add next barrel layer first
    if ( i+1 != theBarrelLayers.end()) reachableBL.push_back(*(i+1));
 
    // Add closest reachable forward layer (except for last BarrelLayer)
    if (i != theBarrelLayers.end() - 1) {
      linkNextForwardLayer( *i, rightFL);
    }

    // Add next BarrelLayer with length larger than the current BL
    if ( i+2 < theBarrelLayers.end()) {
      linkNextLargerLayer( i, theBarrelLayers.end(), reachableBL);
    }

    theBarrelNLC.push_back( new 
       SimpleBarrelNavigableLayer( *i, reachableBL,
				   symFinder.mirror(rightFL),
				   rightFL,theField, 5.));
  }
}

void HICSimpleNavigationSchool::linkNextForwardLayer( BarrelDetLayer* bl, 
						   FDLC& rightFL)
{
  // find first forward layer with larger Z and larger outer radius
 // float length = bl->surface().bounds().length() / 2.;
//  float radius = bl->specificSurface().radius();
  for ( FDLI fli = theRightLayers.begin();
	fli != theRightLayers.end(); fli++) {
   // if ( length < (**fli).position().z() &&
   //	 radius < (**fli).specificSurface().outerRadius()) 
  if(fabs((**fli).position().z())>130. && fabs((**fli).position().z())<132.)
    {
#ifdef DEBUG
      cout<<" Add to barrel layer "<<radius<<" Forward layer "<<(**fli).position().z()<<endl;
#endif
      rightFL.push_back( *fli);
      return;
    }
  }
}

void HICSimpleNavigationSchool::linkNextLargerLayer( BDLI bli, BDLI end,
						  BDLC& reachableBL)
{
  // compare length of next layer with length of following ones
  float length = (**(bli+1)).surface().bounds().length();
  float epsilon = 0.1;

  for ( BDLI i = bli+2; i < end; i++) {
    if ( length + epsilon < (**i).surface().bounds().length()) {
      reachableBL.push_back( *i);
      return;
    }
  }
}

void HICSimpleNavigationSchool::
linkForwardLayers( SymmetricLayerFinder& symFinder)
{

  // handle right side first, groups are only on the right 
  vector<FDLC> groups = splitForwardLayers();

  LogDebug("TkNavigation") << "SimpleNavigationSchool, Forward groups size = " << groups.size() ;
  for (vector<FDLC>::iterator g = groups.begin(); g != groups.end(); g++) {
    LogDebug("TkNavigation") << "group " << g - groups.begin() << " has " 
			     << g->size() << " layers " ;
  }

  for ( vector<FDLC>::iterator group = groups.begin();
	group != groups.end(); group++) {

    for ( FDLI i = group->begin(); i != group->end(); i++) {

      BDLC reachableBL;
      FDLC reachableFL;
 
      // Always connect to next barrel layer first, if exists
      linkNextBarrelLayer( *i, reachableBL);

      // Then always connect to next forward layer of "same" size, 
      // and layers of larger inner Radius
      linkNextLayerInGroup( i, *group, reachableFL);

      // Then connect to next N fw layers of next size
      if ( group+1 != groups.end()) {
	linkOuterGroup( *i, *(group+1), reachableFL);
      }

      // or connect within the group if outer radius increases
      linkWithinGroup( i, *group, reachableFL);

      theForwardNLC.push_back( new SimpleForwardNavigableLayer( *i,reachableBL,
								reachableFL,
								theField,
								5.));
      theForwardNLC.push_back( new SimpleForwardNavigableLayer( symFinder.mirror(*i),
								reachableBL,
								symFinder.mirror(reachableFL),
								theField,
								5.));

    }
  }

//    // now the left side by symmetry
//    for ( FDLI ileft = theLeftLayers.begin(); 
//  	ileft != theLeftLayers.end(); ileft++) {
//      ForwardDetLayer* right = symFinder.mirror( *ileft);
    
//      theForwardNLC.push_back( new 
//         SimpleForwardNavigableLayer( *ileft , right->nextBarrelLayers(),
//  	                      symFinder.mirror(right->nextForwardLayers())));
//    }
}

void HICSimpleNavigationSchool::linkNextBarrelLayer( ForwardDetLayer* fl,
						  BDLC& reachableBL)
{
  if ( fl->position().z() > barrelLength()) return;

 // float outerRadius = fl->specificSurface().outerRadius();
  float zpos        = fl->position().z();
  for ( BDLI bli = theBarrelLayers.begin(); bli != theBarrelLayers.end(); bli++) {
//    if ( outerRadius < (**bli).specificSurface().radius() &&
//	 zpos        < (**bli).surface().bounds().length() / 2.)
   if( fabs(zpos) > 130. && fabs(zpos) < 132. ) 
    {
      cout<<" Forward layer "<<fl->position().z()<<" to Barrel layer "<<(**bli).specificSurface().radius()<<endl;
      reachableBL.push_back( *bli);
      return;
    }
  }
}


void HICSimpleNavigationSchool::linkNextLayerInGroup( FDLI fli,
						   const FDLC& group,
						   FDLC& reachableFL)
{
  // Always connect to next forward layer of "same" size, if exists
  if ( fli+1 != group.end()) {
    reachableFL.push_back( *(fli+1));
    // If that layer has an inner radius larger then the current one
    // also connect ALL next disks of same radius.
    float innerRThis = (**fli).specificSurface().innerRadius();
    float innerRNext =  (**(fli+1)).specificSurface().innerRadius();
    const float epsilon = 2.f;

    if (innerRNext > innerRThis + epsilon) {
      // next disk is smaller, so it doesn't cover fully subsequent ones
      // of same radius

      int i = 2;
      while ( (fli+i) != group.end()) {
	if ( (**(fli+i)).specificSurface().innerRadius() < 
	     innerRNext + epsilon) {
	  // following disk has not increased in ineer radius 
	  reachableFL.push_back( *(fli+i));
	  i++;
	} else {
	  break;
	}
      }
    }
  }
}


void HICSimpleNavigationSchool::linkOuterGroup( ForwardDetLayer* fl,
					     const FDLC& group,
					     FDLC& reachableFL)
{

  // insert N layers with Z grater than fl

  ConstFDLI first = find_if( group.begin(), group.end(), 
			     not1( DetBelowZ( fl->position().z())));
  if ( first != group.end()) {

    // Hard-wired constant!!!!!!
    ConstFDLI last = min( first + 7, group.end());

    reachableFL.insert( reachableFL.end(), first, last);
  }
}

void HICSimpleNavigationSchool::linkWithinGroup( FDLI fl,
					      const FDLC& group,
					      FDLC& reachableFL)
{
  ConstFDLI biggerLayer = outerRadiusIncrease( fl, group);
  if ( biggerLayer != group.end() && biggerLayer != fl+1) {
    reachableFL.push_back( *biggerLayer);
  }
}

HICSimpleNavigationSchool::ConstFDLI
HICSimpleNavigationSchool::outerRadiusIncrease( FDLI fl, const FDLC& group)
{
  const float epsilon = 5.f;
  float outerRadius = (**fl).specificSurface().outerRadius();
  while ( ++fl != group.end()) {
    if ( (**fl).specificSurface().outerRadius() > outerRadius + epsilon) {
      return fl;
    }
  }
  return fl;
}

vector<HICSimpleNavigationSchool::FDLC> 
HICSimpleNavigationSchool::splitForwardLayers() 
{
  // only work on positive Z side; negative by mirror symmetry later

  FDLC myRightLayers( theRightLayers);
  FDLI begin = myRightLayers.begin();
  FDLI end   = myRightLayers.end();

  // sort according to inner radius
  sort ( begin, end, DiskLessInnerRadius()); 

  // partition in cylinders
  vector<FDLC> result;
  FDLC current;
  current.push_back( *begin);
  for ( FDLI i = begin+1; i != end; i++) {

    LogDebug("TkNavigation") << "(**i).specificSurface().innerRadius()      = "
			     << (**i).specificSurface().innerRadius() << endl
			     << "(**(i-1)).specificSurface().outerRadius()) = "
			     << (**(i-1)).specificSurface().outerRadius() ;

    // if inner radius of i is larger than outer radius of i-1 then split!
    if ( (**i).specificSurface().innerRadius() > 
	 (**(i-1)).specificSurface().outerRadius()) {

      LogDebug("TkNavigation") << "found break between groups" ;

      // sort layers in group along Z
      sort ( current.begin(), current.end(), DetLessZ());

      result.push_back(current);
      current.clear();
    }
    current.push_back(*i);
  }
  result.push_back(current); // save last one too 

  // now sort subsets in Z
  for ( vector<FDLC>::iterator ivec = result.begin();
	ivec != result.end(); ivec++) {
    sort( ivec->begin(), ivec->end(), DetLessZ());
  }

  return result;
}

float HICSimpleNavigationSchool::barrelLength() 
{
  if ( theBarrelLength < 1.) {
    for (BDLI i=theBarrelLayers.begin(); i!=theBarrelLayers.end(); i++) {
      theBarrelLength = max( theBarrelLength,
			     (**i).surface().bounds().length() / 2.f);
    }

    LogDebug("TkNavigation") << "The barrel length is " << theBarrelLength ;
  }
  return theBarrelLength;
}

void HICSimpleNavigationSchool::establishInverseRelations() {

  NavigationSetter setter(*this);

    // find for each layer which are the barrel and forward
    // layers that point to it
    typedef map<const DetLayer*, vector<BarrelDetLayer*>, less<const DetLayer*> > BarrelMapType;
    typedef map<const DetLayer*, vector<ForwardDetLayer*>, less<const DetLayer*> > ForwardMapType;

   // cout<<"SimpleNavigationSchool::establishInverseRelations::point 0 "<<endl;
    BarrelMapType reachedBarrelLayersMap;
    ForwardMapType reachedForwardLayersMap;


    for ( BDLI bli = theBarrelLayers.begin();
        bli!=theBarrelLayers.end(); bli++) {
      DLC reachedLC = (**bli).nextLayers( insideOut);
      for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
        reachedBarrelLayersMap[*i].push_back( *bli);
      }
    }
  //  cout<<"SimpleNavigationSchool::establishInverseRelations::point 1 "<<endl;
    for ( FDLI fli = theForwardLayers.begin();
        fli!=theForwardLayers.end(); fli++) {
      DLC reachedLC = (**fli).nextLayers( insideOut);
      for ( DLI i = reachedLC.begin(); i != reachedLC.end(); i++) {
        reachedForwardLayersMap[*i].push_back( *fli);
      }
    }

  //  cout<<"SimpleNavigationSchool::establishInverseRelations::point 2:: size od detlayers "<<theDetLayers.size()<<endl;
//    vector<DetLayer*> lc = theTracker->allLayers();
    for ( vector<DetLayer*>::iterator i = theDetLayers.begin(); i != theDetLayers.end(); i++) {
      SimpleNavigableLayer* navigableLayer =
	dynamic_cast<SimpleNavigableLayer*>((**i).navigableLayer());
      navigableLayer->setInwardLinks( reachedBarrelLayersMap[*i],reachedForwardLayersMap[*i] );
    }
  // cout<<"SimpleNavigationSchool::establishInverseRelations::point 3 "<<endl; 
}

