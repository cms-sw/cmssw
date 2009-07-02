#ifndef RecoHIMuon_HICSimpleNavigationSchool_H
#define RecoHIMuon_HICSimpleNavigationSchool_H

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include <vector>

class DetLayer;
class BarrelDetLayer;
class ForwardDetLayer;
class SymmetricLayerFinder;
class SimpleBarrelNavigableLayer;
class SimpleForwardNavigableLayer;
class MagneticField;

/** Concrete navigation school for the Tracker
 */

class HICSimpleNavigationSchool : public NavigationSchool {
public:
  
  HICSimpleNavigationSchool() : theField(0),theTracker(0){};
  HICSimpleNavigationSchool(const GeometricSearchTracker* theTracker,
			    const MagneticField* field);
  HICSimpleNavigationSchool(const GeometricSearchTracker* theTracker,
			    const MagneticField* field,int j, int l);

  void setExcludedBarrelLayer(int& j){excludedBarrelLayer=j;}; 

  int getExcludedBarrelLayer() {return excludedBarrelLayer;};

  // from base class
  virtual StateType navigableLayers() const;

//private:
protected:

  int                                             excludedBarrelLayer;
  int                                             excludedForwardLayer;
  typedef std::vector<const DetLayer*>            DLC;
  typedef std::vector<BarrelDetLayer*>            BDLC;
  typedef std::vector<ForwardDetLayer*>           FDLC;
  typedef DLC::iterator                           DLI;
  typedef BDLC::iterator                          BDLI;
  typedef FDLC::iterator                          FDLI;
  typedef BDLC::const_iterator                    ConstBDLI;
  typedef FDLC::const_iterator                    ConstFDLI;
 
  BDLC theBarrelLayers;
  FDLC theForwardLayers;  
  FDLC theRightLayers;
  FDLC theLeftLayers;
  float theBarrelLength;

  typedef std::vector< SimpleBarrelNavigableLayer*>   BNLCType;
  typedef std::vector< SimpleForwardNavigableLayer*>  FNLCType;
  BNLCType  theBarrelNLC;
  FNLCType  theForwardNLC;

  virtual void linkBarrelLayers( SymmetricLayerFinder& symFinder);
  virtual void linkForwardLayers( SymmetricLayerFinder& symFinder);

  virtual void linkNextForwardLayer( BarrelDetLayer*, FDLC&);

  virtual void linkNextLargerLayer( BDLI, BDLI, BDLC&);

  virtual void linkNextBarrelLayer( ForwardDetLayer* fl, BDLC&);

  virtual void linkOuterGroup( ForwardDetLayer* fl,
		       const FDLC& group,
		       FDLC& reachableFL);

  virtual void linkWithinGroup( FDLI fl, const FDLC& group, FDLC& reachableFL);
  
  virtual ConstFDLI outerRadiusIncrease( FDLI fl, const FDLC& group);

  virtual std::vector<FDLC> splitForwardLayers();

  virtual float barrelLength();

  virtual void establishInverseRelations();

  virtual void linkNextLayerInGroup( FDLI fli, const FDLC& group, FDLC& reachableFL);

  std::vector<DetLayer*>                    theDetLayers;
  const MagneticField* theField;
  const GeometricSearchTracker* theTracker;
};

#endif // SimpleNavigationSchool_H
