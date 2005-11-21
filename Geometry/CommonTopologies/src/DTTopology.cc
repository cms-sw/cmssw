#include "Geometry/CommonTopologies/interface/DTTopology.h"
#include "Geometry/Vector/interface/LocalPoint.h"

#include <iostream>

const float DTTopology::theWidth=4.2;  // cm FIXME now put by hand, check the number!
const float DTTopology::theHeight=1.3; // cm ...
 

DTTopology::DTTopology(int firstWire, int nChannels,float lenght): theFirstChannel(firstWire),
								   theNChannels(nChannels),theLength(lenght){
  theOffSet = Local2DPoint(-theNChannels/2. * theWidth, -theLength/2.);
  
#ifdef VERBOSE
  cout <<"Constructing DTTopology with:"<<endl
       <<"number of wires = "<<theNChannels
       <<", first wire number = "<<theFirstChannel<<endl
       <<", width = "<<theWidth
       <<", height = "<<theHeight
       <<", length = "<<theLength
       <<endl;
#endif
}

float DTTopology::sensibleWidth(){
  const float IBeamThickness = 0.1;  // I-beam : 1 mm    
  return theWidth-IBeamThickness;
  }

float DTTopology::sensibleHeight(){
  const float plateThickness = 0.15; // aluminium plate: 1.5 mm   
  return theHeight-plateThickness;
}

//FIXME the x from tdrift is missing!!Now only wire position in the layer.
LocalPoint DTTopology::localPosition( const MeasurementPoint& mp) const{
    return LocalPoint( (mp.x() - 0.5)*theWidth + theOffSet.x() , 
		       (1-mp.y())*theLength + theOffSet.y());
}

LocalError DTTopology::localError( const MeasurementPoint& mp, const MeasurementError& me) const{
  return LocalError(me.uu()*(theWidth*theWidth), 0,
		    me.vv()*(theLength*theLength));
}

MeasurementPoint DTTopology::measurementPosition( const LocalPoint& lp) const{
  return MeasurementPoint( static_cast<int>( (lp.x()-theOffSet.x())/theWidth + 0.5),
			   1 - (lp.y()-theOffSet.y())/theLength);
}

MeasurementError DTTopology::measurementError( const LocalPoint& lp, const LocalError& le) const{
  return MeasurementError(le.xx()/(theWidth*theWidth),0,
			  le.yy()/(theLength*theLength));
}

int DTTopology::channel( const LocalPoint& lp) const{
  return static_cast<int>( (lp.x()-theOffSet.x())/theWidth + 0.5);
}

// return the x wire position in the layer, starting from its wire number.
float DTTopology::wirePosition(int wireNumber){
  return  (wireNumber - 0.5)*theWidth + theOffSet.x();
}
  
