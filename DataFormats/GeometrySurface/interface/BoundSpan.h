#ifndef Geom_BoundSpan_H
#define Geom_BoundSpan_H
/**
 *  compute the span of a bounded surface in the global space
 *
 *
 */


namespace boundSpan {
  
  pair<float, float> computeDetPhiRange( const BoundSurface& plane) const 
{  
  const TrapezoidalPlaneBounds* trapezoidalBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  const RectangularPlaneBounds* rectangularBounds( dynamic_cast<const RectangularPlaneBounds*>(&(plane.bounds())));  

  std::vector<GlobalPoint> corners;
  if (trapezoidalBounds) {
    vector<float> parameters = (*trapezoidalBounds).parameters();
    if ( parameters[0] == 0 ) 
      edm::LogError("TkDetLayers") << "CompositeTkPetalWedge: something weird going on with trapezoidal Plane Bounds!" ;
    // edm::LogInfo(TkDetLayers) << " Parameters of DetUnit (L2/L1/T/H): " ;
    // for (int i = 0; i < 4; i++ ) { edm::LogInfo(TkDetLayers) << "  " << 2.*parameters[i]; }
    // edm::LogInfo(TkDetLayers) ;
    
    float hbotedge = parameters[0];
    float htopedge = parameters[1];
    float hapothem = parameters[3];   

    corners.push_back( plane.toGlobal( LocalPoint( -htopedge, hapothem, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint(  htopedge, hapothem, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint(  hbotedge, -hapothem, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint( -hbotedge, -hapothem, 0.)));

  }else if(rectangularBounds) {
    float length = rectangularBounds->length();
    float width  = rectangularBounds->width();   
  
    corners.push_back( plane.toGlobal( LocalPoint( -width/2, -length/2, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint( -width/2, +length/2, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint( +width/2, -length/2, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint( +width/2, +length/2, 0.)));
  }else{  
    string errmsg="TkForwardRing: problems with dynamic cast to rectangular or trapezoidal bounds for Det";
    throw DetLayerException(errmsg);
    edm::LogError("TkDetLayers") << errmsg ;
  }
 
  float phimin = corners[0].phi();
  float phimax = phimin;
  for ( int i = 1; i < 4; i++ ) {
    float cPhi = corners[i].phi();
    if ( PhiLess()( cPhi, phimin)) { phimin = cPhi; }
    if ( PhiLess()( phimax, cPhi)) { phimax = cPhi; }
  }
  return make_pair( phimin, phimax);
  
}


}

#endif
