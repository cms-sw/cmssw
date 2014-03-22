/**
 *  Class: StateSegmentMatcher, Tsos4D, Tsos2DPhi, Tsos2DZed
 *
 *  Description:
 *  utility classes for the dynamical truncation algorithm
 *
 *  Authors :
 *  D. Pagano & G. Bruno - UCL Louvain
 *
 **/

#include "RecoMuon/GlobalTrackingTools/interface/StateSegmentMatcher.h"

using namespace std;



StateSegmentMatcher::StateSegmentMatcher(TrajectoryStateOnSurface const &tsos, DTRecSegment4D const &dtseg4d, LocalError const &apeLoc)
{
  if (dtseg4d.hasPhi() && dtseg4d.hasZed()) {
    setAPE4d(apeLoc);
    match2D = false;
    AlgebraicVector dtseg = dtseg4d.parameters();                                                         
    v1[0] = dtseg[0]; 
    v1[1] = dtseg[1]; 
    v1[2] = dtseg[2]; 
    v1[3] = dtseg[3]; 

    AlgebraicSymMatrix rhErr_vect       = dtseg4d.parametersError(); 
    m1(0,0) = rhErr_vect(1,1); m1(0,1) = rhErr_vect(1,2); m1(0,2) = rhErr_vect(1,3);m1(0,3) = rhErr_vect(1,4); 
    m1(1,0) = rhErr_vect(2,1); m1(1,1) = rhErr_vect(2,2); m1(1,2) = rhErr_vect(2,3);m1(1,3) = rhErr_vect(2,4); 
    m1(2,0) = rhErr_vect(3,1); m1(2,1) = rhErr_vect(3,2); m1(2,2) = rhErr_vect(3,3);m1(2,3) = rhErr_vect(3,4); 
    m1(3,0) = rhErr_vect(4,1); m1(3,1) = rhErr_vect(4,2); m1(3,2) = rhErr_vect(4,3);m1(3,3) = rhErr_vect(4,4);
    
    Tsos4D tsos4d = Tsos4D(tsos);                                                                                                                                             
    v2 = tsos4d.paramVector();                                                            
    m2 = tsos4d.errorMatrix();  
  } else {
    setAPE2d(apeLoc);
    match2D = true;
    AlgebraicVector dtseg = dtseg4d.parameters();
    v1_2d[0] = dtseg[0];
    v1_2d[1] = dtseg[1];

    AlgebraicSymMatrix rhErr_vect = dtseg4d.parametersError();
    m1_2d(0,0) = rhErr_vect(1,1); m1_2d(0,1) = rhErr_vect(1,2); 
    m1_2d(1,0) = rhErr_vect(2,1); m1_2d(1,1) = rhErr_vect(2,2); 
            
    if (dtseg4d.hasPhi()) {
      Tsos2DPhi  tsos2d = Tsos2DPhi(tsos);
      v2_2d = tsos2d.paramVector();
      m2_2d = tsos2d.errorMatrix();
    }

    if (dtseg4d.hasZed()) {
      Tsos2DZed  tsos2d = Tsos2DZed(tsos);
      v2_2d = tsos2d.paramVector();
      m2_2d = tsos2d.errorMatrix();
    }
  }
}



StateSegmentMatcher::StateSegmentMatcher(TrajectoryStateOnSurface const &tsos, CSCSegment const &cscseg4d, LocalError const &apeLoc)
{
  setAPE4d(apeLoc);
  match2D = false;
  AlgebraicVector cscseg = cscseg4d.parameters();
  v1[0] = cscseg[0];
  v1[1] = cscseg[1];
  v1[2] = cscseg[2];
  v1[3] = cscseg[3];

  AlgebraicSymMatrix rhErr_vect = cscseg4d.parametersError();
  m1(0,0) = rhErr_vect(1,1);m1(0,1) = rhErr_vect(1,2);m1(0,2) = rhErr_vect(1,3);m1(0,3) = rhErr_vect(1,4);
  m1(1,0) = rhErr_vect(2,1);m1(1,1) = rhErr_vect(2,2);m1(1,2) = rhErr_vect(2,3);m1(1,3) = rhErr_vect(2,4);
  m1(2,0) = rhErr_vect(3,1);m1(2,1) = rhErr_vect(3,2);m1(2,2) = rhErr_vect(3,3);m1(2,3) = rhErr_vect(3,4);
  m1(3,0) = rhErr_vect(4,1);m1(3,1) = rhErr_vect(4,2);m1(3,2) = rhErr_vect(4,3);m1(3,3) = rhErr_vect(4,4);

  Tsos4D tsos4d = Tsos4D(tsos);
  v2 = tsos4d.paramVector();
  m2 = tsos4d.errorMatrix();
}



double StateSegmentMatcher::value() 
{
  if (match2D) {
    AlgebraicVector2 v3(v1_2d - v2_2d);
    AlgebraicSymMatrix22 m3(m1_2d + m2_2d + ape_2d);
    bool m3i = !m3.Invert();
    if ( m3i ) {
      return 1e7;
    } else {
      estValue = ROOT::Math::Similarity(v3,m3);
    }
    
  } else {
    AlgebraicVector4 v3(v1 - v2);
    AlgebraicSymMatrix44 m3(m1 + m2 + ape);
    bool m3i = !m3.Invert();
    if ( m3i ) {
      return 1e7;
    } else {
      estValue = ROOT::Math::Similarity(v3,m3);
    }
  }
  return estValue;
}


Tsos4D::Tsos4D(TrajectoryStateOnSurface const &tsos)
{
  AlgebraicVector5 tsos_v = tsos.localParameters().vector();
  tsos_4d[0] = tsos_v[1];
  tsos_4d[1] = tsos_v[2];
  tsos_4d[2] = tsos_v[3];
  tsos_4d[3] = tsos_v[4];

  AlgebraicSymMatrix55 E = tsos.localError().matrix();
  tsosErr_44(0,0) = E(1,1);tsosErr_44(0,1) = E(1,2);tsosErr_44(0,2) = E(1,3);tsosErr_44(0,3) = E(1,4);
  tsosErr_44(1,0) = E(2,1);tsosErr_44(1,1) = E(2,2);tsosErr_44(1,2) = E(2,3);tsosErr_44(1,3) = E(2,4);
  tsosErr_44(2,0) = E(3,1);tsosErr_44(2,1) = E(3,2);tsosErr_44(2,2) = E(3,3);tsosErr_44(2,3) = E(3,4);
  tsosErr_44(3,0) = E(4,1);tsosErr_44(3,1) = E(4,2);tsosErr_44(3,2) = E(4,3);tsosErr_44(3,3) = E(4,4);
}


AlgebraicVector4 Tsos4D::paramVector() const {return tsos_4d;}


AlgebraicSymMatrix44 Tsos4D::errorMatrix() const {return tsosErr_44;}



Tsos2DPhi::Tsos2DPhi(TrajectoryStateOnSurface const &tsos)
{
  AlgebraicVector5 tsos_v = tsos.localParameters().vector();
  tsos_2d_phi[0] = tsos_v[1];
  tsos_2d_phi[1] = tsos_v[3];
  
  AlgebraicSymMatrix55 E = tsos.localError().matrix();
  tsosErr_22_phi(0,0) = E(1,1);tsosErr_22_phi(0,1) = E(1,3);
  tsosErr_22_phi(1,0) = E(3,1);tsosErr_22_phi(1,1) = E(3,3);
}


AlgebraicVector2 Tsos2DPhi::paramVector() const {return tsos_2d_phi;}


AlgebraicSymMatrix22 Tsos2DPhi::errorMatrix() const {return tsosErr_22_phi;}



Tsos2DZed::Tsos2DZed(TrajectoryStateOnSurface const &tsos)
{
  AlgebraicVector5 tsos_v = tsos.localParameters().vector();
  tsos_2d_zed[0] = tsos_v[2];
  tsos_2d_zed[1] = tsos_v[4];

  AlgebraicSymMatrix55 E = tsos.localError().matrix();
  tsosErr_22_zed(0,0) = E(2,2);tsosErr_22_zed(0,1) = E(2,4);
  tsosErr_22_zed(1,0) = E(4,2);tsosErr_22_zed(1,1) = E(4,4);
}


AlgebraicVector2 Tsos2DZed::paramVector() const {return tsos_2d_zed;}


AlgebraicSymMatrix22 Tsos2DZed::errorMatrix() const {return tsosErr_22_zed;}
