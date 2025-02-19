#ifndef FTSFROMVERTEXTOPOINTFACTORY_H
#define FTSFROMVERTEXTOPOINTFACTORY_H
// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      FTSFromVertexToPointFactory
// 
/**
 *  Generates a FreeTrajectoryState from a given measured point, vertex
 *  momentum and charge.
 *  FTSFromVertexToPointFactory myFTS; myFTS(xmeas, xvert, momentum, charge);
 *  gives a FreeTrajectoryState of a track which comes from xvert to xmeas. 
 *  The curvature of myFTS is computed taken into account the bend in 
 *  the magnetic field.       

 Implementation:
     should go somewhere else in the future
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: FTSFromVertexToPointFactory.h,v 1.1 2006/06/02 16:21:02 uberthon Exp $
//
//
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
class MagneticField;

class FTSFromVertexToPointFactory{
public:
  FTSFromVertexToPointFactory() { };
  FreeTrajectoryState operator()(const MagneticField *magField, const GlobalPoint& xmeas,  
                                 const GlobalPoint& xvert, 
                                 float momentum, TrackCharge charge);

};

#endif
