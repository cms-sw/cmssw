#ifndef CondFormats_CTPPSReadoutObjects_CTPPSBeamParameters_h
#define CondFormats_CTPPSReadoutObjects_CTPPSBeamParameters_h
// -*- C++ -*-
//
// Package:    CTPPSReadoutObjects
// Class:      CTPPSBeamParameters
// 
/**\class CTPPSBeamParameters CTPPSBeamParameters.h CondFormats/CTPPSRedoutObjects/src/CTPPSBeamParameters.cc

 Description: Beam parameters for proton reconstruction

 Implementation:
     <Notes on implementation>
*/
// Original Author:  Wagner Carvalho
//         Created:  20 Nov 2018
//

#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class CTPPSBeamParameters {

 public:
  
  // Constructor
  CTPPSBeamParameters() ;
  // Destructor
  ~CTPPSBeamParameters() ;
  
  // Getters
  
  double getBeamMom45() const ;
  double getBeamMom56() const ;
  
  double getBetaStarX45() const ;
  double getBetaStarY45() const ;
  double getBetaStarX56() const ;
  double getBetaStarY56() const ;
  
  double getBeamDivergenceX45() const ;
  double getBeamDivergenceY45() const ;
  double getBeamDivergenceX56() const ;
  double getBeamDivergenceY56() const ;
  
  double getHalfXangleX45() const ;
  double getHalfXangleY45() const ;
  double getHalfXangleX56() const ;
  double getHalfXangleY56() const ;
  
  double getVtxOffsetX45() const ;
  double getVtxOffsetY45() const ;
  double getVtxOffsetZ45() const ;
  double getVtxOffsetX56() const ;
  double getVtxOffsetY56() const ;
  double getVtxOffsetZ56() const ;
  
  double getVtxStddevX() const ;
  double getVtxStddevY() const ;
  double getVtxStddevZ() const ;

  // Setters
  
  void setBeamMom45( double mom ) ;
  void setBeamMom56( double mom ) ;

  void setBetaStarX45( double beta ) ;
  void setBetaStarY45( double beta ) ;
  void setBetaStarX56( double beta ) ;
  void setBetaStarY56( double beta ) ;

  void setBeamDivergenceX45( double div ) ;
  void setBeamDivergenceY45( double div ) ;
  void setBeamDivergenceX56( double div ) ;
  void setBeamDivergenceY56( double div ) ;

  void setHalfXangleX45( double angle ) ;
  void setHalfXangleY45( double angle ) ;
  void setHalfXangleX56( double angle ) ;
  void setHalfXangleY56( double angle ) ;

  void setVtxOffsetX45( double offset ) ;
  void setVtxOffsetY45( double offset ) ;
  void setVtxOffsetZ45( double offset ) ;
  void setVtxOffsetX56( double offset ) ;
  void setVtxOffsetY56( double offset ) ;
  void setVtxOffsetZ56( double offset ) ;

  void setVtxStddevX( double stddev ) ;
  void setVtxStddevY( double stddev ) ;
  void setVtxStddevZ( double stddev ) ;

  void printInfo(std::stringstream & s) ;
  

 private:
  
    // LHC sector 45 corresponds to beam 2, sector 56 to beam 1
    double beam_momentum_45_ ;    // GeV
    double beam_momentum_56_ ;    // GeV

    double beta_star_x_45_ , beta_star_x_56_;    // cm
    double beta_star_y_45_ , beta_star_y_56_;

    double beam_divergence_x_45_ , beam_divergence_x_56_ ;    // rad
    double beam_divergence_y_45_ , beam_divergence_y_56_ ;

    double half_crossing_angle_x_45_ , half_crossing_angle_x_56_ ;    // rad
    double half_crossing_angle_y_45_ , half_crossing_angle_y_56_ ;

    // splitting between 45 and 56 may effectively account for magnet misalignment
    double vtx_offset_x_45_ , vtx_offset_x_56_ ; // cm
    double vtx_offset_y_45_ , vtx_offset_y_56_ ; // cm
    double vtx_offset_z_45_ , vtx_offset_z_56_ ; // cm

    // the following variables might possibly be in another CMS record already,
    // but we might want to keep them for completeness/independence
    double vtx_stddev_x_ ; // cm
    double vtx_stddev_y_ ; // cm
    double vtx_stddev_z_ ; // cm
  
    
  COND_SERIALIZABLE;
  
};

std::ostream & operator<<( std::ostream &, CTPPSBeamParameters );

#endif