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

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSBeamParameters.h"
#include <iostream>

  // Constructors
  
  CTPPSBeamParameters::CTPPSBeamParameters() : 
     beam_momentum_45_ (0.) , beam_momentum_56_ (0.) ,
     beta_star_x_45_ (0.) , beta_star_x_56_ (0.) , 
     beta_star_y_45_ (0.) , beta_star_y_56_ (0.) ,
     beam_divergence_x_45_ (0.) , beam_divergence_x_56_ (0.) , 
     beam_divergence_y_45_ (0.) , beam_divergence_y_56_ (0.) ,
     half_crossing_angle_x_45_ (0.) , half_crossing_angle_x_56_ (0.) , 
     half_crossing_angle_y_45_ (0.) , half_crossing_angle_y_56_ (0.) ,
     vtx_offset_x_45_ (0.) , vtx_offset_x_56_ (0.) , 
     vtx_offset_y_45_ (0.) , vtx_offset_y_56_ (0.) , 
     vtx_offset_z_45_ (0.) , vtx_offset_z_56_ (0.) , 
     vtx_stddev_x_ (0.) , vtx_stddev_y_ (0.) , vtx_stddev_z_ (0.)  
  {}
  
  // Destructor
  CTPPSBeamParameters::~CTPPSBeamParameters() {}
  
  // Getters
  
  double CTPPSBeamParameters::getBeamMom45() const {return beam_momentum_45_;}
  double CTPPSBeamParameters::getBeamMom56() const {return beam_momentum_56_;}
  
  double CTPPSBeamParameters::getBetaStarX45() const {return beta_star_x_45_;}
  double CTPPSBeamParameters::getBetaStarY45() const {return beta_star_y_45_;}
  double CTPPSBeamParameters::getBetaStarX56() const {return beta_star_x_56_;}
  double CTPPSBeamParameters::getBetaStarY56() const {return beta_star_y_56_;}
  
  double CTPPSBeamParameters::getBeamDivergenceX45() const {return beam_divergence_x_45_;}
  double CTPPSBeamParameters::getBeamDivergenceY45() const {return beam_divergence_y_45_;}
  double CTPPSBeamParameters::getBeamDivergenceX56() const {return beam_divergence_x_56_;}
  double CTPPSBeamParameters::getBeamDivergenceY56() const {return beam_divergence_y_56_;}
  
  double CTPPSBeamParameters::getHalfXangleX45() const {return half_crossing_angle_x_45_;}
  double CTPPSBeamParameters::getHalfXangleY45() const {return half_crossing_angle_y_45_;}
  double CTPPSBeamParameters::getHalfXangleX56() const {return half_crossing_angle_x_56_;}
  double CTPPSBeamParameters::getHalfXangleY56() const {return half_crossing_angle_y_56_;}
  
  double CTPPSBeamParameters::getVtxOffsetX45() const {return vtx_offset_x_45_;}
  double CTPPSBeamParameters::getVtxOffsetY45() const {return vtx_offset_y_45_;}
  double CTPPSBeamParameters::getVtxOffsetZ45() const {return vtx_offset_z_45_;}
  double CTPPSBeamParameters::getVtxOffsetX56() const {return vtx_offset_x_56_;}
  double CTPPSBeamParameters::getVtxOffsetY56() const {return vtx_offset_y_56_;}
  double CTPPSBeamParameters::getVtxOffsetZ56() const {return vtx_offset_z_56_;}
  
  double CTPPSBeamParameters::getVtxStddevX() const {return vtx_stddev_x_;}
  double CTPPSBeamParameters::getVtxStddevY() const {return vtx_stddev_y_;}
  double CTPPSBeamParameters::getVtxStddevZ() const {return vtx_stddev_z_;}

  // Setters
  
  void CTPPSBeamParameters::setBeamMom45( double mom ) {beam_momentum_45_ = mom;}
  void CTPPSBeamParameters::setBeamMom56( double mom ) {beam_momentum_56_ = mom;}

  void CTPPSBeamParameters::setBetaStarX45( double beta ) {beta_star_x_45_ = beta;}
  void CTPPSBeamParameters::setBetaStarY45( double beta ) {beta_star_y_45_ = beta;}
  void CTPPSBeamParameters::setBetaStarX56( double beta ) {beta_star_x_56_ = beta;}
  void CTPPSBeamParameters::setBetaStarY56( double beta ) {beta_star_y_56_ = beta;}

  void CTPPSBeamParameters::setBeamDivergenceX45( double div ) {beam_divergence_x_45_ = div;}
  void CTPPSBeamParameters::setBeamDivergenceY45( double div ) {beam_divergence_y_45_ = div;}
  void CTPPSBeamParameters::setBeamDivergenceX56( double div ) {beam_divergence_x_56_ = div;}
  void CTPPSBeamParameters::setBeamDivergenceY56( double div ) {beam_divergence_y_56_ = div;}

  void CTPPSBeamParameters::setHalfXangleX45( double angle ) {half_crossing_angle_x_45_ = angle;}
  void CTPPSBeamParameters::setHalfXangleY45( double angle ) {half_crossing_angle_y_45_ = angle;}
  void CTPPSBeamParameters::setHalfXangleX56( double angle ) {half_crossing_angle_x_56_ = angle;}
  void CTPPSBeamParameters::setHalfXangleY56( double angle ) {half_crossing_angle_y_56_ = angle;}

  void CTPPSBeamParameters::setVtxOffsetX45( double offset ) {vtx_offset_x_45_ = offset;}
  void CTPPSBeamParameters::setVtxOffsetY45( double offset ) {vtx_offset_y_45_ = offset;}
  void CTPPSBeamParameters::setVtxOffsetZ45( double offset ) {vtx_offset_z_45_ = offset;}
  void CTPPSBeamParameters::setVtxOffsetX56( double offset ) {vtx_offset_x_56_ = offset;}
  void CTPPSBeamParameters::setVtxOffsetY56( double offset ) {vtx_offset_y_56_ = offset;}
  void CTPPSBeamParameters::setVtxOffsetZ56( double offset ) {vtx_offset_z_56_ = offset;}

  void CTPPSBeamParameters::setVtxStddevX( double stddev ) {vtx_stddev_x_ = stddev;}
  void CTPPSBeamParameters::setVtxStddevY( double stddev ) {vtx_stddev_y_ = stddev;}
  void CTPPSBeamParameters::setVtxStddevZ( double stddev ) {vtx_stddev_z_ = stddev;}


  void CTPPSBeamParameters::printInfo(std::stringstream & s) 
  {
     s << "\n Beam parameters : \n" 
       << "\n   beam_momentum_45 = " << beam_momentum_45_ << " GeV" 
       << "\n   beam_momentum_56 = " << beam_momentum_56_ << " GeV" 
       << "\n   beta_star_x_45 = " << beta_star_x_45_ << " cm" 
       << "\n   beta_star_y_45 = " << beta_star_y_45_ << " cm" 
       << "\n   beta_star_x_56 = " << beta_star_x_56_ << " cm" 
       << "\n   beta_star_y_56 = " << beta_star_y_56_ << " cm" 
       << "\n   beam_divergence_x_45 = " << beam_divergence_x_45_ << " rad" 
       << "\n   beam_divergence_y_45 = " << beam_divergence_y_45_ << " rad"
       << "\n   beam_divergence_x_56 = " << beam_divergence_x_56_ << " rad"
       << "\n   beam_divergence_y_56 = " << beam_divergence_y_56_ << " rad"
       << "\n   half_crossing_angle_x_45 = " << half_crossing_angle_x_45_ << " rad"
       << "\n   half_crossing_angle_y_45 = " << half_crossing_angle_y_45_ << " rad"
       << "\n   half_crossing_angle_x_56 = " << half_crossing_angle_x_56_ << " rad"
       << "\n   half_crossing_angle_y_56 = " << half_crossing_angle_y_56_ << " rad"
       << "\n   vtx_offset_x_45 = " << vtx_offset_x_45_ << " cm"
       << "\n   vtx_offset_y_45 = " << vtx_offset_y_45_ << " cm"
       << "\n   vtx_offset_z_45 = " << vtx_offset_z_45_ << " cm"
       << "\n   vtx_offset_x_56 = " << vtx_offset_x_56_ << " cm"
       << "\n   vtx_offset_y_56 = " << vtx_offset_y_56_ << " cm"
       << "\n   vtx_offset_z_56 = " << vtx_offset_z_56_ << " cm"
       << "\n   vtx_stddev_x = " << vtx_stddev_x_ << " cm"
       << "\n   vtx_stddev_y = " << vtx_stddev_y_ << " cm"
       << "\n   vtx_stddev_z = " << vtx_stddev_z_ << " cm"
       << std::endl ;

  }

std::ostream & operator<<( std::ostream & os, CTPPSBeamParameters info ) {
  std::stringstream ss;
  info.printInfo( ss );
  os << ss.str();
  return os;
}  
