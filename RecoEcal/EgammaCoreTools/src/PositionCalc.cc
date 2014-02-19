#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

PositionCalc::PositionCalc(const edm::ParameterSet& par) :
  param_LogWeighted_   ( par.getParameter<bool>("LogWeighted")) ,
  param_T0_barl_       ( par.getParameter<double>("T0_barl")) , 
  param_T0_endc_       ( par.getParameter<double>("T0_endc")) , 
  param_T0_endcPresh_  ( par.getParameter<double>("T0_endcPresh")) , 
  param_W0_            ( par.getParameter<double>("W0")) ,
  param_X0_            ( par.getParameter<double>("X0")) ,
  m_esGeom             ( 0 ) ,
  m_esPlus             ( false ) ,
  m_esMinus            ( false )
{
}

const PositionCalc& PositionCalc::operator=( const PositionCalc& rhs ) 
{
   param_LogWeighted_ = rhs.param_LogWeighted_;
   param_T0_barl_ = rhs.param_T0_barl_;
   param_T0_endc_ = rhs.param_T0_endc_;
   param_T0_endcPresh_ = rhs.param_T0_endcPresh_;
   param_W0_ = rhs.param_W0_;
   param_X0_ = rhs.param_X0_;

   m_esGeom = rhs.m_esGeom ;
   m_esPlus = rhs.m_esPlus ;
   m_esMinus = rhs.m_esMinus ;
   return *this;
}
