#ifndef Fireworks_Tracks_CmsMagField_h
#define Fireworks_Tracks_CmsMagField_h
// -*- C++ -*-
// 
// Simplified model of the CMS detector magnetic field
// 
// $id$
#include "TEveTrackPropagator.h"
#include <iostream>

class CmsMagField: public TEveMagField
{
  bool m_magnetIsOn;
  bool m_reverse;
  bool m_simpleModel;
  double m_nominalField;
 public:
 CmsMagField():
    m_magnetIsOn(true),
    m_reverse(false),
    m_simpleModel(false),
    m_nominalField(3.8)
 {}

  virtual ~CmsMagField(){}
  virtual TEveVector GetField(Float_t x, Float_t y, Float_t z) const;
  virtual Float_t    GetMaxFieldMag() const { return m_magnetIsOn ? m_nominalField : 0.0; }
  void               setMagnetState( bool state )
  { 
    if ( state != m_magnetIsOn){
      if ( state )
	std::cout << "Magnet state is changed to ON" << std::endl;
      else
	std::cout << "Magnet state is changed to OFF" << std::endl;
    }
    m_magnetIsOn = state; 
  }
  void               setNominalFieldValue( double value ) { m_nominalField = value; }
  bool               isMagnetOn() const { return m_magnetIsOn;}
  void               setReverseState( bool state ){ m_reverse = state; }
  bool               isReverse() const { return m_reverse;}
  void               setSimpleModel( bool simpleModel ){ m_simpleModel = simpleModel; }
  bool               isSimpleModel() const { return m_simpleModel;}
};

#endif
