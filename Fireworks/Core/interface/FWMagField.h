#ifndef Fireworks_Tracks_FWMagField_h
#define Fireworks_Tracks_FWMagField_h
// -*- C++ -*-
// 
// Simplified model of the CMS detector magnetic field
// 
#include "TEveTrackPropagator.h"
class TH1F;

namespace edm
{
   class EventBase;
}

class FWMagField: public TEveMagField
{
   Float_t GetFieldMag() const;

public:
   enum ESource { kNone, kEvent, kUser };

   FWMagField();
   virtual ~FWMagField();

   // get field values
   virtual TEveVector GetField(Float_t x, Float_t y, Float_t z) const;
   virtual Float_t    GetMaxFieldMag() const;

   // auto/user behaviour
   void   setUserField(float b) { m_userField = b; }
   float  getUserField() const { return  m_userField; }
   void   setSource(ESource x) { m_source = x; }
   ESource  getSource() const { return m_source; }

   // field model
   void   setReverseState( bool state ){ m_reverse = state; }
   bool   isReverse() const { return m_reverse;}
   void   setSimpleModel( bool simpleModel ){ m_simpleModel = simpleModel; }
   bool   isSimpleModel() const { return m_simpleModel;}

   // field estimate
   void   guessFieldIsOn( bool guess ) const;
   void   guessField( float estimate ) const;
   void   resetFieldEstimate() const;

   void   checkFieldInfo(const edm::EventBase*);
   void   setFFFieldMag(float);

private:
   FWMagField(const FWMagField&); // stop default
   const FWMagField& operator=(const FWMagField&); // stop default

   ESource   m_source;
   float     m_userField;
   float     m_eventField;

   bool   m_reverse;
   bool   m_simpleModel;

   // runtime estimate , have to be mutable becuse of GetField() is const
   mutable TH1F  *m_guessValHist;
   mutable int    m_numberOfFieldIsOnEstimates;
   mutable int    m_numberOfFieldEstimates;
   mutable bool   m_updateFieldEstimate;
   mutable float  m_guessedField;
};

#endif
