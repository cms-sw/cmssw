#ifndef Fireworks_Core_FW3DViewDistanceMeasureTool_h
#define Fireworks_Core_FW3DViewDistanceMeasureTool_h

#include "TGLUtil.h"

class TGCompositeFrame;
class FW3DViewBase;
class TGTextButton;
class TGLabel;
class FW3DViewDistanceMeasureTool
{
   friend class FW3DViewBase;

public:

   enum EPickAction { kPnt1, kPnt2, kNone};

   FW3DViewDistanceMeasureTool();
   virtual ~FW3DViewDistanceMeasureTool(){};

   void resetAction ();

   void Print() const;
   TGLVector3&   refCurrentVertex();

   void setActionPnt1();
   void setActionPnt2();

   TGCompositeFrame* buildGUI(TGCompositeFrame* p);

 protected:
   TGLVector3 m_pnt1;
   TGLVector3 m_pnt2;
   EPickAction m_action;

 private:
   TGTextButton* m_bp1;   
   TGTextButton* m_bp2;   
   TGLabel* m_lp1;   
   TGLabel* m_lp2;
   TGLabel* m_ldist;



   ClassDef(FW3DViewDistanceMeasureTool, 0);
};
#endif
