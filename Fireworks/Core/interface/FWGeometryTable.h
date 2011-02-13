#ifndef Fireworks_Core_FWGeometryTable_h
#define Fireworks_Core_FWGeometryTable_h

#ifndef __CINT__
#include <boost/shared_ptr.hpp>
#endif
#include "TGFrame.h"

#ifndef __CINT__
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWStringParameter.h"
#include "Fireworks/Core/interface/FWEnumParameter.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterSetterEditorBase.h"
#endif

class FWGUIManager;
class FWTableWidget;
class FWGeometryTableManager;
class TFile;
class TGTextButton;
class TGeoNode;
class TGeoVolume;
class TGTextEntry;
class TGComboBox;

class FWConfiguration;
class FWParameterBase;

class FWGeometryTable : public TGMainFrame
#ifndef __CINT__
                      , public FWConfigurableParameterizable , public FWParameterSetterEditorBase
#endif
{
   friend class FWGeometryTableManager;

public:
   enum EMode { kNode, kVolume };

   FWGeometryTable(FWGUIManager*);
   virtual ~FWGeometryTable();
  
   void cellClicked(Int_t iRow, Int_t iColumn, 
                    Int_t iButton, Int_t iKeyMod, 
                    Int_t iGlobalX, Int_t iGlobalY);
  
   void newIndexSelected(int,int);
   void windowIsClosing();

   void browse();
   void readFile();

   virtual void setFrom(const FWConfiguration&);
   Bool_t HandleKey(Event_t *event);
   // ---------- const member functions --------------------- 

   virtual void addTo(FWConfiguration&) const;
   
protected:
#ifndef __CINT__
   FWEnumParameter         m_mode;
   FWStringParameter       m_filter; 
   FWLongParameter         m_autoExpand; 
   FWLongParameter         m_maxDaughters; 
#endif

private:
   FWGeometryTable(const FWGeometryTable&);
   const FWGeometryTable& operator=(const FWGeometryTable&);

   FWGUIManager           *m_guiManager;

   FWTableWidget          *m_tableWidget;
   FWGeometryTableManager *m_tableManager;

   TFile                  *m_geometryFile;
   TGTextButton           *m_fileOpen;

   TGCompositeFrame       *m_settersFrame;

#ifndef __CINT__
   std::vector<boost::shared_ptr<FWParameterSetterBase> > m_setters;
#endif
   void resetSetters();
   void makeSetter(TGCompositeFrame* frame, FWParameterBase* param);


   ClassDef(FWGeometryTable, 0);
};

#endif

