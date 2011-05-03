#ifndef Fireworks_FWInterface_FWPSetCellEditor_h
#define Fireworks_FWInterface_FWPSetCellEditor_h
// -*- C++ -*-
//
// Package:     FWInterface
// Class  :     FWPSetCellEditor
// 
/**\class FWPSetCellEditor FWPSetCellEditor.h Fireworks/FWInterface/interface/FWPSetCellEditor.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Mon Feb 28 20:45:02 CET 2011
// $Id: FWPSetCellEditor.h,v 1.1 2011/02/28 20:37:39 amraktad Exp $
//


#include "TGTextEntry.h"
#include "Fireworks/FWInterface/src/FWPSetTableManager.h"

class FWPSetCellEditor : public TGTextEntry
{
public:
   FWPSetCellEditor(const TGWindow* w, const  char* txt) : TGTextEntry(w, txt) {};
   virtual ~FWPSetCellEditor() {};
   virtual bool HandleKey(Event_t*event);
   bool apply(FWPSetTableManager::PSetData &data, FWPSetTableManager::PSetData &parent );
};

#endif
