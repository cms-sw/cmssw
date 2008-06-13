// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUIEventDataAdder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 13 09:58:53 EDT 2008
// $Id$
//

// system include files
#include "TGFrame.h"
#include "TGTextEntry.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TClass.h"

// user include files
#include "Fireworks/Core/src/FWGUIEventDataAdder.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"

//
// constants, enums and typedefs
//
/*
namespace {
   class HideableWindow : public TGTransientFrame {
   public:
      HideableWindow(const TGWindow *p = 0, const TGWindow *main = 0, UInt_t w = 1, UInt_t h = 1,
                     UInt_t options = kVerticalFrame) :
      TGTransientFrame(p,main,w,h,options) {}
      
      virtual void DestroyWindow() {
         UnmapWindow();
      }
   };
}
*/
//
// static data member definitions
//

//
// constructors and destructor
//
static void addToFrame(TGVerticalFrame* iParent, const char* iName, TGTextEntry*& oSet)
{
   TGCompositeFrame* hf = new TGHorizontalFrame(iParent);
   hf->AddFrame(new TGLabel(hf,iName),new TGLayoutHints(kLHintsLeft|kLHintsCenterY));
   oSet = new TGTextEntry(hf,"");
   hf->AddFrame(oSet,new TGLayoutHints(kLHintsExpandX));
   iParent->AddFrame(hf);
}

FWGUIEventDataAdder::FWGUIEventDataAdder(
                                         UInt_t iWidth,UInt_t iHeight, 
                                         FWEventItemsManager* iManager):
m_manager(iManager)
{
   createWindow();
}

// FWGUIEventDataAdder::FWGUIEventDataAdder(const FWGUIEventDataAdder& rhs)
// {
//    // do actual copying here;
// }

FWGUIEventDataAdder::~FWGUIEventDataAdder()
{
}

//
// assignment operators
//
// const FWGUIEventDataAdder& FWGUIEventDataAdder::operator=(const FWGUIEventDataAdder& rhs)
// {
//   //An exception safe implementation is
//   FWGUIEventDataAdder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWGUIEventDataAdder::addNewItem()
{
   TClass* theClass = TClass::GetClass(m_type->GetText());
   if(0==theClass) {
      return;
   }
   const std::string moduleLabel = m_moduleLabel->GetText();
   if(moduleLabel.empty()) {
      return;
   }
   
   const std::string name = m_name->GetText();
   if(name.empty()) {
      return;
   }
   
   FWPhysicsObjectDesc desc(name, theClass, m_purpose->GetText(),
                            FWDisplayProperties(),
                            moduleLabel,
                            m_productInstanceLabel->GetText(),
                            m_processName->GetText());
   m_manager->add( desc);
}

void 
FWGUIEventDataAdder::show()
{
   // Map main frame 
   if(0==m_frame) {
      createWindow();
   }
   m_frame->MapWindow(); 
}

void
FWGUIEventDataAdder::windowIsClosing()
{
   m_frame->Cleanup();
   delete m_frame;
   m_frame=0;
}


void
FWGUIEventDataAdder::createWindow()
{
   m_frame = new TGMainFrame(0,10,10);
   m_frame->Connect("CloseWindow()","FWGUIEventDataAdder",this,"windowIsClosing()");
   TGVerticalFrame* vf = new TGVerticalFrame(m_frame);
   m_frame->AddFrame(vf, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY,10,10,10,1));
   
   addToFrame(vf, "Name:", m_name);
   addToFrame(vf, "Purpose:", m_purpose);
   addToFrame(vf,"C++ type:",m_type);
   addToFrame(vf,"Module label:",m_moduleLabel);
   addToFrame(vf,"Product instance label:",m_productInstanceLabel);
   addToFrame(vf,"Process name:",m_processName);
   
   m_apply = new TGTextButton(vf,"Add Data");
   vf->AddFrame(m_apply);
   m_apply->Connect("Clicked()","FWGUIEventDataAdder",this,"addNewItem()");
   
   // Set a name to the main frame 
   m_frame->SetWindowName("Show Additional Event Data"); 
   
   // Map all subwindows of main frame 
   m_frame->MapSubwindows(); 
   
   // Initialize the layout algorithm 
   m_frame->Layout();    

   // Initialize the layout algorithm 
   m_frame->Resize(m_frame->GetDefaultSize()); 
   
}


//
// const member functions
//

//
// static member functions
//
