#ifndef Fireworks_Core_FWViewContextMenuHandlerBase_h
#define Fireworks_Core_FWViewContextMenuHandlerBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewContextMenuHandlerBase
// 
/**\class FWViewContextMenuHandlerBase FWViewContextMenuHandlerBase.h Fireworks/Core/interface/FWViewContextMenuHandlerBase.h

 Description: Base class for handling context menu's from views

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Nov  2 13:46:26 CST 2009
// $Id: FWViewContextMenuHandlerBase.h,v 1.1 2009/11/02 23:59:52 chrjones Exp $
//

// system include files

// user include files

// forward declarations
class TEveViewer;
class FWModelContextMenuHandler;


class FWViewContextMenuHandlerBase
{
public:
   FWViewContextMenuHandlerBase();
   virtual ~FWViewContextMenuHandlerBase();
   
   class MenuEntryAdder {
      friend class FWViewContextMenuHandlerBase;
      MenuEntryAdder(FWModelContextMenuHandler&);
      FWModelContextMenuHandler* m_handler;
      int m_lastIndex;
   public:
      /**Add an entry by name of iEntryName to the context menu.  Returns the entry index
       which simply increments after each 'addEntry' call and starts at 0 */
      int addEntry(const char* iEntryName);
   };

   void addTo(FWModelContextMenuHandler&);
   
   /**Called when a menu item was selected
    iEntryIndex: the index of the selected menu item.  Same as returned from 'addEntry'
    iX, iY: Screen coordinates of where mouse was clicked
    */
   virtual void select(int iEntryIndex, int iX, int iY) = 0;
   
private:
   FWViewContextMenuHandlerBase(const FWViewContextMenuHandlerBase&); // stop default   
   const FWViewContextMenuHandlerBase& operator=(const FWViewContextMenuHandlerBase&); // stop default
   
   virtual void init(MenuEntryAdder&) = 0;
};

//==============================================================================
//==============================================================================

class FWViewContextMenuHandlerGL
{
public:
   enum GLViewerAction { kAnnotate, kPickCenter, kNone };

   FWViewContextMenuHandlerGL(TEveViewer* v): m_viewer(v) {}
   virtual ~FWViewContextMenuHandlerGL() {}
   virtual void select(int iEntryIndex, int iX, int iY);

private:
   FWViewContextMenuHandlerGL(const FWViewContextMenuHandlerGL&); // stop default   
   const FWViewContextMenuHandlerGL& operator=(const FWViewContextMenuHandlerGL&); // stop default

   virtual void init(FWViewContextMenuHandlerBase::MenuEntryAdder&);
 
   TEveViewer* m_viewer;
};

#endif
