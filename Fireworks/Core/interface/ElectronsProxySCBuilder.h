#ifndef Fireworks_Calo_ElectronsProxySCBuilder_h
#define Fireworks_Calo_ElectronsProxySCBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronsProxySCBuilder
//
/**\class ElectronsProxySCBuilder ElectronsProxySCBuilder.h Fireworks/Calo/interface/ElectronsProxySCBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: ElectronsProxySCBuilder.h,v 1.3 2008/03/06 22:48:31 jmuelmen Exp $
//

// system include files

#include "Rtypes.h"

// user include files

// forward declarations

class FWEventItem;
class TEveElementList;

class ElectronsProxySCBuilder {

public:
     ElectronsProxySCBuilder();
     virtual ~ElectronsProxySCBuilder();

     // ---------- const member functions ---------------------

     // ---------- static member functions --------------------

     // ---------- member functions ---------------------------

  //    virtual void buildRhoPhi(const FWEventItem* iItem,
// 			      TEveElementList** product);

//      virtual void buildRhoZ(const FWEventItem* iItem,
// 			    TEveElementList** product);
     void setItem (const FWEventItem *iItem) { m_item = iItem; }
     void build (TEveElementList **product);
     void getCenter( Double_t* vars )
     {
	vars[0] = rotation_center[0];
	vars[1] = rotation_center[1];
	vars[2] = rotation_center[2];
     }
     static ElectronsProxySCBuilder *the_electron_sc_proxy;

private:
     ElectronsProxySCBuilder(const ElectronsProxySCBuilder&); // stop default

     const ElectronsProxySCBuilder& operator=(const ElectronsProxySCBuilder&); // stop default

     // ---------- member data --------------------------------
     const FWEventItem* m_item;
     void resetCenter() {
	rotation_center[0] = 0;
	rotation_center[1] = 0;
	rotation_center[2] = 0;
     }

     Double_t rotation_center[3];
};


#endif
