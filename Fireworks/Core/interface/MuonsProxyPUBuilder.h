#ifndef Fireworks_Calo_MuonsProxyPUBuilder_h
#define Fireworks_Calo_MuonsProxyPUBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     MuonsProxyPUBuilder
//
/**\class MuonsProxyPUBuilder MuonsProxyPUBuilder.h Fireworks/Calo/interface/MuonsProxyPUBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: MuonsProxyPUBuilder.h,v 1.1 2008/03/07 04:01:58 tdaniels Exp $
//

// system include files

#include "Rtypes.h"

// user include files
// forward declarations

class FWEventItem;
class TEveElementList;

class MuonsProxyPUBuilder {

public:
     MuonsProxyPUBuilder();
     virtual ~MuonsProxyPUBuilder();

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

private:
     MuonsProxyPUBuilder(const MuonsProxyPUBuilder&); // stop default

     const MuonsProxyPUBuilder& operator=(const MuonsProxyPUBuilder&); // stop default

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
