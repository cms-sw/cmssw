// -*- C++ -*-
#ifndef Fireworks_Core_PhotonDetailView_h
#define Fireworks_Core_PhotonDetailView_h
//
// Package:     Calo
// Class  :     PhotonDetailView
//
/**\class PhotonDetailView PhotonDetailView.h Fireworks/Calo/interface/PhotonDetailView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PhotonDetailView.h,v 1.5 2008/11/06 22:05:22 amraktad Exp $
//

// system include files

#include "Rtypes.h"

// user include files
#include "TEveBoxSet.h"
#include "Fireworks/Core/interface/FWDetailView.h"

// forward declarations

class FWEventItem;
class TEveElementList;
namespace reco {
     class Photon;
}

class PhotonDetailView : public FWDetailView {

public:
     PhotonDetailView();
     virtual ~PhotonDetailView();

     virtual void build (TEveElementList **product, const FWModelId &id);

protected:
     void setItem (const FWEventItem *iItem) { m_item = iItem; }
     void build_3d (TEveElementList **product, const FWModelId &id);
     void build_projected (TEveElementList **product, const FWModelId &id);
     void getCenter( Double_t* vars )
     {
	vars[0] = rotation_center[0];
	vars[1] = rotation_center[1];
	vars[2] = rotation_center[2];
     }
     TEveElementList *makeLabels (const reco::Photon &);
     TEveElementList *getEcalCrystalsBarrel (const class DetIdToMatrix &,
					     const std::vector<class DetId> &);
     TEveElementList *getEcalCrystalsBarrel (const class DetIdToMatrix &,
					     double eta, double phi,
					     int n_eta = 5, int n_phi = 10);
     TEveElementList *getEcalCrystalsEndcap (const class DetIdToMatrix &,
					     const std::vector<class DetId> &);
     TEveElementList *getEcalCrystalsEndcap (const class DetIdToMatrix &,
					     double x, double y, int iz,
					     int n_x = 5, int n_y = 5);

private:
     PhotonDetailView(const PhotonDetailView&); // stop default
     const PhotonDetailView& operator=(const PhotonDetailView&); // stop default

     // ---------- member data --------------------------------
     const FWEventItem* m_item;
     void resetCenter() {
	rotation_center[0] = 0;
	rotation_center[1] = 0;
	rotation_center[2] = 0;
     }

};

#endif
