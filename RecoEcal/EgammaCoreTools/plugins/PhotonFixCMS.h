#ifndef PhotonFixCMS_Defined_hh
#define PhotonFixCMS_Defined_hh

//-------------------------------------------------------//
// Project: PhotonFix
// Author: Paul Dauncey (p.dauncey@imperial.ac.uk)
// Modified: 11/07/2011
// Admins: Paul Dauncey (p.dauncey@imperial.ac.uk)
//         Matt Kenzie (matthew.william.kenzie@cern.ch)
//-------------------------------------------------------//

/*
  Does post-reco fixes to ECAL photon energy and estimates resolution.
  This can run outside of the usual CMS software framework but requires 
  access to a file 'PhotonFix.dat' which must be in the same directory as 
  that used to run.

  To run independently of CMSSW use PhotonFix.h directly - go to 
  "RecoEcal/EgammaCoreTools/plugins/PhotonFix.h" for details.

  Before instantiating any objects of PhotonFix, the constants must be
  initialised in the first event using
    PhotonFixCMS::initialise("3_8");
  
  The string gives the reco version used. Valid strings are 
  "3_8", "3_11", "4_2" and "Nominal", where the latter gives no correction 
  to the energy and a nominal resolution value.

  Make objects using
    PhotonFixCMS a(p);
  where p is a reco::Photon reference 

  Get the corrected energy using
    a.fixedEnergy();
  and the resolution using
    a.sigmaEnergy();

*/

#include <iostream>
#include <string>
#include "PhotonFix.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class PhotonFixCMS {
 public:
  PhotonFixCMS(const reco::Photon &p);

  // Must be called before instantiating any PhotonFix objects
  static bool initialise(const edm::EventSetup &iSetup, const std::string &s="Nominal");

  // Corrected energy and sigma
  double fixedEnergy() const;
  double sigmaEnergy() const;

  const PhotonFix &photonFix() const;
  
 private:
  PhotonFix pf;
};
#endif
