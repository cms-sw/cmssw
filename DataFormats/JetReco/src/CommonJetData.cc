//
// F.Ratnikov UMd
// Apr 27, 2006
//

#include "DataFormats/JetReco/interface/CommonJetData.h"

CommonJetData::CommonJetData(const LorentzVector& p4, int n) 
  : 
  mP4 (p4), 
  numberOfConstituents (n) 
{
  init ();
}

  CommonJetData::CommonJetData(double px, double py, double pz, double e, int n)
    :
    mP4 (px, py, pz, e),
    numberOfConstituents (n)
{
  init ();
}

void CommonJetData::init () {
  px = mP4.Px ();
  py = mP4.Py ();
  pz = mP4.Pz ();
  e = mP4.E ();
  p = mP4.P ();
  pt = mP4.Pt ();
  et = mP4.Et ();
  m = mP4.M ();
  phi = mP4.Phi ();
  eta = mP4.Eta ();
}
