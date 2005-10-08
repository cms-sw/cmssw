/*
 * \file EBMonitorUtils.cc
 *
 *  $Date: 2005/10/06 17:47:01 $
 *  $Revision: 1.2 $
 * \author B. Gobbo
 *
 */

#include <DQM/EcalBarrelMonitorTasks/interface/EBMonitorUtils.h>
#include <DQM/EcalBarrelMonitorTasks/interface/H4Geom.h>

int EBMonitorUtils::getSuperModuleID( const int phi, const int zed ) {

  int sm = -1;

  // Just a small check...
  if( phi < 1 || phi > 360 || zed == 0 ) return sm;

  // 
  sm = ( phi - 1 ) / 20 + 1;
  if( zed < 0 ) sm += 19;
  return sm;
}

int EBMonitorUtils::getCrystalID( const int eta, const int phi ) {

  // Just a small check...
  if( eta < 1 || eta > 85 || phi < 1 || phi > 360 ) return -1;

  H4Geom geo;
  int localPhi = (phi-1) % 20;
  int localEta = eta -1;
  return( geo.getSMCrystalFromCoord( localEta, localPhi ) );
  
}

void EBMonitorUtils::getEtaPhi( const int crystal, const int sm, int &eta, int &phi ) {

  // Just a small check...
  if( crystal < 1 || crystal > 1700 || sm < 1 || sm > 36 ) {
    eta = -1;
    phi = -1;
    return;
  }

  H4Geom geo;
  geo.getCrystalCoord( eta, phi, crystal );
  eta++;
  phi++;
  if( sm > 18 ) {
    phi = phi * (sm-18);
  }
  else {
    phi = phi * sm;
  } 
}
