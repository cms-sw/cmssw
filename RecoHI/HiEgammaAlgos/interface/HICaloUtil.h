#ifndef HICaloUtil_h
#define HICaloUtil_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

class HICaloUtil {
public:
   HICaloUtil() {}

   static double              EcalEta(const reco::Candidate &p);
   static double              EcalPhi(const reco::Candidate &p);
   static double              EcalEta(double EtaParticle, double Zvertex, double plane_Radius);
   static double              EcalPhi(double PtParticle, double EtaParticle, 
                                        double PhiParticle, int ChargeParticle, double Rstart);

   static const double        kEEtaBarrelEndcap; //eta boundary for between barrel and endcap
   static const double        kER_ECAL;          //radius ecal barrel begin
   static const double        kEZ_Endcap;        //z distance for endcap begin
   static const double        kERBARM;           //magnetic field was 3.15, updated on 16122003
   static const double        kEZENDM;           //magnetic field was 1.31, updated on 16122003
   
};

#endif /*HIROOT_HICaloUtil*/
