#ifndef HSCPIsolation_H
#define HSCPIsolation_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <vector>
#include "DataFormats/Common/interface/ValueMap.h"

namespace susybsm {

 class HSCPIsolation 
  {
   public:
      // constructor
      HSCPIsolation(){
         TK_Count     = -1;
         TK_SumEt     = -1;
         ECAL_Energy  = -1;
         HCAL_Energy  = -1;
      }

   void   Set_TK_Count   (double value){TK_Count    = value;}
   void   Set_TK_SumEt   (double value){TK_SumEt    = value;}
   void   Set_ECAL_Energy(double value){ECAL_Energy = value;}
   void   Set_HCAL_Energy(double value){HCAL_Energy = value;}

   double Get_TK_Count   ()            {return TK_Count    ;}
   double Get_TK_SumEt   ()            {return TK_SumEt    ;}
   double Get_ECAL_Energy()            {return ECAL_Energy ;}
   double Get_HCAL_Energy()            {return HCAL_Energy ;}

   public:
      double TK_Count;
      double TK_SumEt;
      double ECAL_Energy;
      double HCAL_Energy;
  };

  typedef  std::vector<HSCPIsolation>    HSCPIsolationCollection;
  typedef  edm::ValueMap<HSCPIsolation>  HSCPIsolationValueMap;  
}

#endif
