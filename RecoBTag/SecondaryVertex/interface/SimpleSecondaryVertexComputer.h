#ifndef TrackCounting_SimpleSecondaryVertexComputer_h
#define TrackCounting_SimpleSecondaryVertexComputer_h
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "Math/GenVector/VectorUtil.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"

class SimpleSecondaryVertexComputer : public JetTagComputer
{
 public:
  SimpleSecondaryVertexComputer(const edm::ParameterSet  & parameters )
  {
  }


  float discriminator(const reco::BaseTagInfo & ti) const
   {
    const reco::SecondaryVertexTagInfo * svti = dynamic_cast<const reco::SecondaryVertexTagInfo *>(&ti);
      if(svti!=0)  {
           TrackKinematics kinematics(svti->secondaryVertex());
           float gamma=kinematics.vectorSum().M()/kinematics.vectorSum().mag();
           if(svti->nVertices() == 0 ) 
            {
           return -1.;
           } 
           else 
               { 
	         return    svti->flightDistance().value()/gamma;    
/*               if( svti->flightDistance().value() > 0) 
                 return   log(1+ svti->flightDistance().significance());    
                 else
                 return   -log(1-svti->flightDistance().significance());    
  */          
               }
        }
        else
          {
            //FIXME: report some error?
            return -1. ;
          }
   }
};

#endif
