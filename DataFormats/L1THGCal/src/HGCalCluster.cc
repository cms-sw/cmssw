#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace l1t;

HGCalCluster::HGCalCluster( const LorentzVector p4, 
                            int pt,
                            int eta,
                            int phi )
   : HGCalClusterT<l1t::HGCalTriggerCell>(p4, pt, eta, phi),
   module_(0)
{
}


HGCalCluster::HGCalCluster( const edm::Ptr<l1t::HGCalTriggerCell> &tcSeed )
    : HGCalClusterT<l1t::HGCalTriggerCell>(tcSeed),
    module_(0)
{
}


HGCalCluster::~HGCalCluster()
{
}


//void HGCalCluster::removeUnconnectedConstituents( const HGCalTriggerGeometryBase & triggerGeometry ){
//
//    /* get the constituents and the centre of the seed tc (considered as the first of the constituents) */
//    const edm::PtrVector<l1t::HGCalTriggerCell>& constituents = this->constituents(); 
//    Basic3DVector<float> seedCentre( constituents[0]->position() );
//    
//    /* distances from the seed */
//    vector<pair<int,float>> distances;
//    for( unsigned itc=1; itc<constituents.size(); itc++ )
//    {
//        Basic3DVector<float> tcCentre( constituents[itc]->position() );
//        float distance = ( seedCentre - tcCentre ).mag();
//        distances.push_back( pair<int,float>( itc-1, distance ) );
//    }
//
//    /* sorting (needed in order to be sure that we are skipping any tc) */
//    /* FIXME: better sorting needed!!! */
//    for( unsigned i=0; i<distances.size(); i++ ){
//        for( unsigned j=0; j<(distances.size()-1); j++ ){
//            if( distances[j].second > distances[j+1].second )
//            {
//                iter_swap( distances.begin() + j, distances.begin() + (j+1) );
//            }
//        }        
//    }
//    
//    /* checking if the tc is connected to the seed */
//    bool toRemove[constituents.size()];
//    toRemove[0] = false; // this is the seed
//    for( unsigned itc=0; itc<distances.size(); itc++ ){
//        /* get the tc under study */
//        l1t::HGCalTriggerCell tcToStudy = constituents.at()
//        
//        
//        /* compare with the tc in the cluster */
//        
//        
//
//
//    }
//    
//
//}



