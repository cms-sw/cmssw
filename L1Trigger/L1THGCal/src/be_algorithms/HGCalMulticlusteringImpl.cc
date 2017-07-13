#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"
#include "DataFormats/Math/interface/deltaR.h"


HGCalMulticlusteringImpl::HGCalMulticlusteringImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster")),
    ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
    calibSF_(conf.getParameter<double>("calibSF_multicluster"))
{    
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR for Near Neighbour search: " << dr_;  
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster minimum transverse-momentum: " << ptC3dThreshold_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster global calibration factor: " << calibSF_;

}


bool HGCalMulticlusteringImpl::isPertinent( const l1t::HGCalCluster & clu, 
                                            const l1t::HGCalMulticluster & mclu, 
                                            double dR ) const
{
    HGCalDetId cluDetId( clu.detId() );
    HGCalDetId firstClusterDetId( mclu.detId() );
    
    if( cluDetId.zside() != firstClusterDetId.zside() ){
        return false;
    }
    if( ( mclu.centreProj() - clu.centreProj() ).mag() < dR ){
        return true;
    }
    return false;

}


void HGCalMulticlusteringImpl::clusterize( const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters)
{
           
    std::vector<l1t::HGCalMulticluster> multiclustersTmp;

    int iclu = 0;
    for(edm::PtrVector<l1t::HGCalCluster>::const_iterator clu = clustersPtrs.begin(); clu != clustersPtrs.end(); ++clu, ++iclu){
        
        int imclu=0;
        vector<int> tcPertinentMulticlusters;
        for( const auto& mclu : multiclustersTmp ){
            if( this->isPertinent(**clu, mclu, dr_) ){
                tcPertinentMulticlusters.push_back(imclu);
            }
            ++imclu;
        }
        if( tcPertinentMulticlusters.size() == 0 ){
            multiclustersTmp.emplace_back( *clu );
        }
        else{
            unsigned minDist = 1;
            unsigned targetMulticlu = 0; 
            for( int imclu : tcPertinentMulticlusters ){
                double d = ( multiclustersTmp.at(imclu).centreProj() - (*clu)->centreProj() ).mag() ;
                if( d < minDist ){
                    minDist = d;
                    targetMulticlu = imclu;
                }
            } 

            multiclustersTmp.at( targetMulticlu ).addConstituent( *clu );
            
        }        
    }

    /* making the collection of multiclusters */
    for( unsigned i(0); i<multiclustersTmp.size(); ++i ){
        math::PtEtaPhiMLorentzVector calibP4(  multiclustersTmp.at(i).pt() * calibSF_, 
                                               multiclustersTmp.at(i).eta(), 
                                               multiclustersTmp.at(i).phi(), 
                                               0. );
        // overwriting the 4p with the calibrated 4p     
        multiclustersTmp.at(i).setP4( calibP4 );
        if( multiclustersTmp.at(i).pt() > ptC3dThreshold_ ){
            multiclusters.push_back( 0, multiclustersTmp.at(i));  
        }
    }
    
}


void HGCalMulticlusteringImpl::showerShape3D(const edm::PtrVector<l1t::HGCalCluster> & clustersPtr){

    Nlayers_=0;
    EMax_=0; //Maximum energy deposited in a layer
    SeeTot_=0; //SigmaEtaEta considering all TC in 3DC
    SeeMax_=0; //Maximum SigmaEtaEta in a layer
    SppTot_=0; //same but for SigmaPhiPhi
    SppMax_=0;

/*    std::vector<int> layer ; // Size : ncl2D
    std::vector<int> subdetID ;
    std::vector<float> cl2D_energy ;
    std::vector<int> nTC ;
    std::vector<float> tc_energy ; // Size : ncl2D*nTCi
    std::vector<float> tc_eta ;
    std::vector<float> tc_phi ;

    for(edm::PtrVector<l1t::HGCalCluster>::const_iterator clu = clustersPtrs.begin(); clu != clustersPtrs.end(); ++clu){
        
        	layer.emplace_back((*clu)->layer());
        	subdetID.emplace_back((*clu)->subdetId());
        	cl2D_energy.emplace_back((*clu)->energy());

		const edm::PtrVector<l1t::HGCalTriggerCell> triggerCells = (*clu)->constituents();
    		unsigned int ncells = triggerCells.size();
		nTC.emplace_back(ncells);
		for(unsigned int itc=0; itc<ncells;itc++){

    			l1t::HGCalTriggerCell thistc = *triggerCells[itc];

        		tc_energy.emplace_back(thistc.energy());
        		tc_eta.emplace_back(thistc.eta());
        		tc_phi.emplace_back(thistc.phi());

		}
    }
*/
    HGCalShowerShape *shape=new HGCalShowerShape();
    shape->Init3D(clustersPtr);
    shape->makeHGCalProfile();
    Nlayers_=shape->nLayers();
    SeeTot_=shape->SigmaEtaEta();
    SppTot_=shape->SigmaPhiPhi();
    std::vector<float> Energy=shape->EnergyVector();
    std::vector<float> See=shape->SigmaEtaEtaVector();
    std::vector<float> Spp=shape->SigmaPhiPhiVector();


    for(int ilayer=0;ilayer<Nlayers_;ilayer++){
    	if(Energy[ilayer]>EMax_) EMax_= Energy[ilayer];
    	if(See[ilayer]>SeeMax_) SeeMax_= See[ilayer];
    	if(Spp[ilayer]>SppMax_) SppMax_= Spp[ilayer];

    }


}



