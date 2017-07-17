#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"
#include "TMath.h"
#include <cmath>
#include "TH3F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TAttFill.h"
#include "TPrincipal.h"
#include "TMatrixD.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

#include "DataFormats/Math/interface/deltaR.h"


#include <iostream>
#include <sstream>
#include <vector>
#include <string>


/*
void HGCalShowerShape::Init2D(const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs){

    std::vector<float> tc_energy ; 
    std::vector<float> tc_eta ;
    std::vector<float> tc_phi ;

    for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc = triggerCellsPtrs.begin(); tc != triggerCellsPtrs.end(); ++tc){
   
	tc_energy.emplace_back((*tc)->energy());
	tc_eta.emplace_back((*tc)->eta());
	tc_phi.emplace_back((*tc)->phi());

    }

    tc_energy_ = tc_energy;
    tc_eta_ = tc_eta;
    tc_phi_ = tc_phi;

}
*/
////////////////////////////////////////////////////////////////////////////////////////////



/*
void HGCalShowerShape::Init3D(const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs){

    std::vector<int> layer ; // Size : ncl2D
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

  ncl2D_=layer.size();

  layer_=layer;
  subdetID_ = subdetID;
  cl2D_energy_ = cl2D_energy;
  nTC_ = nTC;
  tc_energy_ = tc_energy;
  tc_eta_ = tc_eta;
  tc_phi_ = tc_phi;

}
*/



////////////////////////////////////////////////////////////////////////////////////////////


float HGCalShowerShape::SigmaEtaEta(std::vector<float> energy, std::vector<float> eta){

	int ntc=energy.size();
	//compute weighted eta mean
	float eta_sum=0;
	float w_sum=0; //here weight is energy
	for(int i=0;i<ntc;i++){
		//std::cout<<i<<" energy "<<energy.at(i)<<" eta "<<eta.at(i)<<std::endl;
		eta_sum+=energy.at(i)*eta.at(i);
		w_sum+=energy.at(i);
	}
	float eta_mean=eta_sum/w_sum;
	//compute weighted eta RMS
	float deltaeta2_sum=0;
	for(int i=0;i<ntc;i++) deltaeta2_sum+=energy.at(i)*pow((eta.at(i)-eta_mean),2);
	float eta_RMS=deltaeta2_sum/w_sum;
	float See=sqrt(eta_RMS);
	//std::cout<<"See "<<See<<std::endl;
	return See;
	
}


float HGCalShowerShape::dEta(std::vector<float> energy, std::vector<float> eta){

	int ntc=energy.size();
	//compute weighted eta mean
	float eta_sum=0;
	float w_sum=0; //here weight is energy
	for(int i=0;i<ntc;i++){
		//std::cout<<i<<" energy "<<energy.at(i)<<" eta "<<eta.at(i)<<std::endl;
		eta_sum+=energy.at(i)*eta.at(i);
		w_sum+=energy.at(i);
	}
	float eta_mean=eta_sum/w_sum;
	//compute weighted eta RMS
	float deltaeta=0;
    float deta;
	for(int i=0;i<ntc;i++){
    deta=fabs(eta_mean-eta.at(i));
	if (deta>deltaeta) deltaeta=deta;
    }
    return deltaeta;
	
}


///////////////////////////////////////////////////////////////////////////////////////////////


float HGCalShowerShape::SigmaPhiPhi(std::vector<float> energy, std::vector<float> phi){

	int ntc=energy.size();
	//compute weighted phi mean
	float phi_sum=0;
	float w_sum=0; //here weight is energy
	for(int i=0;i<ntc;i++){
		//std::cout<<i<<" energy "<<energy.at(i)<<" phi "<<phi.at(i)<<std::endl;
		if(phi.at(i)>0) phi_sum+=energy.at(i)*phi.at(i);
		else phi_sum+=energy.at(i)*(phi.at(i)+2*TMath::Pi());
		w_sum+=energy.at(i);
	}
	float phi_mean=phi_sum/w_sum;
	if(phi_mean>TMath::Pi()) phi_mean-=2*TMath::Pi();
	//compute weighted eta RMS
	float deltaphi2_sum=0;
	for(int i=0;i<ntc;i++){
		float deltaPhi=fabs(phi.at(i)-phi_mean);
		if (deltaPhi>TMath::Pi()) deltaPhi=2*TMath::Pi()-deltaPhi;
		deltaphi2_sum+=energy.at(i)*pow(deltaPhi,2);
	}		
	float phi_RMS=deltaphi2_sum/w_sum;
	float Spp=sqrt(phi_RMS);
	//std::cout<<"Spp "<<Spp<<std::endl<<std::endl;
	return Spp;
	
}


float HGCalShowerShape::dPhi(std::vector<float> energy, std::vector<float> phi){

	int ntc=energy.size();
	//compute weighted phi mean
	float phi_sum=0;
	float w_sum=0; //here weight is energy
	for(int i=0;i<ntc;i++){
		//std::cout<<i<<" energy "<<energy.at(i)<<" phi "<<phi.at(i)<<std::endl;
		if(phi.at(i)>0) phi_sum+=energy.at(i)*phi.at(i);
		else phi_sum+=energy.at(i)*(phi.at(i)+2*TMath::Pi());
		w_sum+=energy.at(i);
	}
	float phi_mean=phi_sum/w_sum;
	if(phi_mean>TMath::Pi()) phi_mean-=2*TMath::Pi();
	//compute weighted eta RMS
	float deltaphi=0;
    float dphi;
	for(int i=0;i<ntc;i++){
		dphi=fabs(phi.at(i)-phi_mean);
		if (dphi>TMath::Pi()) dphi=2*TMath::Pi()-dphi;
        if (dphi>deltaphi) deltaphi=dphi;
	}		
	return deltaphi;
	
}


///////////////////////////////////////////////////////////////////////////////////////////////

float HGCalShowerShape::SigmaZZ(std::vector<float> energy, std::vector<float> z){

	int ntc=energy.size();
	//compute weighted eta mean
	float z_sum=0;
	float w_sum=0; //here weight is energy
	for(int i=0;i<ntc;i++){
		//std::cout<<i<<" energy "<<energy.at(i)<<" z "<<z.at(i)<<std::endl;
		z_sum+=energy.at(i)*z.at(i);
		w_sum+=energy.at(i);
	}
	float z_mean=z_sum/w_sum;
	//compute weighted eta RMS
	float deltaz2_sum=0;
	for(int i=0;i<ntc;i++) deltaz2_sum+=energy.at(i)*pow((z.at(i)-z_mean),2);
	float z_RMS=deltaz2_sum/w_sum;
	float Szz=sqrt(z_RMS);
	//std::cout<<"Szz "<<Szz<<std::endl<<std::endl;
	return Szz;
	
}

///////////////////////////////////////////////////////////////////////////////////////////////


float HGCalShowerShape::Zmean(std::vector<float> energy, std::vector<float> z){

	int ntc=energy.size();
	//compute weighted eta mean
	float z_sum=0;
	float w_sum=0; //here weight is energy
	for(int i=0;i<ntc;i++){
		//std::cout<<i<<" energy "<<energy.at(i)<<" z "<<z.at(i)<<std::endl;
		z_sum+=energy.at(i)*z.at(i);
		w_sum+=energy.at(i);
	}
	float z_mean=z_sum/w_sum;
	return fabs(z_mean);
	
}


////////////////////////////////////////////////////////////////////////////////////////////////

void HGCalShowerShape::make2DshowerShape(const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs){

    std::vector<float> tc_energy ; 
    std::vector<float> tc_eta ;
    std::vector<float> tc_phi ;

    for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc = triggerCellsPtrs.begin(); tc != triggerCellsPtrs.end(); ++tc){
   
	tc_energy.emplace_back((*tc)->energy());
	tc_eta.emplace_back((*tc)->eta());
	tc_phi.emplace_back((*tc)->phi());

    }

    SigmaEtaEta_=SigmaEtaEta(tc_energy,tc_eta);	
    SigmaPhiPhi_=SigmaPhiPhi(tc_energy,tc_phi);	

}

////////////////////////////////////////////////////////////////////////////////////////////////

void HGCalShowerShape::makeHGCalProfile(const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs){


    std::vector<int> layer ; // Size : ncl2D
    std::vector<int> subdetID ;
    std::vector<float> cl2D_energy ;
    std::vector<int> nTC ;
    std::vector<float> tc_energy ; // Size : ncl2D*nTCi
    std::vector<float> tc_eta ;
    std::vector<float> tc_phi ;
    std::vector<float> tc_z ;

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
        		tc_z.emplace_back(thistc.position().z());

		    }
    }

    int ncl2D=layer.size();		

    std::vector<float> EnergyVector; // Size : 40 (HGCal layers)
    std::vector<float> SigmaEtaEtaVector; // Size : 40 (HGCal layers)
    std::vector<float> SigmaPhiPhiVector; // Size : 40 (HGCal layers)
    

   //Auxiliary Containers to compute shower shapes
    std::vector<float> energy_layer ;
    std::vector<float> eta_layer ;
    std::vector<float> phi_layer ;
    std::vector<float> energy ;
    std::vector<float> eta ;
    std::vector<float> phi ;
    std::vector<float> x ;
    std::vector<float> y ;
    std::vector<float> z ;

	if(subdetID.at(0)==3) firstLayer_=layer.at(0);//EE
	if(subdetID.at(0)==4) firstLayer_=layer.at(0)+28;//FH

	if(subdetID.at(ncl2D-1)==3) lastLayer_=layer.at(ncl2D-1);
	if(subdetID.at(ncl2D-1)==4) lastLayer_=layer.at(ncl2D-1)+28;

	nLayers_=lastLayer_-firstLayer_+1;

    SigmaEtaEtaMax_=0;
    SigmaPhiPhiMax_=0;
    EMax_=0;

    //std::cout<<" 1st layer "<<firstLayer_<<" Last layer "<<lastLayer_<<" Nlayers "<<nLayers_<<endl;

	for(int ilayer=0;ilayer<40;ilayer++){   //Loop on HGCal layers

			int Layer_found=0;
			float Layer_energy=0;
			float Layer_See=0;
			float Layer_Spp=0;
            float Layer_dEta=0;
			float Layer_dPhi=0;

		    int tc_index=0; // trigger cell index inside cl2D vector

			for(int i2d=0;i2d<ncl2D;i2d++){   // Loop on cl2D inside 3DC
	
				int cl2D_layer=-999;

				if(subdetID.at(i2d)==3) cl2D_layer=layer.at(i2d);
				if(subdetID.at(i2d)==4) cl2D_layer=layer.at(i2d)+28;

				if (cl2D_layer==ilayer){

					Layer_found=1; //+=1 il want to count cl2D per layer

					Layer_energy+=cl2D_energy.at(i2d);

					int ntc=nTC.at(i2d);

			        	for(int itc=0; itc<ntc ; itc++) {   //Loop on TC inside cl2D

						    energy_layer.emplace_back(tc_energy.at(tc_index));
						    energy.emplace_back(tc_energy.at(tc_index));
						    eta_layer.emplace_back(tc_eta.at(tc_index));
						    phi_layer.emplace_back(tc_phi.at(tc_index));
                            eta.emplace_back(tc_eta.at(tc_index));
						    phi.emplace_back(tc_phi.at(tc_index));
						    z.emplace_back(tc_z.at(tc_index));
						    tc_index++;

			        	}

				}

				else{

					if (Layer_found==1) break; //Go to next ilayer
					tc_index+=nTC.at(i2d);
		
				}

			}
		

            EnergyVector.emplace_back(Layer_energy);

			if(Layer_energy>EMax_){
				EMax_=Layer_energy;
				EMaxLayer_=ilayer;
			}

			if(Layer_found==1){
                Layer_See=SigmaEtaEta(energy_layer,eta_layer);	
			    Layer_Spp=SigmaPhiPhi(energy_layer,phi_layer);	
                Layer_dEta=dEta(energy_layer,eta_layer);	
			    Layer_dPhi=dPhi(energy_layer,phi_layer);

            //std::cout<<"Layer "<<ilayer<<" See "<<Layer_See<<" See Max "<<SigmaEtaEtaMax_<<endl;
            //std::cout<<"Layer "<<ilayer<<" Spp "<<Layer_Spp<<" Spp Max "<<SigmaPhiPhiMax_<<endl;
            }	

            SigmaEtaEtaVector.emplace_back(Layer_See);
            SigmaPhiPhiVector.emplace_back(Layer_Spp);

			if(Layer_See>SigmaEtaEtaMax_){
				SigmaEtaEtaMax_=Layer_See;
				SigmaEtaEtaMaxLayer_=ilayer;
			}

		    if(Layer_Spp>SigmaPhiPhiMax_){
				SigmaPhiPhiMax_=Layer_Spp;
				SigmaPhiPhiMaxLayer_=ilayer;
			}

            if(Layer_dEta>dEtaMax_) dEtaMax_=Layer_dEta;
            if(Layer_dPhi>dPhiMax_) dPhiMax_=Layer_dPhi;

			energy_layer.clear();
			eta_layer.clear();
			phi_layer.clear();
		
		}


        

        E0_=EnergyVector.at(firstLayer_-1);
        
        SigmaEtaEta0_=SigmaEtaEtaVector.at(firstLayer_-1);
        SigmaPhiPhi0_=SigmaPhiPhiVector.at(firstLayer_-1);

        EnergyVector.clear();
        SigmaEtaEtaVector.clear();
        SigmaPhiPhiVector.clear();

        Zmean_=Zmean(energy,z);
        SigmaZZ_=SigmaZZ(energy,z);
        SigmaEtaEta_=SigmaEtaEta(energy,eta);
        SigmaPhiPhi_=SigmaPhiPhi(energy,phi);

        //std::cout<<" See Tot "<<SigmaEtaEta_<<endl;
        //std::cout<<" Spp Tot "<<SigmaPhiPhi_<<endl;

        energy.clear();
        z.clear();
        eta.clear();
        phi.clear();

        layer.clear();
		subdetID.clear();
		cl2D_energy.clear();
		nTC.clear();
		tc_energy.clear();
		tc_eta.clear();
		tc_phi.clear();
		tc_z.clear();

			

}





// -----------------------------------
// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
