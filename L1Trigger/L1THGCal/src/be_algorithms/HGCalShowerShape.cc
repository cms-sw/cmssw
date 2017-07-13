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

////////////////////////////////////////////////////////////////////////////////////////////




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

void HGCalShowerShape::make2DshowerShape(){

    SigmaEtaEta_=SigmaEtaEta(tc_energy_,tc_eta_);	
    SigmaPhiPhi_=SigmaPhiPhi(tc_energy_,tc_phi_);	

}

////////////////////////////////////////////////////////////////////////////////////////////////

void HGCalShowerShape::makeHGCalProfile(){

		
		if(subdetID_.at(0)==3) firstLayer_=layer_.at(0);//EE
		if(subdetID_.at(0)==4) firstLayer_=layer_.at(0)+28;//FH

		if(subdetID_.at(ncl2D_-1)==3) lastLayer_=layer_.at(ncl2D_-1);
		if(subdetID_.at(ncl2D_-1)==4) lastLayer_=layer_.at(ncl2D_-1)+28;

		nLayers_=lastLayer_-firstLayer_+1;

		for(int ilayer=0;ilayer<40;ilayer++){   //Loop on HGCal layers

			int Layer_found=0;
			float Layer_energy=0;
			float Layer_See=0;
			float Layer_Spp=0;
            float Layer_dEta=0;
			float Layer_dPhi=0;

		    int tc_index=0; // trigger cell index inside cl2D vector

			for(int i2d=0;i2d<ncl2D_;i2d++){   // Loop on cl2D inside 3DC
	
				int cl2D_layer=-999;

				if(subdetID_.at(i2d)==3) cl2D_layer=layer_.at(i2d);
				if(subdetID_.at(i2d)==4) cl2D_layer=layer_.at(i2d)+28;

				if (cl2D_layer==ilayer){

					Layer_found=1; //+=1 il want to count cl2D per layer

					Layer_energy+=cl2D_energy_.at(i2d);

					int ntc=nTC_.at(i2d);

			        	for(int itc=0; itc<ntc ; itc++) {   //Loop on TC inside cl2D

						    energy_layer_.emplace_back(tc_energy_.at(tc_index));
						    energy_.emplace_back(tc_energy_.at(tc_index));
						    eta_layer_.emplace_back(tc_eta_.at(tc_index));
						    phi_layer_.emplace_back(tc_phi_.at(tc_index));
                            eta_.emplace_back(tc_eta_.at(tc_index));
						    phi_.emplace_back(tc_phi_.at(tc_index));
						    z_.emplace_back(tc_z_.at(tc_index));
						    tc_index++;

			        	}

				}

				else{

					if (Layer_found==1) break; //Go to next ilayer
					tc_index+=nTC_.at(i2d);
		
				}

			}
		

            EnergyVector_.emplace_back(Layer_energy);

			if(Layer_energy>EMax_){
				EMax_=Layer_energy;
				EMaxLayer_=ilayer;
			}

			if(Layer_found==1){
                Layer_See=SigmaEtaEta(energy_layer_,eta_layer_);	
			    Layer_Spp=SigmaPhiPhi(energy_layer_,phi_layer_);	
                Layer_dEta=dEta(energy_layer_,eta_layer_);	
			    Layer_dPhi=dPhi(energy_layer_,phi_layer_);
            }	

            SigmaEtaEtaVector_.emplace_back(Layer_See);
            SigmaPhiPhiVector_.emplace_back(Layer_Spp);

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

			energy_layer_.clear();
			eta_layer_.clear();
			phi_layer_.clear();
		
		}


        E0_=EnergyVector_.at(firstLayer_);
        
        SigmaEtaEta0_=SigmaEtaEtaVector_.at(firstLayer_);
        SigmaPhiPhi0_=SigmaPhiPhiVector_.at(firstLayer_);

        Zmean_=Zmean(energy_,z_);
        SigmaZZ_=SigmaZZ(energy_,z_);
        SigmaEtaEta_=SigmaEtaEta(energy_,eta_);
        SigmaPhiPhi_=SigmaPhiPhi(energy_,phi_);

        energy_.clear();
        z_.clear();
        eta_.clear();
        phi_.clear();

}





// -----------------------------------
// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
