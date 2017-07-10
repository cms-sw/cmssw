#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"
//#include "L1Trigger/L1THGCal/interface/LinkDef.h"
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

#include "DataFormats/Math/interface/deltaR.h"


#include <iostream>
#include <sstream>
#include <vector>
#include <string>


void HGCalShowerShape::Init2D(std::vector<float> tc_energy, std::vector<float> tc_eta, std::vector<float> tc_phi/*, std::vector<float> tc_x, std::vector<float> tc_y, std::vector<float> tc_z*/){
    
  tc_energy_ = tc_energy;
  tc_eta_ = tc_eta;
  tc_phi_ = tc_phi;
 /* tc_x_ = tc_x;
  tc_y_ = tc_y;
  tc_z_ = tc_z;*/


  SigmaEtaEta_=0;

  SigmaPhiPhi_=0;
 

}


////////////////////////////////////////////////////////////////////////////////////////////


void HGCalShowerShape::Init3D(std::vector<int> layer, std::vector<int> subdetID, std::vector<float> cl2D_energy, std::vector<int> nTC, std::vector<float> tc_energy, std::vector<float> tc_eta, std::vector<float> tc_phi/*, std::vector<float> tc_x, std::vector<float> tc_y, std::vector<float> tc_z*/){
    

  ncl2D_=layer.size();

  layer_=layer;
  subdetID_ = subdetID;
  cl2D_energy_ = cl2D_energy;
  nTC_ = nTC;
  tc_energy_ = tc_energy;
  tc_eta_ = tc_eta;
  tc_phi_ = tc_phi;
 /* tc_x_ = tc_x;
  tc_y_ = tc_y;
  tc_z_ = tc_z;*/

  EnergyVector_.clear();
  SigmaZZ_=0; 
  SigmaEtaEtaVector_.clear();
  SigmaPhiPhiVector_.clear();

  EMax_=0;
  E0_=0;
  EMaxLayer_=0;

  SigmaEtaEta_=0;
  SigmaEtaEtaMax_=0;
  SigmaEtaEta0_=0;
  SigmaEtaEta10_=0;
  SigmaEtaEtaMaxLayer_=0;

  SigmaPhiPhi_=0;
  SigmaPhiPhiMax_=0;
  SigmaPhiPhi0_=0;
  SigmaPhiPhi10_=0;
  SigmaPhiPhiMaxLayer_=0;

  dEtaMax_=0;
  dPhiMax_=0;

  firstLayer_=0;
  nLayers_=0;

 /* showerEta_=0;
  showerPhi_=0;
  SigmaEtaEtaTotCor_=0;
  SigmaPhiPhiTotCor_=0;*/

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
       // SigmaEtaEta10_=SigmaEtaEta_.at(10);
       // SigmaPhiPhi10_=SigmaPhiPhi_.at(10);

        Zmean_=Zmean(energy_,z_);
        SigmaZZ_=SigmaZZ(energy_,z_);
        SigmaEtaEta_=SigmaEtaEta(energy_,eta_);
        SigmaPhiPhi_=SigmaPhiPhi(energy_,phi_);

        energy_.clear();
        z_.clear();
        eta_.clear();
        phi_.clear();

}


///////////////////////////////////////////////////////////////////////////////////////////////

/*void HGCalShowerShape::showerAxisAnalysis(){

    TPrincipal *principal= new TPrincipal(3,"D"); 
  
    double variables[3] = {0.,0.,0.};

    // minimal rechit value
    float mip_ = 0.000040;

    for(unsigned int itc=0; itc<tc_energy_.size() ; itc++) {   //Loop on TC inside cl2D
     
	variables[0] = tc_x_.at(itc);
	variables[1] = tc_y_.at(itc);
	variables[2] = tc_z_.at(itc);
	
	// energy weighting
	for (int i=0; i<int(tc_energy_.at(itc)/mip_); i++) principal->AddRow(variables); 
	
    }


    principal->MakePrincipals();
  
    TMatrixD matrix = *principal->GetEigenVectors();
  
    GlobalPoint pcaShowerPos((*principal->GetMeanValues())[0],(*principal->GetMeanValues())[1],(*principal->GetMeanValues())[2]);	 
    GlobalVector pcaShowerDir(matrix(0,0),matrix(1,0),matrix(2,0));

    std::cout<<" Shower Pos Eta "<<pcaShowerPos.eta()<<" Phi "<<pcaShowerPos.phi()<<std::endl;
    std::cout<<" Shower Dir Eta "<<pcaShowerDir.eta()<<" Phi "<<pcaShowerDir.phi()<<std::endl;

    showerEta_=pcaShowerPos.eta();						   
    showerPhi_=pcaShowerPos.phi();		
				   

}
*/
///////////////////////////////////////////////////////////////////////////////////////////////

/*
void HGCalShowerShape::make3DHistogram(std::string name, std::vector<float> x, std::vector<float> y){


    
    tc_x_=x;
    tc_y_=y;


    float x0=tc_x_.at(0);
    float y0=tc_y_.at(0);

    h3DShowerEE = new TH3F("", "", 40,x0-40,x0+40,40,y0-40,y0+40,40,0,40);   //not very elegant for xy range ...
    h3DShowerFH = new TH3F("", "", 40,x0-40,x0+40,40,y0-40,y0+40,40,0,40);   
	   

	for(int ilayer=0;ilayer<28;ilayer++){   //Loop on HGCal layers

		int Layer_found=0;

	    int tc_index=0; // trigger cell index inside cl2D vector

		for(int i2d=0;i2d<ncl2D_;i2d++){   // Loop on cl2D inside 3DC
	
			int cl2D_layer=-999;

			if(subdetID_.at(i2d)==3) cl2D_layer=layer_.at(i2d);
			if(subdetID_.at(i2d)==4) cl2D_layer=layer_.at(i2d)+28;

			if (cl2D_layer==ilayer){

				Layer_found=1; //+=1 il want to count cl2D per layer

				int ntc=nTC_.at(i2d);

		        	for(int itc=0; itc<ntc ; itc++) {   //Loop on TC inside cl2D

                        h3DShowerEE->Fill(tc_x_.at(tc_index),tc_y_.at(tc_index),ilayer,tc_energy_.at(tc_index));

					    tc_index++;

		        	}

			}

			else{

				if (Layer_found==1) break; //Go to next ilayer
				tc_index+=nTC_.at(i2d);
		
			}

		}
		
		
	}


for(int ilayer=28;ilayer<40;ilayer++){   //Loop on HGCal layers

		int Layer_found=0;

	    int tc_index=0; // trigger cell index inside cl2D vector

		for(int i2d=0;i2d<ncl2D_;i2d++){   // Loop on cl2D inside 3DC
	
			int cl2D_layer=-999;

			if(subdetID_.at(i2d)==3) cl2D_layer=layer_.at(i2d);
			if(subdetID_.at(i2d)==4) cl2D_layer=layer_.at(i2d)+28;

			if (cl2D_layer==ilayer){

				Layer_found=1; //+=1 il want to count cl2D per layer

				int ntc=nTC_.at(i2d);

		        	for(int itc=0; itc<ntc ; itc++) {   //Loop on TC inside cl2D

                        h3DShowerFH->Fill(tc_x_.at(tc_index),tc_y_.at(tc_index),ilayer,tc_energy_.at(tc_index));

					    tc_index++;

		        	}

			}

			else{

				if (Layer_found==1) break; //Go to next ilayer
				tc_index+=nTC_.at(i2d);
		
			}

		}
		
		
	}

*/

/*

    h3DShowerEE = new TH3F("", "", 40,x0-40,x0+40,40,y0-40,y0+40,100,320,420);   //not very elegant for xy range ...
    h3DShowerFH = new TH3F("", "", 40,x0-40,x0+40,40,y0-40,y0+40,100,320,420);   


                    int ntc=tc_energy_.size();

		        	for(int itc=0; itc<ntc ; itc++) {   //Loop on TC inside cl2D

                        if(fabs(tc_z_.at(itc))<350) h3DShowerEE->Fill(tc_x_.at(itc),tc_y_.at(itc),fabs(tc_z_.at(itc)),tc_energy_.at(itc));
                        else h3DShowerFH->Fill(tc_x_.at(itc),tc_y_.at(itc),fabs(tc_z_.at(itc)),tc_energy_.at(itc));

		        	}






    TCanvas *can=new TCanvas;

    h3DShowerEE->GetXaxis()->SetTitle("x (cm)");
    h3DShowerEE->GetYaxis()->SetTitle("y (cm)");
    h3DShowerEE->GetZaxis()->SetTitle("z (cm)");
    h3DShowerEE->GetXaxis()->SetTitleOffset(1.5);
    h3DShowerEE->GetYaxis()->SetTitleOffset(1.5);
    h3DShowerEE->GetZaxis()->SetTitleOffset(1.5);
    h3DShowerEE->Draw("ISO");
    h3DShowerEE->SetFillColor(kAzure-9);
    h3DShowerFH->Draw("ISO,same");
    h3DShowerFH->SetFillColor(kBlue-1);
    can->SaveAs(Form("3DplotsZ/%s_iso.png",name.c_str()));
    can->SaveAs(Form("3DplotsZ/%s_iso.root",name.c_str()));
    //h3DShower->Draw("BOX");
    //can->SaveAs(Form("3Dplots/%s_box.png",name.c_str()));
    //can->SaveAs(Form("3Dplots/%s_box.root",name.c_str()));

    delete can;
    delete h3DShowerEE;
    delete h3DShowerFH;

}

*/


// -----------------------------------
// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
