#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

#include "TH2F.h"
#include <string>

enum {kEBChannels = 61200, kEEChannels = 14648};
enum {MIN_IETA = 1, MIN_IPHI = 1, MAX_IETA = 85, MAX_IPHI = 360, EBhistEtaMax = 171};   // barrel lower and upper bounds on eta and phi
enum {IX_MIN = 1, IY_MIN = 1, IX_MAX = 100, IY_MAX = 100, EEhistXMax = 220};           // endcaps lower and upper bounds on x and y


template <class floatCondObj>
void fillEBMap_SingleIOV(std::shared_ptr<floatCondObj> payload, TH2F* & barrel){

	for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
		uint32_t rawid = EBDetId::unhashIndex(cellid);
		EcalCondObjectContainer<float>::const_iterator value_ptr =  payload->find(rawid);
		if (value_ptr == payload->end())
			continue; // cell absent from payload
	          
		float weight = (float)(*value_ptr);
		Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
		Double_t eta = (Double_t)(EBDetId(rawid)).ieta();

		if(eta > 0.)
			eta = eta - 0.5;   //   0.5 to 84.5
		else
			eta  = eta + 0.5;         //  -84.5 to -0.5
	          
		barrel->Fill(phi, eta, weight);
	}// loop over cellid

}


template <class floatCondObj>
void fillEEMap_SingleIOV(std::shared_ptr<floatCondObj> payload, TH2F* & endc_m, TH2F* & endc_p){
	// looping over the EE channels
	for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
		for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
			for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
				if(EEDetId::validDetId(ix, iy, iz)) {
					EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
	                uint32_t rawid = myEEId.rawId();
	                EcalCondObjectContainer<float>::const_iterator value_ptr =  payload->find(rawid);
	                if (value_ptr == payload->end())
						continue; // cell absent from payload
	                
	                float weight = (float)(*value_ptr);
	                if(iz == 1)
						endc_p->Fill(ix, iy, weight);
	                else
						endc_m->Fill(ix, iy, weight);

				}  // validDetId
}


template <class floatCondObj>
void fillEBMap_DiffIOV(std::shared_ptr<floatCondObj> payload, TH2F* & barrel,int irun,
	float pEB[], float & pEBmin, float & pEBmax){

	for(int cellid = EBDetId::MIN_HASH; cellid < EBDetId::kSizeForDenseIndexing; ++cellid) {
		uint32_t rawid = EBDetId::unhashIndex(cellid);
		EcalCondObjectContainer<float>::const_iterator value_ptr =  payload->find(rawid);
		if (value_ptr == payload->end())
			continue; // cell absent from payload

		float weight = (float)(*value_ptr);
		if(irun == 0) 
			pEB[cellid] = weight;
		else {
			Double_t phi = (Double_t)(EBDetId(rawid)).iphi() - 0.5;
			Double_t eta = (Double_t)(EBDetId(rawid)).ieta();
              
		if(eta > 0.)
			eta = eta - 0.5;   //   0.5 to 84.5
		else
			eta  = eta + 0.5;  //  -84.5 to -0.5
              
		double diff = weight - pEB[cellid];
              
		if(diff < pEBmin)
			pEBmin = diff;
		if(diff > pEBmax)
			pEBmax = diff;

		barrel->Fill(phi, eta, diff);
		}

	}// loop over cellid


}


template <class floatCondObj>
void fillEEMap_DiffIOV(std::shared_ptr<floatCondObj> payload, TH2F* & endc_m, TH2F* & endc_p, int irun,
	float pEE[], float & pEEmin, float & pEEmax){

	// looping over the EE channels
	for(int iz = -1; iz < 2; iz = iz + 2)   // -1 or +1
		for(int iy = IY_MIN; iy < IY_MAX+IY_MIN; iy++)
			for(int ix = IX_MIN; ix < IX_MAX+IX_MIN; ix++)
				if(EEDetId::validDetId(ix, iy, iz)) {
					EEDetId myEEId = EEDetId(ix, iy, iz, EEDetId::XYMODE);
					uint32_t cellid = myEEId.hashedIndex();
					uint32_t rawid = myEEId.rawId();
					EcalCondObjectContainer<float>::const_iterator value_ptr =  payload->find(rawid);
                  
				if (value_ptr == payload->end())
					continue; // cell absent from payload
				float weight = (float)(*value_ptr);

				if(irun == 0)
					pEE[cellid] = weight;
				else {
					double diff = weight - pEE[cellid];
					if(diff < pEEmin)
						pEEmin = diff;

					if(diff > pEEmax)
						pEEmax = diff;
					if(iz == 1)
						endc_p->Fill(ix, iy, diff);
					else
						endc_m->Fill(ix, iy, diff);
				}

			}  // validDetId 

}


void fillTableWithSummary(TH2F* & align, std::string title, 
 const float & mean_x_EB,const float & rms_EB,const int & num_x_EB,
 const float & mean_x_EE,const float & rms_EE,const int & num_x_EE){
	
	int NbRows=2;
	align = new TH2F(title.c_str(),"EB/EE      mean_x      rms        num_x", 4, 0, 4, NbRows, 0, NbRows);

	double row = NbRows-0.5;

	align->Fill(0.5,row,1);
	align->Fill(1.5,row,mean_x_EB);
	align->Fill(2.5,row,rms_EB);
	align->Fill(3.5,row,num_x_EB);
      
	row--;
      
	align->Fill(0.5,row,2);
	align->Fill(1.5,row,mean_x_EE);
	align->Fill(2.5,row,rms_EE);
	align->Fill(3.5,row,num_x_EE);


	align->GetXaxis()->SetTickLength(0.);
	align->GetXaxis()->SetLabelSize(0.);
	align->GetYaxis()->SetTickLength(0.);
	align->GetYaxis()->SetLabelSize(0.);

}