
#include "CalibMuon/CSCCalibration/interface/CSCFakeNoiseMatrixConditions.h"

void CSCFakeNoiseMatrixConditions::prefillNoiseMatrix(){

  const CSCDetId& detId = CSCDetId();
  cnmatrix = new CSCNoiseMatrix();
  
  int max_istrip,id_layer,max_ring,max_cham;
  //endcap=1 to 2,station=1 to 4, ring=1 to 4,chamber=1 to 36,layer=1 to 6 
  
  for(int iendcap=detId.minEndcapId(); iendcap<=detId.maxEndcapId(); iendcap++){
    for(int istation=detId.minStationId() ; istation<=detId.maxStationId(); istation++){
      max_ring=detId.maxRingId();
      //station 4 ring 4 not there(36 chambers*2 missing)
      //3 rings max this way of counting (ME1a & b)
      if(istation==1)    max_ring=3;
      if(istation==2)    max_ring=2;
      if(istation==3)    max_ring=2;
      if(istation==4)    max_ring=1;
	
      for(int iring=detId.minRingId(); iring<=max_ring; iring++){
	max_istrip=80;
	max_cham=detId.maxChamberId(); 
	if(istation==1 && iring==1)    max_cham=36;
	if(istation==1 && iring==2)    max_cham=36;
	if(istation==1 && iring==3)    max_cham=36;
	if(istation==2 && iring==1)    max_cham=18;
	if(istation==2 && iring==2)    max_cham=36;
	if(istation==3 && iring==1)    max_cham=18;
	if(istation==3 && iring==2)    max_cham=36;
	if(istation==4 && iring==1)    max_cham=18;
	
	for(int ichamber=detId.minChamberId(); ichamber<=max_cham; ichamber++){
	  for(int ilayer=detId.minLayerId(); ilayer<=detId.maxLayerId(); ilayer++){
	    //station 1 ring 3 has 64 strips per layer instead of 80 
	    if(istation==1 && iring==3)   max_istrip=64;

	    std::vector<CSCNoiseMatrix::Item> itemvector;
	    itemvector.resize(max_istrip);
	    id_layer = 100000*iendcap + 10000*istation + 1000*iring + 10*ichamber + ilayer;
	    
	    for(int istrip=0;istrip<max_istrip;istrip++){
	      
	      if(istation==1 && iring==1){
		itemvector[istrip].elem33 = 7.86675;
		itemvector[istrip].elem34 = 2.07075;
		itemvector[istrip].elem44 = 6.93875;
		itemvector[istrip].elem35 = 1.42525;
		itemvector[istrip].elem45 = 2.51025;
		itemvector[istrip].elem55 = 7.93975;
		itemvector[istrip].elem46 = 0.94725;
		itemvector[istrip].elem56 = 2.39275;
		itemvector[istrip].elem66 = 6.46475;
		itemvector[istrip].elem57 = 1.86325;
		itemvector[istrip].elem67 = 2.08025;
		itemvector[istrip].elem77 = 6.67975;
		cnmatrix->matrix[id_layer]=itemvector;
	      }

	      if(istation==1 && iring==2){
		itemvector[istrip].elem33 = 9.118;
		itemvector[istrip].elem34 = 3.884;
		itemvector[istrip].elem44 = 7.771;
		itemvector[istrip].elem35 = 1.8225;
		itemvector[istrip].elem45 = 3.7505;
		itemvector[istrip].elem55 = 8.597;
		itemvector[istrip].elem46 = 1.651;
		itemvector[istrip].elem56 = 2.5225;
		itemvector[istrip].elem66 = 6.583;
		itemvector[istrip].elem57 = 1.5055;
		itemvector[istrip].elem67 = 2.733;
		itemvector[istrip].elem77 = 6.988;
		cnmatrix->matrix[id_layer]=itemvector;
	      }

	      if(istation==1 && iring==3){
		itemvector[istrip].elem33 = 9.5245;
		itemvector[istrip].elem34 = 3.2415;
		itemvector[istrip].elem44 = 7.6265;
		itemvector[istrip].elem35 = 1.7225;
		itemvector[istrip].elem45 = 3.6075;
		itemvector[istrip].elem55 = 8.7275;
		itemvector[istrip].elem46 = 1.663;
		itemvector[istrip].elem56 = 2.592;
		itemvector[istrip].elem66 = 7.5685;
		itemvector[istrip].elem57 = 1.7905;
		itemvector[istrip].elem67 = 2.409;
		itemvector[istrip].elem77 = 7.1495;
		cnmatrix->matrix[id_layer]=itemvector;
	      }

	      if(istation==2 && iring==1){
		itemvector[istrip].elem33 = 9.06825;
		itemvector[istrip].elem34 = 3.32025;
		itemvector[istrip].elem44 = 7.52925;
		itemvector[istrip].elem35 = 3.66125;
		itemvector[istrip].elem45 = 3.39125;
		itemvector[istrip].elem55 = 9.97625;
		itemvector[istrip].elem46 = 1.32725;
		itemvector[istrip].elem56 = 3.99025;
		itemvector[istrip].elem66 = 8.10125;
		itemvector[istrip].elem57 = 2.56456;
		itemvector[istrip].elem67 = 2.96625;
		itemvector[istrip].elem77 = 7.30925;
		cnmatrix->matrix[id_layer]=itemvector;
	      }	

	      if(istation==2 &&iring==2){
		itemvector[istrip].elem33 = 16.7442;
		itemvector[istrip].elem34 = 7.96925;
		itemvector[istrip].elem44 = 14.1643;
		itemvector[istrip].elem35 = 4.67975;
		itemvector[istrip].elem45 = 8.44075;
		itemvector[istrip].elem55 = 17.2243;
		itemvector[istrip].elem46 = 3.68575;
		itemvector[istrip].elem56 = 7.48825;
		itemvector[istrip].elem66 = 14.4902;
		itemvector[istrip].elem57 = 4.4482;
		itemvector[istrip].elem67 = 6.47875;
		itemvector[istrip].elem77 = 14.6733;
		cnmatrix->matrix[id_layer]=itemvector;
	      }	

	      if(istation==3 && iring==1){
		itemvector[istrip].elem33 = 9.3495;
		itemvector[istrip].elem34 = 3.529;
		itemvector[istrip].elem44 = 7.8715;
		itemvector[istrip].elem35 = 3.8155;
		itemvector[istrip].elem45 = 3.858;
		itemvector[istrip].elem55 = 10.8205;
		itemvector[istrip].elem46 = 1.8585;
		itemvector[istrip].elem56 = 4.445;
		itemvector[istrip].elem66 = 8.0175;
		itemvector[istrip].elem57 = 3.29479;
		itemvector[istrip].elem67 = 3.625;
		itemvector[istrip].elem77 = 8.3895;
		cnmatrix->matrix[id_layer]=itemvector;
	      }	
	
	      if(istation==3 && iring==2){
		itemvector[istrip].elem33 = 13.6193;
		itemvector[istrip].elem34 = 5.91025;
		itemvector[istrip].elem44 = 11.3842;
		itemvector[istrip].elem35 = 3.31775;
		itemvector[istrip].elem45 = 5.69775;
		itemvector[istrip].elem55 = 11.6652;
		itemvector[istrip].elem46 = 2.46175;
		itemvector[istrip].elem56 = 4.48325;
		itemvector[istrip].elem66 = 9.95725;
		itemvector[istrip].elem57 = 2.10561;
		itemvector[istrip].elem67 = 4.04625;
		itemvector[istrip].elem77 = 9.51625;
		cnmatrix->matrix[id_layer]=itemvector;
	      }	

	      if(istation==4 && iring==1){
		itemvector[istrip].elem33 = 10.0;
		itemvector[istrip].elem34 = 4.0;
		itemvector[istrip].elem44 = 10.0;
		itemvector[istrip].elem35 = 3.0;
		itemvector[istrip].elem45 = 8.0;
		itemvector[istrip].elem55 = 10.0;
		itemvector[istrip].elem46 = 2.0;
		itemvector[istrip].elem56 = 5.0;
		itemvector[istrip].elem66 = 10.0;
		itemvector[istrip].elem57 = 3.0;
		itemvector[istrip].elem67 = 4.0;
		itemvector[istrip].elem77 = 10.0;
		cnmatrix->matrix[id_layer]=itemvector;
	      }
	    }
	  }
	}
      }
    }
  }
}

CSCFakeNoiseMatrixConditions::CSCFakeNoiseMatrixConditions(const edm::ParameterSet& iConfig)
{
  
  //tell the framework what data is being produced
  prefillNoiseMatrix();  
  setWhatProduced(this,&CSCFakeNoiseMatrixConditions::produceNoiseMatrix);
  
  findingRecord<CSCNoiseMatrixRcd>();
  
  //now do what ever other initialization is needed
  
}


CSCFakeNoiseMatrixConditions::~CSCFakeNoiseMatrixConditions()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  delete cnmatrix; // since not made persistent so we still own it.
}

//
// member functions
//

// ------------ method called to produce the data  ------------
CSCFakeNoiseMatrixConditions::ReturnType
CSCFakeNoiseMatrixConditions::produceNoiseMatrix(const CSCNoiseMatrixRcd& iRecord)
{
  return cnmatrix;
}

void CSCFakeNoiseMatrixConditions::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&,
						  edm::ValidityInterval & oValidity)
{
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
  
}
