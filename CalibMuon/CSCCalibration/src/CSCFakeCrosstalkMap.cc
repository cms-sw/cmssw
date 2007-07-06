#include "CalibMuon/CSCCalibration/interface/CSCFakeCrosstalkMap.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "FWCore/Utilities/interface/Exception.h"

CSCFakeCrosstalkMap::CSCFakeCrosstalkMap()
: theMean(-0.0009),
  theMin(0.035),
  theMinChi(1.5),
  theM(1000)
{
  
  const CSCDetId& detId = CSCDetId();
  cncrosstalk = new CSCcrosstalk();
  
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
	    
	    std::vector<CSCcrosstalk::Item> itemvector;
	    itemvector.resize(max_istrip);
	    id_layer = 100000*iendcap + 10000*istation + 1000*iring + 10*ichamber + ilayer;
	    
	    for(int istrip=0;istrip<max_istrip;istrip++){		  
	    
	      itemvector[istrip].xtalk_slope_right= theMean;
	      itemvector[istrip].xtalk_intercept_right= theMin;
	      itemvector[istrip].xtalk_chi2_right= theMinChi;
	      itemvector[istrip].xtalk_slope_left= theMean;
	      itemvector[istrip].xtalk_intercept_left= theMin;
	      itemvector[istrip].xtalk_chi2_left= theMinChi;
	      cncrosstalk->crosstalk[id_layer]=itemvector;

	      if(istrip==0){
		itemvector[istrip].xtalk_slope_left=0.0;
		itemvector[istrip].xtalk_intercept_left=0.0;
		itemvector[istrip].xtalk_chi2_left=0.0;
	      }
	      
	      if(istrip==79){
		itemvector[istrip].xtalk_slope_right=0.0;
		itemvector[istrip].xtalk_intercept_right=0.0;
		itemvector[istrip].xtalk_chi2_right=0.0;
	      }
              cncrosstalk->crosstalk[id_layer]=itemvector;
	    }
	  }
	}
      }
    }
  }
} 


void CSCFakeCrosstalkMap::smear() 
{
  //FIXME make memebrs
  float theMeanSigma = 1/10000.;
  float theMinSigma = 1/100.;
  float theMinChiSigma = 1;
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
       << "CSCFakeCrosstalkMap requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
  }
  CLHEP::RandGaussQ randGauss( rng->getEngine() );

  for(std::map< int,std::vector<CSCcrosstalk::Item> >::iterator layerItr = cncrosstalk->crosstalk.begin(),
                                                                lastLayer = cncrosstalk->crosstalk.end();
      layerItr != lastLayer; ++layerItr)
  {
    for(std::vector<CSCcrosstalk::Item>::iterator itemItr = layerItr->second.begin(),
                                                  lastItem = layerItr->second.end();
        itemItr != lastItem; ++itemItr)
    {
      itemItr->xtalk_slope_left  += randGauss.fire(0.,theMeanSigma);
      itemItr->xtalk_slope_right += randGauss.fire(0.,theMeanSigma);
      itemItr->xtalk_intercept_left  += randGauss.fire(0.,theMinSigma);
      itemItr->xtalk_intercept_right += randGauss.fire(0.,theMinSigma);
      itemItr->xtalk_chi2_left  += randGauss.fire(0.,theMinChiSigma);
      itemItr->xtalk_chi2_right += randGauss.fire(0.,theMinChiSigma);


    }
  }
}

