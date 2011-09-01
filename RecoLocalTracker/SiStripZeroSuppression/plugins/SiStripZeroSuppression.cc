#include "RecoLocalTracker/SiStripZeroSuppression/plugins/SiStripZeroSuppression.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include <memory>

SiStripZeroSuppression::
SiStripZeroSuppression(edm::ParameterSet const& conf)
  : inputTags(conf.getParameter<std::vector<edm::InputTag> >("RawDigiProducersList")),
    algorithms(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
    storeCM(conf.getParameter<bool>("storeCM")),
    doAPVRestore(conf.getParameter<bool>("doAPVRestore")),
    mergeCollections(conf.getParameter<bool>("mergeCollections"))
	
{
  
  if(doAPVRestore){
    produceRawDigis = conf.getParameter<bool>("produceRawDigis");
    produceCalculatedBaseline = conf.getParameter<bool>("produceCalculatedBaseline");
    produceBaselinePoints = conf.getParameter<bool>("produceBaselinePoints");
    storeInZScollBadAPV = conf.getParameter<bool>("storeInZScollBadAPV");
    useCMMeanMap = conf.getParameter<bool>("useCMMeanMap");
    fixCM = conf.getParameter<bool>("fixCM");  
  } else {
    produceRawDigis = false;
    produceCalculatedBaseline = false;
    produceBaselinePoints = false;
    storeInZScollBadAPV = false;
    useCMMeanMap = false;
    fixCM = false;	
  }
  
  if(mergeCollections){
    storeCM = false;
    produceRawDigis = false;
    DigisToMergeZS = conf.getParameter<edm::InputTag>("DigisToMergeZS");
    DigisToMergeVR = conf.getParameter<edm::InputTag>("DigisToMergeVR");
    produces< edm::DetSetVector<SiStripDigi> > ("ZeroSuppressed");
  }
  
  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ){
    produces< edm::DetSetVector<SiStripDigi> > (inputTag->instance());
    if(produceRawDigis)
      produces< edm::DetSetVector<SiStripRawDigi> > (inputTag->instance());
  } 
  
  if(produceCalculatedBaseline) 
    produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("BADAPVBASELINE");
	
  if(produceBaselinePoints) 
    produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("BADAPVBASELINEPOINTS");
	  
  if(storeCM)
    produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("APVCM");
  
  
}

void SiStripZeroSuppression::
produce(edm::Event& e, const edm::EventSetup& es) {
    
  algorithms->initialize(es);
  if( doAPVRestore && useCMMeanMap) algorithms->restorer->LoadMeanCMMap( e );
   
  if(mergeCollections){
    //this->CollectionMergedZeroSuppression(e);
	this->MergeCollectionsZeroSuppression(e);
  }else{ 
    this->StandardZeroSuppression(e);
  }
}

inline void SiStripZeroSuppression::StandardZeroSuppression(edm::Event& e){

  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ) {

    edm::Handle< edm::DetSetVector<SiStripRawDigi> > input;
    e.getByLabel(*inputTag,input);

    std::vector<edm::DetSet<SiStripDigi> > output_base; 
    std::vector<edm::DetSet<SiStripRawDigi> > output_base_raw; 
    
    if (input->size()) 
      processRaw(*inputTag, *input, output_base, output_base_raw );
    
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(output_base) );
    e.put( output, inputTag->instance() );
    
	
    if(produceRawDigis){
      std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > outputraw(new edm::DetSetVector<SiStripRawDigi>(output_base_raw) );
      e.put(outputraw, inputTag->instance() );
    }
  }

  
  if(produceCalculatedBaseline){
    std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > outputbaseline(new edm::DetSetVector<SiStripProcessedRawDigi>(output_baseline) );
    e.put(outputbaseline, "BADAPVBASELINE" );
  }
  
  if(produceBaselinePoints){
    std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > outputbaselinepoints(new edm::DetSetVector<SiStripProcessedRawDigi>(output_baseline_points) );
    e.put(outputbaselinepoints, "BADAPVBASELINEPOINTS" );
  }
  
  if(storeCM){
    std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > outputAPVCM(new edm::DetSetVector<SiStripProcessedRawDigi>(output_apvcm) );
    e.put( outputAPVCM,"APVCM");
  }  
}

inline void SiStripZeroSuppression::MergeCollectionsZeroSuppression(edm::Event& e){
    
    std::cout<< "starting Merging" << std::endl;
    edm::Handle< edm::DetSetVector<SiStripDigi> > inputdigi;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > inputraw;
    e.getByLabel(DigisToMergeZS,inputdigi);
    e.getByLabel(DigisToMergeVR,inputraw);
	
    std::cout << inputdigi->size() << " " << inputraw->size() << std::endl;
	
    if (inputraw->size()){
		
		std::vector<edm::DetSet<SiStripDigi> > outputdigi; 
		outputdigi.clear();
        
    	//std::cout << "copying the input ZS to the output ZS collection" << std::endl;
    	for ( edm::DetSetVector<SiStripDigi>::const_iterator Digis = inputdigi->begin(); Digis != inputdigi->end(); ++Digis)  outputdigi.push_back(*Digis);
       
	   	    
        std::cout << "looping over the raw data collection" << std::endl;
    	for ( edm::DetSetVector<SiStripRawDigi>::const_iterator rawDigis = inputraw->begin(); rawDigis != inputraw->end(); ++rawDigis) {
    	   
			edm::DetSet<SiStripRawDigi>::const_iterator itRawDigis = rawDigis->begin();
			uint16_t nAPV = rawDigis->size()/128;
			uint32_t rawDetId = rawDigis->id;
          
			std::vector<bool> restoredAPV;
			restoredAPV.clear();
			restoredAPV.insert(restoredAPV.begin(), nAPV, false); 
          
          
			bool isModuleRestored = false;
			for( uint16_t strip =0; strip < rawDigis->size();++strip){
				if(itRawDigis[strip].adc()!=0){
				  restoredAPV[strip/128] = true;       
				  isModuleRestored = true;
				}
			}
 
   		  	
			if(isModuleRestored){
				std::cout << "apply the ZS to the raw data collection" << std::endl;
				edm::DetSet<SiStripDigi> suppressedDigis(rawDetId);
				std::vector<int16_t> processedRawDigis(rawDigis->size());
				algorithms->subtractorPed->subtract(*rawDigis, processedRawDigis);
				//processedRawDigisCopy.assign(processedRawDigis.begin(), processedRawDigis.end());
				algorithms->subtractorCMN->subtract(rawDetId, processedRawDigis);
				//nAPVflagged = algorithms->restorer->inspect( rawDigis->id, processedRawDigisCopy, algorithms->subtractorCMN->getAPVsCM() );
				//algorithms->restorer->restore( processedRawDigis );
				algorithms->suppressor->suppress(processedRawDigis, suppressedDigis);
			   
				if(suppressedDigis.size()){	  
					std::cout << "looking for the detId with the new ZS in the collection of the zero suppressed data" << std::endl; 
					std::vector<edm::DetSet<SiStripDigi> >::iterator zsModule = outputdigi.begin();
					//std::vector<edm::DetSet<SiStripDigi> >::iterator LastLowerIdDigis = zsModule;
					
					uint32_t zsDetId = zsModule->id;
					bool isModuleInZscollection = false;
					while((zsDetId <= rawDetId)&&(zsModule != outputdigi.end())&&(!isModuleInZscollection)){
						//std::cout << rawDetId << " ==== " <<  zsDetId << std::endl;
						if( zsDetId == rawDetId){
							isModuleInZscollection = true;
						}else{
							++zsModule;
							zsDetId = zsModule->id;
						}
					}         
					std::cout << "after the look " << rawDetId << " ==== " <<  zsDetId << std::endl;
					std::cout << "exiting looking for the detId with the new ZS in the collection of the zero suppressed data" << std::endl; 
		
					//creating the map containing the digis (in rawdigi format) merged
					std::vector<uint16_t> MergedRawDigis;
					MergedRawDigis.clear();
					MergedRawDigis.insert(MergedRawDigis.begin(), nAPV*128, 0);
					
					uint32_t count=0; // to be removed...
					
					edm::DetSet<SiStripDigi> newDigiToIndert(rawDetId);
					if(!isModuleInZscollection){
					  std::cout << "WE HAVE A PROBLEM, THE MODULE IS NTOT FOUND" << std::endl;
					  outputdigi.insert(zsModule, newDigiToIndert);
						--zsModule;	
					  std::cout << "new module id -1 " << zsModule->id << std::endl;
						++zsModule;
					  std::cout << "new module id " << zsModule->id << std::endl;
						++zsModule;
						std::cout << "new module id +1 " << zsModule->id << std::endl;
						--zsModule;
					  	
					} else {
						std::cout << "inserting only the digis for not restored APVs" << std::endl;
						std::cout << "size : " << zsModule->size() << std::endl;
						edm::DetSet<SiStripDigi>::iterator itZsModule = zsModule->begin(); 
						for(; itZsModule != zsModule->end(); ++itZsModule){
							uint16_t adc = itZsModule->adc();
							uint16_t strip = itZsModule->strip();
							if(!restoredAPV[strip/128]){
								MergedRawDigis[strip] = adc;
								++count;
								std::cout << "original count: "<< count << " strip: " << strip << " adc: " << adc << std::endl;  
							}
						} 
						
					}
										 
					std::cout << "size of digis to keep: " << count << std::endl;				
					std::cout << "inserting only the digis for the restored APVs" << std::endl;
					std::cout << "size : " << suppressedDigis.size() << std::endl;
					edm::DetSet<SiStripDigi>::iterator itSuppDigi = suppressedDigis.begin();
					for(; itSuppDigi != suppressedDigis.end(); ++itSuppDigi){
					  uint16_t adc = itSuppDigi->adc();
					  uint16_t strip = itSuppDigi->strip();
					  if(restoredAPV[strip/128]){
					  	MergedRawDigis[strip] = adc;
					  	std::cout << "new suppressed strip: " << strip << " adc: " << adc << std::endl;
					  }
					} 
				  
				  
				  
					std::cout << "suppressing the raw digis" << std::endl;
					zsModule->clear();
					for(uint16_t strip=0; strip < MergedRawDigis.size(); ++strip){
					  uint16_t adc = MergedRawDigis[strip];
					  if(adc) zsModule->push_back(SiStripDigi(strip, adc));
					}
					std::cout << "size zsModule after the merging: " << zsModule->size() << std::endl;
					if((count + suppressedDigis.size()) != zsModule->size()) std::cout << "WE HAVE A PROBLEM!!!! THE NUMBER OF DIGIS IS NOT RIGHT==============" << std::endl;
					std::cout << "exiting suppressing the raw digis" << std::endl;
				}//if new ZS digis size
			} //if module restored
		}//loop over raw data collection
		
		uint32_t oldid =0;
		for(edm::DetSetVector<SiStripDigi>::const_iterator dg = outputdigi.begin(); dg != outputdigi.end(); ++dg){
			uint32_t iddg = dg->id;
			if(iddg < oldid){
				std::cout<< "NOT IN THE RIGHT ORGER" << std:: endl;
				std::cout<< "======================="<< std:: endl;
			}
			oldid = iddg; 
		}
		
		
		std::cout << "write the output vector" << std::endl;
		std::auto_ptr< edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(outputdigi) );
		e.put( output, "ZeroSuppressed" );  
		
		
    }//if inputraw.size


   
}

inline void SiStripZeroSuppression::CollectionMergedZeroSuppression(edm::Event& e){

  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ) {

    edm::Handle< edm::DetSetVector<SiStripDigi> > inputdigi;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > inputraw;
    e.getByLabel(*inputTag,inputdigi);
    e.getByLabel(*inputTag,inputraw);
	
    std::vector<edm::DetSet<SiStripDigi> > outputdigi; 
    std::vector<edm::DetSet<SiStripRawDigi> > outputraw;     
	
    if (inputraw->size())	
      processRaw(*inputTag, *inputraw, outputdigi, outputraw );
    
	
    for ( std::vector<edm::DetSet<SiStripDigi> >::const_iterator itinputdigi = inputdigi->begin(); itinputdigi !=inputdigi->end(); ++itinputdigi) {
      outputdigi.push_back(*itinputdigi);	
    }
	
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(outputdigi) );
    e.put( output, inputTag->instance() );
  	
  }

}

inline
void SiStripZeroSuppression::
processRaw(const edm::InputTag& inputTag, const edm::DetSetVector<SiStripRawDigi>& input, std::vector<edm::DetSet<SiStripDigi> >& output, std::vector<edm::DetSet<SiStripRawDigi> >& outputraw) {

  if(storeCM){
    output_apvcm.clear();
    output_apvcm.reserve(16000);
  }
  
  if(produceCalculatedBaseline){
    output_baseline.clear();
    output_baseline.reserve(16000);
  }
  
  if(produceBaselinePoints){
    output_baseline_points.clear();
    output_baseline_points.reserve(16000);
  }
  
  output.reserve(10000);    
  outputraw.reserve(10000);
  
  
  for ( edm::DetSetVector<SiStripRawDigi>::const_iterator 
	  rawDigis = input.begin(); rawDigis != input.end(); ++rawDigis) {
    	
      edm::DetSet<SiStripDigi> suppressedDigis(rawDigis->id);
    
    int16_t nAPVflagged = 0;
		
    if ( "ProcessedRaw" == inputTag.instance()) {
      std::vector<int16_t> processedRawDigis, processedRawDigisCopy ;
      transform(rawDigis->begin(), rawDigis->end(), back_inserter(processedRawDigis), boost::bind(&SiStripRawDigi::adc , _1));
	  if( doAPVRestore ){
	   	processedRawDigisCopy.assign(processedRawDigis.begin(), processedRawDigis.end());
	  }
      algorithms->subtractorCMN->subtract( rawDigis->id, processedRawDigis);
      if( doAPVRestore ){
	    nAPVflagged = algorithms->restorer->inspect( rawDigis->id, processedRawDigisCopy, algorithms->subtractorCMN->getAPVsCM());
		algorithms->restorer->restore( processedRawDigis );
	  }
      algorithms->suppressor->suppress( processedRawDigis, suppressedDigis );
      
      const std::vector< std::pair<short,float> >& vmedians= algorithms->subtractorCMN->getAPVsCM();
      if(storeCM) this->storeCMN(rawDigis->id, vmedians);
      if(produceCalculatedBaseline&& nAPVflagged > 0){
	     std::map< uint16_t, std::vector < int16_t> >& baselinemap = algorithms->restorer->GetBaselineMap();
	     this->storeBaseline(rawDigis->id, vmedians, baselinemap);
      }
      if(produceBaselinePoints&& nAPVflagged > 0){
      	std::vector< std::map< uint16_t, int16_t> >&  baselinpoints = algorithms->restorer->GetSmoothedPoints();
      	this->storeBaselinePoints(rawDigis->id, baselinpoints);
      }
    } else if ( "VirginRaw" == inputTag.instance()) {
      std::vector<int16_t> processedRawDigis(rawDigis->size()), processedRawDigisCopy;
      algorithms->subtractorPed->subtract( *rawDigis, processedRawDigis);
       if( doAPVRestore ){
	    processedRawDigisCopy.assign(processedRawDigis.begin(), processedRawDigis.end());
	  }
      algorithms->subtractorCMN->subtract( rawDigis->id, processedRawDigis);
      if( doAPVRestore ){
	      nAPVflagged = algorithms->restorer->inspect( rawDigis->id, processedRawDigisCopy, algorithms->subtractorCMN->getAPVsCM());
    	  algorithms->restorer->restore( processedRawDigis );
	  }
      algorithms->suppressor->suppress( processedRawDigis, suppressedDigis );
      
      const std::vector< std::pair<short,float> >& vmedians = algorithms->subtractorCMN->getAPVsCM();
      if(storeCM) this->storeCMN(rawDigis->id, vmedians);
      if(produceCalculatedBaseline&& nAPVflagged > 0){
	     std::map< uint16_t, std::vector < int16_t> >& baselinemap = algorithms->restorer->GetBaselineMap();
	     this->storeBaseline(rawDigis->id, vmedians, baselinemap);
      }	  
      if(produceBaselinePoints&& nAPVflagged > 0){
     	std::vector< std::map< uint16_t, int16_t> >&  baselinpoints = algorithms->restorer->GetSmoothedPoints();
     	this->storeBaselinePoints(rawDigis->id, baselinpoints);
      }
      
    } else 
      
      throw cms::Exception("Unknown input type") 
	<< inputTag.instance() << " unknown.  SiStripZeroZuppression can only process types \"VirginRaw\" and \"ProcessedRaw\" ";
    
    if (suppressedDigis.size() && (storeInZScollBadAPV || nAPVflagged ==0)) 
      output.push_back(suppressedDigis); 
    
    if (produceRawDigis && nAPVflagged > 0){  
      std::vector<bool> apvf;
      algorithms->restorer->GetAPVFlags(apvf);
      edm::DetSet<SiStripRawDigi> outRawDigis(rawDigis->id);
      edm::DetSet<SiStripRawDigi>::const_iterator itRawDigis = rawDigis->begin(); 
    
     
	  //std::cout << "detId: " << rawDigis->id << std::endl;
      for(size_t APVn=0; APVn < apvf.size(); ++APVn){
	     //std::cout << "APV: " << APVn <<  " " << apvf[APVn] << std::endl;
         if(apvf[APVn]==0){
           for(size_t strip =0; strip < 128; ++strip) outRawDigis.push_back(SiStripRawDigi(0));
		 } else {
		   for(size_t strip =0; strip < 128; ++strip) outRawDigis.push_back(itRawDigis[APVn*128+strip]);
         }
      }
                                  
      //outputraw.push_back(*rawDigis); 
      outputraw.push_back(outRawDigis);
	  
    }
   
}
  
}


inline 
void SiStripZeroSuppression::storeBaseline(uint32_t id, const std::vector< std::pair<short,float> >& vmedians, std::map< uint16_t, std::vector < int16_t> >& baselinemap){
	
  edm::DetSet<SiStripProcessedRawDigi> baselineDetSet(id);
  std::map< uint16_t, std::vector < int16_t> >::iterator itBaselineMap;
  
  for(size_t i=0;i<vmedians.size();++i){
    uint16_t APVn = vmedians[i].first;
    float median = vmedians[i].second;
    itBaselineMap = baselinemap.find(APVn);
    if(itBaselineMap==baselinemap.end()){
      for(size_t strip=0; strip < 128; ++strip)  baselineDetSet.push_back(SiStripProcessedRawDigi(median));
    } else {
      for(size_t strip=0; strip < 128; ++strip) baselineDetSet.push_back(SiStripProcessedRawDigi((itBaselineMap->second)[strip]));
    }
    
  }
  
  if(baselineDetSet.size())
    output_baseline.push_back(baselineDetSet);
  
}

inline
void SiStripZeroSuppression::storeBaselinePoints(uint32_t id, std::vector< std::map< uint16_t, int16_t> >& BasPointVec){

    edm::DetSet<SiStripProcessedRawDigi> baspointDetSet(id);
    std::vector< std::map< uint16_t, int16_t> >::iterator itBasPointVec;
    std::map< uint16_t, int16_t>::iterator itBaselinePointMap;
    
    uint16_t APVn =0; 
    for(itBasPointVec= BasPointVec.begin();  itBasPointVec != BasPointVec.end();++itBasPointVec){
    	for(size_t strip=0; strip < 128; ++strip) baspointDetSet.push_back(SiStripProcessedRawDigi(0));
          
        if(itBasPointVec->size()){
             itBaselinePointMap = itBasPointVec->begin();    
             for(;itBaselinePointMap != itBasPointVec->end(); ++itBaselinePointMap){
                  uint16_t bpstrip = (itBaselinePointMap->first) + APVn*128;
             	  int16_t  bp = itBaselinePointMap->second;
                  baspointDetSet[bpstrip] = bp;
               
             } 
        }
      	++APVn; 
	}    
		
    
    
    if(baspointDetSet.size())
    output_baseline_points.push_back(baspointDetSet);
  
}

inline 
void SiStripZeroSuppression::storeCMN(uint32_t id, const std::vector< std::pair<short,float> >& vmedians){
	
  edm::DetSet<SiStripProcessedRawDigi> apvDetSet(id);
  short apvNb=0;
  for(size_t i=0;i<vmedians.size();++i){
    if(vmedians[i].first>apvNb){
      for(int i=0;i<vmedians[i].first-apvNb;++i){
	apvDetSet.push_back(SiStripProcessedRawDigi(0.));
	apvNb++;
      }
    }
    apvDetSet.push_back(SiStripProcessedRawDigi(vmedians[i].second));
    apvNb++;
  }
  if(fixCM) algorithms->restorer->fixAPVsCM( apvDetSet ); 
  if(apvDetSet.size())
    output_apvcm.push_back(apvDetSet);
  
}
