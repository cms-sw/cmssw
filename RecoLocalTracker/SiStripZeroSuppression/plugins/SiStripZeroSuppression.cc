#include "RecoLocalTracker/SiStripZeroSuppression/plugins/SiStripZeroSuppression.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"
#include "FWCore/Utilities/interface/transform.h"
#include <memory>

SiStripZeroSuppression::
SiStripZeroSuppression(edm::ParameterSet const& conf)
  : inputTags(conf.getParameter<std::vector<edm::InputTag> >("RawDigiProducersList")),
    algorithms(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
    storeCM(conf.getParameter<bool>("storeCM")),
    mergeCollections(conf.getParameter<bool>("mergeCollections"))
	
{
   
    produceRawDigis = conf.getParameter<bool>("produceRawDigis");
    produceCalculatedBaseline = conf.getParameter<bool>("produceCalculatedBaseline");
    produceBaselinePoints = conf.getParameter<bool>("produceBaselinePoints");
    storeInZScollBadAPV = conf.getParameter<bool>("storeInZScollBadAPV");
    fixCM = conf.getParameter<bool>("fixCM");  
  
  if(mergeCollections){
    storeCM = false;
    produceRawDigis = false;
    DigisToMergeZS = consumes< edm::DetSetVector<SiStripDigi> >(conf.getParameter<edm::InputTag>("DigisToMergeZS"));
    DigisToMergeVR = consumes< edm::DetSetVector<SiStripRawDigi> >(conf.getParameter<edm::InputTag>("DigisToMergeVR"));
    produces< edm::DetSetVector<SiStripDigi> > ("ZeroSuppressed");
  }
  
  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag ){
    produces< edm::DetSetVector<SiStripDigi> > (inputTag->instance());
    if(produceRawDigis) produces< edm::DetSetVector<SiStripRawDigi> > (inputTag->instance());
    if(produceCalculatedBaseline) produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("BADAPVBASELINE"+inputTag->instance());
    if(produceBaselinePoints) produces< edm::DetSetVector<SiStripDigi> > ("BADAPVBASELINEPOINTS"+inputTag->instance());
    if(storeCM) produces< edm::DetSetVector<SiStripProcessedRawDigi> > ("APVCM"+inputTag->instance());
//     tokens consumes<reco::BeamSpot>(
  } 
  inputTokens = edm::vector_transform( inputTags, [this](edm::InputTag const & tag) { return consumes< edm::DetSetVector<SiStripRawDigi> >(tag);} );
  
  
  
}

void SiStripZeroSuppression::
produce(edm::Event& e, const edm::EventSetup& es) {
    
  algorithms->initialize(es, e);
     
  if(mergeCollections){
    this->MergeCollectionsZeroSuppression(e);
  }else{ 
    this->StandardZeroSuppression(e);
  }
}

inline void SiStripZeroSuppression::StandardZeroSuppression(edm::Event& e){

  token_iterator_t inputToken = inputTokens.begin();
  for(tag_iterator_t inputTag = inputTags.begin(); inputTag != inputTags.end(); ++inputTag,++inputToken ) {

    edm::Handle< edm::DetSetVector<SiStripRawDigi> > input;
    e.getByToken(*inputToken,input);

    if (input->size())
      processRaw(*inputTag, *input);
    
      std::auto_ptr< edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(output_base) );
      e.put( output, inputTag->instance() );
    	
      if(produceRawDigis){
	std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > outputraw(new edm::DetSetVector<SiStripRawDigi>(output_base_raw) );
	e.put(outputraw, inputTag->instance() );
      }
    
      if(produceCalculatedBaseline){
	std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > outputbaseline(new edm::DetSetVector<SiStripProcessedRawDigi>(output_baseline) );
	e.put(outputbaseline, "BADAPVBASELINE"+inputTag->instance() );
      }
  
      if(produceBaselinePoints){
	std::auto_ptr< edm::DetSetVector<SiStripDigi> > outputbaselinepoints(new edm::DetSetVector<SiStripDigi>(output_baseline_points) );
	e.put(outputbaselinepoints, "BADAPVBASELINEPOINTS"+inputTag->instance() );
      }
  
      if(storeCM){
	std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > outputAPVCM(new edm::DetSetVector<SiStripProcessedRawDigi>(output_apvcm) );
	e.put( outputAPVCM,"APVCM"+inputTag->instance());
      }
    
  }
}




inline
void SiStripZeroSuppression::
processRaw(const edm::InputTag& inputTag, const edm::DetSetVector<SiStripRawDigi>& input ) {

  output_apvcm.clear();
  output_baseline.clear();
  output_baseline_points.clear();
  output_base.clear(); 
  output_base_raw.clear();

  if(storeCM) output_apvcm.reserve(16000);
  if(produceCalculatedBaseline) output_baseline.reserve(16000);
  if(produceBaselinePoints) output_baseline_points.reserve(16000);
  if(produceRawDigis) output_base_raw.reserve(16000);
  output_base.reserve(16000);    
 
  
  for ( edm::DetSetVector<SiStripRawDigi>::const_iterator 
      rawDigis = input.begin(); rawDigis != input.end(); ++rawDigis) {
    	
      edm::DetSet<SiStripDigi> suppressedDigis(rawDigis->id);
      int16_t nAPVflagged = 0;
        
      if ( "ProcessedRaw" == inputTag.instance()) nAPVflagged = algorithms->SuppressProcessedRawData(*rawDigis, suppressedDigis);
      else if ( "VirginRaw" == inputTag.instance()) nAPVflagged = algorithms->SuppressVirginRawData(*rawDigis, suppressedDigis); 
      else     
      throw cms::Exception("Unknown input type") 
	<< inputTag.instance() << " unknown.  SiStripZeroZuppression can only process types \"VirginRaw\" and \"ProcessedRaw\" ";

      //here storing the output
      this->storeExtraOutput(rawDigis->id, nAPVflagged);
      if (suppressedDigis.size() && (storeInZScollBadAPV || nAPVflagged ==0)) 
	output_base.push_back(suppressedDigis); 
         
      if (produceRawDigis && nAPVflagged > 0){  
	edm::DetSet<SiStripRawDigi> outRawDigis(rawDigis->id);
	this->formatRawDigis(rawDigis, outRawDigis);
	output_base_raw.push_back(outRawDigis);
      }
         
  }
  
}

inline 
void SiStripZeroSuppression::formatRawDigis(edm::DetSetVector<SiStripRawDigi>::const_iterator 
					    rawDigis, edm::DetSet<SiStripRawDigi>& outRawDigis){
     
      const std::vector<bool>& apvf = algorithms->GetAPVFlags();
      edm::DetSet<SiStripRawDigi>::const_iterator itRawDigis = rawDigis->begin(); 
         
      uint32_t strip=0;
      for (; itRawDigis != rawDigis->end(); ++itRawDigis){
	int16_t APVn = strip/128;
        if(apvf[APVn]) outRawDigis.push_back(*itRawDigis); 
        else outRawDigis.push_back(SiStripRawDigi(0));
        ++strip;
       }
          
}


inline 
void SiStripZeroSuppression::storeExtraOutput(uint32_t id, int16_t nAPVflagged){

      const std::vector< std::pair<short,float> >& vmedians = algorithms->getAPVsCM();
      if(storeCM) this->storeCMN(id, vmedians);
      if(nAPVflagged > 0){
	if(produceCalculatedBaseline) this->storeBaseline(id, vmedians);
	if(produceBaselinePoints) this->storeBaselinePoints(id);
      }
}


inline 
void SiStripZeroSuppression::storeBaseline(uint32_t id, const std::vector< std::pair<short,float> >& vmedians){
  
  std::map< uint16_t, std::vector < int16_t> >& baselinemap = algorithms->GetBaselineMap();	

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
void SiStripZeroSuppression::storeBaselinePoints(uint32_t id){

    std::map< uint16_t, std::map< uint16_t, int16_t> >&  BasPointVec = algorithms->GetSmoothedPoints();
    std::map< uint16_t, std::map< uint16_t, int16_t> >::iterator itBasPointVect = BasPointVec.begin() ;
    std::map< uint16_t, int16_t>::iterator itBaselinePointMap;
    
    edm::DetSet<SiStripDigi> baspointDetSet(id);
          
    for(; itBasPointVect != BasPointVec.end(); ++itBasPointVect){
        uint16_t APVn= itBasPointVect->first;
        itBaselinePointMap =itBasPointVect->second.begin();
        for(;itBaselinePointMap != itBasPointVect->second.end(); ++itBaselinePointMap){
                  uint16_t bpstrip = (itBaselinePointMap->first) + APVn*128;
             	  int16_t  bp = itBaselinePointMap->second;
                  baspointDetSet.push_back(SiStripDigi(bpstrip,bp+128));
               
          } 
      }    

    
    if(baspointDetSet.size())
    output_baseline_points.push_back(baspointDetSet);
  
}

inline 
void SiStripZeroSuppression::storeCMN(uint32_t id, const std::vector< std::pair<short,float> >& vmedians){
	
  edm::DetSet<SiStripProcessedRawDigi> apvDetSet(id);
  short apvNb=0;
  
  std::vector<bool> apvf;
  apvf.clear();
  apvf.insert(apvf.begin(), 6, false);

  if(fixCM){
    std::vector<bool>& apvFlagged = algorithms->GetAPVFlags();
    for(uint16_t it=0; it< apvFlagged.size(); ++it) apvf[it] = apvFlagged[it];
  }

  for(size_t i=0;i<vmedians.size();++i){
    if(vmedians[i].first>apvNb){
      for(int i=0;i<vmedians[i].first-apvNb;++i){
	apvDetSet.push_back(SiStripProcessedRawDigi(-999.));
	apvNb++;
      }
    }

    if(fixCM&&apvf[vmedians[i].first]){
      apvDetSet.push_back(SiStripProcessedRawDigi(-999.));
    }else{
      apvDetSet.push_back(SiStripProcessedRawDigi(vmedians[i].second));
    }
    apvNb++;
  }
   
  if(apvDetSet.size())
    output_apvcm.push_back(apvDetSet);
  
}


inline void SiStripZeroSuppression::MergeCollectionsZeroSuppression(edm::Event& e){
    
    std::cout<< "starting Merging" << std::endl;
    edm::Handle< edm::DetSetVector<SiStripDigi> > inputdigi;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > inputraw;
    e.getByToken(DigisToMergeZS,inputdigi);
    e.getByToken(DigisToMergeVR,inputraw);
	
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
                                algorithms->SuppressVirginRawData(*rawDigis, suppressedDigis);
		  	   
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
      processRaw(*inputTag, *inputraw);
    
	
    for ( std::vector<edm::DetSet<SiStripDigi> >::const_iterator itinputdigi = inputdigi->begin(); itinputdigi !=inputdigi->end(); ++itinputdigi) {
      output_base.push_back(*itinputdigi);	
    }
	
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(output_base) );
    e.put( output, inputTag->instance() );
  	
  }

}
