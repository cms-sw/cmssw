#include "CondTools/SiStrip/plugins/SiStripApvGainFromFileBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <sstream>


struct clean_up {
   void operator()(SiStripApvGainFromFileBuilder::Gain* el) {
       if (el!=0) {
           el->clear();
           delete el;
           el = 0;
       }
   }
} CleanUp;

SiStripApvGainFromFileBuilder::~SiStripApvGainFromFileBuilder() {
    for_each(gains_.begin(), gains_.end(), CleanUp);
    for_each(negative_gains_.begin(), negative_gains_.end(), CleanUp);
    for_each(null_gains_.begin(), null_gains_.end(), CleanUp);
}

SiStripApvGainFromFileBuilder::SiStripApvGainFromFileBuilder( const edm::ParameterSet& iConfig ):
  gfp_(iConfig.getUntrackedParameter<edm::FileInPath>("geoFile",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  tfp_(iConfig.getUntrackedParameter<edm::FileInPath>("tickFile",edm::FileInPath("CondTools/SiStrip/data/tickheight.txt"))),
  gainThreshold_(iConfig.getUntrackedParameter<double>("gainThreshold",0.)), 
  dummyAPVGain_(iConfig.getUntrackedParameter<double>("dummyAPVGain",690./640.)),
  putDummyIntoUncabled_(iConfig.getUntrackedParameter<bool>("putDummyIntoUncabled",0)),
  putDummyIntoUnscanned_(iConfig.getUntrackedParameter<bool>("putDummyIntoUnscanned",0)),
  putDummyIntoOffChannels_(iConfig.getUntrackedParameter<bool>("putDummyIntoOffChannels",0.)),
  putDummyIntoBadChannels_(iConfig.getUntrackedParameter<bool>("putDummyIntoBadChannels",0.)),
  outputMaps_(iConfig.getUntrackedParameter<bool>("outputMaps",0)),
  outputSummary_(iConfig.getUntrackedParameter<bool>("outputSummary",0)){}

void SiStripApvGainFromFileBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup){

  //unsigned int run=evt.id().run();

  edm::LogInfo("Workflow") << "@SUB=analyze" <<  "Insert SiStripApvGain Data." << std::endl;
  this->read_tickmark();

  if (outputMaps_) {
    try {
      this->output_maps(&gains_,"tickmark_heights");
      this->output_maps(&negative_gains_,"negative_tickmark");
      this->output_maps(&null_gains_,"zero_tickmark");
    } catch ( std::exception& e ) {
      std::cerr << e.what() << std::endl;
    }
  }

  // Retrieve the SiStripDetCabling description
  iSetup.get<SiStripDetCablingRcd>().get( detCabling_ );

  // APV gain record to be filled with gains and delivered into the database.
  SiStripApvGain* obj = new SiStripApvGain();

  SiStripDetInfoFileReader reader(gfp_.fullPath());
  
  const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > DetInfos  = reader.getAllData();

  LogTrace("Debug") << "  det id  |APVOF| CON |APVON| FED |FEDCH|i2cAd|tickGain|" << std::endl;

  for(std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator it = DetInfos.begin(); it != DetInfos.end(); it++){    
    // check if det id is correct and if it is actually cabled in the detector
    if( it->first==0 || it->first==0xFFFFFFFF ) {
      edm::LogError("DetIdNotGood") << "@SUB=analyze" << "Wrong det id: " << it->first 
                                    << "  ... neglecting!" << std::endl;
      continue;
    }


    // For the cabled det_id retrieve the number of APV connected
    // to the module with the FED cabling
    uint16_t nAPVs = 0;
    const std::vector<const FedChannelConnection*> connection = detCabling_->getConnections(it->first);
    for (unsigned int ca = 0; ca<connection.size(); ca++) {
      if ( connection[ca]!=0 )  {
        nAPVs+=( connection[ca] )->nApvs();
        break;
      }
    }

    // check consistency among FED cabling and ideal cabling, exit on error
    if (connection.size()!=0 && nAPVs != (uint16_t)it->second.nApvs) {
      edm::LogError("SiStripCablingError") << "@SUB=analyze"
                     << "det id " << it->first << ": APV number from FedCabling (" << nAPVs
                     << ") is different from the APV number retrieved from the ideal cabling ("
                     << it->second.nApvs << ")." << std::endl;
      throw("Inconsistency on the number of APVs.");
    }


    // eventually separate the processing for the module that are fully
    // uncabled. This is worth only if we decide not tu put the record
    // in the DB for the uncabled det_id.
    //if( !detCabling_->IsConnected(it->first) ) {
    //
    //  continue;
    //}
 

    //Gather the APV online id
    std::vector<std::pair<int,float>> tickmark_for_detId(it->second.nApvs,std::pair<int,float>(-1,999999.));
    for (unsigned int ca = 0; ca<connection.size(); ca++) {
      if( connection[ca]!=0 ) {
        uint16_t id1 = (connection[ca])->i2cAddr( 0 )%32;
        uint16_t id2 = (connection[ca])->i2cAddr( 1 )%32;
        tickmark_for_detId[ online2offline(id1,it->second.nApvs) ].first = id1;
        tickmark_for_detId[ online2offline(id2,it->second.nApvs) ].first = id2;
      }
    }
    gain_from_maps(it->first, it->second.nApvs, tickmark_for_detId);
    std::vector<float> theSiStripVector;

    // Fill the gain in the DB object, apply the logic for the dummy values
    for(unsigned short j=0; j<it->second.nApvs; j++){
      Summary summary;
      summary.det_id = it->first;
      summary.offlineAPV_id = j;
      summary.onlineAPV_id  = tickmark_for_detId.at(j).first;
      summary.is_connected = false;
      summary.FED_id = -1;
      summary.FED_ch = -1;
      summary.i2cAdd = -1;

      for (unsigned int ca = 0; ca<connection.size(); ca++) { 
        if( connection[ca]!=0 && (connection[ca])->i2cAddr( j%2 )%32==summary.onlineAPV_id ) {
          summary.is_connected = (connection[ca])->isConnected();
          summary.FED_id = (connection[ca])->fedId();
          summary.FED_ch = (connection[ca])->fedCh();
          summary.i2cAdd = (connection[ca])->i2cAddr( j%2 );
        }
      }

      try {
        float gain = tickmark_for_detId[j].second;
        summary.gain_from_scan = gain;
        LogTrace("Debug") << it->first << "  " << std::setw(3) << j << "   "
                          << std::setw(3) << connection.size() << "   "
                          << std::setw(3) << summary.onlineAPV_id << "    "
                          << std::setw(3) << summary.FED_id << "   "
                          << std::setw(3) << summary.FED_ch << "   "
                          << std::setw(3) << summary.i2cAdd << "   "
                          << std::setw(7) << summary.gain_from_scan << std::endl;

        if ( gain!=999999. ) {
          summary.is_scanned = true;
          if( gain>gainThreshold_ ) {
            summary.gain_in_db = gain;
            if ( !summary.is_connected ) ex_summary_.push_back( summary );
            else summary_.push_back( summary );
          } else {
            if ( gain==0. ) {
              if( putDummyIntoOffChannels_ ) summary.gain_in_db = dummyAPVGain_;
              else                           summary.gain_in_db = 0.;
              ex_summary_.push_back( summary );
            }
            if (gain<0. )   {
              if( putDummyIntoBadChannels_ ) summary.gain_in_db = dummyAPVGain_;
              else                           summary.gain_in_db = 0.;
              ex_summary_.push_back( summary );
            }
          } 
        } else {
          summary.is_scanned = false;
          if( !summary.is_connected ) {
            if( putDummyIntoUncabled_ ) summary.gain_in_db = dummyAPVGain_;
            else                        summary.gain_in_db = 0.;
          } else {
            if( putDummyIntoUnscanned_ ) summary.gain_in_db = dummyAPVGain_;
            else                         summary.gain_in_db = 0.;
          }
          ex_summary_.push_back( summary );
        }

        theSiStripVector.push_back( summary.gain_in_db );

      } catch ( std::exception& e ) {
        std::cerr << e.what() << std::endl;
        edm::LogError("MappingError") << "@SUB=analyze" << "Job end prematurely." << std::endl;
        return;
      }
    }
  	    
    SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( ! obj->put(it->first,range) )
      edm::LogError("IndexError")<< "@SUB=analyze" << "detid already exists." << std::endl;
  }

  if( outputSummary_ ) output_summary();

  
  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  
  if( mydbservice.isAvailable() ){
    if( mydbservice->isNewTagRequest("SiStripApvGainRcd") ){
      mydbservice->createNewIOV<SiStripApvGain>(obj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripApvGainRcd");      
    } else {
      mydbservice->appendSinceTime<SiStripApvGain>(obj,mydbservice->currentTime(),"SiStripApvGainRcd");      
    }
  }else{
    edm::LogError("DBServiceNotAvailable") << "@SUB=analyze" << "DB Service is unavailable" << std::endl;
  }
}
     
void
SiStripApvGainFromFileBuilder::read_tickmark() {
  // Connect file for input
  const char* filename = tfp_.fullPath().c_str();
  std::ifstream thickmark_heights ( filename );

  if ( !thickmark_heights.is_open() ) {
    edm::LogError("FileNotFound") << "@SUB=read_ticlmark" << "File with thickmark height file " 
                                  << filename << " cannot be opened!" << std::endl;
    return;
  }

  // clear internal maps
  for_each(gains_.begin(), gains_.end(), CleanUp);
  for_each(negative_gains_.begin(), negative_gains_.end(), CleanUp);
  for_each(null_gains_.begin(), null_gains_.end(), CleanUp);

  // read file and fill internal map
  uint32_t det_id = 0;
  uint32_t APV_id = 0;
  float    tick_h = 0.;

  int count = -1;

  for ( ; ; ) {
    count++;
    thickmark_heights >> det_id >> APV_id >> tick_h;

    if ( ! (thickmark_heights.eof() || thickmark_heights.fail()) ) {

      if (count==0) {
        LogTrace("Debug") << "Reading " << filename << " for gathering the tickmark heights" << std::endl;
        LogTrace("Debug") << "|  Det Id   |  APV Id  |  Tickmark" << std::endl;
        LogTrace("Debug") << "+-----------+----------+----------" << std::endl;
      }
      LogTrace("Debug") << std::setw(11) << det_id
                        << std::setw(8)  << APV_id 
                        << std::setw(14) << tick_h << std::endl;

      // retrieve the map corresponding to the gain collection
      Gain* map = 0;
      if ( tick_h>0. ) {
        map = get_map(&gains_, APV_id); 
      } else if ( tick_h<0. ) {
        map = get_map(&negative_gains_, APV_id);
      } else if ( tick_h==0.) {
        map = get_map(&null_gains_, APV_id);
      }

      // insert the gain value in the map
      std::pair<Gain::iterator,bool> ret = map->insert( std::pair<uint32_t,float>(det_id,tick_h) );
      if ( ret.second == false ) {
        edm::LogError("MapError") << "@SUB=read_tickmark" << "Cannot not insert gain for detector id "
                                  << det_id << " into the internal map: detector id already in the map." << std::endl;
      }

    } else if (thickmark_heights.eof()) {
      edm::LogInfo("Workflow") << "@SUB=read_tickmark" << "EOF of " << filename << " reached." << std::endl;
      break;
    } else if (thickmark_heights.fail()) {
      edm::LogError("FileiReadError") << "@SUB=read_tickmark" << "error while reading " << filename << std::endl;
      break;
    }
  }

  thickmark_heights.close();
}

SiStripApvGainFromFileBuilder::Gain* 
SiStripApvGainFromFileBuilder::get_map(std::vector<Gain*>* maps, int onlineAPV_id) {
    Gain* map=0;
    if( onlineAPV_id<0 || onlineAPV_id>5 ) return map;

    try {
        map = maps->at(onlineAPV_id);
    } catch (std::out_of_range) {
        if ( maps->size()<static_cast<unsigned int>(onlineAPV_id) ) maps->resize( onlineAPV_id );
        maps->insert( maps->begin()+onlineAPV_id, new Gain() );
        map = (*maps)[onlineAPV_id];
    }

    if ( map==0 ) {
        (*maps)[onlineAPV_id] = new Gain();
        map = (*maps)[onlineAPV_id];
    }

    return map;
}


void SiStripApvGainFromFileBuilder::output_maps(std::vector<Gain*>* maps, const char* basename) const {
    for (unsigned int APV=0; APV<maps->size(); APV++) {
        Gain* map = (*maps)[APV];
        if ( map!=0 ) {
            // open output file
            std::stringstream name;
            name << basename << "_APV" << APV << ".txt";
            std::ofstream* ofile = new std::ofstream(name.str(), std::ofstream::trunc);
            if ( !ofile->is_open() ) throw "cannot open output file!";
            for(Gain::const_iterator el = map->begin(); el!=map->end(); el++) {
                (*ofile) << (*el).first << "    " << (*el).second << std::endl;
            }
            ofile->close();
            delete ofile;
        }
    }
}


void SiStripApvGainFromFileBuilder::output_summary() const {
    std::ofstream* ofile = new std::ofstream("SiStripApvGainSummary.txt",std::ofstream::trunc);
    (*ofile) << "  det id  | APV | isConnected | FED |FEDCH|i2cAd|APVON| is_scanned |tickGain|gainInDB|" << std::endl;
    (*ofile) << "----------+-----+-------------+-----+-----+-----+-----+------------+--------+--------+" << std::endl;
    for (unsigned int s=0; s<summary_.size(); s++) {
        Summary summary = summary_[s];

        std::stringstream line;

        format_summary(line, summary);

        (*ofile) << line.str() << std::endl;
    }
    ofile->close();
    delete ofile;

    ofile = new std::ofstream("SiStripApvGainExceptionSummary.txt",std::ofstream::trunc);
    (*ofile) << "  det id  | APV | isConnected | FED |FEDCH|i2cAd|APVON| is_scanned |tickGain|gainInDB|" << std::endl;
    (*ofile) << "----------+-----+-------------+-----+-----+-----+-----+------------+--------+--------+" << std::endl;
    for (unsigned int s=0; s<ex_summary_.size(); s++) {
        Summary summary = ex_summary_[s];

        std::stringstream line;

        format_summary(line, summary);

        (*ofile) << line.str() << std::endl;
    }
    ofile->close();
    delete ofile;
}


void SiStripApvGainFromFileBuilder::format_summary(std::stringstream& line, Summary summary) const {
    std::string conn = (summary.is_connected)? "CONNECTED"    : "NOT_CONNECTED";
    std::string scan = (summary.is_scanned)?   "IN_SCAN" : "NOT_IN_SCAN";

    line << summary.det_id << "  " << std::setw(3) << summary.offlineAPV_id << "  "
         << std::setw(13)  << conn << "   " << std::setw(3) << summary.FED_id
         << "   " << std::setw(3)   << summary.FED_ch << "   " << std::setw(3)
         << summary.i2cAdd << "  " << std::setw(3) << summary.onlineAPV_id << "   "
         << std::setw(11)  << scan << "  "
         << std::setw(7) << summary.gain_from_scan << "  " << std::setw(7)
         << summary.gain_in_db;
}


bool SiStripApvGainFromFileBuilder::gain_from_maps(uint32_t det_id, int onlineAPV_id, float& gain) {

    Gain* map = 0;

    // search det_id and APV in the good scan map
    map = get_map(&gains_, onlineAPV_id);
    if( map!=0 ) {
      Gain::const_iterator el = map->find( det_id );
      if ( el!=map->end() ) {
        gain = el->second;
        return true;
      }
    }

    // search det_id and APV in the zero gain scan map
    map = get_map(&negative_gains_, onlineAPV_id);
    if( map!=0 ) {
      Gain::const_iterator el = map->find( det_id );
      if ( el!=map->end() ) {
        gain = el->second;
        return true;
      }
    }

    //search det_id and APV in the negative gain scan map
    map = get_map(&null_gains_, onlineAPV_id);
    if( map!=0 ) {
      Gain::const_iterator el = map->find( det_id );
      if ( el!=map->end() ) {
        gain = el->second;
        return true;
      }
    }

    return false; 
}

void SiStripApvGainFromFileBuilder::gain_from_maps(uint32_t det_id, uint16_t totalAPVs, 
                                                   std::vector<std::pair<int,float>>& gain) const {
  std::stringstream ex_msg;
  ex_msg << "two APVs with the same online id for det id " << det_id 
         << ". Please check the tick mark file or the read_tickmark routine." << std::endl;

  for (unsigned int i=0; i<6; i++) {
    int offlineAPV_id = online2offline(i, totalAPVs);
    try {
      Gain* map =  gains_.at(i);
      if( map!=0 ) {
        Gain::const_iterator el = map->find( det_id );
        if ( el!=map->end() ) {
	  if( gain[offlineAPV_id].second!=999999. ) throw( ex_msg.str() );
          gain[offlineAPV_id].second=el->second;
        }
      }
    } catch (std::out_of_range) {
      // nothing to do, just pass over
    } 

    try {
      Gain* map =  negative_gains_.at(i);
      if( map!=0 ) {
        Gain::const_iterator el = map->find( det_id );
        if ( el!=map->end() ) {
          if( gain[offlineAPV_id].second!=999999. ) throw( ex_msg.str() );
          gain[offlineAPV_id].second=el->second;
        }
      }
    } catch (std::out_of_range) {
      // nothing to do, just pass over
    }

    try {
      Gain* map =  null_gains_.at(i);
      if( map!=0 ) {
        Gain::const_iterator el = map->find( det_id );
        if ( el!=map->end() ) {
          if( gain[offlineAPV_id].second!=999999. ) throw( ex_msg.str() );
          gain[offlineAPV_id].second=el->second;
        }
      }
    } catch (std::out_of_range) {
      // nothing to do, just pass over
    }
  }
}

int SiStripApvGainFromFileBuilder::online2offline(uint16_t onlineAPV_id, uint16_t totalAPVs) const {
    return (onlineAPV_id>=totalAPVs)? onlineAPV_id -2 : onlineAPV_id;
}
