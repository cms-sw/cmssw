#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/DataRecord/interface/Phase2TrackerCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/DetSetVector.h"

namespace Phase2Tracker
{
  const int MAX_NP = 31;
  const int MAX_NS = 31;

  Phase2TrackerDigiToRaw::Phase2TrackerDigiToRaw(const Phase2TrackerCabling * cabling, std::map<int,int> stackmap, edm::Handle< edmNew::DetSetVector<SiPixelCluster> > digis_handle, int mode):
    cabling_(cabling),
    stackMap_(stackmap),
    digishandle_(digis_handle),
    mode_(mode),
    FedDaqHeader_(0,0,0,DAQ_EVENT_TYPE_SIMULATED), // TODO : add L1ID
    FedDaqTrailer_(0,0)
  {
    FedHeader_.setDataFormatVersion(2);
    FedHeader_.setDebugMode(SUMMARY); 
    FedHeader_.setEventType((uint8_t)0x04);
  }

  void Phase2TrackerDigiToRaw::buildFEDBuffers(std::auto_ptr<FEDRawDataCollection>& rcollection)
  {
    // store digis for a given fedid
    std::vector<edmNew::DetSet<SiPixelCluster>> digis_t;
    // store active connections for a given fedid
    std::vector<bool> festatus (72,false);
    // iterate on all possible channels 
    Phase2TrackerCabling::cabling conns = cabling_->orderedConnections(0);
    Phase2TrackerCabling::cabling::const_iterator iconn;
    int fedid_current = -1;
    for(iconn = conns.begin(); iconn != conns.end(); iconn++) 
    {
      std::pair<unsigned int, unsigned int> fedch = (*iconn)->getCh();
      int detid = (*iconn)->getDetid();
      edmNew::DetSetVector<SiPixelCluster>::const_iterator  digis = digishandle_->find(detid);
      if(((int)fedch.first != fedid_current or (conns.end()-iconn)==1) and fedid_current >= 0)
      {
        std::cout << "XCHECK: starting buffer " << (int)fedch.first << std::endl;
        FedHeader_.setFrontendStatus(festatus);
        std::vector<uint64_t> fedbuffer = makeBuffer(digis_t);
        FEDRawData& frd = rcollection->FEDData(fedid_current);
        int size = fedbuffer.size()*8;
        uint8_t arrtemp[size];
        vec_to_array(fedbuffer,arrtemp);
        frd.resize(size);
        memcpy(frd.data(),arrtemp,size);
        digis_t.clear();
        festatus.assign(72,false);
      }
      if (digis != digishandle_->end())
      {
        digis_t.push_back(*digis);
        // add digis from second plane
        digis = digishandle_->find(detid+4);
        if (digis != digishandle_->end()) { digis_t.push_back(*digis); }
        // set fe status for header
        festatus[fedch.second] = true;
        fedid_current = (int)fedch.first;
      }
    }
  }

  std::vector<uint64_t> Phase2TrackerDigiToRaw::makeBuffer(std::vector<edmNew::DetSet<SiPixelCluster>> digis)
  {
    uint64_t bitindex = 0;
    int moduletype = 0;
    std::vector<uint64_t> fedbuffer;
    // add daq header
    fedbuffer.push_back(*(uint64_t*)FedDaqHeader_.data());
    bitindex += 64;
    // add fed header
    uint8_t* feh = FedHeader_.data();
    fedbuffer.push_back(*(uint64_t*)feh);
    fedbuffer.push_back(*(uint64_t*)(feh+8));
    bitindex += 128;
    // looping on detids
    std::vector<edmNew::DetSet<SiPixelCluster>>::const_iterator idigi;
    for (idigi = digis.begin(); idigi != digis.end(); idigi++ )
    {
      std::cout << "XCHECK: Current ID : " << idigi->detId() << std::endl;
      // determine module type
      moduletype = cabling_->findDetid(idigi->detId()).getModuleType();
      int ns = 0; 
      int np = 0;
      STACK_LAYER layer = LAYER_UNUSED;

      // pair modules if there are digis for both
      if(stackMap_[idigi->detId()] > 0)
      {
        // digis for inner plane (P in case of PS)
	if( (idigi+1) != digis.end() and (int)(idigi+1)->detId() == stackMap_[idigi->detId()])
        {
          // next digi is the corresponding outer plane : join them
          if (moduletype == 0)
          {
            // 2S module
            std::vector<std::pair<const SiPixelCluster*, int>> digs_mod;
            edmNew::DetSet<SiPixelCluster>::const_iterator it;
            for (it = idigi->begin(); it != idigi->end(); it++)
            {
              // associate with odd strip number
              digs_mod.push_back(std::pair<const SiPixelCluster*, int>(&*it,it->minPixelRow()*2+1));
            }
            idigi++;
            for (it = idigi->begin(); it != idigi->end(); it++)
            {
              // associate with even strip number
              digs_mod.push_back(std::pair<const SiPixelCluster*, int>(&*it,it->minPixelRow()*2));
            }
            // do not overflow max number of clusters
            if((int)digs_mod.size() > MAX_NS) { digs_mod.resize(MAX_NS); }
            writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,0,digs_mod.size());
            // sort digis of both channels according to strip
            std::sort(digs_mod.begin(),digs_mod.end(),second_sort());
            std::vector<std::pair<const SiPixelCluster*, int>>::iterator its;
            for(its = digs_mod.begin(); its != digs_mod.end(); its++)
            {
              layer = (its->second % 2 == 1) ? LAYER_INNER : LAYER_OUTER;
              writeCluster(fedbuffer, bitindex, its->first, moduletype, layer);
            }
          }
          else
          {
            // PS module
            np = idigi->size();
            ns = (idigi+1)->size();
            writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,np,ns);
            edmNew::DetSet<SiPixelCluster>::const_iterator it;
            for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NP; it++)
            {
              writeCluster(fedbuffer, bitindex, it, moduletype, LAYER_INNER); 
            }
            idigi++;
            for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NS; it++)
            {
              writeCluster(fedbuffer, bitindex, it, moduletype, LAYER_OUTER); 
            }
          }
        }
        else
        {
          // next digi is from another stack, process only the current INNER one
          if (moduletype == 0)
          {
            writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,0,idigi->size());
          }
          else
          {
            writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,idigi->size(),0);
          }
          edmNew::DetSet<SiPixelCluster>::const_iterator it;
          for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NP; it++)
          {
            writeCluster(fedbuffer, bitindex, it, moduletype, LAYER_INNER);
          }
        }
      }
      else
      {
        // digis from outer plane (S in case of PS) 
        writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,0,idigi->size());
        edmNew::DetSet<SiPixelCluster>::const_iterator it;
        for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NS; it++)
        {
          writeCluster(fedbuffer, bitindex, it, moduletype, LAYER_OUTER);
        }
      }
    } // end idigi loop
    // add daq trailer 
    fedbuffer.push_back(*(uint64_t*)FedDaqTrailer_.data());
    return fedbuffer;
  }

  void Phase2TrackerDigiToRaw::writeFeHeaderSparsified(std::vector<uint64_t> & buffer, uint64_t & bitpointer, int modtype, int np, int ns)
  {
    uint8_t  length = 6;
    uint16_t header = (uint16_t)modtype & 0x01;
    // np and ns are on 5 bits : trunk them
    if(np > MAX_NP) { np = MAX_NP; }
    if(ns > MAX_NS) { ns = MAX_NS; }
    // module type switch
    if (modtype == 1)
    {
      header |= ((uint16_t)np & 0x1F)<<1;
      header |= ((uint16_t)ns & 0x1F)<<6;
      length = 11;
    }
    else 
    {
      header |= ((uint16_t)ns & 0x1F)<<1;
    }
    write_n_at_m(buffer,length,bitpointer,header);
    bitpointer += length;
  }

  // layer = 0 for inner, 1 for outer (-1 if irrelevant)
  void Phase2TrackerDigiToRaw::writeCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, const SiPixelCluster * digi, int moduletype, STACK_LAYER layer)
  {
    std::cout << "XCHECK:" << (int)layer << " " << (int)moduletype << " " << digi->minPixelRow() << std::endl;
    if(moduletype == 0)
    {
      // 2S module
      writeSCluster(buffer,bitpointer,digi,layer);
    } 
    else
    {
      // PS module
      if(layer == 0)
      {
        writePCluster(buffer,bitpointer,digi);
      }
      else
      {
        // layer set to UNUSED tells there's only one S layer (SonPS) 
        writeSCluster(buffer,bitpointer,digi,LAYER_UNUSED);   
      }
    }
  }


  void Phase2TrackerDigiToRaw::writeSCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, const SiPixelCluster * digi, STACK_LAYER layer)
  {
    std::pair<int,int> chipstrip = calcChipId(digi, layer);
    uint16_t scluster = (chipstrip.first & 0x0F) << 11;
    scluster |= (chipstrip.second & 0xFF) << 3;
    scluster |= (digi->sizeX() & 0x07);
    write_n_at_m(buffer,15,bitpointer,scluster);
    bitpointer += 15;
  }

  void Phase2TrackerDigiToRaw::writePCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, const SiPixelCluster * digi)
  {
    std::pair<int,int> chipstrip = calcChipId(digi,LAYER_UNUSED);
    uint32_t pcluster = (chipstrip.first & 0x0F) << 14;
    pcluster |= (chipstrip.second & 0x7F) << 7;
    pcluster |= ((int)digi->y() & 0x0F) << 3;
    pcluster |= (digi->sizeX() & 0x07);
    write_n_at_m(buffer,18,bitpointer,pcluster);
    bitpointer += 18;
  }

  std::pair<int,int> Phase2TrackerDigiToRaw::calcChipId(const SiPixelCluster * digi, STACK_LAYER layer)
  {
    int x = digi->minPixelRow(); 
    int chipid, strip;
    if (layer < 0)
    {
      chipid = x/127;
      strip  = x%127;
    }
    else 
    {
      x *= 2;
      if(layer == 1) { x += 1; }
      chipid = x/254;
      strip  = x%254;
    }
    std::pair<int,int> id (chipid,strip);
    return id;
  }
}
