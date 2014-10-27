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

  std::map<std::string,int> calcChipId(const SiPixelCluster * digi, STACK_LAYER layer, int moduletype)
  {
    int x = digi->minPixelRow(); 
    int y = digi->minPixelCol();
    std::map<std::string,int> id;
    if (moduletype== 1)
    {
      if(layer == LAYER_INNER)
      {
        // PonPS
        id["chipid"] = x/PS_ROWS;
        // which side ?
        if (y >= PS_COLS/2) 
        { 
          id["chipid"] += 8; 
          id["rawy"] = y - PS_COLS/2;
        }
        id["rawx"] = x%PS_ROWS;
      }
      else
      {
        // SonPS
        id["chipid"] = x/PS_ROWS;
        if (y > 0) { id["chipid"] += 8; }
        id["rawx"] = x%PS_ROWS;
        id["rawy"] = y;
      }
    }
    else 
    {
      x *= 2;
      if(layer == LAYER_OUTER) { x += 1; }
      id["chipid"] = x/STRIPS_PER_CBC;
      // which side ? 
      if (y > 0) { id["chipid"] += 8; }
      id["rawx"] = x%STRIPS_PER_CBC;
      id["rawy"] = y;
    }
    return id;
  }

  class stackedDigi {
    public:
      stackedDigi() {}
      stackedDigi(const SiPixelCluster *, STACK_LAYER, int);
      ~stackedDigi() {}
      std::map<std::string,int> chipid() const ;
      bool operator<(stackedDigi) const ;
      inline const SiPixelCluster * getDigi() { return digi_; } 
      inline STACK_LAYER getLayer() { return layer_; } 
      inline int getModuleType() { return moduletype_; }
    private:
      const SiPixelCluster * digi_;
      STACK_LAYER layer_;
      int moduletype_;
  };

  stackedDigi::stackedDigi(const SiPixelCluster * digi, STACK_LAYER layer, int moduletype) : digi_(digi), layer_(layer), moduletype_(moduletype) {}

  std::map<std::string,int> stackedDigi::chipid() const
  {
    return calcChipId(digi_,layer_,moduletype_);
  }

  bool stackedDigi::operator<(const stackedDigi d2) const
  {
    std::map<std::string,int> c1 = chipid();
    std::map<std::string,int> c2 = d2.chipid();
    if (c1["chipid"] <  c2["chipid"]) { return true; }
    if (c1["chipid"] == c2["chipid"] and c1["rawx"] < c2["rawx"] ) { return true; } 
    return false;
  }

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
      if((int)fedch.first != fedid_current and fedid_current >= 0)
      {
        std::cout << "XCHECK: saving buffer " << fedid_current << std::endl;
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
      }
      fedid_current = (int)fedch.first;
      // save last fed
      if ((conns.end()-iconn)==1)
      {
        std::cout << "XCHECK: saving buffer " <<  fedid_current << std::endl;
        FedHeader_.setFrontendStatus(festatus);
        std::vector<uint64_t> fedbuffer = makeBuffer(digis_t);
        FEDRawData& frd = rcollection->FEDData(fedid_current);
        int size = fedbuffer.size()*8;
        uint8_t arrtemp[size];
        vec_to_array(fedbuffer,arrtemp);
        frd.resize(size);
        memcpy(frd.data(),arrtemp,size);
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
      // determine module type
      moduletype = cabling_->findDetid(idigi->detId()).getModuleType();
      int ns = 0; 
      int np = 0;

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
            std::vector<stackedDigi> digs_mod;
            edmNew::DetSet<SiPixelCluster>::const_iterator it;
            for (it = idigi->begin(); it != idigi->end(); it++)
            {
              // associate with odd strip number
              digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
            }
            idigi++;
            for (it = idigi->begin(); it != idigi->end(); it++)
            {
              // associate with even strip number
              digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
            }
            // do not overflow max number of clusters
            if((int)digs_mod.size() > MAX_NS) { digs_mod.resize(MAX_NS); }
            writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,0,digs_mod.size());
            // sort digis of both channels according to strip
            std::sort(digs_mod.begin(),digs_mod.end());
            std::vector<stackedDigi>::iterator its;
            for(its = digs_mod.begin(); its != digs_mod.end(); its++)
            {
              writeCluster(fedbuffer, bitindex, its->getDigi(), its->getModuleType(), its->getLayer());
            }
          }
          else
          {
            // PS module
            np = idigi->size();
            ns = (idigi+1)->size();
            writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,np,ns);
            std::vector<stackedDigi> digs_mod;
            edmNew::DetSet<SiPixelCluster>::const_iterator it;
            for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NP; it++)
            {
              digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
            }
            std::sort(digs_mod.begin(),digs_mod.end());
            std::vector<stackedDigi>::iterator its;
            for(its = digs_mod.begin(); its != digs_mod.end(); its++)
            {
              writeCluster(fedbuffer, bitindex, its->getDigi(), its->getModuleType(), its->getLayer()); 
            }
            idigi++;
            digs_mod.clear();
            for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NS; it++)
            {
              digs_mod.push_back(stackedDigi(it,LAYER_OUTER,moduletype));
            }
            std::sort(digs_mod.begin(),digs_mod.end());
            for(its = digs_mod.begin(); its != digs_mod.end(); its++)
            {
              writeCluster(fedbuffer, bitindex, its->getDigi(), its->getModuleType(), its->getLayer()); 
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
          std::vector<stackedDigi> digs_mod;
          edmNew::DetSet<SiPixelCluster>::const_iterator it;
          for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NP; it++)
          {
            digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
          }
          std::vector<stackedDigi>::iterator its;
          for(its = digs_mod.begin(); its != digs_mod.end(); its++)
          {
            writeCluster(fedbuffer, bitindex, its->getDigi(), its->getModuleType(), its->getLayer());
          }
        }
      }
      else
      {
        // digis from outer plane (S in case of PS) 
        writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,0,idigi->size());
        edmNew::DetSet<SiPixelCluster>::const_iterator it;
        std::vector<stackedDigi> digs_mod;
        for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NS; it++)
        {
          digs_mod.push_back(stackedDigi(it,LAYER_OUTER,moduletype));
        }
        std::vector<stackedDigi>::iterator its;
        for(its = digs_mod.begin(); its != digs_mod.end(); its++)
        {
          writeCluster(fedbuffer, bitindex, its->getDigi(), its->getModuleType(), its->getLayer());
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
    if(moduletype == 0)
    {
      // 2S module
      writeSCluster(buffer,bitpointer,digi,layer,moduletype);
    } 
    else
    {
      // PS module
      if(layer == LAYER_INNER)
      {
        writePCluster(buffer,bitpointer,digi);
      }
      else
      {
        writeSCluster(buffer,bitpointer,digi,LAYER_OUTER,moduletype);   
      }
    }
  }


  void Phase2TrackerDigiToRaw::writeSCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, const SiPixelCluster * digi, STACK_LAYER layer, int moduletype)
  {
    std::map<std::string,int> chipstrip = calcChipId(digi, layer,moduletype);
    std::cout << "S " << layer << " digi x= " <<digi->minPixelRow()<< " chip: " <<chipstrip["chipid"]<< " raw strip: " << chipstrip["rawx"] << std::endl;
    uint16_t scluster = (chipstrip["chipid"] & 0x0F) << 11;
    scluster |= (chipstrip["rawx"] & 0xFF) << 3;
    scluster |= ((digi->sizeX()-1) & 0x07);
    write_n_at_m(buffer,15,bitpointer,scluster);
    bitpointer += 15;
  }

  void Phase2TrackerDigiToRaw::writePCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, const SiPixelCluster * digi)
  {
    std::map<std::string,int> chipstrip = calcChipId(digi,LAYER_INNER,1);
    std::cout << "P digi x= " <<digi->minPixelRow()<< " chip: " <<chipstrip["chipid"]<< " raw strip: " << chipstrip["rawx"] << std::endl;
    uint32_t pcluster = (chipstrip["chipid"] & 0x0F) << 14;
    pcluster |= (chipstrip["rawx"] & 0x7F) << 7;
    pcluster |= (chipstrip["rawy"] & 0x0F) << 3;
    pcluster |= ((digi->sizeX()-1) & 0x07);
    write_n_at_m(buffer,18,bitpointer,pcluster);
    bitpointer += 18;
  }
}
