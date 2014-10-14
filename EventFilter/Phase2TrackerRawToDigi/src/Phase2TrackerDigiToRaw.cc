#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/DataRecord/interface/Phase2TrackerCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/DetSetVector.h"

namespace Phase2Tracker
{
  Phase2TrackerDigiToRaw::Phase2TrackerDigiToRaw(const Phase2TrackerCabling * cabling, const TrackerTopology * topo, edm::Handle< edmNew::DetSetVector<SiPixelCluster> > digis_handle, int mode):
    cabling_(cabling),
    topo_(topo),
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
        FedHeader_.setFrontendStatus(festatus);
        std::vector<uint64_t> fedbuffer = makeBuffer(digis_t);
        FEDRawData& frd = rcollection->FEDData(fedid_current);
        int size = fedbuffer.size()*8;
        uint8_t arrtemp[size];
        vec_to_array(fedbuffer,arrtemp);
        frd.resize(size);
        memcpy(frd.data(),arrtemp,size);
        /*
        std::cout << std::showbase << std::internal << std::setfill('0');
        for ( int i = 0;  i < size; i += 8)
        {
          uint64_t word  = read64(i,arrtemp);
          std::cout << std::hex << std::setw(18) << word << std::dec << std::endl;
        }
        */
        digis_t.clear();
        festatus.assign(72,false);
      }
      if (digis != digishandle_->end())
      {
        digis_t.push_back(*digis);
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
      // determine module type
      moduletype = cabling_->findDetid(idigi->detId()).getModuleType();
      int ns = 0; 
      int np = 0;
      int layer = 0;
      // detids should always come by two, odd then even (top/bottom or S/P)
      if (idigi->detId()%2)
      {
        if((idigi+1)->detId() - idigi->detId() != 1)
        {
          // the next module does not have a detid following this one : this should never happen
          std::cout << "the next module does not have a detid following this one : this should never happen : Skipping " << std::endl;
          continue;
        }
        if (moduletype == 0)
        {
          std::vector<std::pair<const SiPixelCluster*, int>> digs_mod;
          edmNew::DetSet<SiPixelCluster>::const_iterator it;
          for (it = idigi->begin(); it != idigi->end(); it++)
          {
            digs_mod.push_back(std::pair<const SiPixelCluster*, int>(&*it,it->x()*2+1));
          }
          idigi++;
          for (it = idigi->begin(); it != idigi->end(); it++)
          {
            digs_mod.push_back(std::pair<const SiPixelCluster*, int>(&*it,it->x()*2));
          }
          writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,0,np+ns);
          // sort digis of both channels according to strip
          std::sort(digs_mod.begin(),digs_mod.end(),second_sort());
          std::vector<std::pair<const SiPixelCluster*, int>>::iterator its;
          for(its = digs_mod.begin(); its != digs_mod.end(); its++)
          {
            layer = (its->second % 2 == 1) ? 0 : 1;
            writeCluster(fedbuffer, bitindex, its->first, moduletype, layer);
          }
        }
        else
        {
          ns = idigi->size();
          np = (idigi+1)->size();
          writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,np,ns);
          edmNew::DetSet<SiPixelCluster>::const_iterator it;
          for (it = (idigi+1)->begin(); it != (idigi+1)->end(); it++)
          {
            writeCluster(fedbuffer, bitindex, it, moduletype, 1); 
          }
          for (it = idigi->begin(); it != idigi->end(); it++)
          {
            writeCluster(fedbuffer, bitindex, it, moduletype, 0); 
          }
        }
      }
    }
    // add daq trailer 
    fedbuffer.push_back(*(uint64_t*)FedDaqTrailer_.data());
    return fedbuffer;
  }

  void Phase2TrackerDigiToRaw::writeFeHeaderSparsified(std::vector<uint64_t> & buffer, uint64_t & bitpointer, int modtype, int np, int ns)
  {
    uint8_t  length = 6;
    uint16_t header = (uint16_t)modtype & 0x01;
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

  void Phase2TrackerDigiToRaw::writeCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, const SiPixelCluster * digi, int moduletype, int layer)
  {
    if(moduletype == 0)
    {
      writeSCluster(buffer,bitpointer,digi,layer);
    } 
    else
    {
      if(layer == 0)
      {
        // layer set to -1 tells there's only one S layer (SonPS) 
        writeSCluster(buffer,bitpointer,digi,-1);   
      }
      else
      {
        writePCluster(buffer,bitpointer,digi);
      }
    }
  }


  void Phase2TrackerDigiToRaw::writeSCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, const SiPixelCluster * digi, int layer)
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
    std::pair<int,int> chipstrip = calcChipId(digi,-1);
    uint16_t pcluster = (chipstrip.first & 0x0F) << 14;
    pcluster |= (chipstrip.second & 0x7F) << 7;
    pcluster |= ((int)digi->y() & 0x0F) << 3;
    pcluster |= (digi->sizeX() & 0x07);
    write_n_at_m(buffer,18,bitpointer,pcluster);
    bitpointer += 18;
  }

  std::pair<int,int> Phase2TrackerDigiToRaw::calcChipId(const SiPixelCluster * digi, int layer)
  {
    int x = (int)digi->x();
    if(layer >= 0) { x *= 2; }
    if(layer == 0) { x += 1; }
    int chipid = x/254;
    int strip  = x%254;
    std::pair<int,int> id (chipid,strip);
    return id;
  }
}
