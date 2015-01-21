#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "CondFormats/DataRecord/interface/Phase2TrackerCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/DetSetVector.h"

namespace Phase2Tracker
{
  const int MAX_NP = 31; // max P clusters per concentrator i.e. per side
  const int MAX_NS = 31; // same for S clusters

  std::pair<int,int> SortExpandAndLimitClusters(std::vector<stackedDigi> & digis, int max_ns, int max_np)
  {
    std::vector<stackedDigi> processed;
    // number of clusters allowed : P-left, P-right, S-left, S-right 
    int roomleft[4] = {max_ns,max_ns,max_np,max_np};
    // fill left and right vectors, expand big clusters
    for(auto dig = digis.begin(); dig < digis.end(); dig++)
    {
      if(dig->getSizeX() > 8)
      {
        int pos = dig->getDigiX();
        int end = pos + dig->getSizeX();
        while (pos < end)
        {
          // compute size of cluster to add
          int isize = std::min(8,end-pos);
          // add cluster
          stackedDigi ndig(*dig);
          ndig.setPosSizeX(pos,isize); 
          if(roomleft[ndig.getSideType()] > 0) 
          {
            processed.push_back(ndig); 
            roomleft[ndig.getSideType()] -= 1;
          }
          pos += isize;
        } 
      }
      else
      {
        if(roomleft[dig->getSideType()] > 0) 
        { 
          processed.push_back(*dig); 
          roomleft[dig->getSideType()] -= 1;
        }
      }
    }
    // Sort vector
    std::sort(processed.begin(),processed.end());
    // replace input vector
    digis.swap(processed);
    // return number of S and P clusters
    return std::make_pair(2*max_ns - roomleft[2] - roomleft[3], 2*max_np - roomleft[0] - roomleft[1]); 
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
    // testing better way to gather stuff:
    Phase2TrackerCabling::cabling::const_iterator iconn = conns.begin(), end = conns.end(), icon2;
    while(iconn != end)
    {
      unsigned int fedid = (*iconn)->getCh().first;
      for (icon2 = iconn; icon2 != end && (*icon2)->getCh().first == fedid; icon2++)
      {
        int detid = (*icon2)->getDetid();
        edmNew::DetSetVector<SiPixelCluster>::const_iterator  digis;
        digis = digishandle_->find(detid);
        if (digis != digishandle_->end())
        {
          digis_t.push_back(*digis);
          festatus[(*icon2)->getCh().second] = true;
        }
        // store digis from other module plane
        digis = digishandle_->find(stackMap_[detid]);
        if (digis != digishandle_->end())
        {
          digis_t.push_back(*digis);
          festatus[(*icon2)->getCh().second] = true;
        }
      }
      // save buffer
      FedHeader_.setFrontendStatus(festatus);
      // write digis to buffer
      std::vector<uint64_t> fedbuffer = makeBuffer(digis_t);
      FEDRawData& frd = rcollection->FEDData(fedid);
      int size = fedbuffer.size()*8;
      uint8_t arrtemp[size];
      vec_to_array(fedbuffer,arrtemp);
      frd.resize(size);
      memcpy(frd.data(),arrtemp,size);
      festatus.assign(72,false);
      digis_t.clear();
      // advance connections pointer
      iconn = icon2;
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
      int detid = idigi->detId();
      if(stackMap_[detid] < 0) { detid = - stackMap_[detid]; }
      moduletype = cabling_->findDetid(detid).getModuleType();
      // container for digis, to be sorted afterwards
      std::vector<stackedDigi> digs_mod;
      edmNew::DetSet<SiPixelCluster>::const_iterator it;
      // pair modules if there are digis for both
      if(stackMap_[idigi->detId()] > 0)
      {
        // digis for inner plane (P in case of PS)
	if( (idigi+1) != digis.end() and (int)(idigi+1)->detId() == stackMap_[idigi->detId()])
        {
          // next digi is the corresponding outer plane : join them
          for (it = idigi->begin(); it != idigi->end(); it++)
          {
            digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
          }
          idigi++;
          for (it = idigi->begin(); it != idigi->end(); it++)
          {
            digs_mod.push_back(stackedDigi(it,LAYER_OUTER,moduletype));
          }
        }
        else
        {
          // next digi is from another module, only use this one
          for (it = idigi->begin(); it != idigi->end(); it++)
          {
            digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
          }
        }
      }
      else
      {
        // digis from outer plane (S in case of PS) 
        for (it = idigi->begin(); it != idigi->end(); it++)
        {
          digs_mod.push_back(stackedDigi(it,LAYER_OUTER,moduletype));
        }
      }
      // here we should:
      // - sort all digis
      std::sort(digs_mod.begin(),digs_mod.end());
      // - divide big clusters into 8-strips parts
      // - count digis on each side/layer (concentrator)
      // - remove extra digis
      std::pair<int,int> nums = SortExpandAndLimitClusters(digs_mod, MAX_NS, MAX_NP);
      // - write appropriate header
      writeFeHeaderSparsified(fedbuffer, bitindex, moduletype, nums.second, nums.first);
      // - write the digis
      std::vector<stackedDigi>::iterator its;
      for(its = digs_mod.begin(); its != digs_mod.end(); its++)
      {
        writeCluster(fedbuffer, bitindex, *its);
      }

    } // end idigi (FE) loop
    // add daq trailer 
    fedbuffer.push_back(*(uint64_t*)FedDaqTrailer_.data());
    return fedbuffer;
  }

  void Phase2TrackerDigiToRaw::writeFeHeaderSparsified(std::vector<uint64_t> & buffer, uint64_t & bitpointer, int modtype, int np, int ns)
  {
    uint8_t  length = 7;
    uint16_t header = (uint16_t)modtype & 0x01;
    // module type switch
    if (modtype == 1)
    {
      header |= ((uint16_t)np & 0x3F)<<1;
      header |= ((uint16_t)ns & 0x3F)<<7;
      length = 13;
    }
    else 
    {
      header |= ((uint16_t)ns & 0x3F)<<1;
    }
    write_n_at_m(buffer,length,bitpointer,header);
    bitpointer += length;
  }

  // layer = 0 for inner, 1 for outer (-1 if irrelevant)
  void Phase2TrackerDigiToRaw::writeCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, stackedDigi digi)
  {
    if(digi.getModuleType() == 0)
    {
      // 2S module
      writeSCluster(buffer,bitpointer,digi);
    } 
    else
    {
      // PS module
      if(digi.getLayer() == LAYER_INNER)
      {
        writePCluster(buffer,bitpointer,digi);
      }
      else
      {
        writeSCluster(buffer,bitpointer,digi);   
      }
    }
  }


  void Phase2TrackerDigiToRaw::writeSCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, stackedDigi digi)
  {
    // std::cout << "S " << digi.getRawX() << " " << digi.getSizeX() << std::endl; 
    std::cout << "S " << digi.getChipId() << " " << digi.getRawX() << " " << digi.getSizeX() << std::endl; 
    uint16_t scluster = (digi.getChipId() & 0x0F) << 11;
    scluster |= (digi.getRawX() & 0xFF) << 3;
    scluster |= ((digi.getSizeX()-1) & 0x07);
    write_n_at_m(buffer,15,bitpointer,scluster);
    bitpointer += 15;
  }

  void Phase2TrackerDigiToRaw::writePCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, stackedDigi digi)
  {
    // std::cout << "P " << digi.getRawX() << " " << digi.getSizeX() << " " << digi.getRawY() << std::endl; 
    std::cout << "P " << digi.getChipId() << " " << digi.getRawX() << " " << digi.getSizeX() << " " << digi.getRawY() << std::endl; 
    uint32_t pcluster = (digi.getChipId() & 0x0F) << 14;
    pcluster |= (digi.getRawX() & 0x7F) << 7;
    pcluster |= (digi.getRawY() & 0x0F) << 3;
    pcluster |= ((digi.getSizeX()-1) & 0x07);
    write_n_at_m(buffer,18,bitpointer,pcluster);
    bitpointer += 18;
  }
}
