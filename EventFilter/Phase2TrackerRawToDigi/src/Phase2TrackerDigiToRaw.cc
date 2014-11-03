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
      std::cout << "XCHECK: saving buffer " << fedid << std::endl;
      // save FE status
      FedHeader_.setFrontendStatus(festatus);
      festatus.assign(72,false);
      // write digis to buffer
      std::vector<uint64_t> fedbuffer = makeBuffer(digis_t);
      FEDRawData& frd = rcollection->FEDData(fedid);
      int size = fedbuffer.size()*8;
      uint8_t arrtemp[size];
      vec_to_array(fedbuffer,arrtemp);
      frd.resize(size);
      memcpy(frd.data(),arrtemp,size);
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
      std::cout << "id : " << detid << " type: " << moduletype << std::endl; 
      // container for digis, to be sorted afterwards
      std::vector<stackedDigi> digs_mod;
      edmNew::DetSet<SiPixelCluster>::const_iterator it;
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
            for (it = idigi->begin(); it != idigi->end(); it++)
            {
              digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
            }
            idigi++;
            for (it = idigi->begin(); it != idigi->end(); it++)
            {
              digs_mod.push_back(stackedDigi(it,LAYER_OUTER,moduletype));
            }
            // do not overflow max number of clusters
            if((int)digs_mod.size() > MAX_NS) 
            { 
              digs_mod.resize(MAX_NS); 
            }
            writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,0,digs_mod.size());
          }
          else
          {
            // PS module
            np = idigi->size();
            ns = (idigi+1)->size();
            writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,np,ns);
            for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NP; it++)
            {
              digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
            }
            idigi++;
            for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NS; it++)
            {
              digs_mod.push_back(stackedDigi(it,LAYER_OUTER,moduletype));
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
          for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NP; it++)
          {
            digs_mod.push_back(stackedDigi(it,LAYER_INNER,moduletype));
          }
        }
      }
      else
      {
        // digis from outer plane (S in case of PS) 
        writeFeHeaderSparsified(fedbuffer,bitindex,moduletype,0,idigi->size());
        for (it = idigi->begin(); it != idigi->end() and std::distance(idigi->begin(),it) < MAX_NS; it++)
        {
          digs_mod.push_back(stackedDigi(it,LAYER_OUTER,moduletype));
        }
      }
      // sort according to strip, side and chip id
      std::sort(digs_mod.begin(),digs_mod.end());
      std::vector<stackedDigi>::iterator its;
      // write all clusters
      for(its = digs_mod.begin(); its != digs_mod.end(); its++)
      {
        writeCluster(fedbuffer, bitindex, *its);
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
    std::cout << "S " << digi.getLayer() << " digi x= " <<digi.getDigi()->minPixelRow()<< " chip: " << digi.getChipId() << " raw strip: " << digi.getRawX() << std::endl;
    uint16_t scluster = (digi.getChipId() & 0x0F) << 11;
    scluster |= (digi.getRawX() & 0xFF) << 3;
    scluster |= ((digi.getDigi()->sizeX()-1) & 0x07);
    write_n_at_m(buffer,15,bitpointer,scluster);
    bitpointer += 15;
  }

  void Phase2TrackerDigiToRaw::writePCluster(std::vector<uint64_t> & buffer, uint64_t & bitpointer, stackedDigi digi)
  {
    std::cout << "P digi x= " <<digi.getDigi()->minPixelRow()<< " chip: " << digi.getChipId() << " raw strip: " << digi.getRawX() << std::endl;
    uint32_t pcluster = (digi.getChipId() & 0x0F) << 14;
    pcluster |= (digi.getRawX() & 0x7F) << 7;
    pcluster |= (digi.getRawY() & 0x0F) << 3;
    pcluster |= ((digi.getDigi()->sizeX()-1) & 0x07);
    write_n_at_m(buffer,18,bitpointer,pcluster);
    bitpointer += 18;
  }
}
