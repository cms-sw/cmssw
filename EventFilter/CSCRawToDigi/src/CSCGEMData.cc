#include "EventFilter/CSCRawToDigi/interface/CSCGEMData.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <cstdio>
#include <strings.h>  // for bzero
#include <cstring>

#ifdef LOCAL_UNPACK
bool CSCGEMData::debug = false;
#else
std::atomic<bool> CSCGEMData::debug{false};
#endif

CSCGEMData::CSCGEMData(int ntbins, int gems_fibers) : ntbins_(ntbins), size_(0) {
  theData[0] = 0x6C04;
  ntbins_ = ntbins;
  gems_enabled_ = gems_fibers;
  ngems_ = 0;
  /* Not implemented in the firmware yet */
  /*
  for (int i=0; i<4; i++)
    ngems_ += (gems_fibers>>i) & 0x1;
  */
  ngems_ = 4;
  size_ = 2 + ntbins_ * ngems_ * 4;
  theData[size_ - 1] = 0x6D04;
}

CSCGEMData::CSCGEMData(const unsigned short* buf, int length, int gems_fibers) : size_(length) {
  size_ = length;
  ngems_ = 0;
  /* Not implemented in the firmware yet */
  /*
  for (int i=0; i<4; i++)
    ngems_ += (gems_fibers>>i) & 0x1;
  */
  ngems_ = 4;
  ntbins_ = (size_ - 2) / (4 * ngems_);
  gems_enabled_ = gems_fibers;
  memcpy(theData, buf, size_ * 2);
}

void CSCGEMData::print() const {
  LogTrace("CSCGEMData|CSCRawToDigi") << "CSCGEMData.Print";
  for (int line = 0; line < ((size_)); ++line) {
    LogTrace("CSCGEMData|CSCRawToDigi") << std::hex << theData[line];
  }
}

int CSCGEMData::getPartitionNumber(int addr, int npads) const { return addr / (npads - 1); }

int CSCGEMData::getPartitionStripNumber(int address, int nPads, int etaPart) const {
  return address - (nPads * etaPart);
}

std::vector<GEMPadDigiCluster> CSCGEMData::digis(int gem_chamber) const {
  /// GEM data format v2
  std::vector<GEMPadDigiCluster> result;
  result.clear();
  int nPads = 192;  // From geometry
  int maxClusters = 4;
  int nGEMs = 4;
  // nGEMs = ngems_; // based on enabled fibers. not implemented in the firmware yet
  for (int i = 0; i < ntbins_; i++) {
    for (int fiber = 0; fiber < nGEMs; fiber++) {
      for (int cluster = 0; cluster < maxClusters; cluster++) {
        int dataAddr = 1 + (i * nGEMs + fiber) * maxClusters + cluster;
        int gem_layer = (theData[dataAddr] >> 14) & 0x1;  // gemA=0 or gemB=1
        if (gem_layer == gem_chamber) {
          int cl_word = theData[dataAddr] & 0x3fff;
          int pad = theData[dataAddr] & 0xff;
          int eta = (theData[dataAddr] >> 8) & 0x7;
          int cluster_size = (theData[dataAddr] >> 11) & 0x7;
          if (pad < nPads) {
            int padInPart = eta * nPads + pad;
            if (debug)
              LogTrace("CSCGEMData|CSCRawToDigi")
                  << "GEMlayer" << gem_layer << " cl_word" << dataAddr << ": 0x" << std::hex << cl_word << std::dec
                  << " tbin: " << i << " fiber#: " << (fiber + 1) << " cluster#: " << (cluster + 1)
                  << " padInPart: " << padInPart << " pad: " << pad << " eta: " << eta
                  << " cluster_size: " << cluster_size << std::endl;
            std::vector<short unsigned int> pads;
            for (int iP = 0; iP <= cluster_size; ++iP)
              pads.push_back(padInPart + iP);
            GEMPadDigiCluster pad_cluster(pads, i);
            result.push_back(pad_cluster);
          }
        }
      }
    }
  }

  /// GEM data format v1
  /// It is not used in the production
  /// Keeping this commented code just for v1 unpacking algo reference
  /*
  std::vector<GEMPadDigiCluster> result;
  result.clear();
  int nPads = 192; // From geometry
  int maxAddr = 1536;
  int nGEMs = 2;
  int maxClusters = 8;
  // std::cout << std::hex << "markers " << theData[0] << ": " << theData[size_-1] << std::dec << " size: " << size_ << std::endl;
  for (int i=0; i<ntbins_; i++)
    {
      for (int gem=0; gem<nGEMs; gem++)
        {
          if (gem==gem_chamber) // Return only digis for specified GEM chamber
            {
              for (int TMBCluster=0; TMBCluster<maxClusters; TMBCluster++)
                {
                  int dataAddr = 1 + (i*nGEMs+gem)*maxClusters + TMBCluster;
                  int address = theData[dataAddr] & 0x7ff;
                  // std::cout << dataAddr << ": " << address <<std::endl;
                  int nExtraPads = (theData[dataAddr] >>11)&0x7;
                  if (address<maxAddr)
                    {
                      int  etaPart   = getPartitionNumber(address,nPads);
                      int  padInPart = getPartitionStripNumber(address,nPads,etaPart);
                      vector<short unsigned int> pads;
                      for(int iP = 0; iP <= nExtraPads; ++iP)
                        pads.push_back(padInPart + iP );
                      GEMPadDigiCluster cluster ( pads, i);
                      result.push_back(cluster);
                    }
                }
            }
        }
    }
  */
  return result;
}

std::vector<GEMPadDigiCluster> CSCGEMData::etaDigis(int gem_chamber, int eta_roll) const {
  /// GEM data format v2
  std::vector<GEMPadDigiCluster> result;
  result.clear();
  int nPads = 192;  // From geometry
  int maxClusters = 4;
  int nGEMs = 4;
  // nGEMs = ngems_; // based on enabled fibers. not implemented in the firmware yet
  for (int i = 0; i < ntbins_; i++) {
    for (int fiber = 0; fiber < nGEMs; fiber++) {
      for (int cluster = 0; cluster < maxClusters; cluster++) {
        int dataAddr = 1 + (i * nGEMs + fiber) * maxClusters + cluster;
        int gem_layer = (theData[dataAddr] >> 14) & 0x1;  // gemA=0 or gemB=1
        if (gem_layer == gem_chamber) {
          int cl_word = theData[dataAddr] & 0x3fff;
          int pad = theData[dataAddr] & 0xff;
          int eta = (theData[dataAddr] >> 8) & 0x7;
          int cluster_size = (theData[dataAddr] >> 11) & 0x7;
          if ((pad < nPads) && (eta == eta_roll)) {
            int padInPart = pad;
            if (debug)
              LogTrace("CSCGEMData|CSCRawToDigi")
                  << "GEMlayer" << gem_layer << " cl_word" << dataAddr << ": 0x" << std::hex << cl_word << std::dec
                  << " tbin: " << i << " fiber#: " << (fiber + 1) << " cluster#: " << (cluster + 1)
                  << " padInPart: " << padInPart << " pad: " << pad << " eta: " << eta
                  << " cluster_size: " << cluster_size << std::endl;
            std::vector<short unsigned int> pads;
            for (int iP = 0; iP <= cluster_size; ++iP)
              pads.push_back(padInPart + iP);
            GEMPadDigiCluster pad_cluster(pads, i);
            result.push_back(pad_cluster);
          }
        }
      }
    }
  }
  return result;
}
