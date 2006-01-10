#ifndef EventFilter_SiStripDigiToRaw_H
#define EventFilter_SiStripDigiToRaw_H

#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "Fed9UUtils.hh"

class SiStripReadoutCabling;
class StripDigiCollection;
class FEDRawDataCollection;

/**
   \class SiStripDigiToRaw
   \brief Takes a StripDigiCollection as input and creates a
   FEDRawDataCollection.
   \author M.Wingham, R.Bainbridge
*/
class SiStripDigiToRaw {
  
 public: // ----- public interface -----
  
  SiStripDigiToRaw( unsigned short verbosity );
  ~SiStripDigiToRaw();
  
  /** Takes a StripDigiCollection as input and creates a
      FEDRawDataCollection. */
  void createFedBuffers( edm::ESHandle<SiStripReadoutCabling>& cabling,
			 edm::Handle<StripDigiCollection>& digis,
			 std::auto_ptr<FEDRawDataCollection>& buffers );

  inline void fedReadoutPath( std::string readout_path );
  inline void fedReadoutMode( std::string readout_mode );

 private: // ----- private data members -----

  /** */
  unsigned short verbosity_;

  /** */
  std::string readoutPath_;
  /** */
  std::string readoutMode_;

  // some debug counters
  std::vector<unsigned int> position_;
  std::vector<unsigned int> landau_;
  unsigned long nFeds_;
  unsigned long nDigis_;

};

#endif // EventFilter_SiStripDigiToRaw_H

// inline methods 

inline void SiStripDigiToRaw::fedReadoutPath( std::string rpath ) { 
  readoutPath_ = rpath;
}

inline void SiStripDigiToRaw::fedReadoutMode( std::string rmode ) { 
  readoutMode_ = rmode;
}
