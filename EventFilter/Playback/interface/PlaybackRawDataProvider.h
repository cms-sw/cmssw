#ifndef PLAYBACKRAWDATAPROVIDER_H
#define PLAYBACKRAWDATAPROVIDER_H 1

class FEDRawDataCollection;

class PlaybackRawDataProvider 
{
public:
  //
  // construction/destruction
  //
  virtual ~PlaybackRawDataProvider(){}
  
  
  // provide cached fed collection (and run/evt number, if needed!)
  virtual FEDRawDataCollection* getFEDRawData() = 0;
  virtual FEDRawDataCollection* getFEDRawData(unsigned int& runNumber,
					      unsigned int& evtNumber) = 0;
  
  
  static PlaybackRawDataProvider* instance();
  
  
protected:

  static PlaybackRawDataProvider* instance_;
  
};


//
// implementation of inline functions
//

//______________________________________________________________________________
inline
PlaybackRawDataProvider* PlaybackRawDataProvider::instance()
{
  return instance_;
}


#endif
