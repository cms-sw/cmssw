#ifndef raw_FEDRawDataCollection_H
#define raw_FEDRawDataCollection_H

/** \class DaqRawDataCollection
 *  An EDCollection storing the raw data for all  FEDs in a Event.
 *  
 *  Reference: DaqPrototype/DaqPersistentData/interface/DaqFedOpaqueData.h
 *
 *  $Date: 2005/04/11 14:34:45 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - S. Argiro'
 */

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <vector>



namespace raw{

  class FEDRawDataCollection  {
  public:

    // obviously will need to get this parameter from somewhere
    static const unsigned int lastfedid = 1023;

    FEDRawDataCollection();

    virtual ~FEDRawDataCollection();
    
    /// retrieve data for fed @param fedid
    const FEDRawData&  FEDData(int fedid) const;

    /// retrieve data for fed @param fedid
    FEDRawData&        FEDData(int fedid);

  private:
    
    std::vector<FEDRawData> data_; ///< the raw data 

  };
}

#endif

