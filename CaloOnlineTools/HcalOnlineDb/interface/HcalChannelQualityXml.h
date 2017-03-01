#ifndef CaloOnlineTools_HcalOnlineDb_HcalChannelQualityXml_h
#define CaloOnlineTools_HcalOnlineDb_HcalChannelQualityXml_h
// -*- C++ -*-
//
// Package:     CaloOnlineTools/HcalOnlineDb
// Class  :     HcalChannelQualityXml
// 
/**\class HcalChannelQualityXml HcalChannelQualityXml.h CaloOnlineTools/HcalOnlineDb/interface/HcalChannelQualityXml.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Wed Jul 01 06:42:00 CDT 2009
//

#include <map>
#include <string>
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalChannelDataXml.h"

class HcalChannelQualityXml : public HcalChannelDataXml
{
  
 public:

  typedef struct _ChannelQuality{
    _ChannelQuality();
    int status;
    int onoff;
    std::string comment;
  } ChannelQuality;
      
  HcalChannelQualityXml();
  virtual ~HcalChannelQualityXml();
  
  // read ASCII stream with channel status word (cat ascii file)
  // and generate corresponding XML
  int makeXmlFromAsciiStream(int _runnumber,
			     int _iov_begin,
			     int _iov_end,
			     std::string _tag,
			     std::string _elements_comment,
			     std::string _base="hex"
			     );

  // get baseline channel status from a tag for a given IOV
  // and output it in the ASCII format to stdout
  int writeBaseLineFromOmdsToStdout(std::string _tag, int _iov_begin, std::string base="hex");

  // read ASCII stream with channel status word (cat ascii file)
  int readStatusWordFromStdin(std::string base="hex");

  // write ASCII stream with channel status word
  int writeStatusWordToStdout(std::string base = "hex"); // status and detId in hex or dec

  // add dataset to the XML document
  XERCES_CPP_NAMESPACE::DOMNode * add_hcal_channel_dataset( int ieta, int iphi, int depth, std::string subdetector,
				      int _channel_status, int _on_off, std::string _comment );

  // add channel to the geomid_cq map (XML has not changed)
  int addChannelToGeomIdMap( int ieta, int iphi, int depth, std::string subdetector,
			     int _channel_status, int _on_off, std::string _comment );

  // add data tag inside a dataset tag
  XERCES_CPP_NAMESPACE::DOMElement * add_data( XERCES_CPP_NAMESPACE::DOMNode * _dataset, int _channel_status, int _on_off, std::string _comment );

  // add XML for all HCAL channels onoff entries
  int set_all_channels_on_off( int _hb, int _he, int _hf, int _ho);

  // add XML for all HCAL channels status entries
  int set_all_channels_status( uint32_t _hb, uint32_t _he, uint32_t _hf, uint32_t _ho);

  // get baseline channel status from a tag for a given IOV
  int getBaseLineFromOmds(std::string _tag, int _iov_begin);

  // adds to XML datasets for channels in the map
  int addChannelQualityGeom(std::map<int,ChannelQuality> & _cq); // the map key is geom hash as in HcalAssistant

  // reads tags from OMDS and dumps them to stdout newest first
  // returns number of available tags
  int dumpTagsFromOmdsToStdout(void);

  // reads a sorted list of tags, newest first
  std::vector<std::string> getTagsFromOmds(void);

  // Reads IOVs available for a given tag from OMDS and dumps them
  // to stdout newest first.
  // Returns the number of available IOVs
  int dumpIovsFromOmdsToStdout(std::string tag);

  // reads a sorted list of IOVs for a given tag, newest first
  std::vector<int> getIovsFromOmds(std::string tag);


  std::map<int,ChannelQuality> detid_cq; // HcalDetId as key
  std::map<int,ChannelQuality> geomid_cq; // geomId as key

 private:
  std::string get_random_comment( void );
  HcalAssistant hAss; // HcalDetId and proprietary channel hash
};

#endif
