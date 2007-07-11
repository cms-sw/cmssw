#include "EventFilter/CSCRawToDigi/interface/CSCALCTHeader2007.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>
#include <bitset>

bool CSCALCTHeader2007::debug=false;


CSCALCTHeader2007::CSCALCTHeader2007(const unsigned short * buf, const unsigned short size) 
{
  memcpy(this, buf, size);
}

CSCALCTHeader2007::CSCALCTHeader2007(const CSCALCTStatusDigi & digi)
{
  memcpy(this, digi.header(), sizeof(* digi.header()));///check this
}


std::vector<CSCALCTDigi> CSCALCTHeader2007::ALCTDigis() const 
{ 

  std::vector<CSCALCTDigi> result;
  
  ///loop over ALCT words: 
  for (unsigned short int i = 0; i < lctBins()*2 ; ++i) 
    {
      CSCALCT alct = getALCT(i);
      if (debug) 
	edm::LogInfo("CSCALCTHeader2007") << "ALCT DIGI" <<i<<" valid = " << alct.Valid 
					  << "  quality = "  << alct.Quality
					  << "  accel = " << alct.Accel
					  << "  pattern = " << alct.Pattern 
					  << "  Key Wire Group = " << alct.KeyWire 
					  << "  BX = " << alct.BXN;  

      CSCALCTDigi digi(alct.Valid, alct.Quality, alct.Accel, alct.Pattern,
		       alct.KeyWire, alct.BXN, 1);
      digi.setFullBX(BXNCount());
      result.push_back(digi);
    }

  return result;
}



std::ostream & operator<<(std::ostream & os, const CSCALCTHeader2007 & header) 
{
  os << "ALCT HEADER CSCID " << header.CSCID()
     << "  L1ACC " << header.L1Acc() << std::endl;
  os << "# ALCT chips read : "  << header.nLCTChipRead() 
     << " time samples " << header.NTBins() << std::endl;
  return os;
}
