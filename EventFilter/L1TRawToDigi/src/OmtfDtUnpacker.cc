#include "EventFilter/L1TRawToDigi/interface/OmtfDtUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/L1TRawToDigi/interface/OmtfDtDataWord64.h"


namespace omtf {

void DtUnpacker::unpack(unsigned int fed, unsigned int amc, const DtDataWord64 &data, std::vector<L1MuDTChambPhDigi> & phi_Container, std::vector<L1MuDTChambThDigi> & the_Container)
{
  LogTrace("") <<"HERE OMTF->DT " << std::endl;
  LogTrace("") << data << std::endl;
  if (data.sector()==0) {
     LogTrace("") << "...data skipped, since from oberlaping chambers."<< std::endl;
     return; // skip signal from chamber fiber exchange
  }
  int bx = data.bxNum()-3;
  int whNum = (fed==1380) ? -2 : 2;
  int sector =   (amc-1)*2 + data.sector();
  if (sector==12) sector=0;
  int station =  data.station()+1;
  LogTrace("") <<"DT_AMC#  "<<amc<<" RAW_SECTOR: "<<data.sector()<<" DT_SECTOR: "<<sector<<std::endl;
  phi_Container.push_back( L1MuDTChambPhDigi( bx, whNum, sector, station,
                           data.phi(), data.phiB(), data.quality(),
                           data.fiber(),               // utag/Ts2Tag
                           data.bcnt_st())); //ucnt/BxCnt
  int pos[7];
  int posQual[7];
  for (unsigned int i=0; i<7; i++) { pos[i] = (data.eta() >> i & 1); posQual[i] = (data.etaQuality() >> i & 1); }
  if (data.eta()) LogTrace("") <<"HERE DATA DT ETA";
  if(data.eta())the_Container.push_back(L1MuDTChambThDigi(bx,whNum, sector, station, pos, posQual));
 
}
 
}
