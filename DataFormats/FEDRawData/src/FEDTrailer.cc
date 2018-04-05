/** \file
 *
 *  \author N. Amapane - CERN, R. Mommsen - FNAL
 */

#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/src/fed_trailer.h"


FEDTrailer::FEDTrailer(const unsigned char* trailer) :
  theTrailer(reinterpret_cast<const fedt_t*>(trailer)) {}


FEDTrailer::~FEDTrailer() {}


uint32_t FEDTrailer::fragmentLength() const {
  return FED_EVSZ_EXTRACT(theTrailer->eventsize);
}


uint16_t FEDTrailer::crc() const {
  return FED_CRCS_EXTRACT(theTrailer->conscheck);
}


uint8_t FEDTrailer::evtStatus() const {
  return FED_STAT_EXTRACT(theTrailer->conscheck);
}


uint8_t FEDTrailer::ttsBits() const {
  return FED_TTSI_EXTRACT(theTrailer->conscheck);
}


bool FEDTrailer::moreTrailers() const {
  return ( FED_MORE_TRAILERS_EXTRACT(theTrailer->conscheck) != 0 );
}


bool FEDTrailer::crcModified() const {
  return ( FED_CRC_MODIFIED_EXTRACT(theTrailer->conscheck) != 0 );
}


bool FEDTrailer::slinkError() const {
  return ( FED_SLINK_ERROR_EXTRACT(theTrailer->conscheck) != 0 );
}


bool FEDTrailer::wrongFedId() const {
  return ( FED_WRONG_FEDID_EXTRACT(theTrailer->conscheck) != 0 );
}

uint32_t FEDTrailer::conscheck() const {
  return theTrailer->conscheck;
}

void FEDTrailer::set(unsigned char* trailer,
		     uint32_t lenght,
		     uint16_t crc,
		     uint8_t evtStatus,
		     uint8_t ttsBits,
		     bool moreTrailers) {
  // FIXME: should check that input ranges are OK!!!
  fedt_t* t = reinterpret_cast<fedt_t*>(trailer);

  t->eventsize =
    (FED_SLINK_END_MARKER << FED_TCTRLID_SHIFT) |
    ( (lenght    << FED_EVSZ_SHIFT) & FED_EVSZ_MASK);

  t->conscheck =
    ( (crc       << FED_CRCS_SHIFT) & FED_CRCS_MASK ) |
    ( (evtStatus << FED_STAT_SHIFT) & FED_STAT_MASK ) |
    ( (ttsBits   << FED_TTSI_SHIFT) & FED_TTSI_MASK );

  if (moreTrailers)
    t->conscheck |= (FED_MORE_TRAILERS_WIDTH << FED_MORE_TRAILERS_SHIFT);
}


bool FEDTrailer::check() const {
  return ( FED_TCTRLID_EXTRACT(theTrailer->eventsize) == FED_SLINK_END_MARKER );
}


const uint32_t FEDTrailer::length = sizeof(fedt_t);
