#ifndef CastorGainWidths_h
#define CastorGainWidths_h

/** 
\class CastorGainWidths
\author Radek Ofierzynski
Modified by L.Mundim (Mar/2009)
POOL container to store GainWidth values 4xCapId
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"
#include "CondFormats/CastorObjects/interface/CastorGainWidth.h"

//typedef CastorCondObjectContainer<CastorGainWidth> CastorGainWidths;

class CastorGainWidths : public CastorCondObjectContainer<CastorGainWidth> {
public:
  CastorGainWidths() : CastorCondObjectContainer<CastorGainWidth>() {}

  std::string myname() const { return (std::string) "CastorGainWidths"; }

private:
  COND_SERIALIZABLE;
};

#endif
