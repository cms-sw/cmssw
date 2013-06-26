#if !defined(_CINT_)

Int_t triggerSector(Int_t station, Int_t ring, Int_t chamber) const
{
  // taken from /CMSSW/DataFormats/MuonDetId/src/CSCDetId.cc on July 23, 2008
  // and modified

  Int_t result;

    if(station > 1 && ring > 1 ) {
      result = ((static_cast<unsigned>(chamber-3) & 0x7f) / 6) + 1; // ch 3-8->1, 9-14->2, ... 1,2 -> 6
    }
    else {
      result =  (station != 1) ? ((static_cast<unsigned>(chamber-2) & 0x1f) / 3) + 1 : // ch 2-4-> 1, 5-7->2, ...
	                         ((static_cast<unsigned>(chamber-3) & 0x7f) / 6) + 1;
    }

  return (result <= 6) ? result : 6; // max sector is 6, some calculations give a value greater than six but this is expected.
}

Int_t triggerCscId(Int_t station, Int_t ring, Int_t chamber) const
{
  // taken from /CMSSW/DataFormats/MuonDetId/src/CSCDetId.cc on July 23, 2008
  // and modified

  Int_t result;

  if( station == 1 ) {
    result = (chamber) % 3 + 1; // 1,2,3
    switch (ring) {
    case 1:
      break;
    case 2:
      result += 3; // 4,5,6
      break;
    case 3:
      result += 6; // 7,8,9
      break;
    }
  }
  else {
    if( ring == 1 ) {
      result = (chamber+1) % 3 + 1; // 1,2,3
    }
    else {
      result = (chamber+3) % 6 + 4; // 4,5,6,7,8,9
    }
  }
  return result;
}

#endif
