#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"

using namespace egHLT;

const ComCodes EgCutCodes::codes_(EgCutCodes::setCodes_());

//horribly inefficient I know but its done once
ComCodes EgCutCodes::setCodes_()
{
  ComCodes codes;
  codes.setCode("et",int(ET));
  codes.setCode("pt",int(PT));
  codes.setCode("detEta",int(DETETA));
  codes.setCode("crack",int(CRACK));
 
  codes.setCode("dEtaIn",int(DETAIN));
  codes.setCode("dPhiIn",int(DPHIIN));
  codes.setCode("invEInvP",int(INVEINVP));
  
  codes.setCode("hadem",int(HADEM));
  codes.setCode("sigmaIEtaIEta",int(SIGMAIETAIETA)); 
  codes.setCode("sigmaEtaEta",int(SIGMAETAETA));
  codes.setCode("e2x5Over5x5",int(E2X5OVER5X5));
  //---Morse-------
  //codes.setCode("r9",int(R9));
  codes.setCode("minr9",int(MINR9));
  codes.setCode("maxr9",int(MAXR9));
  //---------------

  codes.setCode("isolEm",int(ISOLEM));
  codes.setCode("isolHad",int(ISOLHAD));
  codes.setCode("isolPtTrks",int(ISOLPTTRKS));
  codes.setCode("isolNrTrks",int(ISOLNRTRKS));

  codes.setCode("hltIsolTrksEle",int(HLTISOLTRKSELE));
  codes.setCode("hltIsolTrksPho",int(HLTISOLTRKSPHO));
  codes.setCode("hltIsolHad",int(HLTISOLHAD));
  codes.setCode("hltIsolEm",int(HLTISOLEM));
  
  codes.setCode("ctfTrack",int(CTFTRACK));
  codes.setCode("hltDEtaIn",int(HLTDETAIN));
  codes.setCode("hltDPhiIn",int(HLTDPHIIN));
  codes.setCode("hltInvEInvP",int(HLTINVEINVP));

  codes.setCode("invalid",int(INVALID));
  codes.sort();
  return codes;
}

