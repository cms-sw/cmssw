#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"

using namespace egHLT;

ComCodes EgCutCodes::codes_(EgCutCodes::setCodes_());

//horribly inefficient I know but its done once
ComCodes EgCutCodes::setCodes_()
{
  ComCodes codes;
  codes.setCode("et",ET);
  codes.setCode("pt",PT);
  codes.setCode("detEta",DETETA);
  codes.setCode("crack",CRACK);
 
  codes.setCode("dEtaIn",DETAIN);
  codes.setCode("dPhiIn",DPHIIN);
  codes.setCode("invEInvP",INVEINVP);
  
  codes.setCode("hadem",HADEM);
  codes.setCode("sigmaIEtaIEta",SIGMAIETAIETA);
  codes.setCode("e2x5Over5x5",E2X5OVER5X5);
  codes.setCode("r9",R9);

  codes.setCode("isolEm",ISOLEM);
  codes.setCode("isolHad",ISOLHAD);
  codes.setCode("isolPtTrks",ISOLPTTRKS);
  codes.setCode("isolNrTrks",ISOLNRTRKS);

  codes.setCode("hltIsolTrksEle",HLTISOLTRKSELE);
  codes.setCode("hltIsolTrksPho",HLTISOLTRKSPHO);
  codes.setCode("hltIsolHad",HLTISOLHAD);

  codes.setCode("invalid",INVALID);
  codes.sort();
  return codes;
}

