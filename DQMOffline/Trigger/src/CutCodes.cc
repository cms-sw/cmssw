#include "DQMOffline/Trigger/interface/CutCodes.h"


ComCodes CutCodes::codes_(CutCodes::setCodes_());

//horribly inefficient I know but its done once
ComCodes CutCodes::setCodes_()
{
  ComCodes codes;
  codes.setCode("et",ET);
  codes.setCode("pt",PT);
  codes.setCode("detEta",DETETA);
  codes.setCode("crack",CRACK);
  codes.setCode("epIn",EPIN);
  codes.setCode("dEtaIn",DETAIN);
  codes.setCode("dPhiIn",DPHIIN);
  codes.setCode("hadem",HADEM);
  codes.setCode("epOut",EPOUT);
  codes.setCode("dPhiOut",DPHIOUT);
  codes.setCode("invEInvP",INVEINVP);
  codes.setCode("bremFrac",BREMFRAC);
  codes.setCode("e9OverE25",E9OVERE25);
  codes.setCode("sigmaEtaEta",SIGMAETAETA);
  codes.setCode("sigmaPhiPhi",SIGMAPHIPHI);
  codes.setCode("isolEm",ISOLEM);
  codes.setCode("isolHad",ISOLHAD);
  codes.setCode("isolPtTrks",ISOLPTTRKS);
  codes.setCode("isolNrTrks",ISOLNRTRKS);
  codes.setCode("invalid",INVALID);
  codes.sort();
  return codes;
}

