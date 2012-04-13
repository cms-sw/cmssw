// --> >// $Id: rootlogon.C,v 1.2 2012/04/11 16:30:21 arapyan Exp $
{
 {
  TString libstr(Form("%s/lib/%s/%s",
                      gSystem->Getenv("CMSSW_BASE"),
                      gSystem->Getenv("SCRAM_ARCH"),
                      "libCondFormatsEgammaObjects.so"));

  gSystem->Load(libstr);
 }

  gSystem->AddIncludePath("-I$CMSSW_BASE/src/CondFormats/EgammaObjects/interface");

  gInterpreter->AddIncludePath((TString(":")+TString(gSystem->Getenv("CMSSW_BASE"))+
				TString("/src/CondFormats/EgammaObjects/interface")).Data());

}
