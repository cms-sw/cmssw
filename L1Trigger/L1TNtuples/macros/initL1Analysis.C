{
  std::cout << "FWCoreFWLite library loading ..." <<std::endl;
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();

  std::cout << "Macros & ntuple producer paths setting ..." <<std::endl;
  std::string tmp=gEnv->GetValue("Unix.*.Root.MacroPath","");
  //tmp=tmp+std::string(":$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros:$CAF_TRIGGER");
  tmp=tmp+std::string(":$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros");
  gEnv->SetValue("Unix.*.Root.MacroPath",tmp.c_str());
  tmp=gROOT->GetMacroPath();
  //tmp=tmp+std::string(":$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros:$CAF_TRIGGER");
  tmp=tmp+std::string(":$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros");
 
  // L1 rates 
  //tmp=tmp+std::string(":$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros/L1Rates:$CAF_TRIGGER");	
  //tmp=tmp+std::string(":$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros/L1Rates/toolbox:$CAF_TRIGGER");
  tmp=tmp+std::string(":$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros/L1Rates");	
  tmp=tmp+std::string(":$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros/L1Rates/toolbox");
  
  gROOT->SetMacroPath(tmp.c_str());
  gSystem->AddIncludePath(" -I$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/interface");
  gSystem->AddIncludePath(" -I$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros");
  gROOT->ProcessLine(".include $CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/interface");
  gROOT->ProcessLine(".include $CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros"); 

  gSystem->AddIncludePath(" -I$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros/L1Rates");
  gSystem->AddIncludePath(" -I$CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros/L1Rates/toolbox");
  gROOT->ProcessLine(".include $CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros/L1Rates");
  gROOT->ProcessLine(".include $CMSSW_BASE/src/L1TriggerDPG/L1Ntuples/macros/L1Rates/toolbox");

  std::cout << "L1Ntuple library loading ..." <<std::endl;
  gROOT->ProcessLine(".L L1Ntuple.C+");
  gROOT->ProcessLine(".L L1GtNtuple.C+");
  gROOT->ProcessLine(".L L1AlgoFactory.h+");

  std::cout << "---- initialization done ----"<<std::endl;
}
