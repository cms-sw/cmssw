#include "OHltConfig.h"

OHltConfig::OHltConfig(TString cfgfile,OHltMenu *omenu)
{

  try {
    /* Load the configuration.. */
    cout << "Loading "<<cfgfile;
    cfg.readFile(cfgfile);
    cout << " ... ok" << endl;

    // temporary vars
    const char* stmp; float ftmp; bool btmp;
    
    /**** General Menu & Run conditions ****/ 
    cfg.lookupValue("run.nEntries",nEntries);    
    cfg.lookupValue("run.nPrintStatusEvery",nPrintStatusEvery);    
    cfg.lookupValue("run.isRealData",isRealData);
    cfg.lookupValue("run.menuTag",stmp); menuTag = TString(stmp);    
    cfg.lookupValue("run.versionTag",stmp); versionTag = TString(stmp);    
    cfg.lookupValue("run.alcaCondition",stmp); alcaCondition = TString(stmp);    
    cfg.lookupValue("run.doPrintAll",doPrintAll);    
    /**********************************/
  
    /**** Beam conditions ****/ 
    cfg.lookupValue("beam.iLumi",iLumi);    
    cfg.lookupValue("beam.bunchCrossingTime",bunchCrossingTime);
    cfg.lookupValue("beam.maxFilledBunches",maxFilledBunches);    
    cfg.lookupValue("beam.nFilledBunches",nFilledBunches);    
    cfg.lookupValue("beam.cmsEnergy",cmsEnergy);    
    /**********************************/

    /**** Real data conditions ****/   
    cfg.lookupValue("data.liveTimeRun",liveTimeRun);
    cfg.lookupValue("data.nL1AcceptsRun",nL1AcceptsRun); 
    /******************************/  
  
    /**** Samples & Processes ****/ 
    Setting &p = cfg.lookup("process.names");
    const int nproc = (const int)p.getLength();
    //cout << nproc << endl;
    Setting &xs = cfg.lookup("process.sigmas");
    Setting &pa = cfg.lookup("process.paths");
    Setting &fn = cfg.lookup("process.fnames");
    Setting &muc = cfg.lookup("process.doMuonCuts");
    Setting &ec = cfg.lookup("process.doElecCuts");


    for (int i=0;i<nproc;i++) {
      stmp = p[i];
      pnames.push_back(TString(stmp));
      stmp = pa[i];
      ppaths.push_back(TString(stmp));
      stmp = fn[i];
      pfnames.push_back(TString(stmp));
      ftmp = xs[i];
      psigmas.push_back(ftmp);
      btmp = muc[i];
      pdomucuts.push_back(btmp);
      btmp = ec[i];
      pdoecuts.push_back(btmp);
    }
    /**********************************/

    /**** Branch Selections ****/ 
    // Only for experts:
    // Select certain branches to speed up code.
    // Modify only if you know what you do!
    cfg.lookupValue("branch.doSelectBranches",doSelectBranches);    
    cfg.lookupValue("branch.selectBranchL1",selectBranchL1);    
    cfg.lookupValue("branch.selectBranchHLT",selectBranchHLT);    
    cfg.lookupValue("branch.selectBranchOpenHLT",selectBranchOpenHLT);    
    cfg.lookupValue("branch.selectBranchL1extra",selectBranchL1extra);    
    cfg.lookupValue("branch.selectBranchReco",selectBranchReco);    
    cfg.lookupValue("branch.selectBranchMC",selectBranchMC);    
    /**********************************/

    
print();
    fillMenu(omenu);   
    printMenu(omenu);   

    //cout << "Done!" << endl;
  }
  catch (...) {
    cout << endl << "Reading cfg file "<< cfgfile <<" failed. Exit!" << endl;
  }

  convert();  // Convert cross-sections to cm^2
  
}

void OHltConfig::fillMenu(OHltMenu *omenu)
{
  // temporary vars
  const char* stmp; float ftmp; int itmp; //bool btmp; 
  const char* seedstmp;
    
  /**** Menu ****/ 
  cfg.lookupValue("menu.isL1Menu",isL1Menu); 
  omenu->SetIsL1Menu(isL1Menu);
  cfg.lookupValue("menu.doL1preloop",doL1preloop); 
  omenu->SetDoL1preloop(doL1preloop);

  Setting &m = cfg.lookup("menu.triggers");
  const int nm = (const int)m.getLength();
  //cout << nm << endl;
  for (int i=0;i<nm;i++) {
    TString ss0 = "menu.triggers.["; ss0 +=i; ss0=ss0+"].[0]";
    Setting &tt0 = cfg.lookup(ss0.Data());
    stmp = tt0;
    
    TString ss3 = "menu.triggers.["; ss3 +=i; ss3=ss3+"].[1]";
    Setting &tt3 = cfg.lookup(ss3.Data());
    seedstmp = tt3;

    
    //cout << stmp << endl;
    TString ss1 = "menu.triggers.["; ss1 +=i; ss1=ss1+"].[2]";
    Setting &tt1 = cfg.lookup(ss1.Data());
    itmp = tt1;
    //cout << itmp << endl;
    TString ss2 = "menu.triggers.["; ss2 +=i; ss2=ss2+"].[3]";
    Setting &tt2 = cfg.lookup(ss2.Data());
    ftmp = tt2;
    //cout << ftmp << endl;

    //omenu->AddTrigger(stmp,itmp,ftmp);
    omenu->AddTrigger(stmp,seedstmp,itmp,ftmp);
  }

  if (doL1preloop) {
    Setting &lm = cfg.lookup("menu.L1triggers");
    const int lnm = (const int)lm.getLength();
    //cout << lnm << endl;
    for (int i=0;i<lnm;i++) {
      TString ss0 = "menu.L1triggers.["; ss0 +=i; ss0=ss0+"].[0]";
      Setting &tt0 = cfg.lookup(ss0.Data());
      stmp = tt0;
      //cout << stmp << endl;
      TString ss1 = "menu.L1triggers.["; ss1 +=i; ss1=ss1+"].[1]";
      Setting &tt1 = cfg.lookup(ss1.Data());
      itmp = tt1;
      //cout << itmp << endl;
      
      omenu->AddL1forPreLoop(stmp,itmp);
    }
  }
  

  /**********************************/
}

void OHltConfig::print()
{
  cout << "---------------------------------------------" <<  endl;
  cout << "Configuration settings: " <<  endl;
  cout << "---------------------------------------------" <<  endl;
  cout << "nEntries: " << nEntries << endl;
  cout << "nPrintStatusEvery: " << nPrintStatusEvery << endl;
  cout << "menuTag: " << menuTag << endl;
  cout << "versionTag: " << versionTag << endl;
  cout << "isRealData: " << isRealData << endl;
  if(isRealData == true)
    {
      cout << "nL1AcceptsRun: " << nL1AcceptsRun << endl;
      cout << "liveTimeRun: " << liveTimeRun << endl;
    }
  cout << "alcaCondition: " << alcaCondition << endl;
  cout << "doPrintAll: " << doPrintAll << endl;
  cout << "---------------------------------------------" <<  endl;
  cout << "iLumi: " << iLumi << endl;
  cout << "bunchCrossingTime: " << bunchCrossingTime << endl;
  cout << "maxFilledBunches: " << maxFilledBunches << endl;
  cout << "nFilledBunches: " << nFilledBunches << endl;
  cout << "cmsEnergy: " << cmsEnergy << endl;
  cout << "---------------------------------------------" <<  endl;
  cout << "doSelectBranches: " << doSelectBranches << endl;
  cout << "selectBranchL1: " << selectBranchL1 << endl;
  cout << "selectBranchHLT: " << selectBranchHLT << endl;
  cout << "selectBranchOpenHLT: " << selectBranchOpenHLT << endl;
  cout << "selectBranchL1extra: " << selectBranchL1extra << endl;
  cout << "selectBranchReco: " << selectBranchReco << endl;
  cout << "selectBranchMC: " << selectBranchMC << endl;
  cout << "---------------------------------------------" <<  endl;
  
  cout << endl;
  cout << "Number of Samples: "<<pnames.size()<<  endl;
  cout << "**********************************" <<  endl;
  for (unsigned int i=0;i<pnames.size();i++) {
    cout << "pnames["<<i<<"]: " << pnames[i] << endl;
    cout << "ppaths["<<i<<"]: " << ppaths[i] << endl;
    cout << "pfnames["<<i<<"]: " << pfnames[i] << endl;
    cout << "psigmas["<<i<<"]: " << psigmas[i] << endl;
    cout << "pdomucuts["<<i<<"]: " << pdomucuts[i] << endl;
    cout << "pdoecuts["<<i<<"]: " << pdoecuts[i] << endl;
    cout << endl;
  }
  cout << "**********************************" <<  endl;

  cout << "---------------------------------------------" <<  endl;
}

void OHltConfig::printMenu(OHltMenu *omenu)
{
  omenu->print();
}



void OHltConfig::convert()
{
  // Convert cross-sections to cm^2
  for (unsigned int i = 0; i < psigmas.size(); i++){psigmas[i] *= 1.E-36;}
}
