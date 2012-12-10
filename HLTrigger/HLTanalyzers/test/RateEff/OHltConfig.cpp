#include "OHltConfig.h"
#include "HLTDatasets.h"

using namespace std;
using namespace libconfig;

OHltConfig::OHltConfig(TString cfgfile, OHltMenu *omenu)
{

   // init
   /**** General Menu & Run conditions ****/
   nEntries = -1;
   nPrintStatusEvery = 10000;
   isRealData= false;
   menuTag = "";
   preFilterLogicString = "";
   versionTag = "";
   doPrintAll = true;
   doDeterministicPrescale = false;
   useNonIntegerPrescales = false;
   readRefPrescalesFromNtuple = false;
   nonlinearPileupFit = "";
   lumiBinsForPileupFit = 150;
   dsList = "";
   iLumi = 1.E31;
   bunchCrossingTime = 25.0E-09;
   maxFilledBunches = 3564;
   nFilledBunches = 156;
   cmsEnergy = 10.;
   lumiSectionLength = 23.3;
   lumiScaleFactor = 1.0;
   prescaleNormalization = 1;
   isL1Menu = false;
   doL1preloop = true;
   doSelectBranches = false;
   selectBranchL1 = true;
   selectBranchHLT = true;
   selectBranchOpenHLT = true;
   selectBranchReco = true;
   selectBranchL1extra = true;
   selectBranchMC = true;

   try
   {
      /* Load the configuration.. */
      cout << "Loading "<<cfgfile;
      cfg.readFile(cfgfile);
      cout << " ... ok" << endl;

      // temporary vars
      const char* stmp; float ftmp; bool btmp; int itmp;

      /**** General Menu & Run conditions ****/
      cfg.lookupValue("run.nEntries",nEntries);
      cfg.lookupValue("run.nPrintStatusEvery",nPrintStatusEvery);
      cfg.lookupValue("run.isRealData",isRealData);
      omenu->SetIsRealData(isRealData);
      cfg.lookupValue("run.menuTag",stmp); menuTag = TString(stmp);
      cfg.lookupValue("run.versionTag",stmp); versionTag = TString(stmp);
      cfg.lookupValue("run.doPrintAll",doPrintAll);
      cfg.lookupValue("run.dsList",stmp); dsList= TString(stmp);
      cfg.lookupValue("run.doDeterministicPrescale",doDeterministicPrescale);
      cfg.lookupValue("run.useNonIntegerPrescales",useNonIntegerPrescales);
      cfg.lookupValue("run.readRefPrescalesFromNtuple",readRefPrescalesFromNtuple);
      cfg.lookupValue("run.nonlinearPileupFit",stmp); nonlinearPileupFit = TString(stmp);
      cfg.lookupValue("run.lumiBinsForPileupFit",lumiBinsForPileupFit);
      cout << "General Menu & Run conditions...ok"<< endl;
      /**********************************/

      /**** Beam conditions ****/
      cfg.lookupValue("beam.iLumi",iLumi);
      cfg.lookupValue("beam.bunchCrossingTime",bunchCrossingTime);
      cfg.lookupValue("beam.maxFilledBunches",maxFilledBunches);
      cfg.lookupValue("beam.nFilledBunches",nFilledBunches);
      cfg.lookupValue("beam.cmsEnergy",cmsEnergy);
      cout << "Beam conditions...ok"<< endl;
      /**********************************/

      /**** Real data conditions ****/
      cfg.lookupValue("data.lumiSectionLength",lumiSectionLength);
      cfg.lookupValue("data.lumiScaleFactor",lumiScaleFactor);
      cfg.lookupValue("data.prescaleNormalization",prescaleNormalization);

      getPreFilter();
      fillRunBlockList();

      cout << "Real data conditions...ok"<< endl;
      /******************************/

      /**** Samples & Processes ****/
      Setting &p = cfg.lookup("process.names");
      const int nproc = (const int)p.getLength();
      //cout << nproc << endl;
      Setting &isPS = cfg.lookup("process.isPhysicsSample");
      Setting &xs = cfg.lookup("process.sigmas");
      Setting &pa = cfg.lookup("process.paths");
      Setting &fn = cfg.lookup("process.fnames");
      Setting &muc = cfg.lookup("process.doMuonCuts");
      Setting &ec = cfg.lookup("process.doElecCuts");

      for (int i=0;i<nproc;i++)
      {
         stmp = p[i];
         pnames.push_back(TString(stmp));
         itmp = isPS[i];
         pisPhysicsSample.push_back(itmp);
         stmp = pa[i];
         // LA add trailing slash to directories if missing
         string ppath = stmp;
         string lastChar=ppath.substr(ppath.size()-1);
         if (lastChar.compare("/") != 0 ) ppath.append("/");

         ppaths.push_back(TString(ppath));
         stmp = fn[i];
         pfnames.push_back(TString(stmp));
         ftmp = xs[i];
         psigmas.push_back(ftmp);
         btmp = muc[i];
         pdomucuts.push_back(btmp);
         btmp = ec[i];
         pdoecuts.push_back(btmp);
      }
      cout << "Samples & Processes...ok"<< endl;

      for (int i=0;i<nproc;i++)
      { //RR

         printf("Name [%d]: %s\n",i,pnames.at(i).Data());
         printf("Path [%d]: %s\n",i,ppaths.at(i).Data());
         printf("File [%d]: %s\n",i,pfnames.at(i).Data());
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
      cout << "Branch Selections...ok"<< endl;
      /**********************************/

      print();
      fillMenu(omenu);
      printMenu(omenu);

      //cout << "Done!" << endl;
   }
   catch (...)
   {
      cout << endl << "Reading cfg file "<< cfgfile <<" failed. Exit!" << endl;
   }

   convert(); // Convert cross-sections to cm^2

}

void OHltConfig::getPreFilter()
{
   // Lookup in prefilter in LogicParser String format

   // temporary vars
   const char* stmp;
   try
   {
      if (cfg.lookupValue("menu.preFilterByBits",stmp))
      preFilterLogicString = TString(stmp);

   }
   catch (...)
   {
      cout << endl << "No PreFilter String found! " << endl;
   }
}

void OHltConfig::fillRunBlockList()
{
   // Lookup runLumiblockList, format: (runnr, minLumiBlock, maxLumiBlock)

   // temporary vars
   int itmp1;
   int itmp2;
   int itmp3;

   try
   {
      Setting &m = cfg.lookup("data.runLumiblockList");
      const int nm = (const int)m.getLength();
      for (int i=0;i<nm;i++)
      {
         TString ss0 = "data.runLumiblockList.["; ss0 +=i; ss0=ss0+"].[0]";
         Setting &tt0 = cfg.lookup(ss0.Data());
         itmp1 = tt0;

         TString ss3 = "data.runLumiblockList.["; ss3 +=i; ss3=ss3+"].[1]";
         Setting &tt3 = cfg.lookup(ss3.Data());
         itmp2 = tt3;

         TString ss1 = "data.runLumiblockList.["; ss1 +=i; ss1=ss1+"].[2]";
         Setting &tt1 = cfg.lookup(ss1.Data());
         itmp3 = tt1;

         vector<int> tmpItem;
         tmpItem.push_back(itmp1);
         tmpItem.push_back(itmp2);
         tmpItem.push_back(itmp3);
         runLumiblockList.push_back(tmpItem);
      }
   }
   catch (...)
   {
      cout << endl << "No runLumiblock list! " << endl;
   }
}

void OHltConfig::fillMenu(OHltMenu *omenu)
{
   // temporary vars
   const char* stmp;
   float ftmp;
   float itmpfloat;
   int itmp;
   int refprescaletmp; //bool btmp; 
   const char* seedstmp;
   /**** Menu ****/
   cfg.lookupValue("menu.isL1Menu", isL1Menu);
   omenu->SetIsL1Menu(isL1Menu);
   cfg.lookupValue("menu.doL1preloop", doL1preloop);
   omenu->SetDoL1preloop(doL1preloop);
   Setting &m = cfg.lookup("menu.triggers");
   const int nm = (const int)m.getLength();
   //cout << nm << endl;
   for (int i=0; i<nm; i++)
   {
      TString ss0 = "menu.triggers.[";
      ss0 +=i;
      ss0=ss0+"].[0]";
      Setting &tt0 = cfg.lookup(ss0.Data());
      stmp = tt0;
      //cout << "Trigger: " << stmp << endl;

      if (isL1Menu)
      {
         TString ss1 = "menu.triggers.[";
         ss1 +=i;
         ss1=ss1+"].[1]";
         Setting &tt1 = cfg.lookup(ss1.Data());
         if (useNonIntegerPrescales == true)
            itmpfloat = tt1;
         else
            itmp = tt1;
         //cout << itmp << endl;
         TString ss2 = "menu.triggers.[";
         ss2 +=i;
         ss2=ss2+"].[2]";
         Setting &tt2 = cfg.lookup(ss2.Data());
         ftmp = tt2;
         //cout << ftmp << endl;

         if (useNonIntegerPrescales == true)
            omenu->AddTrigger(stmp, itmpfloat, ftmp);
         else
            omenu->AddTrigger(stmp, itmp, ftmp);

         omenu->AddL1forPreLoop(stmp, itmp);
      }
      else
      {

         TString ss3 = "menu.triggers.[";
         ss3 +=i;
         ss3=ss3+"].[1]";
         Setting &tt3 = cfg.lookup(ss3.Data());
         seedstmp = tt3;
         //cout << "Seed: " << seedstmp << endl;

         TString ss1 = "menu.triggers.[";
         ss1 +=i;
         ss1=ss1+"].[2]";
         Setting &tt1 = cfg.lookup(ss1.Data());
         if (useNonIntegerPrescales == true)
            itmpfloat = tt1;
         else
            itmp = tt1;

         //cout << "Prescale: "<< itmp << endl;
         TString ss2 = "menu.triggers.[";
         ss2 +=i;
         ss2=ss2+"].[3]";
         Setting &tt2 = cfg.lookup(ss2.Data());
         ftmp = tt2;
         //cout << "SampleSize: "<< ftmp << endl;

         // JH - testing reference prescale
         refprescaletmp = 1;
         TString ss4 = "menu.triggers.[";
         ss4 +=i;
         ss4=ss4+"].[4]";
         if (cfg.exists(ss4.Data()))
         {
            Setting &tt4 = cfg.lookup(ss4.Data());
            refprescaletmp = tt4;
         }
         else
         {
            refprescaletmp = 1;
         }

         if (useNonIntegerPrescales == true)
            omenu->AddTrigger(stmp, seedstmp, itmpfloat, ftmp, refprescaletmp);
         else
            omenu->AddTrigger(stmp, seedstmp, itmp, ftmp, refprescaletmp);
      }
   }

   if (!isL1Menu)
   {
      Setting &lm = cfg.lookup("menu.L1triggers");
      const int lnm = (const int)lm.getLength();
      //cout << lnm << endl;
      for (int i=0; i<lnm; i++)
      {
         TString ss0 = "menu.L1triggers.[";
         ss0 +=i;
         ss0=ss0+"].[0]";
         Setting &tt0 = cfg.lookup(ss0.Data());
         stmp = tt0;
         //cout << stmp << endl;
         if (doL1preloop)
         {
            TString ss1 = "menu.L1triggers.[";
            ss1 +=i;
            ss1=ss1+"].[1]";
            Setting &tt1 = cfg.lookup(ss1.Data());
            if (useNonIntegerPrescales == true)
               itmpfloat = tt1;
            else
               itmp = tt1;
         }
         else
         {
            itmp = 1;
         }
         if (useNonIntegerPrescales == true)
            omenu->AddL1forPreLoop(stmp, itmpfloat);
         else
            omenu->AddL1forPreLoop(stmp, itmp);
      }
   }
   /**********************************/
}

void OHltConfig::print()
{
   cout << "---------------------------------------------" << endl;
   cout << "Configuration settings: " << endl;
   cout << "---------------------------------------------" << endl;
   cout << "nEntries: " << nEntries << endl;
   cout << "nPrintStatusEvery: " << nPrintStatusEvery << endl;
   cout << "menuTag: " << menuTag << endl;
   cout << "versionTag: " << versionTag << endl;
   cout << "isRealData: " << isRealData << endl;
   if (isRealData == true)
   {
      cout << "Time of one LumiSection: " << lumiSectionLength << endl;
      if (fabs(lumiScaleFactor-1.) > 0.001)
         cout << "Luminosity scaled by: " << lumiScaleFactor << endl;
      cout << "PD prescale factor: " << prescaleNormalization << endl;
   }
   cout << "doPrintAll: " << doPrintAll << endl;
   cout << "doDeterministicPrescale: " << doDeterministicPrescale << endl;
   cout << "useNonIntegerPrescales: " << useNonIntegerPrescales << endl;
   cout << "readRefPrescalesFromNtuple: " << readRefPrescalesFromNtuple << endl;
   cout << "nonlinearPileupFit: " << nonlinearPileupFit << endl;
   cout << "lumiBinsForPileupFit: " << lumiBinsForPileupFit << endl;
   cout << "preFilterLogicString: " << preFilterLogicString << endl;
   cout << "---------------------------------------------" << endl;
   cout << "iLumi: " << iLumi << endl;
   cout << "bunchCrossingTime: " << bunchCrossingTime << endl;
   cout << "maxFilledBunches: " << maxFilledBunches << endl;
   cout << "nFilledBunches: " << nFilledBunches << endl;
   cout << "cmsEnergy: " << cmsEnergy << endl;
   cout << "---------------------------------------------" << endl;
   cout << "doSelectBranches: " << doSelectBranches << endl;
   cout << "selectBranchL1: " << selectBranchL1 << endl;
   cout << "selectBranchHLT: " << selectBranchHLT << endl;
   cout << "selectBranchOpenHLT: " << selectBranchOpenHLT << endl;
   cout << "selectBranchL1extra: " << selectBranchL1extra << endl;
   cout << "selectBranchReco: " << selectBranchReco << endl;
   cout << "selectBranchMC: " << selectBranchMC << endl;
   cout << "---------------------------------------------" << endl;

   cout << endl;
   cout << "Number of Samples: "<<pnames.size()<< endl;
   cout << "**********************************" << endl;
   for (unsigned int i=0; i<pnames.size(); i++)
   {
      cout << "pnames["<<i<<"]: " << pnames[i] << endl;
      cout << "ppaths["<<i<<"]: " << ppaths[i] << endl;
      cout << "pfnames["<<i<<"]: " << pfnames[i] << endl;
      cout << "psigmas["<<i<<"]: " << psigmas[i] << endl;
      cout << "pdomucuts["<<i<<"]: " << pdomucuts[i] << endl;
      cout << "pdoecuts["<<i<<"]: " << pdoecuts[i] << endl;
      cout << endl;
   }
   cout << "**********************************" << endl;

   unsigned int nrunLumiList = runLumiblockList.size();
   if (nrunLumiList>0)
   {
      cout << endl;
      cout << "List of (runNo, minLumiblockID, maxLumiblockID) "<< endl;
      cout << "**********************************" << endl;
      for (unsigned int i=0; i<nrunLumiList; i++)
      {
         cout<<runLumiblockList[i][0]<< ", " << runLumiblockList[i][1]<< ", "
               << runLumiblockList[i][2]<<endl;
      }
      cout << "**********************************" << endl;
   }

   cout << "---------------------------------------------" << endl;
}

void OHltConfig::printMenu(OHltMenu *omenu)
{
   omenu->print();
}

void OHltConfig::convert()
{
   // Convert cross-sections to cm^2
   for (unsigned int i = 0; i < psigmas.size(); i++)
   {
      psigmas[i] *= 1.E-36;
   }
}
