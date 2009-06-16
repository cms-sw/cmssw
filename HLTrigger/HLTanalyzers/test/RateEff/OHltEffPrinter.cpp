#include <iostream>
#include <iomanip>
#include <fstream>
#include <TMath.h>
#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TString.h>
#include "OHltEffPrinter.h"
#include "OHltTree.h"

void OHltEffPrinter::SetupAll(vector<float> tEff,vector<float> tEffErr,vector<float> tspureEff,
			       vector<float> tspureEffErr,vector<float> tpureEff,
			       vector<float> tpureEffErr,vector< vector<float> >tcoMa, float tDenEff) {
  Eff = tEff;
  EffErr = tEffErr;
  spureEff = tspureEff;
  spureEffErr = tspureEffErr;
  pureEff = tpureEff;
  pureEffErr = tpureEffErr;
  coMa = tcoMa;
  DenEff = tDenEff;
}

/* ********************************************** */
// Print out eff as ascii
/* ********************************************** */
void OHltEffPrinter::printEffASCII(OHltConfig *cfg, OHltMenu *menu) {

  cout.setf(ios::floatfield,ios::fixed);
  cout<<setprecision(3);

  cout << "\n";
  cout << "Trigger Effs : " << "\n";
  cout << "                                          Name        Prescale           Indiv.          Pure   Cumulative\n";
  cout << "----------------------------------------------------------------------------------------------------------\n";

  float cumulEff = 0.;
  float cumulEffErr = 0.;
  for (unsigned int i=0;i<menu->GetTriggerSize();i++) {

    TString tempTrigSeeds; 
    TString tempTrigSeedPrescales; 
    std::map<TString, std::vector<TString> > 
      mapL1seeds = menu->GetL1SeedsOfHLTPathMap(); // mapping to all seeds 

    TString stmp; 
    vector<int> itmp; 
    typedef map< TString, vector<TString> >  mymap; 
    for(mymap::const_iterator it = mapL1seeds.begin();it != mapL1seeds.end(); ++it) {
      if (it->first.CompareTo(menu->GetTriggerName(i)) == 0) { 
        for (unsigned int j=0;j<it->second.size();j++) { 
          itmp.push_back(menu->GetL1Prescale((it->second)[j])); 
        } 
      } 
    }
    tempTrigSeeds = menu->GetSeedCondition(menu->GetTriggerName(i));
    for (unsigned int j=0;j<itmp.size();j++) { 
      tempTrigSeedPrescales += itmp[j]; 
      if (j<(itmp.size()-1)) { 
	tempTrigSeedPrescales = tempTrigSeedPrescales + ", "; 
      } 
    }
    if (itmp.size()>2) {
      tempTrigSeedPrescales = "-";
    }

    cumulEff += spureEff[i];
    //    cumulEffErr += pow(spureEffErr[i],2.);
    cumulEffErr = sqrt(cumulEff*(1-cumulEff)/DenEff);
    cout<<setw(50)<<menu->GetTriggerName(i)<<" ("
	<<setw(4)<<menu->GetPrescale(i)<<"*" <<tempTrigSeedPrescales<<setw(5)<<")  "
	<<setw(8)<<Eff[i]<<" +- "
	<<setw(7)<<EffErr[i]<<"  "
	<<setw(8)<<spureEff[i]<<"  "
	<<setw(8)<<cumulEff
	<<endl;
  }

  cumulEffErr = sqrt(cumulEffErr);
  cout << "\n";
  cout << setw(60) << "TOTAL EFF : " << setw(5) << cumulEff << " +- " << cumulEffErr <<" For "<< DenEff <<" events"<<"\n";
  cout << "----------------------------------------------------------------------------------------------\n";
  
}


