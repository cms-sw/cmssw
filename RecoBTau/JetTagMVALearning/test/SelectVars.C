//The code uses as inputs the Train*.xml files in the folder ./TrainExtraVars
//Then it outputs the modified files in the current directory. It will overwrite there the Train*.xml files
//The folder TrainExtraVars contains the training with the standard set of variables plus 6 extra ones
// found to have good discrimination power for different jet flavors
//The Train_*.xml files in CSV_LR are the xml that are used in the training process. This macro allows to replace
// those xml file by the one in TrainExtraVars while vetoing the variables that you don't want to include from that
// set of Std + 6 variables
//You are supposed to run this macro from the folder CSV_LR with:
// root -q -l SelectVars.C


std::vector<TString> VetoVariablesReco;
std::vector<TString> VetoVariablesPseudo;
std::vector<TString> VetoVariablesNo;

void InitVetoVariables() {
//Select here the variables you want to veto. By default we
//  veto the following 6 extra variables to recover the standard
//  set of CSV variables
  VetoVariablesReco.push_back("trackDeltaR");
  VetoVariablesReco.push_back("trackPtRatio");
  VetoVariablesReco.push_back("trackJetDist");
  VetoVariablesReco.push_back("trackDecayLenVal");
  VetoVariablesReco.push_back("trackSumJetEtRatio");
  VetoVariablesReco.push_back("vertexJetDeltaR");
  VetoVariablesReco.push_back("PFchargedHadronEnergy");
  VetoVariablesReco.push_back("PFchargedHadronEnergyFraction");
  VetoVariablesReco.push_back("PFchargedHadronMultiplicity");
  VetoVariablesReco.push_back("PFneutralHadronEnergy");
  VetoVariablesReco.push_back("PFneutralHadronEnergyFraction");
  VetoVariablesReco.push_back("PFneutralHadronMultiplicity");
  VetoVariablesReco.push_back("PFphotonEnergy");
  VetoVariablesReco.push_back("PFphotonEnergyFraction");
  VetoVariablesReco.push_back("PFphotonMultiplicity");
  VetoVariablesReco.push_back("PFmuonEnergy");
  VetoVariablesReco.push_back("PFmuonEnergyFraction");
  VetoVariablesReco.push_back("PFmuonMultiplicity");
  VetoVariablesReco.push_back("PFelectronEnergy");
  VetoVariablesReco.push_back("PFelectronEnergyFraction");
  VetoVariablesReco.push_back("PFelectronMultiplicity");
  VetoVariablesReco.push_back("PFtrackid");
  VetoVariablesReco.push_back("PF_T_FROM_DISP");
  VetoVariablesReco.push_back("PF_T_FROM_GAMMACONV");
  VetoVariablesReco.push_back("PF_T_FROM_V0");
  VetoVariablesReco.push_back("PF_T_TO_DISP");			

  VetoVariablesPseudo.push_back("trackDeltaR");
  VetoVariablesPseudo.push_back("trackPtRatio");
  VetoVariablesPseudo.push_back("trackJetDist");
  VetoVariablesPseudo.push_back("trackDecayLenVal");
  VetoVariablesPseudo.push_back("trackSumJetEtRatio");
  VetoVariablesPseudo.push_back("vertexJetDeltaR");
  VetoVariablesPseudo.push_back("PFchargedHadronEnergy");
  VetoVariablesPseudo.push_back("PFchargedHadronEnergyFraction");
  VetoVariablesPseudo.push_back("PFchargedHadronMultiplicity");
  VetoVariablesPseudo.push_back("PFneutralHadronEnergy");
  VetoVariablesPseudo.push_back("PFneutralHadronEnergyFraction");
  VetoVariablesPseudo.push_back("PFneutralHadronMultiplicity");
  VetoVariablesPseudo.push_back("PFphotonEnergy");
  VetoVariablesPseudo.push_back("PFphotonEnergyFraction");
  VetoVariablesPseudo.push_back("PFphotonMultiplicity");
  VetoVariablesPseudo.push_back("PFmuonEnergy");
  VetoVariablesPseudo.push_back("PFmuonEnergyFraction");
  VetoVariablesPseudo.push_back("PFmuonMultiplicity");
  VetoVariablesPseudo.push_back("PFelectronEnergy");
  VetoVariablesPseudo.push_back("PFelectronEnergyFraction");
  VetoVariablesPseudo.push_back("PFelectronMultiplicity");
  VetoVariablesPseudo.push_back("PFtrackid");
  VetoVariablesPseudo.push_back("PF_T_FROM_DISP");
  VetoVariablesPseudo.push_back("PF_T_FROM_GAMMACONV");
  VetoVariablesPseudo.push_back("PF_T_FROM_V0");
  VetoVariablesPseudo.push_back("PF_T_TO_DISP");			


  VetoVariablesNo.push_back("trackDeltaR");
  VetoVariablesNo.push_back("trackPtRatio");
  VetoVariablesNo.push_back("trackJetDist");
  VetoVariablesNo.push_back("trackDecayLenVal");
  VetoVariablesNo.push_back("trackSumJetEtRatio");
  VetoVariablesNo.push_back("PFchargedHadronEnergy");
  VetoVariablesNo.push_back("PFchargedHadronEnergyFraction");
  VetoVariablesNo.push_back("PFchargedHadronMultiplicity");
  VetoVariablesNo.push_back("PFneutralHadronEnergy");
  VetoVariablesNo.push_back("PFneutralHadronEnergyFraction");
  VetoVariablesNo.push_back("PFneutralHadronMultiplicity");
  VetoVariablesNo.push_back("PFphotonEnergy");
  VetoVariablesNo.push_back("PFphotonEnergyFraction");
  VetoVariablesNo.push_back("PFphotonMultiplicity");
  VetoVariablesNo.push_back("PFmuonEnergy");
  VetoVariablesNo.push_back("PFmuonEnergyFraction");
  VetoVariablesNo.push_back("PFmuonMultiplicity");
  VetoVariablesNo.push_back("PFelectronEnergy");
  VetoVariablesNo.push_back("PFelectronEnergyFraction");
  VetoVariablesNo.push_back("PFelectronMultiplicity");
  VetoVariablesNo.push_back("PFtrackid");
  VetoVariablesNo.push_back("PF_T_FROM_DISP");
  VetoVariablesNo.push_back("PF_T_FROM_GAMMACONV");
  VetoVariablesNo.push_back("PF_T_FROM_V0");
  VetoVariablesNo.push_back("PF_T_TO_DISP");			
}

//In principle no need to modify anything below this point

void SelectVars() {
  InitVetoVariables();

  AdaptOneFile("Train_NoVertex.xml", VetoVariablesNo);
  AdaptOneFile("Train_NoVertex_B_C.xml", VetoVariablesNo);
  AdaptOneFile("Train_NoVertex_B_DUSG.xml", VetoVariablesNo);
  AdaptOneFile("Train_PseudoVertex.xml", VetoVariablesPseudo);
  AdaptOneFile("Train_PseudoVertex_B_C.xml", VetoVariablesPseudo);
  AdaptOneFile("Train_PseudoVertex_B_DUSG.xml", VetoVariablesPseudo);
  AdaptOneFile("Train_RecoVertex.xml", VetoVariablesReco);
  AdaptOneFile("Train_RecoVertex_B_C.xml", VetoVariablesReco);
  AdaptOneFile("Train_RecoVertex_B_DUSG.xml", VetoVariablesReco);

}

void AdaptOneFile(TString file = "", std::vector<TString> VetoVars) {
  
  TString dir = "TrainExtraVars/";
  TString outputdir = "./";

  
  
  ifstream ifile(dir + file);
  ofstream of(outputdir + file);

  //Initialize a vector of first appearences in the text of vetoed variable
  std::vector<int> IsFirstAppearence;
  for (unsigned int  i = 0; i < VetoVars.size(); i++)
    IsFirstAppearence.push_back(1);
  

  TString line;

  while (ifile) {
    line.ReadLine(ifile,false);
    
    //If line contains a line with the variable we are copying
    // then the line is not printed. Unless it is the first appearence of the variable
    // (all the variables in the input root file should be included at the beginning
    // of the xml file. Event if they are not used)
    bool printline = true;
    for (unsigned int i = 0; i < VetoVars.size(); i++) {
      
      //if line contains one of the vetoed variables, then the line is not written
      if (line.Contains(VetoVars[i]) && !(VetoVars[i].Contains("PF") && VetoVars[i].EndsWith("Energy") && line.Contains("EnergyFraction")) ) {
        if (IsFirstAppearence[i] != 0) IsFirstAppearence[i] = 0;
        else printline = false;
	
      }
    
    }//end loop to veto variables vector
    
    if (printline) of << line << endl;
    
  }//end loop to input file lines
}
