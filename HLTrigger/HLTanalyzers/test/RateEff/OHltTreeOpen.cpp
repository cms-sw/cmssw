//////////////////////////////////////////////////////////////////
// OpenHLT definitions
//////////////////////////////////////////////////////////////////

#define OHltTreeOpen_cxx

#include "TVector2.h"
#include "OHltTree.h"
#include <stdio.h>

#include "TPRegexp.h"

using namespace std;


typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVector;


/**
 * method trying to match pattern with trigger name
 * returns result as boolean
 * returns by reference vector of matched strings 
 *   (bracketed groups in regex in order of appearance)
 */
bool triggerNamePatternMatch(
      const string& triggerName,
      const string& pattern,
      vector<string> &vecMatch)
{
   TPRegexp re(pattern);
   
   // does the regular expression match the triggerName given? 
   bool doesMatch= re.MatchB(triggerName);

   // if so, extract matches
   if (doesMatch)
   {
      // retrieve matches as object array
      TObjArray *objArrMatches = re.MatchS(triggerName);
      
      // extract match objects from array (skip index 0 = whole match)
      // convert each into a string and push back into a vector
      for (int i= 1; i <= objArrMatches->GetLast(); ++i)
      {
         TObject& objMatch= *(objArrMatches->At(i));
         TObjString& objStrMatch= dynamic_cast<TObjString&>(objMatch);
         vecMatch.push_back(objStrMatch.GetString().Data());
      }
      
      // delete object array
      delete objArrMatches;
   }
   
   // return match result
   return doesMatch;
}


bool isJetXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Jet([0-9]+)U){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isJetXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Jet([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isJetX_NoJetIDTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Jet([0-9]+)_NoJetID){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}


bool isDiJetAveXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_DiJetAve([0-9]+)U){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isDiJetAveXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_DiJetAve([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}


bool isMeffXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Meff([0-9]+)U){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdMeff = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdMeff);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isMeffXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Meff([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdMeff = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdMeff);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isHTXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_HT([0-9]+)U){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdHT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isHTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_HT([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdHT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isMHTXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_MHT([0-9]+)U){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdMHT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdMHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isMHTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_MHT([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdMHT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdMHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isMETXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_MET([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdMET = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdMET);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool ispfMHTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_PFMHT([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdpfMHT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdpfMHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isDiJetXU_PTXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_DiJet([0-9]+)U_PT([0-9]+)U)$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdPT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      thresholds.push_back(thresholdPT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isDiJetX_PTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_DiJet([0-9]+)_PT([0-9]+))$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdPT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      thresholds.push_back(thresholdPT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isHTXU_MHTXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_HT([0-9]+)U_MHT([0-9]+)U)$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdHT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMHT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdHT);
      thresholds.push_back(thresholdMHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isHTX_MHTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_HT([0-9]+)_MHT([0-9]+))$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdHT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMHT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdHT);
      thresholds.push_back(thresholdMHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isDoubleJetXU_ForwardBackwardTrigger(
      TString triggerName,
      vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_DoubleJet([0-9]+)U_ForwardBackward){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDoubleJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDoubleJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isDoubleJetX_ForwardBackwardTrigger(
      TString triggerName,
      vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_DoubleJet([0-9]+)_ForwardBackward){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDoubleJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDoubleJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isQuadJetXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_QuadJet([0-9]+)U){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdQuadJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdQuadJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isQuadJetXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_QuadJet([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdQuadJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdQuadJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isQuadJetX_BTagIPTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_QuadJet([0-9]+)_BTagIP){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdQuadJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdQuadJet);
      delete subStrL;
      return true;
   }
   else
      return false;

}

bool isQuadJetX_IsoPFTauXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_QuadJet([0-9]+)_IsoPFTau([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdQuadJet  = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdIsoPFTau = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdQuadJet);
      thresholds.push_back(thresholdIsoPFTau);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isQuadJetX_IsoPFTauX_PFMHTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_QuadJet([0-9]+)_IsoPFTau([0-9]+)_PFMHT([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdQuadJet  = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdIsoPFTau = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      double thresholdIsoPFMHT = (((TObjString *)subStrL->At(4))->GetString()).Atof();
      thresholds.push_back(thresholdQuadJet);
      thresholds.push_back(thresholdIsoPFTau);
      thresholds.push_back(thresholdIsoPFMHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}


bool isR0XU_MRXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_R0([0-9]+)U_MR([0-9]+)U{1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdR = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMR = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdR/100.);
      thresholds.push_back(thresholdMR);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isR0X_MRXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_R0([0-9]+)_MR([0-9]+){1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdR = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMR = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdR/100.);
      thresholds.push_back(thresholdMR);
      return true;
   }
   else
      return false;
}

bool isR0XTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_R0([0-9]+){1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdR = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdR/100.);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isMRXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_MR([0-9]+){1})$";
   TPRegexp matchThreshold(pattern);
   
   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdMR = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdMR);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isPT12U_XUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_PT12U_([0-9]+)U{1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdPT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdPT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isPT12_XTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_PT12_([0-9]+){1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdPT = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdPT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isBTagMu_DiJetXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_isBTagMu_DiJet([0-9]+)U{1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isBTagMu_DiJetXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_isBTagMu_DiJet([0-9]+){1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isBTagMu_JetXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_isBTagMu_DiJet([0-9]+)U{1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isBTagMu_JetXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_isBTagMu_Jet([0-9]+){1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isBTagMu_DiJetXU_MuXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_BTagMu_DiJet([0-9]+)U_Mu([0-9]+){1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMu = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      thresholds.push_back(thresholdMu);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isBTagMu_DiJetX_MuXTrigger(TString triggerName, vector<double> &thresholds)
{

  TString pattern = "(OpenHLT_BTagMu_DiJet([0-9]+)_Mu([0-9]+){1})$";
  TPRegexp matchThreshold(pattern);

  if (matchThreshold.MatchB(triggerName))
    {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMu = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      thresholds.push_back(thresholdMu);
      delete subStrL;
      return true;
    }
  else
    return false;
}


bool isBTagIP_JetXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_BTagIP_Jet([0-9]+){1})$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isSingleIsoTauX_TrkX_METXTrigger(
      TString triggerName,
      vector<double> &thresholds)
{

   TString pattern =
         "(OpenHLT_SingleIsoTau([0-9]+)_Trk([0-9]+)_MET([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdTau = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdTrk = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      double thresholdMET = (((TObjString *)subStrL->At(4))->GetString()).Atof();
      thresholds.push_back(thresholdTau);
      thresholds.push_back(thresholdTrk);
      thresholds.push_back(thresholdMET);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isCentralJetXU_METXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_CentralJet([0-9]+)U_MET([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdCenJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMET = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdCenJet);
      thresholds.push_back(thresholdMET);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isCentralJetX_METXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_CentralJet([0-9]+)_MET([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdCenJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMET = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdCenJet);
      thresholds.push_back(thresholdMET);
      delete subStrL;
      return true;
   }
   else
      return false;
}



bool isDiJetXU_METXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_DiJet([0-9]+)U_MET([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMET = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      thresholds.push_back(thresholdMET);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isDiJetX_METXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_DiJet([0-9]+)_MET([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMET = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      thresholds.push_back(thresholdMET);
      delete subStrL;
      return true;
   }
   else
      return false;
}


bool isMETX_HTXUTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_MET([0-9]+)_HTX([0-9]+)U){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdMET = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdHT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdMET);
      thresholds.push_back(thresholdHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isMETX_HTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_MET([0-9]+)_HTX([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdMET = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdHT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdMET);
      thresholds.push_back(thresholdHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isExclDiJetXU_HFORTrigger(TString triggerName, vector<double> &thresholds)
{
  
  TString pattern = "(OpenHLT_ExclDiJet([0-9]+)U_HFOR){1}$";
  TPRegexp matchThreshold(pattern);
  
   if (matchThreshold.MatchB(triggerName))
     {
       TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
       double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
       thresholds.push_back(thresholdDiJet);
       delete subStrL;
       return true;
     }
   else
     return false;
}

bool isExclDiJetX_HFORTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_ExclDiJet([0-9]+)_HFOR){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isExclDiJetXU_HFANDTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_ExclDiJet([0-9]+)U_HFAND){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isExclDiJetX_HFANDTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_ExclDiJet([0-9]+)_HFAND){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdDiJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdDiJet);
      delete subStrL;
      return true;
   }
   else
      return false;
}


bool isL1SingleMuXTrigger(TString triggerName)
{

  TString pattern = "(OpenHLT_L1SingleMu([0-9]+)){1}$";
  TPRegexp matchThreshold(pattern);

  if (matchThreshold.MatchB(triggerName))
    {
      return true;
    }
  else
    return false;
}

bool isL1DoubleMuXTrigger(TString triggerName)
{

  TString pattern = "(OpenHLT_L1DoubleMu([0-9]+)){1}$";
  TPRegexp matchThreshold(pattern);

  if (matchThreshold.MatchB(triggerName))
    {
      return true;
    }
  else
    return false;
}

bool isL2SingleMuXTrigger(TString triggerName, vector<double> &thresholds)
{
  TString pattern = "(OpenHLT_L2Mu([0-9]+)){1}$";
  TPRegexp matchThreshold(pattern);

  if (matchThreshold.MatchB(triggerName))
    {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdJet);
      delete subStrL;
      return true;
    }
  else
    return false;
}

bool isL2DoubleMuXTrigger(TString triggerName, vector<double> &thresholds)
{
  TString pattern = "(OpenHLT_L2DoubleMu([0-9]+)){1}$";
  TPRegexp matchThreshold(pattern);

  if (matchThreshold.MatchB(triggerName))
    {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdJet = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      thresholds.push_back(thresholdJet);
      delete subStrL;
      return true;
    }
  else
    return false;
}

bool isL1SingleEGXTrigger(TString triggerName)
{

   TString pattern = "(OpenHLT_L1SingleEG([0-9]+)){1}$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      return true;
   }
   else
      return false;
}

bool isPhotonX_HTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Photon([0-9]+)_HT([0-9]+)_L1R)$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdPhoton = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdHT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdPhoton);
      thresholds.push_back(thresholdHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isPhotonX_MHTXTrigger(TString triggerName, vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Photon([0-9]+)_MHT([0-9]+)_L1R)$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdPhoton = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMHT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdPhoton);
      thresholds.push_back(thresholdMHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isPhotonX_CaloIdL_HTXTrigger(
      TString triggerName,
      vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Photon([0-9]+)_CaloIdL_HT([0-9]+))$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdPhoton = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdHT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdPhoton);
      thresholds.push_back(thresholdHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

bool isPhotonX_CaloIdL_MHTXTrigger(
      TString triggerName,
      vector<double> &thresholds)
{

   TString pattern = "(OpenHLT_Photon([0-9]+)_CaloIdL_MHT([0-9]+))$";
   TPRegexp matchThreshold(pattern);

   if (matchThreshold.MatchB(triggerName))
   {
      TObjArray *subStrL = TPRegexp(pattern).MatchS(triggerName);
      double thresholdPhoton = (((TObjString *)subStrL->At(2))->GetString()).Atof();
      double thresholdMHT = (((TObjString *)subStrL->At(3))->GetString()).Atof();
      thresholds.push_back(thresholdPhoton);
      thresholds.push_back(thresholdMHT);
      delete subStrL;
      return true;
   }
   else
      return false;
}

void OHltTree::CheckOpenHlt(
      OHltConfig *cfg,
      OHltMenu *menu,
      OHltRateCounter *rcounter,
      int it)
{
   TString triggerName = menu->GetTriggerName(it);
   vector<double> thresholds;

   //////////////////////////////////////////////////////////////////
   // Check OpenHLT L1 bits for L1 rates

   if (triggerName.CompareTo("OpenL1_ZeroBias") == 0)
   {
      if (prescaleResponse(menu, cfg, rcounter, it))
      {
         triggerBit[it] = true;
      }
   }
   else if (triggerName.CompareTo("OpenL1_EG5_HTT100") == 0)
   {
      if (map_BitOfStandardHLTPath.find(triggerName)->second == 1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenL1_Mu3EG5") == 0)
   {
      if (map_BitOfStandardHLTPath.find(triggerName)->second == 1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenL1_QuadJet8U") == 0)
   {
      if (map_BitOfStandardHLTPath.find(triggerName)->second == 1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }

   //////////////////////////////////////////////////////////////////
   // Example for pass through triggers

   else if (triggerName.CompareTo("OpenHLT_L1Seed1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L1Seed2") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }

   //////////////////////////////////////////////////////////////////
   // Check OpenHLT triggers

   /* Single Jet */
   else if (triggerName.CompareTo("OpenHLT_L1SingleCenJet") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L1SingleForJet") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L1SingleTauJet") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L1Jet6") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L1Jet10") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            for (int i=0; i<NL1CenJet; i++)
               if (L1CenJetEt[i] >= 10.0)
                  rc++;
            for (int i=0; i<NL1ForJet; i++)
               if (L1ForJetEt[i] >= 10.0)
                  rc++;
            for (int i=0; i<NL1Tau; i++)
               if (L1TauEt [i] >= 10.0)
                  rc++;
            if (rc > 0)
               triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L1Jet15") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }

   /*OpenHLT_JetX(U) paths*/

   else if (isJetXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1JetPassed(thresholds[0])>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isJetXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1CorJetPassed(thresholds[0])>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isJetX_NoJetIDTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1CorJetPassedNoJetID(thresholds[0])>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }


   /* DoubleJetX(U)_ForwardBackward */

   else if (isDoubleJetXU_ForwardBackwardTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            int rc1 = 0;
            int rc2 = 0;

            // Loop over all oh jets, select events where both pT of a pair are above threshold and in HF+ and HF-
            for (int i=0; i<NrecoJetCal; i++)
            {
               if (recoJetCalPt[i]/0.7 > thresholds[0] && recoJetCalEta[i]
                     > 3.0 && recoJetCalEta[i] < 5.1)
               { // Jet pT/eta cut
                  ++rc1;
               }
               if (recoJetCalPt[i]/0.7 > thresholds[0] && recoJetCalEta[i]
                     > -5.1 && recoJetCalEta[i] < -3.0)
               { // Jet pT/eta cut
                  ++rc2;
               }
            }
            if (rc1!=0 && rc2!=0)
               rc=1;
            if (rc > 0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isDoubleJetX_ForwardBackwardTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            int rc1 = 0;
            int rc2 = 0;

            // Loop over all oh jets, select events where both pT of a pair are above threshold and in HF+ and HF-
            for (int i=0; i<NrecoJetCorCal; i++)
            {
               if (recoJetCorCalPt[i]/0.7 > thresholds[0]
                     && recoJetCorCalEta[i] > 3.0 && recoJetCorCalEta[i] < 5.1)
               { // Jet pT/eta cut
                  ++rc1;
               }
               if (recoJetCorCalPt[i]/0.7 > thresholds[0]
                     && recoJetCorCalEta[i] > -5.1 && recoJetCorCalEta[i]
                     < -3.0)
               { // Jet pT/eta cut
                  ++rc2;
               }
            }
            if (rc1!=0 && rc2!=0)
               rc=1;
            if (rc > 0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /* DiJetAveX(U) */
   else if (isDiJetAveXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltDiJetAvePassed(thresholds[0])>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isDiJetAveXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltCorDiJetAvePassed(thresholds[0])>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }




   /* Forward & MultiJet */
   else if (triggerName.CompareTo("OpenHLT_FwdJet20U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltFwdJetPassed(20.)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /******ExlDiJetX(U)_HFAND**********/
   else if (isExclDiJetXU_HFANDTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {

            int rcDijetCand = 0;
            double rcHFplusEnergy = 0;
            double rcHFminusEnergy = 0;

            // First loop over all jets and find a pair above threshold and with DeltaPhi/pi > 0.5
            for (int i=0; i<NrecoJetCal; i++)
            {
               if (recoJetCalPt[i]>thresholds[0])
               { // Jet pT cut 
                  for (int j=0; j<NrecoJetCal && j!=i; j++)
                  {
                     if (recoJetCalPt[j]>thresholds[0])
                     {
                        double Dphi=fabs(recoJetCalPhi[i]-recoJetCalPhi[j]);
                        if (Dphi>3.14159)
                           Dphi=2.0*(3.14159)-Dphi;
                        if (Dphi>0.5*3.14159)
                        {
                           rcDijetCand++;
                        }
                     }
                  }
               }
            }

            // Now ask for events with HF energy below threshold
            if (rcDijetCand > 0)
            {
               for (int i=0; i < NrecoTowCal; i++)
               {
                  if ((recoTowEta[i] > 3.0) && (recoTowE[i] > 4.0))
                     rcHFplusEnergy += recoTowE[i];
                  if ((recoTowEta[i] < -3.0) && (recoTowE[i] > 4.0))
                     rcHFminusEnergy += recoTowE[i];
               }
            }

            // Now put them together
            if ((rcDijetCand > 0) && (rcHFplusEnergy < 50) && (rcHFminusEnergy
                  < 50))
               triggerBit[it] = true;
         }
      }
   }

   else if (isExclDiJetX_HFANDTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {

            int rcDijetCand = 0;
            double rcHFplusEnergy = 0;
            double rcHFminusEnergy = 0;

            // First loop over all jets and find a pair above threshold and with DeltaPhi/pi > 0.5
            for (int i=0; i<NrecoJetCorCal; i++)
            {
               if (recoJetCorCalPt[i]>thresholds[0])
               { // Jet pT cut 
                  for (int j=0; j<NrecoJetCorCal && j!=i; j++)
                  {
                     if (recoJetCorCalPt[j]>thresholds[0])
                     {
                        double Dphi=fabs(recoJetCorCalPhi[i]-recoJetCorCalPhi[j]);
                        if (Dphi>3.14159)
                           Dphi=2.0*(3.14159)-Dphi;
                        if (Dphi>0.5*3.14159)
                        {
                           rcDijetCand++;
                        }
                     }
                  }
               }
            }

            // Now ask for events with HF energy below threshold
            if (rcDijetCand > 0)
            {
               for (int i=0; i < NrecoTowCal; i++)
               {
                  if ((recoTowEta[i] > 3.0) && (recoTowE[i] > 4.0))
                     rcHFplusEnergy += recoTowE[i];
                  if ((recoTowEta[i] < -3.0) && (recoTowE[i] > 4.0))
                     rcHFminusEnergy += recoTowE[i];
               }
            }

            // Now put them together
            if ((rcDijetCand > 0) && (rcHFplusEnergy < 50) && (rcHFminusEnergy
                  < 50))
               triggerBit[it] = true;
         }
      }
   }

   /*******ExclDiJetX(U)_HFOR**********/
   else if (isExclDiJetXU_HFORTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {

            int rcDijetCand = 0;
            double rcHFplusEnergy = 0;
            double rcHFminusEnergy = 0;

            // First loop over all jets and find a pair above threshold and with DeltaPhi/pi > 0.5 
            for (int i=0; i<NrecoJetCal; i++)
            {
               if (recoJetCalPt[i]>thresholds[0])
               { // Jet pT cut  
                  for (int j=0; j<NrecoJetCal && j!=i; j++)
                  {
                     if (recoJetCalPt[j]>thresholds[0])
                     {
                        double Dphi=fabs(recoJetCalPhi[i]-recoJetCalPhi[j]);
                        if (Dphi>3.14159)
                           Dphi=2.0*(3.14159)-Dphi;
                        if (Dphi>0.5*3.14159)
                        {
                           rcDijetCand++;
                        }
                     }
                  }
               }
            }

            // Now ask for events with HF energy below threshold 
            if (rcDijetCand > 0)
            {
               for (int i=0; i < NrecoTowCal; i++)
               {
                  if ((recoTowEta[i] > 3.0) && (recoTowE[i] > 4.0))
                     rcHFplusEnergy += recoTowE[i];
                  if ((recoTowEta[i] < -3.0) && (recoTowE[i] > 4.0))
                     rcHFminusEnergy += recoTowE[i];
               }
            }

            // Now put them together 
            if ((rcDijetCand > 0) && ((rcHFplusEnergy < 50) || (rcHFminusEnergy
                  < 50)))
               triggerBit[it] = true;
         }
      }
   }

   else if (isExclDiJetX_HFORTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {

            int rcDijetCand = 0;
            double rcHFplusEnergy = 0;
            double rcHFminusEnergy = 0;

            // First loop over all jets and find a pair above threshold and with DeltaPhi/pi > 0.5 
            for (int i=0; i<NrecoJetCorCal; i++)
            {
               if (recoJetCorCalPt[i]>thresholds[0])
               { // Jet pT cut  
                  for (int j=0; j<NrecoJetCorCal && j!=i; j++)
                  {
                     if (recoJetCorCalPt[j]>thresholds[0])
                     {
                        double Dphi=fabs(recoJetCorCalPhi[i]-recoJetCorCalPhi[j]);
                        if (Dphi>3.14159)
                           Dphi=2.0*(3.14159)-Dphi;
                        if (Dphi>0.5*3.14159)
                        {
                           rcDijetCand++;
                        }
                     }
                  }
               }
            }

            // Now ask for events with HF energy below threshold 
            if (rcDijetCand > 0)
            {
               for (int i=0; i < NrecoTowCal; i++)
               {
                  if ((recoTowEta[i] > 3.0) && (recoTowE[i] > 4.0))
                     rcHFplusEnergy += recoTowE[i];
                  if ((recoTowEta[i] < -3.0) && (recoTowE[i] > 4.0))
                     rcHFminusEnergy += recoTowE[i];
               }
            }

            // Now put them together 
            if ((rcDijetCand > 0) && ((rcHFplusEnergy < 50) || (rcHFminusEnergy
                  < 50)))
               triggerBit[it] = true;
         }
      }
   }
   /********QuadJetX(U)********/
   else if (isQuadJetXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltQuadJetPassed(thresholds[0])>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isQuadJetXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltQuadCorJetPassed(thresholds[0])>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /**************************/

   else if (triggerName.CompareTo("OpenHLT_FwdJet40") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltFwdCorJetPassed(40.)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_PentaJet25U20U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if ((OpenHlt1JetPassed(25)>=4) && (OpenHlt1JetPassed(20)>=5))
               triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_QuadJet50_Jet40") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if ((OpenHlt1CorJetPassed(50)>=4) && (OpenHlt1CorJetPassed(40)>=5))
               triggerBit[it] = true;
         }
      }
   }

   /* MET, HT, SumHT, Razor, PT */
   else if (triggerName.CompareTo("OpenHLT_L1MET20") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            triggerBit[it] = true;
         }
      }
   }

   /***METX******/
   else if (isMETXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (recoMetCal > thresholds[0])
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /*******pfMHTX******/

   else if (ispfMHTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (pfMHT > thresholds[0])
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /*******MeffX(U)********/

   else if (isMeffXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltMeffU(thresholds[0], 20.)==1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isMeffXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltMeff(thresholds[0], 40.)==1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /***********PTX(U)_X**************************/

   else if (isPT12U_XUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPT12U(thresholds[0], 50.)==1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isPT12_XTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPT12(thresholds[0], 50.)==1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /*HTX and HTXU paths*/

   else if (isHTXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumHTPassed(thresholds[0], 20.) == 1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isHTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(thresholds[0], 40.) == 1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_SumET120") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (recoMetCalSum > 120.)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /****RX(U)_MRX(U)**** RPassed not implemented with corrected jets********/

   else if (isR0XU_MRXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (OpenHltRUPassed(thresholds[0], thresholds[1], false, 7, 30.)>0)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               triggerBit[it] = true;
            }
         }
      }
   }

 else if (isR0X_MRXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (OpenHltRPassed(thresholds[0], thresholds[1], false, 7, 56.)>0)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               triggerBit[it] = true;
            }
         }
      }
   }

 else if (isR0XTrigger(triggerName, thresholds))
   {
     if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
       {
         if (OpenHltRPassed(thresholds[0], 0., false, 7, 56.)>0)
	   {
	     if (prescaleResponse(menu, cfg, rcounter, it))
	       {
		 triggerBit[it] = true;
	       }
	   }
       }
   }

 else if (isMRXTrigger(triggerName, thresholds))
   {
     if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
       {
         if (OpenHltRPassed(0., thresholds[0], false, 7, 56.)>0)
	   {
	     if (prescaleResponse(menu, cfg, rcounter, it))
	       {
		 triggerBit[it] = true;
	       }
	   }
       }
   }



   /* Muons */
   else if (isL1SingleMuXTrigger(triggerName))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (true)
            { // passthrough
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isL1DoubleMuXTrigger(triggerName))
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       if (true)
		 { // passthrough
		   triggerBit[it] = true;
		 }
	     }
	 }
     }


   else if (isL2SingleMuXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1L2MuonPassed(0.0, thresholds[0], 9999.0) > 0)
               triggerBit[it] = true;
         }
      }
   }

   else if (isL2DoubleMuXTrigger(triggerName, thresholds))
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       if (OpenHlt1L2MuonPassed(0.0, thresholds[0], 9999.0) > 1)
		 triggerBit[it] = true;
	     }
	 }
     }

   else if (triggerName.CompareTo("OpenHLT_Mu3") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(0., 0., 3., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu0_L1MuOpen") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(0., 0., 0., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu0_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(0., 0., 0., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu3_L1MuOpen") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(0., 3., 3., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_L1MuOpen") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu8") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 3., 8., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu3Mu0") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(0., 0., 0., 2., 0)>=2 && OpenHlt1MuonPassed(
                  0.,
                  3.,
                  3.,
                  2.,
                  0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 3., 5., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu7") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(5., 5., 7., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu9") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 9., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu10") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 10., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu11") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 11., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu12") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 12., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu13") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 13., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu15") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(10., 10., 15., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu17") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(12., 12., 17., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu20") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(12., 12., 20., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu24") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(12., 12., 24., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu30") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(12., 12., 30., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L2Mu0_NoVertex") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1L2MuonNoVertexPassed(0., 0., 9999.)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L2DoubleMu35_NoVertex_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1L2MuonNoVertexPassed(0., 35., 9999.)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu50_NoVertex") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 15., 50., 9999., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_DoubleMu3") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(0., 0., 3., 2., 0)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_DoubleMu4_Acoplanarity03") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            float ptl2 = 3.0;
            float ptl3 = 4.0;
            float drl3 = 2.0;
            float etal3 = 2.4;
            float etal2 = 2.4;
            float deltaphil3 = 0.3;

            for (int i=0; i<NohMuL3; i++)
            {
               for (int j=i+1; j<NohMuL3; j++)
               {
                  if (fabs(i[ohMuL3Eta]) < etal3 && fabs(ohMuL3Eta[j]) < etal3)
                  { // L3 eta cut   
                     if (ohMuL3Pt[i] > ptl3 && ohMuL3Pt[j] > ptl3)
                     { // L3 pT cut         
                        if (ohMuL3Dr[i] < drl3 && ohMuL3Dr[j] < drl3)
                        { // L3 DR cut 
                           if ((ohMuL3Chg[i] * ohMuL3Chg[j]) < 0)
                           { // opposite charge
                              float deltaphi = fabs(ohMuL3Phi[i]-ohMuL3Phi[j]);
                              if (deltaphi > 3.14159)
                                 deltaphi = (2.0 * 3.14159) - deltaphi;

                              deltaphi = 3.14159 - deltaphi;
                              if (deltaphi < deltaphil3)
                              {
                                 int l2match1 = ohMuL3L2idx[i];
                                 int l2match2 = ohMuL3L2idx[j];

                                 if ( (fabs(ohMuL2Eta[l2match1]) < etal2)
                                       && (fabs(ohMuL2Eta[l2match2]) < etal2))
                                 { // L2 eta cut  
                                    if ( (ohMuL2Pt[l2match1] > ptl2)
                                          && (ohMuL2Pt[l2match2] > ptl2))
                                    { // L2 pT cut  
                                       rc++;
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
            if (rc >=1)
               triggerBit[it] = true;
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_DoubleMu5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(0., 0., 5., 2., 0)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_DoubleMu6") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(3., 3., 6., 2., 0)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_DoubleMu7") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(3., 3., 7., 2., 0)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L1DoubleMuOpen") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (1)
            { // Pass through
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_DoubleMu0") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(0., 0., 0., 2., 0)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_TripleMu5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(3., 3., 5., 2., 0)>=3)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu5_DoubleMu5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(3., 4., 5., 2., 0)>=3 && OpenHlt1MuonPassed(
                  3.,
                  4.,
                  5.,
                  2.,
                  1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_TripleMu5_2IsoMu5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(3., 4., 5., 2., 0)>=3 && OpenHlt2MuonPassed(
                  3.,
                  4.,
                  5.,
                  2.,
                  1)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_TripleIsoMu5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(3., 4., 5., 2., 1)>=3)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_TripleMu_2IsoMu5_1Mu8") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(3., 4., 5., 2., 1)>=3&&OpenHlt1MuonPassed(
                  3.,
                  4.,
                  8.,
                  2.,
                  0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_IsoMu3") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 3., 3., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu9") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 9., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu11") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 11., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu12") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 12., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu13") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 13., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu15") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(10., 10., 15., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu17") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(10., 10., 17., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu20") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(10., 10., 20., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu24") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(10., 10., 24., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu30") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(12., 12., 30., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   /* Quarkonia */
   else if (triggerName.CompareTo("OpenHLT_DoubleMu3_Bs_v1") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       if (OpenHlt2MuonOSMassPassed(0., 0., 3., 2., 0, 4.8, 6.0)>=1)
		 {
		   triggerBit[it] = true;
		 }
	     }
	 }
     }
   else if (triggerName.CompareTo("OpenHLT_DoubleMu3_Jpsi_v1") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       if (OpenHlt2MuonOSMassPassed(0., 0., 3., 2., 0, 2.5, 4.0)>=1)
		 {
		   triggerBit[it] = true;
		 }
	     }
	 }
     }
   else if (triggerName.CompareTo("OpenHLT_DoubleMu3_Quarkonium_v1") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       if (OpenHlt2MuonOSMassPassed(0., 0., 3., 2., 0, 1.5, 14.0)>=1)
		 {
		   triggerBit[it] = true;
		 }
	     }
	 }
     }

   else if (triggerName.CompareTo("OpenHLT_Onia") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            //       cout << "checking for Onia " << endl;
            //variables for pixel cuts
            double ptPix = 0.;
            double pPix = 3.;
            double etaPix = 999.;
            double DxyPix = 999.;
            double DzPix = 999.;
            int NHitsPix = 3;
            double normChi2Pix = 999999999.;
            double massMinPix[2] = { 2.6, 7.5 };
            double massMaxPix[2] = { 3.6, 12.0 };
            double DzMuonPix = 999.;
            bool checkChargePix = false;
            //variables for tracker track cuts
            double ptTrack = 0.;
            double pTrack = 3.;
            double etaTrack = 999.;
            double DxyTrack = 999.;
            double DzTrack = 999.;
            int NHitsTrack = 5;
            double normChi2Track = 999999999.;
            double massMinTrack[2] = { 2.8, 8.5 };
            double massMaxTrack[2] = { 3.4, 11.0 };
            double DzMuonTrack = 0.5;
            bool checkChargeTrack = true;
            if ((OpenHlt1MuonPassed(0., 3., 3., 2., 0)>=1) && //check the L3 muon
                  OpenHltMuPixelPassed(
                        ptPix,
                        pPix,
                        etaPix,
                        DxyPix,
                        DzPix,
                        NHitsPix,
                        normChi2Pix,
                        massMinPix,
                        massMaxPix,
                        DzMuonPix,
                        checkChargePix) && //check the L3Mu + pixel
                  OpenHltMuTrackPassed(
                        ptTrack,
                        pTrack,
                        etaTrack,
                        DxyTrack,
                        DzTrack,
                        NHitsTrack,
                        normChi2Track,
                        massMinTrack,
                        massMaxTrack,
                        DzMuonTrack,
                        checkChargeTrack))
            { //check the L3Mu + tracker track
               triggerBit[it] = true;
            }
            //        if (GetIntRandom() % menu->GetPrescale(it) == 0) { triggerBit[it] = true; }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu0_Track0_Ups") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            //       cout << "checking for Onia " << endl;
            //variables for pixel cuts
            double ptPix = 0.;
            double pPix = 3.;
            double etaPix = 999.;
            double DxyPix = 999.;
            double DzPix = 999.;
            int NHitsPix = 3;
            double normChi2Pix = 999999999.;
            double massMinPix[1] = { 7.5 };
            double massMaxPix[1] = { 12.0 };
            double DzMuonPix = 999.;
            bool checkChargePix = false;
            //variables for tracker track cuts
            double ptTrack = 0.;
            double pTrack = 3.;
            double etaTrack = 999.;
            double DxyTrack = 999.;
            double DzTrack = 999.;
            int NHitsTrack = 5;
            double normChi2Track = 999999999.;
            double massMinTrack[1] = { 8.5 };
            double massMaxTrack[1] = { 11.0 };
            double DzMuonTrack = 0.5;
            bool checkChargeTrack = true;
            if ((OpenHlt1MuonPassed(0., 0., 0., 2., 0)>=1) && //check the L3 muon
                  OpenHltMuPixelPassed_Ups(
                        ptPix,
                        pPix,
                        etaPix,
                        DxyPix,
                        DzPix,
                        NHitsPix,
                        normChi2Pix,
                        massMinPix,
                        massMaxPix,
                        DzMuonPix,
                        checkChargePix,
                        7) && //check the L3Mu + pixel
                  OpenHltMuTrackPassed_Ups(
                        ptTrack,
                        pTrack,
                        etaTrack,
                        DxyTrack,
                        DzTrack,
                        NHitsTrack,
                        normChi2Track,
                        massMinTrack,
                        massMaxTrack,
                        DzMuonTrack,
                        checkChargeTrack,
                        7))
            {
               //check the L3Mu + tracker track
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu3_Track0_Ups") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            //       cout << "checking for Onia " << endl;
            //variables for pixel cuts
            double ptPix = 0.;
            double pPix = 3.;
            double etaPix = 999.;
            double DxyPix = 999.;
            double DzPix = 999.;
            int NHitsPix = 3;
            double normChi2Pix = 999999999.;
            double massMinPix[1] = { 7.5 };
            double massMaxPix[1] = { 12.0 };
            double DzMuonPix = 999.;
            bool checkChargePix = false;
            //variables for tracker track cuts
            double ptTrack = 0.;
            double pTrack = 3.;
            double etaTrack = 999.;
            double DxyTrack = 999.;
            double DzTrack = 999.;
            int NHitsTrack = 5;
            double normChi2Track = 999999999.;
            double massMinTrack[1] = { 8.5 };
            double massMaxTrack[1] = { 11.0 };
            double DzMuonTrack = 0.5;
            bool checkChargeTrack = true;
            if ((OpenHlt1MuonPassed(0., 3., 3., 2., 0)>=1) && //check the L3 muon
                  OpenHltMuPixelPassed_Ups(
                        ptPix,
                        pPix,
                        etaPix,
                        DxyPix,
                        DzPix,
                        NHitsPix,
                        normChi2Pix,
                        massMinPix,
                        massMaxPix,
                        DzMuonPix,
                        checkChargePix,
                        8) && //check the L3Mu + pixel
                  OpenHltMuTrackPassed_Ups(
                        ptTrack,
                        pTrack,
                        etaTrack,
                        DxyTrack,
                        DzTrack,
                        NHitsTrack,
                        normChi2Track,
                        massMinTrack,
                        massMaxTrack,
                        DzMuonTrack,
                        checkChargeTrack,
                        8))
            {
               //check the L3Mu + tracker track
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_Track0_Ups") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            //       cout << "checking for Onia " << endl;
            //variables for pixel cuts
            double ptPix = 0.;
            double pPix = 3.;
            double etaPix = 999.;
            double DxyPix = 999.;
            double DzPix = 999.;
            int NHitsPix = 3;
            double normChi2Pix = 999999999.;
            double massMinPix[1] = { 7.5 };
            double massMaxPix[1] = { 12.0 };
            double DzMuonPix = 999.;
            bool checkChargePix = false;
            //variables for tracker track cuts
            double ptTrack = 0.;
            double pTrack = 3.;
            double etaTrack = 999.;
            double DxyTrack = 999.;
            double DzTrack = 999.;
            int NHitsTrack = 5;
            double normChi2Track = 999999999.;
            double massMinTrack[1] = { 8.5 };
            double massMaxTrack[1] = { 11.0 };
            double DzMuonTrack = 0.5;
            bool checkChargeTrack = true;
            if ((OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1) && //check the L3 muon
                  OpenHltMuPixelPassed_Ups(
                        ptPix,
                        pPix,
                        etaPix,
                        DxyPix,
                        DzPix,
                        NHitsPix,
                        normChi2Pix,
                        massMinPix,
                        massMaxPix,
                        DzMuonPix,
                        checkChargePix,
                        9) && //check the L3Mu + pixel
                  OpenHltMuTrackPassed_Ups(
                        ptTrack,
                        pTrack,
                        etaTrack,
                        DxyTrack,
                        DzTrack,
                        NHitsTrack,
                        normChi2Track,
                        massMinTrack,
                        massMaxTrack,
                        DzMuonTrack,
                        checkChargeTrack,
                        9))
            {
               //check the L3Mu + tracker track
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu0_Track0_Jpsi") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            //       cout << "checking for Onia " << endl;
            //variables for pixel cuts
            double ptPix = 0.;
            double pPix = 3.;
            double etaPix = 999.;
            double DxyPix = 999.;
            double DzPix = 999.;
            int NHitsPix = 3;
            double normChi2Pix = 999999999.;
            double massMinPix[1] = { 2.6 };
            double massMaxPix[1] = { 3.6 };
            double DzMuonPix = 999.;
            bool checkChargePix = false;
            //variables for tracker track cuts
            double ptTrack = 0.;
            double pTrack = 3.;
            double etaTrack = 999.;
            double DxyTrack = 999.;
            double DzTrack = 999.;
            int NHitsTrack = 5;
            double normChi2Track = 999999999.;
            double massMinTrack[1] = { 2.8 };
            double massMaxTrack[1] = { 3.4 };
            double DzMuonTrack = 0.5;
            bool checkChargeTrack = true;
            if ((OpenHlt1MuonPassed(0., 0., 0., 2., 0)>=1) && //check the L3 muon
                  OpenHltMuPixelPassed_JPsi(
                        ptPix,
                        pPix,
                        etaPix,
                        DxyPix,
                        DzPix,
                        NHitsPix,
                        normChi2Pix,
                        massMinPix,
                        massMaxPix,
                        DzMuonPix,
                        checkChargePix,
                        0) && //check the L3Mu + pixel
                  OpenHltMuTrackPassed_JPsi(
                        ptTrack,
                        pTrack,
                        etaTrack,
                        DxyTrack,
                        DzTrack,
                        NHitsTrack,
                        normChi2Track,
                        massMinTrack,
                        massMaxTrack,
                        DzMuonTrack,
                        checkChargeTrack,
                        0))
            {
               //check the L3Mu + tracker track
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu3_Track0_Jpsi") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            //       cout << "checking for Onia " << endl;
            //variables for pixel cuts
            double ptPix = 0.;
            double pPix = 3.;
            double etaPix = 999.;
            double DxyPix = 999.;
            double DzPix = 999.;
            int NHitsPix = 3;
            double normChi2Pix = 999999999.;
            double massMinPix[1] = { 2.6 };
            double massMaxPix[1] = { 3.6 };
            double DzMuonPix = 999.;
            bool checkChargePix = false;
            //variables for tracker track cuts
            double ptTrack = 0.;
            double pTrack = 3.;
            double etaTrack = 999.;
            double DxyTrack = 999.;
            double DzTrack = 999.;
            int NHitsTrack = 5;
            double normChi2Track = 999999999.;
            double massMinTrack[1] = { 2.8 };
            double massMaxTrack[1] = { 3.4 };
            double DzMuonTrack = 0.5;
            bool checkChargeTrack = true;
            if ((OpenHlt1MuonPassed(0., 3., 3., 2., 0)>=1) && //check the L3 muon
                  OpenHltMuPixelPassed_JPsi(
                        ptPix,
                        pPix,
                        etaPix,
                        DxyPix,
                        DzPix,
                        NHitsPix,
                        normChi2Pix,
                        massMinPix,
                        massMaxPix,
                        DzMuonPix,
                        checkChargePix,
                        5) && //check the L3Mu + pixel
                  OpenHltMuTrackPassed_JPsi(
                        ptTrack,
                        pTrack,
                        etaTrack,
                        DxyTrack,
                        DzTrack,
                        NHitsTrack,
                        normChi2Track,
                        massMinTrack,
                        massMaxTrack,
                        DzMuonTrack,
                        checkChargeTrack,
                        5))
            {
               //check the L3Mu + tracker track
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu3_Track3_Jpsi") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       //       cout << "checking for Onia " << endl;
	       //variables for pixel cuts
	       double ptPix = 0.;
	       double pPix = 3.;
	       double etaPix = 999.;
	       double DxyPix = 999.;
	       double DzPix = 999.;
	       int NHitsPix = 3;
	       double normChi2Pix = 999999999.;
	       double massMinPix[1] = { 2.6 };
	       double massMaxPix[1] = { 3.6 };
	       double DzMuonPix = 999.;
	       bool checkChargePix = false;
	       //variables for tracker track cuts
	       double ptTrack = 3.;
	       double pTrack = 5.;
	       double etaTrack = 999.;
	       double DxyTrack = 999.;
	       double DzTrack = 999.;
	       int NHitsTrack = 5;
	       double normChi2Track = 999999999.;
	       double massMinTrack[1] = { 2.8 };
	       double massMaxTrack[1] = { 3.4 };
	       double DzMuonTrack = 0.5;
	       bool checkChargeTrack = true;
	       if ((OpenHlt1MuonPassed(0., 3., 3., 2., 0)>=1) && //check the L3 muon
		   OpenHltMuPixelPassed_JPsi(
					     ptPix,
					     pPix,
					     etaPix,
					     DxyPix,
					     DzPix,
					     NHitsPix,
					     normChi2Pix,
					     massMinPix,
					     massMaxPix,
					     DzMuonPix,
					     checkChargePix,
					     5) && //check the L3Mu + pixel
		   OpenHltMuTrackPassed_JPsi(
					     ptTrack,
					     pTrack,
					     etaTrack,
					     DxyTrack,
					     DzTrack,
					     NHitsTrack,
					     normChi2Track,
					     massMinTrack,
					     massMaxTrack,
					     DzMuonTrack,
					     checkChargeTrack,
					     5))
		 {
		   //check the L3Mu + tracker track
		   triggerBit[it] = true;
		 }
	     }
	 }
     }


   else if (triggerName.CompareTo("OpenHLT_Mu3_Track5_Jpsi") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            //       cout << "checking for Onia " << endl;
            //variables for pixel cuts
            double ptPix = 0.;
            double pPix = 3.;
            double etaPix = 999.;
            double DxyPix = 999.;
            double DzPix = 999.;
            int NHitsPix = 3;
            double normChi2Pix = 999999999.;
            double massMinPix[1] = { 2.6 };
            double massMaxPix[1] = { 3.6 };
            double DzMuonPix = 999.;
            bool checkChargePix = false;
            //variables for tracker track cuts
            double ptTrack = 5.;
            double pTrack = 3.;
            double etaTrack = 999.;
            double DxyTrack = 999.;
            double DzTrack = 999.;
            int NHitsTrack = 5;
            double normChi2Track = 999999999.;
            double massMinTrack[1] = { 2.8 };
            double massMaxTrack[1] = { 3.4 };
            double DzMuonTrack = 0.5;
            bool checkChargeTrack = true;
            if ((OpenHlt1MuonPassed(0., 3., 3., 2., 0)>=1) && //check the L3 muon
                  OpenHltMuPixelPassed_JPsi(
                        ptPix,
                        pPix,
                        etaPix,
                        DxyPix,
                        DzPix,
                        NHitsPix,
                        normChi2Pix,
                        massMinPix,
                        massMaxPix,
                        DzMuonPix,
                        checkChargePix,
                        5) && //check the L3Mu + pixel
                  OpenHltMuTrackPassed_JPsi(
                        ptTrack,
                        pTrack,
                        etaTrack,
                        DxyTrack,
                        DzTrack,
                        NHitsTrack,
                        normChi2Track,
                        massMinTrack,
                        massMaxTrack,
                        DzMuonTrack,
                        checkChargeTrack,
                        5))
            {
               //check the L3Mu + tracker track
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_Track0_Jpsi") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            //       cout << "checking for Onia " << endl;
            //variables for pixel cuts
            double ptPix = 0.;
            double pPix = 3.;
            double etaPix = 999.;
            double DxyPix = 999.;
            double DzPix = 999.;
            int NHitsPix = 3;
            double normChi2Pix = 999999999.;
            double massMinPix[1] = { 2.6 };
            double massMaxPix[1] = { 3.6 };
            double DzMuonPix = 999.;
            bool checkChargePix = false;
            //variables for tracker track cuts
            double ptTrack = 0.;
            double pTrack = 3.;
            double etaTrack = 999.;
            double DxyTrack = 999.;
            double DzTrack = 999.;
            int NHitsTrack = 5;
            double normChi2Track = 999999999.;
            double massMinTrack[1] = { 2.8 };
            double massMaxTrack[1] = { 3.4 };
            double DzMuonTrack = 0.5;
            bool checkChargeTrack = true;
            if ((OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1) && //check the L3 muon
                  OpenHltMuPixelPassed_JPsi(
                        ptPix,
                        pPix,
                        etaPix,
                        DxyPix,
                        DzPix,
                        NHitsPix,
                        normChi2Pix,
                        massMinPix,
                        massMaxPix,
                        DzMuonPix,
                        checkChargePix,
                        6) && //check the L3Mu + pixel
                  OpenHltMuTrackPassed_JPsi(
                        ptTrack,
                        pTrack,
                        etaTrack,
                        DxyTrack,
                        DzTrack,
                        NHitsTrack,
                        normChi2Track,
                        massMinTrack,
                        massMaxTrack,
                        DzMuonTrack,
                        checkChargeTrack,
                        6))
            {
               //check the L3Mu + tracker track
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu5_Track5_Jpsi") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       //       cout << "checking for Onia " << endl;
	       //variables for pixel cuts
	       double ptPix = 0.;
	       double pPix = 3.;
	       double etaPix = 999.;
	       double DxyPix = 999.;
	       double DzPix = 999.;
	       int NHitsPix = 3;
	       double normChi2Pix = 999999999.;
	       double massMinPix[1] = { 2.6 };
	       double massMaxPix[1] = { 3.6 };
	       double DzMuonPix = 999.;
	       bool checkChargePix = false;
	       //variables for tracker track cuts
	       double ptTrack = 5.;
	       double pTrack = 3.;
	       double etaTrack = 999.;
	       double DxyTrack = 999.;
	       double DzTrack = 999.;
	       int NHitsTrack = 5;
	       double normChi2Track = 999999999.;
	       double massMinTrack[1] = { 2.8 };
	       double massMaxTrack[1] = { 3.4 };
	       double DzMuonTrack = 0.5;
	       bool checkChargeTrack = true;
	       if ((OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1) && //check the L3 muon
		   OpenHltMuPixelPassed_JPsi(
					     ptPix,
					     pPix,
					     etaPix,
					     DxyPix,
					     DzPix,
					     NHitsPix,
					     normChi2Pix,
					     massMinPix,
					     massMaxPix,
					     DzMuonPix,
					     checkChargePix,
					     6) && //check the L3Mu + pixel
		   OpenHltMuTrackPassed_JPsi(
					     ptTrack,
					     pTrack,
					     etaTrack,
					     DxyTrack,
					     DzTrack,
					     NHitsTrack,
					     normChi2Track,
					     massMinTrack,
					     massMaxTrack,
					     DzMuonTrack,
					     checkChargeTrack,
					     6))
		 {
		   //check the L3Mu + tracker track
		   triggerBit[it] = true;
		 }
	     }
         }
     }

   else if (triggerName.CompareTo("OpenHLT_Mu7_Track5_Jpsi") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
           if (prescaleResponse(menu, cfg, rcounter, it))
             {
               //       cout << "checking for Onia " << endl;
               //variables for pixel cuts
               double ptPix = 0.;
               double pPix = 3.;
               double etaPix = 999.;
               double DxyPix = 999.;
               double DzPix = 999.;
               int NHitsPix = 3;
               double normChi2Pix = 999999999.;
               double massMinPix[1] = { 2.6 };
               double massMaxPix[1] = { 3.6 };
               double DzMuonPix = 999.;
               bool checkChargePix = false;
               //variables for tracker track cuts
               double ptTrack = 5.;
               double pTrack = 3.;
               double etaTrack = 999.;
               double DxyTrack = 999.;
               double DzTrack = 999.;
               int NHitsTrack = 5;
               double normChi2Track = 999999999.;
               double massMinTrack[1] = { 2.8 };
               double massMaxTrack[1] = { 3.4 };
               double DzMuonTrack = 0.5;
               bool checkChargeTrack = true;
               if ((OpenHlt1MuonPassed(3., 4., 7., 2., 0)>=1) && //check the L3 muon
                   OpenHltMuPixelPassed_JPsi(
                                             ptPix,
                                             pPix,
                                             etaPix,
                                             DxyPix,
                                             DzPix,
                                             NHitsPix,
                                             normChi2Pix,
                                             massMinPix,
                                             massMaxPix,
                                             DzMuonPix,
                                             checkChargePix,
                                             6) && //check the L3Mu + pixel
                   OpenHltMuTrackPassed_JPsi(
                                             ptTrack,
                                             pTrack,
                                             etaTrack,
                                             DxyTrack,
                                             DzTrack,
                                             NHitsTrack,
                                             normChi2Track,
                                             massMinTrack,
                                             massMaxTrack,
                                             DzMuonTrack,
                                             checkChargeTrack,
                                             6))
                 {
                   //check the L3Mu + tracker track
                   triggerBit[it] = true;
                 }
             }
         }
     }

   else if (triggerName.CompareTo("OpenHLT_Mu7_Track7_Jpsi") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
           if (prescaleResponse(menu, cfg, rcounter, it))
             {
               //       cout << "checking for Onia " << endl;
               //variables for pixel cuts
               double ptPix = 0.;
               double pPix = 3.;
               double etaPix = 999.;
               double DxyPix = 999.;
               double DzPix = 999.;
               int NHitsPix = 3;
               double normChi2Pix = 999999999.;
               double massMinPix[1] = { 2.6 };
               double massMaxPix[1] = { 3.6 };
               double DzMuonPix = 999.;
               bool checkChargePix = false;
               //variables for tracker track cuts
               double ptTrack = 7.;
               double pTrack = 3.;
               double etaTrack = 999.;
               double DxyTrack = 999.;
               double DzTrack = 999.;
               int NHitsTrack = 5;
               double normChi2Track = 999999999.;
               double massMinTrack[1] = { 2.8 };
               double massMaxTrack[1] = { 3.4 };
               double DzMuonTrack = 0.5;
               bool checkChargeTrack = true;
               if ((OpenHlt1MuonPassed(3., 4., 7., 2., 0)>=1) && //check the L3 muon
                   OpenHltMuPixelPassed_JPsi(
                                             ptPix,
                                             pPix,
                                             etaPix,
                                             DxyPix,
                                             DzPix,
                                             NHitsPix,
                                             normChi2Pix,
                                             massMinPix,
                                             massMaxPix,
                                             DzMuonPix,
                                             checkChargePix,
                                             6) && //check the L3Mu + pixel
                   OpenHltMuTrackPassed_JPsi(
                                             ptTrack,
                                             pTrack,
                                             etaTrack,
                                             DxyTrack,
                                             DzTrack,
                                             NHitsTrack,
                                             normChi2Track,
                                             massMinTrack,
                                             massMaxTrack,
                                             DzMuonTrack,
                                             checkChargeTrack,
                                             6))
                 {
		   {
		     //check the L3Mu + tracker track
		     triggerBit[it] = true;
		   }
		 }
	     }
	 }
     }

   else if (triggerName.CompareTo("OpenHLT_DoubleMu0_Quarkonium") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            TLorentzVector mu1;
            TLorentzVector mu2;
            TLorentzVector diMu;
            const double muMass = 0.105658367;
            int rc = 0;
            for (int i=0; i<NohMuL3; i++)
            {
               for (int j=0; j<NohMuL3 && j != i; j++)
               {

                  mu1.SetPtEtaPhiM(
                        ohMuL3Pt[i],
                        ohMuL3Eta[i],
                        ohMuL3Phi[i],
                        muMass);
                  mu2.SetPtEtaPhiM(
                        ohMuL3Pt[j],
                        ohMuL3Eta[j],
                        ohMuL3Phi[j],
                        muMass);
                  diMu = mu1 + mu2;
                  int dimuCharge = (int) (ohMuL3Chg[i] + ohMuL3Chg[j]);
                  float diMuMass = diMu.M();
                  if (diMuMass > 2.5 && diMuMass < 14.5 && dimuCharge == 0)
                     rc++;
               }
            }
            if (rc >= 1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_DoubleMu0_Quarkonium_LS") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            TLorentzVector mu1;
            TLorentzVector mu2;
            TLorentzVector diMu;
            const double muMass = 0.105658367;
            int rc = 0;
            for (int i=0; i<NohMuL3; i++)
            {
               for (int j=0; j<NohMuL3 && j != i; j++)
               {

                  mu1.SetPtEtaPhiM(
                        ohMuL3Pt[i],
                        ohMuL3Eta[i],
                        ohMuL3Phi[i],
                        muMass);
                  mu2.SetPtEtaPhiM(
                        ohMuL3Pt[j],
                        ohMuL3Eta[j],
                        ohMuL3Phi[j],
                        muMass);
                  diMu = mu1 + mu2;
                  int dimuCharge = (int) (ohMuL3Chg[i] + ohMuL3Chg[j]);
                  float diMuMass = diMu.M();
                  if (diMuMass > 2.5 && diMuMass < 14.5 && dimuCharge != 0)
                     rc++;
               }
            }
            if (rc >= 1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /* Electrons */

   else if (isL1SingleEGXTrigger(triggerName))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (true)
            { // passthrough      
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_L1DoubleEG5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (true)
            { // passthrough        
               triggerBit[it] = true;
            }
         }
      }
   }

  else if (triggerName.CompareTo("OpenHLT_Ele8_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap  
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap  
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap  
                  0.15,
                  0.1, // H/E barrel, H/E endcap  
                  999.,
                  999., // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  999,
                  999, // Deta barrel, Deta endcap  
                  999,
                  999 // Dphi barrel, Dphi endcap  
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Ele8_CaloIdL_CaloIsoVL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999,
                  999, // Track/pT iso barrel, Track/pT iso endcap  
                  0.2,
                  0.2, // H/ET iso barrel, H/ET iso endcap  
                  0.2,
                  0.2, // E/ET iso barrel, E/ET iso endcap  
                  0.15,
                  0.1, // H/E barrel, H/E endcap  
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  999,
                  999, // Deta barrel, Deta endcap  
                  999,
                  999 // Dphi barrel, Dphi endcap  
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
 
   else if (triggerName.CompareTo("OpenHLT_Ele8_CaloIdL_TrkIdVL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap  
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap  
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap  
                  0.15,
                  0.1, // H/E barrel, H/E endcap  
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  0.01,
                  0.01, // Deta barrel, Deta endcap  
                  0.15,
                  0.1 // Dphi barrel, Dphi endcap  
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }


   else if (triggerName.CompareTo("OpenHLT_Ele17_CaloIdL_CaloIsoVL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(17., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999,
                  999, // Track/pT iso barrel, Track/pT iso endcap  
                  0.2,
                  0.2, // H/ET iso barrel, H/ET iso endcap  
                  0.2,
                  0.2, // E/ET iso barrel, E/ET iso endcap  
                  0.15,
                  0.1, // H/E barrel, H/E endcap  
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  999,
                  999, // Deta barrel, Deta endcap  
                  999,
                  999 // Dphi barrel, Dphi endcap  
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(15., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  0.125,
                  0.075, // Track/pT iso barrel, Track/pT iso endcap  
                  0.125,
                  0.075, // H/ET iso barrel, H/ET iso endcap  
                  0.125,
                  0.075, // E/ET iso barrel, E/ET iso endcap  
                  0.05,
                  0.05, // H/E barrel, H/E endcap  
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  0.008,
                  0.008, // Deta barrel, Deta endcap  
                  0.07,
                  0.05 // Dphi barrel, Dphi endcap  
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v1")
         == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(27., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  0.125,
                  0.075, // Track/pT iso barrel, Track/pT iso endcap  
                  0.125,
                  0.075, // H/ET iso barrel, H/ET iso endcap  
                  0.125,
                  0.075, // E/ET iso barrel, E/ET iso endcap  
                  0.05,
                  0.05, // H/E barrel, H/E endcap  
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  0.008,
                  0.008, // Deta barrel, Deta endcap  
                  0.07,
                  0.05 // Dphi barrel, Dphi endcap  
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Ele45_CaloIdVT_TrkIdT_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(45., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap  
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap  
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap  
                  0.05,
                  0.05, // H/E barrel, H/E endcap  
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  0.008,
                  0.008, // Deta barrel, Deta endcap  
                  0.07,
                  0.05 // Dphi barrel, Dphi endcap  
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Ele90_NoSpikeFilter_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(90., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap  
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap  
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap  
                  0.15,
                  0.1, // H/E barrel, H/E endcap  
                  999.,
                  999., // cluster shape barrel, cluster shape endcap  
                  999.,
                  999., // R9 barrel, R9 endcap  
                  999.,
                  999., // Deta barrel, Deta endcap  
                  999.,
                  999. // Dphi barrel, Dphi endcap  
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   // only ele17 ele8 to keep
   else if (triggerName.CompareTo("OpenHLT_Ele17_CaloIdL_CaloIsoVL_Ele8_CaloIdL_CaloIsoVL_v1")
         == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2ElectronsAsymSamHarperPassed(17., 0, // ET, L1isolation 
                  999.,
                  999., // Track iso barrel, Track iso endcap 
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap 
                  0.2,
                  0.2, // H/ET iso barrel, H/ET iso endcap 
                  0.2,
                  0.2, // E/ET iso barrel, E/ET iso endcap 
                  0.15,
                  0.10, // H/E barrel, H/E endcap 
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap 
                  //999., 999.,       // R9 barrel, R9 endcap 
                  0.98,
                  999., // R9 barrel, R9 endcap 
                  999.,
                  999., // Deta barrel, Deta endcap 
                  999.,
                  999., // Dphi barrel, Dphi endcap 
                  8.,
                  0, // ET, L1isolation 
                  999.,
                  999., // Track iso barrel, Track iso endcap 
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap 
                  0.2,
                  0.2, // H/ET iso barrel, H/ET iso endcap 
                  0.2,
                  0.2, // E/ET iso barrel, E/ET iso endcap 
                  0.15,
                  0.10, // H/E barrel, H/E endcap 
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap 
                  //999., 999.,       // R9 barrel, R9 endcap 
                  0.98,
                  999., // R9 barrel, R9 endcap 
                  999.,
                  999., // Deta barrel, Deta endcap 
                  999.,
                  999. // Dphi barrel, Dphi endcap 
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   // triple ele
   else if (triggerName.CompareTo("OpenHLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_v1")
         == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt3ElectronsSamHarperPassed(10., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap
                  0.15,
                  0.1, // H/E barrel, H/E endcap
                  999.,
                  999., // cluster shape barrel, cluster shape endcap
                  0.98,
                  1., // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=3 && OpenHlt2ElectronsSamHarperPassed(10., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap 
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap 
                  0.98,
                  1., // R9 barrel, R9 endcap
                  0.01,
                  0.01, // Deta barrel, Deta endcap 
                  0.15,
                  0.10 // Dphi barrel, Dphi endcap
                  )>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_TripleEle10_CaloIdL_TrkIdVL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt3ElectronsSamHarperPassed(10., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap 
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap 
                  0.98,
                  1., // R9 barrel, R9 endcap 
                  0.01,
                  0.01, // Deta barrel, Deta endcap 
                  0.15,
                  0.10 // Dphi barrel, Dphi endcap
            )>=3)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /* 
    * this is only a proxy to the HLT implementation 
    * - should use SC variable once available in ntuples */
   else if (triggerName.CompareTo("OpenHLT_Ele32_CaloIdL_CaloIsoVL_SC17_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(
                  32., 
                  0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999,
                  999, // Track/pT iso barrel, Track/pT iso endcap  
                  0.2,
                  0.2, // H/ET iso barrel, H/ET iso endcap  
                  0.2,
                  0.2, // E/ET iso barrel, E/ET iso endcap  
                  0.15,
                  0.10, // H/E barrel, H/E endcap  
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap  
                  999.,
                  999., // R9 barrel, R9 endcap  
                  999,
                  999, // Deta barrel, Deta endcap  
                  999,
                  999 // Dphi barrel, Dphi endcap  
            ) >=1
            && 
            OpenHlt1PhotonSamHarperPassed(
                  17., 
                  0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H iso barrel, H iso endcap
                  999.,
                  999., // E iso barrel, E iso endcap
                  0.15,
                  0.1, // H/E barrel, H/E endcap
                  999.,
                  999., // cluster shape barrel, cluster shape endcap
                  999,
                  999, // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
                  ) >= 2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v1") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       std::vector<int> firstVector = VectorOpenHlt1ElectronSamHarperPassed(
										    17., 
										    0, // ET, L1isolation  
										    999.,
										    999., // Track iso barrel, Track iso endcap  
										    0.05,
										    0.05, // Track/pT iso barrel, Track/pT iso endcap  
										    0.05,
										    0.05, // H/ET iso barrel, H/ET iso endcap  
										    0.05,
										    0.05, // E/ET iso barrel, E/ET iso endcap  
										    0.05,
										    0.05, // H/E barrel, H/E endcap  
										    0.011,
										    0.031, // cluster shape barrel, cluster shape endcap  
										    0.98,
										    1.0, // R9 barrel, R9 endcap  
										    0.008,
										    0.008, // Deta barrel, Deta endcap  
										    0.07,
										    0.05 );// Dphi barrel, Dphi endcap  
		 if (firstVector.size()>=1){
		   std::vector<int> secondVector = VectorOpenHlt1PhotonSamHarperPassed(
										       8., 
										       0, // ET, L1isolation
										       999.,
										       999., // Track iso barrel, Track iso endcap
										       999.,
										       999., // Track/pT iso barrel, Track/pT iso endcap
										       999.,
										       999., // H iso barrel, H iso endcap
										       999.,
										       999., // E iso barrel, E iso endcap
										       0.15,
										       0.1, // H/E barrel, H/E endcap
										       999.,
										       999., // cluster shape barrel, cluster shape endcap
										       999.,
										       999., // R9 barrel, R9 endcap
										       999.,
										       999., // Deta barrel, Deta endcap
										       999.,
										       999.); // Dphi barrel, Dphi endcap
		   if (secondVector.size()>=2){


		     // mass condition
		     TLorentzVector ele;
		     TLorentzVector pho;
		     TLorentzVector sum;
		     float mass = 0.;
		     for (unsigned int i=0; i<firstVector.size(); i++)
		       {
			 for (unsigned int j=0; j<secondVector.size() ; j++)
			   {

			     //                  if (firstVector[i] == secondVector[j]) continue;
			     ele.SetPtEtaPhiM(
					     ohEleEt[firstVector[i]],
					     ohEleEta[firstVector[i]],
					     ohElePhi[firstVector[i]],
					     0.);
			     pho.SetPtEtaPhiM(
					     ohPhotEt[secondVector[j]],
					     ohPhotEta[secondVector[j]],
					     ohPhotPhi[secondVector[j]],
					     0.);
			     sum = ele + pho;
			     mass = sum.M();

			     if (mass>30)
			       triggerBit[it] = true;

			   }
		       }

		   }
		 }
	     }
	 }
     }

   
   /* Photons */
   else if (triggerName.CompareTo("OpenHLT_Photon30_CaloIdVL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonSamHarperPassed(30., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H iso barrel, H iso endcap
                  999.,
                  999., // E iso barrel, E iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap
                  0.024,
                  0.040, // cluster shape barrel, cluster shape endcap
                  0.98,
                  999., // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Photon30_CaloIdVL_IsoL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonSamHarperPassed(30., 0, // ET, L1isolation
                  3.5,
                  3.5, // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  3.5,
                  3.5, // H iso barrel, H iso endcap
                  5.5,
                  5.5, // E iso barrel, E iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap
                  0.024,
                  0.040, // cluster shape barrel, cluster shape endcap
                  0.98,
                  999., // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Photon75_CaloIdVL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonSamHarperPassed(75., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H iso barrel, H iso endcap
                  999.,
                  999., // E iso barrel, E iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap
                  0.024,
                  0.040, // cluster shape barrel, cluster shape endcap
                  0.98,
                  999., // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Photon75_CaloIdVL_IsoL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonSamHarperPassed(75., 0, // ET, L1isolation
                  3.5,
                  3.5, // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  3.5,
                  3.5, // H iso barrel, H iso endcap
                  5.5,
                  5.5, // E iso barrel, E iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap
                  0.024,
                  0.040, // cluster shape barrel, cluster shape endcap
                  0.98,
                  999., // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Photon125_NoSpikeFilter_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonSamHarperPassed(125., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H iso barrel, H iso endcap
                  999.,
                  999., // E iso barrel, E iso endcap
                  999.,
                  999., // H/E barrel, H/E endcap
                  999.,
                  999., // cluster shape barrel, cluster shape endcap
                  999.,
                  999., // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Photon32_CaloIdL_Photon26_CaloIdL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPhoCuts(32, 0.15, 0.10, 0.014, 0.035, 999, 999) >= 1
                  && OpenHltPhoCuts(26, 0.15, 0.10, 0.014, 0.035, 999, 999)
                        >= 2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   // to be removed ??
   /*PhotonX_(M)HTX */

   else if (isPhotonX_HTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonPassedRA3(
                  thresholds[0],
                  0,
                  999.,
                  999.,
                  999.,
                  999.,
                  0.075,
                  0.075,
                  0.98,
                  1.0)>=1 && OpenHltSumCorHTPassed(thresholds[1], 30.)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isPhotonX_MHTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonPassedRA3(
                  thresholds[0],
                  0,
                  999.,
                  999.,
                  999.,
                  999.,
                  0.075,
                  0.075,
                  0.98,
                  1.0)>=1 && OpenHltMHT(thresholds[1], 30.)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isPhotonX_CaloIdL_HTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPhoCuts(thresholds[0], 0.15, 0.10, 0.014, 0.034, 999, 999) >= 1
                  && OpenHltSumCorHTPassed(thresholds[1], 30.)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isPhotonX_CaloIdL_MHTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPhoCuts(thresholds[0], 0.15, 0.10, 0.014, 0.034, 999, 999) >= 1
                  && OpenHltMHT(thresholds[1], 30.)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /*
    * 2011-01-26 added by Christian Hartl
    * according to https://twiki.cern.ch/twiki/bin/view/CMS/EgammaWorkingPoints
    * adapted from "OpenHLT_Photon65_CaloEleId_Isol_L1R":
    * - dropped _L1R from name because L1 not relaxed anymore 
    * TODO
    * - Tisobarrel=0.001, Tisoendcap=5.0 ? -- check with Mass.
    */
   // to be removed
   else if (triggerName.CompareTo("OpenHLT_Photon65_CaloEleId_Isol") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonSamHarperPassed( 65., 0, // ET, L1isolation
                  0.001,
                  5.0, // Track iso barrel, Track iso endcap ???
                  0.2,
                  0.1, // Track/pT iso barrel, Track/pT iso endcap
                  0.2,
                  0.1, // H iso barrel, H iso endcap
                  0.2,
                  0.1, // E iso barrel, E iso endcap
                  0.05,
                  0.05, // H/E barrel, H/E endcap
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap
                  0.98,
                  1.0, // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }


   else if (triggerName.CompareTo("OpenHLT_DoublePhoton32_CaloIdL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonSamHarperPassed(32., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H iso barrel, H iso endcap
                  999.,
                  999., // E iso barrel, E iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap
                  0.98,
                  999., // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_DoublePhoton33_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PhotonSamHarperPassed(33., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H iso barrel, H iso endcap
                  999.,
                  999., // E iso barrel, E iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap
                  999.,
                  999., // cluster shape barrel, cluster shape endcap
                  0.98,
                  999., // R9 barrel, R9 endcap
                  999.,
                  999., // Deta barrel, Deta endcap
                  999.,
                  999. // Dphi barrel, Dphi endcap
            )>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /* Taus */
   else if (triggerName.CompareTo("OpenHLT_DoubleIsoTau15_Trk5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltTauL2SCPassed(15., 5., 0, 0., 1, 14., 30.)>=2)
            { //Thresholds are for UNcorrected L1 jets in 8E29
               triggerBit[it] = true;
            }
         }
      }
   }

   /* BTag */

   /**********BTagMu_JetX(U)**********/
   else if (isBTagMu_JetXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            int njets = 0;

            // apply L2 cut on jets 
            for (int i = 0; i < NohBJetL2; i++)
               if (ohBJetL2Et[i] > thresholds[0] && abs(ohBJetL2Eta[i]) < 3.0) // change this ET cut to 20 for the 20U patath 
                  njets++;

            // apply b-tag cut 
            int max = (NohBJetL2 > 4) ? 4 : NohBJetL2;
            for (int i = 0; i < max; i++)
            {
               if (ohBJetL2Et[i] > 10.)
               { // keep this at 10 even for the 20UU path - also, no eta cut here 
                  if (ohBJetMuL25Tag[i] > 0.5)
                  { // Level 2.5 b tag 
                     if (ohBJetPerfL3Tag[i] > 0.5)
                     { // Level 3 b tag 
                        rc++;
                     }
                  }
               }
            }
            if (rc >= 1 && njets>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (isBTagMu_JetXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            int max = (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected;
            for (int i = 0; i < max; i++)
            {
               if (ohBJetL2CorrectedEt[i] > thresholds[0])
               { // ET cut
                  if (ohBJetMuL25Tag[i] > 0.5)
                  { // Level 2.5 b tag
                     if (ohBJetPerfL3Tag[i] > 0.5)
                     { // Level 3 b tag
                        rc++;
                     }
                  }
               }
            }
            if (rc >= 1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /******BTagMu_DiJetXU******/

   else if (isBTagMu_DiJetXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            int njets = 0;

            // apply L2 cut on jets
            for (int i = 0; i < NohBJetL2; i++)
               if (ohBJetL2Et[i] > thresholds[0] && abs(ohBJetL2Eta[i]) < 3.0) // change this ET cut to 20 for the 20U patath
                  njets++;

            // apply b-tag cut
            for (int i = 0; i < NohBJetL2; i++)
            {
               if (ohBJetL2Et[i] > 10.)
               { // keep this at 10 even for the 20UU path - also, no eta cut here
                  if (ohBJetMuL25Tag[i] > 0.5)
                  { // Level 2.5 b tag
                     if (ohBJetPerfL3Tag[i] > 0.5)
                     { // Level 3 b tag
                        rc++;
                     }
                  }
               }
            }
            if (rc >= 1 && njets>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   /**********BTagMu_DiJetXU_MuX***************************************/
   else if (isBTagMu_DiJetXU_MuXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            int njets = 0;

            // apply L2 cut on jets
            for (int i = 0; i < NohBJetL2; i++)
               if (ohBJetL2Et[i] > thresholds[0] && abs(ohBJetL2Eta[i]) < 3.0)
                  njets++;

            // apply b-tag cut
            for (int i = 0; i < NohBJetL2; i++)
            {
               if (ohBJetL2Et[i] > 10.)
               { // keep this at 10 even for all btag mu paths
                  if (ohBJetMuL25Tag[i] > 0.5)
                  { // Level 2.5 b tag
                     if (OpenHlt1L3MuonPassed(thresholds[1], 5.0) >=1)
                     {//require at least one L3 muon
                        if (ohBJetPerfL3Tag[i] > 0.5)
                        { // Level 3 b tag
                           rc++;
                        }
                     }
                  }
               }
            }
            if (rc >= 1 && njets>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isBTagMu_DiJetX_MuXTrigger(triggerName, thresholds))
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       int rc = 0;
	       int njets = 0;

	       // apply L2 cut on jets
	       for (int i = 0; i < NohBJetL2Corrected; i++)
		 if (ohBJetL2CorrectedEt[i] > thresholds[0] && abs(ohBJetL2Eta[i]) < 3.0)
		   njets++;

	       // apply b-tag cut
	       for (int i = 0; i < NohBJetL2Corrected; i++)
		 {
		   if (ohBJetL2CorrectedEt[i] > 10.)
		     { // keep this at 10 even for all btag mu paths
		       if (ohBJetMuL25Tag[i] > 0.5)
			 { // Level 2.5 b tag
			   if (OpenHlt1L3MuonPassed(thresholds[1], 5.0) >=1)
			     {//require at least one L3 muon
			       if (ohBJetPerfL3Tag[i] > 0.5)
				 { // Level 3 b tag
				   rc++;
				 }
			     }
			 }
		     }
		 }
	       if (rc >= 1 && njets>=2)
		 {
		   triggerBit[it] = true;
		 }
	     }
	 }
     }

   /****************BTagIP_JetX*********************************/

   else if (isBTagIP_JetXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            int max = (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected;
            for (int i = 0; i < max; i++)
            {
               if (ohBJetL2CorrectedEt[i] > thresholds[0])
               { // ET cut 
                  if (ohBJetIPL25Tag[i] > 2.5)
                  { // Level 2.5 b tag 
                     if (ohBJetIPL3Tag[i] > 3.5)
                     { // Level 3 b tag 
                        rc++;
                     }
                  }
               }
            }
            if (rc >= 1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /****************QuadJetX_BTagIP*********************************/

   else if (isQuadJetX_BTagIPTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltQuadCorJetPassed(thresholds[0])>=1)
            {
               int rc = 0;
               int max = (NohBJetL2Corrected > 2) ? 2 : NohBJetL2Corrected;
               for (int i = 0; i < max; i++)
               {
                  if (ohBJetL2CorrectedEt[i] > thresholds[0])
                  { // ET cut 
                     if (ohBJetIPL25Tag[i] > 2.5)
                     { // Level 2.5 b tag 
                        if (ohBJetIPL3Tag[i] > 3.5)
                        { // Level 3 b tag 
                           rc++;
                        }
                     }
                  }
               }
               if (rc >= 1)
               {
                  triggerBit[it] = true;
               }
            }
         }
      }
   }


   /**********************************************/
   else if (triggerName.CompareTo("OpenHLT_Mu15_BTagIP_CentJet20U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            for (int i = 0; i < NohBJetL2; i++)
            {
               if (ohBJetL2Et[i] > 20. && fabs(ohBJetL2Eta[i]) < 3.0)
               { // ET and eta cuts
                  if (ohBJetIPL25Tag[i] >= 0)
                  { // Level 2.5 b tag  
                     if (ohBJetIPL3Tag[i] >= 2.0)
                     { // Level 3 b tag  
                        rc++;
                     }
                  }
               }
            }
            if (rc >= 1 && OpenHlt1MuonPassed(7., 7., 15., 2., 0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu15_BTagIP_CentJet20U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int rc = 0;
            for (int i = 0; i < NohBJetL2; i++)
            {
               if (ohBJetL2Et[i] > 20. && fabs(ohBJetL2Eta[i]) < 3.0)
               { // ET and eta cuts
                  if (ohBJetIPL25Tag[i] >= 0)
                  { // Level 2.5 b tag  
                     if (ohBJetIPL3Tag[i] >= 2.0)
                     { // Level 3 b tag  
                        rc++;
                     }
                  }
               }
            }
            if (rc >= 1 && OpenHlt1MuonPassed(7., 7., 15., 2., 1)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_BTagIP_TripleJet20U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            int njets = 0;
            int ntaggedjets = 0;
            int max = (NohBJetL2 > 2) ? 2 : NohBJetL2;
            for (int i = 0; i < max; i++)
            {
               if (ohBJetL2Et[i] > 20.)
               { // ET cut on uncorrected jets 
                  njets++;
                  if (ohBJetPerfL25Tag[i] > 0.5)
                  { // Level 2.5 b tag 
                     if (ohBJetPerfL3Tag[i] > 0.5)
                     { // Level 3 b tag 
                        ntaggedjets++;
                     }
                  }
               }
            }
            if (njets > 2 && ntaggedjets > 0)
            { // Require >= 3 jets, and >= 1 tagged jet
               triggerBit[it] = true;
            }
         }
      }
   }

   /*Electron-jet cross-triggers*/
   //to be removed? cant find anything looking like this in confdb
   else if (triggerName.CompareTo("OpenHLT_Ele15_SW_CaloIdVT_TrkIdT_TrkIsoT_CaloIsoT_L1R_CleanedJet35Jet20_Deta2")
         == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second == 1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it) )
         {
            int EtMaxIt = -1;
            std::vector<int> EleIts;
            if (OpenHlt1ElectronVbfEleIDPassed(15., 12., true, EtMaxIt, &EleIts)
                  >= 1)
            {
               if (OpenHltCleanedDiJetPassed(
                     35.,
                     20.,
                     true,
                     "Calo",
                     2.,
                     0.,
                     false,
                     false,
                     EleIts) >= 1)
                  triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999,
                  999, // Track/pT iso barrel, Track/pT iso endcap  
                  0.2,
                  0.2, // H/ET iso barrel, H/ET iso endcap  
                  0.2,
                  0.2, // E/ET iso barrel, E/ET iso endcap  
                  0.15,
                  0.1, // H/E barrel, H/E endcap  
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  999,
                  999, // Deta barrel, Deta endcap  
                  999,
                  999 // Dphi barrel, Dphi endcap  
            )>=1 && OpenHlt1CorJetPassed(40)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   //  else if(triggerName.CompareTo("OpenHLT_Ele27_SW_TighterEleId_L1R_BTagIP_CentJet20U") == 0) { 
   else if (triggerName.CompareTo("OpenHLT_Ele25_CaloIdVT_TrkIdT_CentralJet40_BTagIP_v1")
         == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(25., 0, // ET, L1isolation 
                  999.,
                  999., // Track iso barrel, Track iso endcap 
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap 
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap 
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap 
                  0.05,
                  0.05, // H/E barrel, H/E endcap 
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap 
                  0.98,
                  1.0, // R9 barrel, R9 endcap 
                  0.008,
                  0.008, // Deta barrel, Deta endcap 
                  0.07,
                  0.05 // Dphi barrel, Dphi endcap 
            )>=1 && OpenHlt1BJetPassedEleRemoval(20., 3.0, 0.3, // jet ET, eta, DrCut
                  0.,
                  2.0, // discL25, discL3
                  27.,
                  0, // ET, L1isolation 
                  999.,
                  999., // Track iso barrel, Track iso endcap 
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap 
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap 
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap 
                  0.05,
                  0.05, // H/E barrel, H/E endcap 
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap 
                  0.98,
                  1.0, // R9 barrel, R9 endcap 
                  0.008,
                  0.007, // Deta barrel, Deta endcap 
                  0.1,
                  0.1 // Dphi barrel, Dphi endcap 
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(25., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap  
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap  
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap  
                  0.05,
                  0.05, // H/E barrel, H/E endcap  
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  0.008,
                  0.008, // Deta barrel, Deta endcap  
                  0.07,
                  0.05 // Dphi barrel, Dphi endcap  
            )>=1 && OpenHlt1CorJetPassed(30, 3.0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Ele25_CaloIdVT_TrkIdT_CentralDiJet30_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(25., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap  
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap  
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap  
                  0.05,
                  0.05, // H/E barrel, H/E endcap  
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  0.008,
                  0.008, // Deta barrel, Deta endcap  
                  0.07,
                  0.05 // Dphi barrel, Dphi endcap  
            )>=1 && OpenHlt1CorJetPassed(30, 3.0)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Ele25_CaloIdVT_TrkIdT_CentralTriJet30_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1ElectronSamHarperPassed(25., 0, // ET, L1isolation  
                  999.,
                  999., // Track iso barrel, Track iso endcap  
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap  
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap  
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap  
                  0.05,
                  0.05, // H/E barrel, H/E endcap  
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap  
                  0.98,
                  1.0, // R9 barrel, R9 endcap  
                  0.008,
                  0.008, // Deta barrel, Deta endcap  
                  0.07,
                  0.05 // Dphi barrel, Dphi endcap  
            )>=1 && OpenHlt1CorJetPassed(30, 3.0)>=3)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /* Minbias */
   else if (triggerName.CompareTo("OpenHLT_MinBiasBSC_OR") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            bool techTriggerBSCOR = (bool) L1Tech_BSC_minBias_OR_v0;
            if (techTriggerBSCOR)
               triggerBit[it] = true;
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_MinBiasBSC") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            bool techTriggerBSC1 = (bool) L1Tech_BSC_minBias_threshold1_v0;
            bool techTriggerBSC2 = (bool) L1Tech_BSC_minBias_threshold2_v0;
            bool techTriggerBS3 = (bool) L1Tech_BSC_minBias_inner_threshold1_v0;
            bool techTriggerBS4 = (bool) L1Tech_BSC_minBias_inner_threshold2_v0;

            if (techTriggerBSC1 || techTriggerBSC2 || techTriggerBS3
                  || techTriggerBS4)
               triggerBit[it] = true;
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_MinBiasPixel_SingleTrack") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PixelTrackPassed(0.0, 1.0, 0.0)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_MinBiasPixel_DoubleTrack") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PixelTrackPassed(0.0, 1.0, 0.0)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_MinBiasPixel_DoubleIsoTrack5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1PixelTrackPassed(5.0, 1.0, 1.0)>=0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_ZeroBias") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (map_BitOfStandardHLTPath.find("OpenL1_ZeroBias")->second == 1)
               triggerBit[it] = true;
         }
      }
   }

   /* AlCa */
   else if (triggerName.CompareTo("OpenAlCa_HcalPhiSym") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (ohHighestEnergyHFRecHit > 0 || ohHighestEnergyHBHERecHit > 0)
            {
               // Require one RecHit with E > 0 MeV in any HCAL subdetector
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoTrackHB") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {

            bool passL2=false;
            for (int itrk=0; itrk<ohIsoPixelTrackHBL2N; itrk++)
            {
               if (ohIsoPixelTrackHBL2P[itrk]>8.0
                     && TMath::Abs(ohIsoPixelTrackHBL2Eta[itrk])>0.0
                     && TMath::Abs(ohIsoPixelTrackHBL2Eta[itrk])<1.3
                     && ohIsoPixelTrackHBL2MaxNearP[itrk]<2.0)
                  passL2=true;
            }

            bool passL3=false;
            for (int itrk=0; itrk<ohIsoPixelTrackHBL3N; itrk++)
            {
               if (ohIsoPixelTrackHBL3P[itrk]>20.0
                     && TMath::Abs(ohIsoPixelTrackHBL3Eta[itrk])>0.0
                     && TMath::Abs(ohIsoPixelTrackHBL3Eta[itrk])<1.3
                     && ohIsoPixelTrackHBL3MaxNearP[itrk]<2.0)
                  passL3=true;
            }

            if (passL2 && passL3)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoTrackHE") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {

            bool passL2=false;
            for (int itrk=0; itrk<ohIsoPixelTrackHEL2N; itrk++)
            {
               if (ohIsoPixelTrackHEL2P[itrk]>12.0
                     && TMath::Abs(ohIsoPixelTrackHEL2Eta[itrk])>0.0
                     && TMath::Abs(ohIsoPixelTrackHEL2Eta[itrk])<2.2
                     && ohIsoPixelTrackHEL2MaxNearP[itrk]<2.0)
                  passL2=true;
            }

            bool passL3=false;
            for (int itrk=0; itrk<ohIsoPixelTrackHEL3N; itrk++)
            {
               if (ohIsoPixelTrackHEL3P[itrk]>38.0
                     && TMath::Abs(ohIsoPixelTrackHEL3Eta[itrk])>0.0
                     && TMath::Abs(ohIsoPixelTrackHEL3Eta[itrk])<2.2
                     && ohIsoPixelTrackHEL3MaxNearP[itrk]<2.0)
                  passL3=true;
            }

            if (passL2 && passL3)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /*Photon-electron cross-triggers*/
   // Not finished yet. 


//   else if (triggerName.CompareTo("OpenHLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v1") == 0)
//    {
//       if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
//       {
//          if (prescaleResponse(menu, cfg, rcounter, it))
//          {
// 	   if ((OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation  
//                   999.,
//                   999., // Track iso barrel, Track iso endcap  
//                   999,
//                   999, // Track/pT iso barrel, Track/pT iso endcap  
//                   0.2,
//                   0.2, // H/ET iso barrel, H/ET iso endcap  
//                   0.2,
//                   0.2, // E/ET iso barrel, E/ET iso endcap  
//                   0.15,
//                   0.1, // H/E barrel, H/E endcap  
//                   0.014,
//                   0.035, // cluster shape barrel, cluster shape endcap  
//                   0.98,
//                   1.0, // R9 barrel, R9 endcap  
//                   999,
//                   999, // Deta barrel, Deta endcap  
//                   999,
//                   999) // Dphi barrel, Dphi endcap  
// 		)>=1
//                   && OpenHlt1PhotonSamHarperPassed(20., 0, // ET, L1isolation
//                         999.,
//                         999., // Track iso barrel, Track iso endcap
//                         3.0,
//                         3.0, // Track/pT iso barrel, Track/pT iso endcap
//                         3.0,
//                         3.0, // H/ET iso barrel, H/ET iso endcap
//                         5.0,
//                         5.0, // E/ET iso barrel, E/ET iso endcap
//                         0.05,
//                         0.05, // H/E barrel, H/E endcap  
//                         0.011,
//                         0.031, // cluster shape barrel, cluster shape endcap  
//                         0.98,
//                         999., // R9 barrel, R9 endcap
//                         999.,
//                         999., // Deta barrel, Deta endcap
//                         999.,
//                         999. // Dphi barrel, Dphi endcap
//                         )>=1)
//             {
//                triggerBit[it] = true;
//             }
//          }
//       }
//    }


   /* muon-jet/MET/HT cross-triggers */
   else if (triggerName.CompareTo("OpenHLT_Mu17_CentralJet30") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 17., 2., 0)>=1
                  && OpenHlt1CorJetPassed( 30, 2.6)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu17_DiCentralJet30") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 17., 2., 0)>=1
                  && OpenHlt1CorJetPassed( 30, 2.6)>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu17_TriCentralJet30") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 17., 2., 0)>=1
                  && OpenHlt1CorJetPassed( 30, 2.6)>=3)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_DoubleMu3_HT100U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(0., 0., 3., 2., 0)>=2 && OpenHltSumHTPassed(
                  100,
                  20)>0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_DoubleMu3_HT160") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(0., 0., 3., 2., 0)>=2
                  && OpenHltSumCorHTPassed( 160., 30.)>0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_DoubleMu3_HT200") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(0., 0., 3., 2., 0)>=2
                  && OpenHltSumCorHTPassed(200., 30.)>0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu5_HT50U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && OpenHltSumHTPassed(
                  50,
                  20)>0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_HT70U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && OpenHltSumHTPassed(
                  70,
                  20)>0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_MET20") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && recoMetCal>45)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_MET45") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && recoMetCal>45)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_Jet30U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1
                  && OpenHlt1JetPassed(30)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu5_Jet35U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1
                  && OpenHlt1JetPassed(35)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_Jet50U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1
                  && OpenHlt1JetPassed(50)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu8_Jet40_v1") == 0)
     {
       if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
	 {
	   if (prescaleResponse(menu, cfg, rcounter, it))
	     {
	       if (OpenHlt1JetPassed(40.)>=1)
		 {
		   if (OpenHlt1MuonPassed(3., 4., 8., 2., 0)>=1)
		     triggerBit[it] = true;
		 }
	     }
	 }
     }
   else if (triggerName.CompareTo("OpenHLT_Mu5_Jet70") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1JetPassed(70.)>=1)
            {
               if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1)
                  triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu5_MET45x") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && recoMetCal>45)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu7_MET20") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 7., 2., 0)>=1 && recoMetCal>20)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu5_HT70U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && OpenHltSumHTPassed(
                  70,
                  20)>0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu5_HT100U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && OpenHltSumHTPassed(
                  100,
                  20)>0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu5_HT120U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && OpenHltSumHTPassed(
               120,
               20)>0)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu5_HT140U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1 && OpenHltSumHTPassed(
                  140,
                  20)>0)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu20_CentralJet20U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 20., 2., 0)>=1 && OpenHlt1JetPassed(
                  20,
                  2.6)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu17_TripleCentralJet20U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 17., 2., 0)>=1 && OpenHlt1JetPassed(
                  20,
                  2.6)>=3)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu5_Jet50U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1
                  && OpenHlt1JetPassed(50) >=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu5_Jet70U") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1
                  && OpenHlt1JetPassed(70) >=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu5_HT50") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(50., 30.) == 1)
            {
               if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1)
                  triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu5_HT80") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(80., 30.) == 1)
            {
               if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1)
                  triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu5_HT200") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(200., 30.) == 1)
            {
               if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1)
                  triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu8_HT50") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(50., 30.) == 1)
            {
               if (OpenHlt1MuonPassed(3., 4., 8., 2., 0)>=1)
                  triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_Mu8_HT200") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(200., 30.) == 1)
            {
               if (OpenHlt1MuonPassed(3., 4., 8., 2., 0)>=1)
                  triggerBit[it] = true;
            }
         }
      }
   }
   
   else if (triggerName.CompareTo("OpenHLT_L2Mu8_HT50") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(50., 30.) == 1)
            {
               if (OpenHlt1L2MuonPassed(7., 8., 9999.)>=1)
               {
                  triggerBit[it] = true;
               }
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_L2Mu10_HT50") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second>0)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(50., 30.) == 1)
            {
               if (OpenHlt1L2MuonPassed(7., 10., 9999.)>=1)
               {
                  triggerBit[it] = true;
               }
            }
         }
      }
   }

   /*muon-Tau cross-triggers*/
   else if (triggerName.CompareTo("OpenHLT_Mu11_PFIsoTau15") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 11., 2., 0)>=1)
               if (OpenHltPFTauPassedNoMuon(15., 1., 1, 1.)>=1)
               {
                  triggerBit[it] = true;
               }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu15_PFIsoTau15") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 15., 2., 0)>=1)
               if (OpenHltPFTauPassedNoMuon(15., 1., 1, 1)>=1)
               {
                  triggerBit[it] = true;
               }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu11_PFIsoTau15") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(7., 7., 11., 2., 1)>=1)
               if (OpenHltPFTauPassedNoMuon(15., 1., 1, 1)>=1)
               {
                  triggerBit[it] = true;
               }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu12_PFIsoTau10_Trk1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(10., 10., 12., 2., 1)>=1)
            {
               if (OpenHltPFTauPassedNoMuon(10., 1., 1., 1.) >=1)
                  triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_Mu17_PFIsoTau15_Trk5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(14., 14., 17., 2., 0)>=1)
            {
               if (OpenHltPFTauPassedNoMuon(15., 5., 1., 1.) >=1)
                  triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_IsoMu15_PFIsoTau20_Trk5") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(10., 10., 15., 2., 1)>=1)
            {
               if (OpenHltPFTauPassedNoMuon(20., 5., 1., 1.) >=1)
                  triggerBit[it] = true;
            }
         }
      }
   }

   /*Electron-Tau cross-triggers*/

   else if (triggerName.CompareTo("OpenHLT_Ele15_CaloIdVT_TrkIdT_LooseIsoPFTau15_v1")
         == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1Ele1PFTauPassed(
                  15.,
                  0,
                  999.,
                  999.,
                  999.,
                  999.,
                  999.,
                  999.,
                  999.,
                  999.,
                  0.05,
                  0.05,
                  0.011,
                  0.031,
                  0.98,
                  1.,
                  0.008,
                  0.008,
                  0.07,
                  0.05,
                  15.,
                  2.5,
                  1.,
                  1.5,
                  1000.,
                  0.) >= 1)
               triggerBit[it] = true;
         }
      }
   }

  else if (triggerName.CompareTo("OpenHLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1Ele1PFTauPassed(
                  15.,
                  0,
                  999.,
                  999.,
                  0.125,
                  0.075,
                  0.125,
                  0.075,
                  0.125,
                  0.075,
                  0.05,
                  0.05,
                  0.011,
                  0.031,
                  0.98,
                  1.,
                  0.008,
                  0.008,
                  0.07,
                  0.05,
                  15.,
                  2.5,
                  1.,
                  1.5,
                  1000.,
                  0.) >= 1)
               triggerBit[it] = true;
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1Ele1PFTauPassed(
                  15.,
                  0,
                  999.,
                  999.,
                  0.125,
                  0.075,
                  0.125,
                  0.075,
                  0.125,
                  0.075,
                  0.05,
                  0.05,
                  0.011,
                  0.031,
                  0.98,
                  1.,
                  0.008,
                  0.008,
                  0.07,
                  0.05,
                  20.,
                  2.5,
                  1.,
                  1.5,
                  1000.,
                  0.) >= 1)
               triggerBit[it] = true;
         }
      }
   }

   /* Tau-jet/MET cross-triggers */
   else if (isQuadJetX_IsoPFTauXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltQuadJetPassedPlusTauPFId(
                  thresholds[0],
                  2.5,
                  thresholds[1]) == 1 && OpenL1QuadJet8(10, 2.5) >= 4)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isQuadJetX_IsoPFTauX_PFMHTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltQuadJetPassedPlusTauPFId(
                  thresholds[0],
                  2.5,
                  thresholds[1]) == 1 && OpenL1QuadJet8(10, 2.5) >= 4)
            {
               if (pfMHT > thresholds[2])
               {
                  triggerBit[it] = true;
               }
            }
         }
      }
   }

   /***********OpenHLT_SingleIsoTauX_TrkX_METX***********/
   else if (isSingleIsoTauX_TrkX_METXTrigger(
         triggerName,
         thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltTauL2SCMETPassed(
                  thresholds[0],
                  thresholds[1],
                  0,
                  0.,
                  1,
                  thresholds[2],
                  20.,
                  30.)>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /*****************************************************/

   else if (triggerName.CompareTo("OpenHLT_SingleTau5_Trk0_MET50_Level1_10") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (recoMetCal>50)
            {
               //                if(OpenHltTauL2SCMETPassed(5.,0.,0,0.,0,50.,10.,10.)>=1) {
               triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_SinglePFIsoTau50_Trk15_PFMHT40") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPFTauPassedNoMuon(50., 15., 1, 1)>=1)
            {
               if (pfMHT > 40)
                  triggerBit[it] = true;
            }
         }
      }
   }
   else if (triggerName.CompareTo("OpenHLT_SinglePFIsoTau30_Trk15_PFMHT50") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPFTauPassedNoMuon(30., 15., 1, 1)>=1)
            {
               if (pfMHT > 50)
                  triggerBit[it] = true;
            }
         }
      }
   }

   /* Electron-MET cross-triggers */
   
   else if (triggerName.CompareTo("OpenHLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_HT200_v2") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if ( (OpenHlt1ElectronSamHarperPassed(10., 0, // ET, L1isolation 
                  999.,
                  999., // Track iso barrel, Track iso endcap 
                  0.2,
                  0.2, // Track/pT iso barrel, Track/pT iso endcap 
                  0.2,
                  0.2, // H/ET iso barrel, H/ET iso endcap 
                  0.2,
                  0.2, // E/ET iso barrel, E/ET iso endcap 
                  0.15,
                  0.10, // H/E barrel, H/E endcap 
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap 
                  0.98,
                  1.0, // R9 barrel, R9 endcap 
                  0.01,
                  0.01, // Deta barrel, Deta endcap 
                  0.15,
                  0.10 // Dphi barrel, Dphi endcap 
            )>=1)
            && 
            (OpenHltSumCorHTPassed(200, 40)>=1))
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Ele10_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT200_v2") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if ( (OpenHlt1ElectronSamHarperPassed(10., 0, // ET, L1isolation 
                  999.,
                  999., // Track iso barrel, Track iso endcap 
                  0.2,
                  0.2, // Track/pT iso barrel, Track/pT iso endcap 
                  0.2,
                  0.2, // H/ET iso barrel, H/ET iso endcap 
                  0.2,
                  0.2, // E/ET iso barrel, E/ET iso endcap 
                  0.1,
                  0.075, // H/E barrel, H/E endcap 
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap 
                  0.98,
                  1.0, // R9 barrel, R9 endcap 
                  0.008,
                  0.008, // Deta barrel, Deta endcap 
                  0.07,
                  0.05 // Dphi barrel, Dphi endcap 
            )>=1)
            && 
            (OpenHltSumCorHTPassed(200, 40)>=1))
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /* Jet-MET/HT cross-triggers*/

   /****CentralJetXU_METX*****/
   else if (isCentralJetXU_METXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1JetPassed(thresholds[0], 2.6)>=1 && recoMetCal
                  >=thresholds[1])
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /****CentralJetX_METX*****/
   else if (isCentralJetX_METXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1JetPassed(thresholds[0], 2.6)>=1 && recoMetCal
                  >=thresholds[1])
            {
               triggerBit[it] = true;
            }
         }
      }
   }
  
 
   /********DiJetX(U)_METX***********************/
   else if (isDiJetXU_METXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1JetPassed(thresholds[0])>=2 && recoMetCal
                  >=thresholds[1])
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isDiJetX_METXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1CorJetPassed(thresholds[0])>=2 && recoMetCal
                  >=thresholds[1])
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   /**DiJetX(U)_PTX(U)**/

   else if (isDiJetXU_PTXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPT12U(thresholds[1], thresholds[0]) == 1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isDiJetX_PTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltPT12(thresholds[1], thresholds[0]) == 1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /***********/

   /*HT-MET/MHT cross-triggers*/
   /*****METX_HTX(U)**************/
   else if (isMETX_HTXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumHTPassed(thresholds[1], 20)>=1 && recoMetCal
                  >=thresholds[0])
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isMETX_HTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(thresholds[1], 40)>=1 && recoMetCal
                  >=thresholds[0])
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /***HTX(U)_MHT(X)U*****/

   else if (isHTXU_MHTXUTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltMHTU(thresholds[1], 20.)==1 && (OpenHltSumHTPassed(
                  thresholds[0],
                  20.) == 1))
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (isHTX_MHTXTrigger(triggerName, thresholds))
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltMHT(thresholds[1], 30.)==1 && (OpenHltSumCorHTPassed(
                  thresholds[0],
                  40.) == 1))
            {
               triggerBit[it] = true;
            }
         }
      }
   }
   /*********************/

   /*Muon-photon cross-triggers*/

   else if (triggerName.CompareTo("OpenHLT_Mu8_Photon20_CaloIdVT_IsoT_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(0., 4., 8., 2., 0)>=1
                  && OpenHlt1PhotonSamHarperPassed(20., 0, // ET, L1isolation
                        999.,
                        999., // Track iso barrel, Track iso endcap
                        3.0,
                        3.0, // Track/pT iso barrel, Track/pT iso endcap
                        3.0,
                        3.0, // H/ET iso barrel, H/ET iso endcap
                        5.0,
                        5.0, // E/ET iso barrel, E/ET iso endcap
                        0.05,
                        0.05, // H/E barrel, H/E endcap  
                        0.011,
                        0.031, // cluster shape barrel, cluster shape endcap  
                        0.98,
                        999., // R9 barrel, R9 endcap
                        999.,
                        999., // Deta barrel, Deta endcap
                        999.,
                        999. // Dphi barrel, Dphi endcap
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu15_Photon20_CaloIdL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(5., 7., 15., 2., 0)>=1
                  && OpenHlt1PhotonSamHarperPassed(20., 0, // ET, L1isolation
                        999.,
                        999., // Track iso barrel, Track iso endcap
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap
                        0.15,
                        0.10, // H/E barrel, H/E endcap
                        0.014,
                        0.035, // cluster shape barrel, cluster shape endcap
                        0.98,
                        999., // R9 barrel, R9 endcap
                        999.,
                        999., // Deta barrel, Deta endcap
                        999.,
                        999. // Dphi barrel, Dphi endcap
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   //else if (triggerName.CompareTo("OpenHLT_Mu15_DiPhoton15_CaloIdL_v1") == 0)
   else if (triggerName.CompareTo("OpenHLT_Mu15_DoublePhoton15_CaloIdL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(5., 7., 15., 2., 0)>=1
                  && OpenHlt1PhotonSamHarperPassed(15., 0, // ET, L1isolation
                        999.,
                        999., // Track iso barrel, Track iso endcap
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap
                        0.15,
                        0.10, // H/E barrel, H/E endcap
                        0.014,
                        0.035, // cluster shape barrel, cluster shape endcap
                        0.98,
                        999., // R9 barrel, R9 endcap
                        999.,
                        999., // Deta barrel, Deta endcap
                        999.,
                        999. // Dphi barrel, Dphi endcap
                  )>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   /*Muon-electron cross-triggers*/

   else if (triggerName.CompareTo("OpenHLT_Mu5_DoubleEle8_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1
                  &&OpenHlt2ElectronsSamHarperPassed(8., 0, // ET, L1isolation 
                        999.,
                        999., // Track iso barrel, Track iso endcap 
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap 
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap 
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap 
                        0.15,
                        0.1, // H/E barrel, H/E endcap 
                        999.,
                        999., // cluster shape barrel, cluster shape endcap 
                        0.98,
                        1., // R9 barrel, R9 endcap 
                        999.,
                        999., // Deta barrel, Deta endcap 
                        999.,
                        999. // Dphi barrel, Dphi endcap 
                  )>=2)
            {
               //OpenHlt1ElectronPassed(8.,0,9999.,9999.)>=2){ 
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("HLT_Mu5_Ele8_CaloIdL_TrkIdVL_Ele8_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 4., 5., 2., 0)>=1
                  &&OpenHlt2ElectronsSamHarperPassed(8., 0, // ET, L1isolation 
                        999.,
                        999., // Track iso barrel, Track iso endcap 
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap 
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap 
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap 
                        0.15,
                        0.1, // H/E barrel, H/E endcap 
                        999.,
                        999., // cluster shape barrel, cluster shape endcap 
                        0.98,
                        1., // R9 barrel, R9 endcap 
                        999.,
                        999., // Deta barrel, Deta endcap 
                        999.,
                        999. // Dphi barrel, Dphi endcap 
						     )>=2 &&
	      OpenHlt1ElectronSamHarperPassed(8.,0,          // ET, L1isolation
					       999., 999.,       // Track iso barrel, Track iso endcap
					       999., 999.,        // Track/pT iso barrel, Track/pT iso endcap
					       999., 999.,       // H/ET iso barrel, H/ET iso endcap
					       999., 999.,       // E/ET iso barrel, E/ET iso endcap
					       0.15, 0.10,       // H/E barrel, H/E endcap 
					       0.014, 0.035,       // cluster shape barrel, cluster shape endcap 
					       0.98, 1.,       // R9 barrel, R9 endcap
					       0.01, 0.01,       // Deta barrel, Deta endcap 
					       0.15, 0.10        // Dphi barrel, Dphi endcap
					       )>=1)
            {
               //OpenHlt1ElectronPassed(8.,0,9999.,9999.)>=2){ 
               triggerBit[it] = true;
            }
         }
      }
   }


   else if (triggerName.CompareTo("OpenHLT_DoubleMu5_Ele8_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(0., 0., 5., 2., 0)>=2
                  && OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation 
                        999.,
                        999., // Track iso barrel, Track iso endcap 
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap 
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap 
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap 
                        0.15,
                        0.1, // H/E barrel, H/E endcap 
                        999.,
                        999., // cluster shape barrel, cluster shape endcap 
                        0.98,
                        1.0, // R9 barrel, R9 endcap 
                        999.,
                        999., // Deta barrel, Deta endcap 
                        999.,
                        999. // Dphi barrel, Dphi endcap 
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_DoubleMu5_Ele8_CaloIdL_TrkIdVL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt2MuonPassed(3., 4., 5., 2., 0)>=2
                  && OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation 
                        999.,
                        999., // Track iso barrel, Track iso endcap 
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap 
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap 
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap 
                        0.15,
                        0.10, // H/E barrel, H/E endcap  
                        0.014,
                        0.035, // cluster shape barrel, cluster shape endcap  
                        0.98,
                        1., // R9 barrel, R9 endcap  
                        0.01,
                        0.01, // Deta barrel, Deta endcap  
                        0.15,
                        0.10 // Dphi barrel, Dphi endcap 
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu3_Ele8_CaloIdL_TrkIdVL_HT160_v2") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 3., 3., 2., 0)>=1 && OpenHltSumCorHTPassed(
                  160,
                  40)>0 && OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap
                  0.15,
                  0.10, // H/E barrel, H/E endcap
                  0.014,
                  0.035, // cluster shape barrel, cluster shape endcap
                  0.98,
                  1.0, // R9 barrel, R9 endcap
                  0.01,
                  0.01, // Deta barrel, Deta endcap
                  0.15,
                  0.10 // Dphi barrel, Dphi endcap
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu3_Ele8_CaloIdT_TrkIdVL_HT160_v2") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 3., 3., 2., 0)>=1 && OpenHltSumCorHTPassed(
                  160,
                  40)>0 && OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation
                  999.,
                  999., // Track iso barrel, Track iso endcap
                  999.,
                  999., // Track/pT iso barrel, Track/pT iso endcap
                  999.,
                  999., // H/ET iso barrel, H/ET iso endcap
                  999.,
                  999., // E/ET iso barrel, E/ET iso endcap
                  0.10,
                  0.075, // H/E barrel, H/E endcap 
                  0.011,
                  0.031, // cluster shape barrel, cluster shape endcap 
                  0.98,
                  1.0, // R9 barrel, R9 endcap
                  0.01,
                  0.01, // Deta barrel, Deta endcap
                  0.15,
                  0.10 // Dphi barrel, Dphi endcap
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_DoubleEle8_CaloIdL_TrkIdVL_HT160_v2") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(160, 40)>0
                  && OpenHlt2ElectronsSamHarperPassed(8., 0, // ET, L1isolation
                        999.,
                        999., // Track iso barrel, Track iso endcap
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap
                        0.15,
                        0.1, // H/E barrel, H/E endcap
                        0.014,
                        0.035, // cluster shape barrel, cluster shape endcap
                        0.98,
                        1.0, // R9 barrel, R9 endcap
                        0.1,
                        0.1, // Deta barrel, Deta endcap
                        0.15,
                        0.1 // Dphi barrel, Dphi endcap
                  )>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }


   else if (triggerName.CompareTo("OpenHLT_DoubleEle8_CaloIdT_TrkIdVL_HT160_v2") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHltSumCorHTPassed(160, 40)>0
                  && OpenHlt2ElectronsSamHarperPassed(8., 0, // ET, L1isolation
                        999.,
                        999., // Track iso barrel, Track iso endcap
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap
                        0.1,
                        0.075, // H/E barrel, H/E endcap
                        0.011,
                        0.031, // cluster shape barrel, cluster shape endcap
                        0.98,
                        1.0, // R9 barrel, R9 endcap
                        0.1,
                        0.1, // Deta barrel, Deta endcap
                        0.15,
                        0.1 // Dphi barrel, Dphi endcap
                  )>=2)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu17_Ele8_CaloIdL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 5., 17., 2., 0)>=1
                  && OpenHlt1ElectronSamHarperPassed(8., 0, // ET, L1isolation 
                        999.,
                        999., // Track iso barrel, Track iso endcap 
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap 
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap 
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap 
                        0.15,
                        0.10, // H/E barrel, H/E endcap 
                        0.014,
                        0.035, // cluster shape barrel, cluster shape endcap 
                        0.98,
                        1.0, // R9 barrel, R9 endcap 
                        999.,
                        999., // Deta barrel, Deta endcap 
                        999.,
                        999. // Dphi barrel, Dphi endcap 
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu10_Ele10_CaloIdL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 5., 10., 2., 0)>=1
                  && OpenHlt1ElectronSamHarperPassed(10., 0, // ET, L1isolation 
                        999.,
                        999., // Track iso barrel, Track iso endcap 
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap 
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap 
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap 
                        0.15,
                        0.10, // H/E barrel, H/E endcap 
                        0.014,
                        0.035, // cluster shape barrel, cluster shape endcap 
                        0.98,
                        1.0, // R9 barrel, R9 endcap 
                        999.,
                        999., // Deta barrel, Deta endcap 
                        999.,
                        999. // Dphi barrel, Dphi endcap 
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.CompareTo("OpenHLT_Mu8_Ele17_CaloIdL_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            if (OpenHlt1MuonPassed(3., 5., 8., 2., 0)>=1
                  && OpenHlt1ElectronSamHarperPassed(17., 0, // ET, L1isolation 
                        999.,
                        999., // Track iso barrel, Track iso endcap 
                        999.,
                        999., // Track/pT iso barrel, Track/pT iso endcap 
                        999.,
                        999., // H/ET iso barrel, H/ET iso endcap 
                        999.,
                        999., // E/ET iso barrel, E/ET iso endcap 
                        0.15,
                        0.10, // H/E barrel, H/E endcap 
                        0.014,
                        0.035, // cluster shape barrel, cluster shape endcap 
                        0.98,
                        1.0, // R9 barrel, R9 endcap 
                        999.,
                        999., // Deta barrel, Deta endcap 
                        999.,
                        999. // Dphi barrel, Dphi endcap 
                  )>=1)
            {
               triggerBit[it] = true;
            }
         }
      }
   }

   else if (triggerName.BeginsWith("OpenHLT_Photon26")==1)
   {
      // Photon Paths (V. Rekovic)

      int lowerEt = 18;
      int upperEt = 26;

      // Names of Photon Paths With Mass Cut
      // -------------------------------------

      // No special cuts
      char pathName_Photon_Photon[100];
      sprintf(
            pathName_Photon_Photon,
            "OpenHLT_Photon%d_Photon%d_v1",
            upperEt,
            lowerEt);
      
      // Both legs IsoVL
      char pathName_Photon_IsoVL_Photon_IsoVL[100];
      sprintf(
            pathName_Photon_IsoVL_Photon_IsoVL,
            "OpenHLT_Photon%d_IsoVL_Photon%d_IsoVL_v1",
            upperEt,
            lowerEt);
      
      // One leg IsoVL
      char pathName_Photon_IsoVL_Photon[100];
      sprintf(
            pathName_Photon_IsoVL_Photon,
            "OpenHLT_Photon%d_IsoVL_Photon%d_v1",
            upperEt,
            lowerEt);
      
      // One leg IsoT + Mass>60
      char pathName_Photon_IsoT_Photon_Mass60[100];
      sprintf(
            pathName_Photon_IsoT_Photon_Mass60,
            "OpenHLT_Photon%d_IsoT_Photon%d_Mass60_v1",
            upperEt,
            lowerEt);
      
      // Both legs IsoT  + Mass>60
      char pathName_Photon_IsoT_Photon_IsoT_Mass60[100];
      sprintf(
            pathName_Photon_IsoT_Photon_IsoT_Mass60,
            "OpenHLT_Photon%d_IsoT_Photon%d_IsoT_Mass60_v1",
            upperEt,
            lowerEt);
      
      // Both legs IsoT
      char pathName_Photon_IsoT_Photon_IsoT[100];
      sprintf(
            pathName_Photon_IsoT_Photon_IsoT,
            "OpenHLT_Photon%d_IsoT_Photon%d_IsoT_v1",
            upperEt,
            lowerEt);
      
      // One leg IsoT
      char pathName_Photon_IsoT_Photon[100];
      sprintf(
            pathName_Photon_IsoT_Photon,
            "OpenHLT_Photon%d_IsoT_Photon%d_v1",
            upperEt,
            lowerEt);

      // One leg IsoL
      char pathName_Photon_IsoL_Photon[100];
      sprintf(
            pathName_Photon_IsoL_Photon,
            "OpenHLT_Photon%d_IsoL_Photon%d_v1",
            upperEt,
            lowerEt);
      
      // One leg CaloIdL
      char pathName_Photon_CaloIdL_Photon[100];
      sprintf(
            pathName_Photon_CaloIdL_Photon,
            "OpenHLT_Photon%d_CaloIdL_Photon%d_v1",
            upperEt,
            lowerEt);
      
      // Both legs CaloIdL + IsoVL
      char pathName_Photon_CaloIdL_IsoVL_Photon_CaloIdL_IsoVL[100];
      sprintf(
            pathName_Photon_CaloIdL_IsoVL_Photon_CaloIdL_IsoVL,
            "OpenHLT_Photon%d_CaloIdL_IsoVL_Photon%d_CaloIdL_IsoVL_v1",
            upperEt,
            lowerEt);
      
      // One leg CaloIdL + IsoVL
      char pathName_Photon_CaloIdL_IsoVL_Photon[100];
      sprintf(
            pathName_Photon_CaloIdL_IsoVL_Photon,
            "OpenHLT_Photon%d_CaloIdL_IsoVL_Photon%d_v1",
            upperEt,
            lowerEt);
      
      // One leg CaloIdL + IsoT + Mass>60
      char pathName_Photon_CaloIdL_IsoT_Photon_Mass60[100];
      sprintf(
            pathName_Photon_CaloIdL_IsoT_Photon_Mass60,
            "OpenHLT_Photon%d_CaloIdL_IsoT_Photon%d_Mass60_v1",
            upperEt,
            lowerEt);
      
      // Both legs CaloIdL + IsoT + Mass>60
      char pathName_Photon_CaloIdL_IsoT_Photon_CaloIdL_IsoT_Mass60[100];
      sprintf(
            pathName_Photon_CaloIdL_IsoT_Photon_CaloIdL_IsoT_Mass60,
            "OpenHLT_Photon%d_CaloIdL_IsoT_Photon%d_CaloIdL_IsoT_Mass60_v1",
            upperEt,
            lowerEt);

      // Both legs CaloIdL + IsoT
      char pathName_Photon_CaloIdL_IsoT_Photon_CaloIdL_IsoT[100];
      sprintf(
            pathName_Photon_CaloIdL_IsoT_Photon_CaloIdL_IsoT,
            "OpenHLT_Photon%d_CaloIdL_IsoT_Photon%d_CaloIdL_IsoT_v1",
            upperEt,
            lowerEt);
      
      // One leg CaloIdL + IsoT
      char pathName_Photon_CaloIdL_IsoT_Photon[100];
      sprintf(
            pathName_Photon_CaloIdL_IsoT_Photon,
            "OpenHLT_Photon%d_CaloIdL_IsoT_Photon%d_v1",
            upperEt,
            lowerEt);

      if (triggerName.CompareTo(pathName_Photon_Photon) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        999,
                        999,
                        999,
                        999,
                        0.15,
                        0.10,
                        0.98);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_IsoVL_Photon_IsoVL) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     4.0,
                     6.0,
                     4.0,
                     4.0,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        4.0,
                        6.0,
                        4.0,
                        4.0,
                        0.15,
                        0.10,
                        0.98);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }

      else if (triggerName.CompareTo(pathName_Photon_IsoVL_Photon) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        4.0,
                        6.0,
                        4.0,
                        4.0,
                        0.15,
                        0.10,
                        0.98);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_IsoT_Photon_Mass60) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3,
                        5,
                        3,
                        3,
                        0.15,
                        0.10,
                        0.98);
                  if (secondVector.size()>=1)
                  {

                     TLorentzVector e1;
                     TLorentzVector e2;
                     TLorentzVector meson;
                     float mass = 0.;
                     for (unsigned int i=0; i<firstVector.size(); i++)
                     {
                        for (unsigned int j=0; j<secondVector.size() ; j++)
                        {

                           if (firstVector[i] == secondVector[j])
                              continue;
                           e1.SetPtEtaPhiM(
                                 ohPhotEt[firstVector[i]],
                                 ohPhotEta[firstVector[i]],
                                 ohPhotPhi[firstVector[i]],
                                 0.);
                           e2.SetPtEtaPhiM(
                                 ohPhotEt[secondVector[j]],
                                 ohPhotEta[secondVector[j]],
                                 ohPhotPhi[secondVector[j]],
                                 0.);
                           meson = e1 + e2;
                           mass = meson.M();

                           if (mass>60)
                              triggerBit[it] = true;

                        }
                     }

                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_IsoT_Photon_IsoT_Mass60) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     3,
                     5,
                     3,
                     3,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3,
                        5,
                        3,
                        3,
                        0.15,
                        0.10,
                        0.98);
                  if (secondVector.size()>=1)
                  {

                     TLorentzVector e1;
                     TLorentzVector e2;
                     TLorentzVector meson;
                     float mass = 0.;
                     for (unsigned int i=0; i<firstVector.size(); i++)
                     {
                        for (unsigned int j=0; j<secondVector.size() ; j++)
                        {

                           if (firstVector[i] == secondVector[j])
                              continue;
                           e1.SetPtEtaPhiM(
                                 ohPhotEt[firstVector[i]],
                                 ohPhotEta[firstVector[i]],
                                 ohPhotPhi[firstVector[i]],
                                 0.);
                           e2.SetPtEtaPhiM(
                                 ohPhotEt[secondVector[j]],
                                 ohPhotEta[secondVector[j]],
                                 ohPhotPhi[secondVector[j]],
                                 0.);
                           meson = e1 + e2;
                           mass = meson.M();

                           if (mass>60)
                              triggerBit[it] = true;

                        }
                     }

                  }
               }
            }
         }
      }

      else if (triggerName.CompareTo(pathName_Photon_IsoT_Photon_IsoT) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     3,
                     5,
                     3,
                     3,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3,
                        5,
                        3,
                        3,
                        0.15,
                        0.10,
                        0.98);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_IsoT_Photon) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3,
                        5,
                        3,
                        3,
                        0.15,
                        0.10,
                        0.98);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_IsoL_Photon) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3.5,
                        5.5,
                        3.5,
                        3.5,
                        0.15,
                        0.10,
                        0.98);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }

      else if (triggerName.CompareTo(pathName_Photon_CaloIdL_Photon) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        999,
                        999,
                        999,
                        999,
                        0.15,
                        0.10,
                        0.98,
                        0.014,
                        0.035);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_CaloIdL_IsoVL_Photon_CaloIdL_IsoVL) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     4.0,
                     6.0,
                     4.0,
                     4.0,
                     0.15,
                     0.10,
                     0.98,
                     0.014,
                     0.035);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        4.0,
                        6.0,
                        4.0,
                        4.0,
                        0.15,
                        0.10,
                        0.98,
                        0.014,
                        0.035);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }

      else if (triggerName.CompareTo(pathName_Photon_CaloIdL_IsoVL_Photon) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        4.0,
                        6.0,
                        4.0,
                        4.0,
                        0.15,
                        0.10,
                        0.98,
                        0.014,
                        0.035);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_CaloIdL_IsoT_Photon_Mass60) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3,
                        5,
                        3,
                        3,
                        0.15,
                        0.10,
                        0.98,
                        0.014,
                        0.035);
                  if (secondVector.size()>=1)
                  {

                     TLorentzVector e1;
                     TLorentzVector e2;
                     TLorentzVector meson;
                     float mass = 0.;
                     for (unsigned int i=0; i<firstVector.size(); i++)
                     {
                        for (unsigned int j=0; j<secondVector.size() ; j++)
                        {

                           if (firstVector[i] == secondVector[j])
                              continue;
                           e1.SetPtEtaPhiM(
                                 ohPhotEt[firstVector[i]],
                                 ohPhotEta[firstVector[i]],
                                 ohPhotPhi[firstVector[i]],
                                 0.);
                           e2.SetPtEtaPhiM(
                                 ohPhotEt[secondVector[j]],
                                 ohPhotEta[secondVector[j]],
                                 ohPhotPhi[secondVector[j]],
                                 0.);
                           meson = e1 + e2;
                           mass = meson.M();

                           if (mass>60)
                              triggerBit[it] = true;

                        }
                     }

                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_CaloIdL_IsoT_Photon_CaloIdL_IsoT_Mass60) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     3,
                     5,
                     3,
                     3,
                     0.15,
                     0.10,
                     0.98,
                     0.014,
                     0.035);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3,
                        5,
                        3,
                        3,
                        0.15,
                        0.10,
                        0.98,
                        0.014,
                        0.035);
                  if (secondVector.size()>=1)
                  {

                     TLorentzVector e1;
                     TLorentzVector e2;
                     TLorentzVector meson;
                     float mass = 0.;
                     for (unsigned int i=0; i<firstVector.size(); i++)
                     {
                        for (unsigned int j=0; j<secondVector.size() ; j++)
                        {

                           if (firstVector[i] == secondVector[j])
                              continue;
                           e1.SetPtEtaPhiM(
                                 ohPhotEt[firstVector[i]],
                                 ohPhotEta[firstVector[i]],
                                 ohPhotPhi[firstVector[i]],
                                 0.);
                           e2.SetPtEtaPhiM(
                                 ohPhotEt[secondVector[j]],
                                 ohPhotEta[secondVector[j]],
                                 ohPhotPhi[secondVector[j]],
                                 0.);
                           meson = e1 + e2;
                           mass = meson.M();

                           if (mass>60)
                              triggerBit[it] = true;

                        }
                     }
                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_CaloIdL_IsoT_Photon_CaloIdL_IsoT) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     3,
                     5,
                     3,
                     3,
                     0.15,
                     0.10,
                     0.98,
                     0.014,
                     0.035);
               if (firstVector.size()>=2)
               {
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3,
                        5,
                        3,
                        3,
                        0.15,
                        0.10,
                        0.98,
                        0.014,
                        0.035);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }
      
      else if (triggerName.CompareTo(pathName_Photon_CaloIdL_IsoT_Photon) == 0)
      {
         if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
         {
            if (prescaleResponse(menu, cfg, rcounter, it))
            {
               std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                     lowerEt,
                     0,
                     999,
                     999,
                     999,
                     999,
                     0.15,
                     0.10,
                     0.98);
               if (firstVector.size()>=2)
               {
                  //if(OpenHlt1PhotonPassed(upperEt,0,3,5,3,3,0.15,0.98,0.014,0.035)>=1) {      
                  std::vector<int> secondVector = VectorOpenHlt1PhotonPassed(
                        upperEt,
                        0,
                        3,
                        5,
                        3,
                        3,
                        0.15,
                        0.10,
                        0.98,
                        0.014,
                        0.035);
                  if (secondVector.size()>=1)
                  {
                     triggerBit[it] = true;
                  }
               }
            }
         }
      }

   }

   else if (triggerName.CompareTo("OpenHLT_Photon20_R9Id_Photon18_R9Id_v1") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            std::vector<int> firstVector = VectorOpenHlt1PhotonPassedR9ID(
                  18,
                  0.8,
                  0,
                  999,
                  999,
                  999,
                  999,
                  0.05,
                  0.98);

            if (firstVector.size()>=2)
            {
               std::vector<int> secondVector =
                     VectorOpenHlt1PhotonPassedR9ID(
                           20,
                           0.8,
                           0,
                           999,
                           999,
                           999,
                           999,
                           0.05,
                           0.98);

               if (secondVector.size()>=1)
               {
                  triggerBit[it] = true;
               }
            }
         }
      }
   }
   
   // to be removed?
   // (added by Hartl when merging in new paths from V. Rekovic; most were obsolete but two using R9ID where new... including this)
   else if (triggerName.CompareTo("OpenHLT_Photon20_R9ID_Photon18_Isol_CaloId_L1R") == 0)
   {
      if (map_L1BitOfStandardHLTPath.find(triggerName)->second==1)
      {
         if (prescaleResponse(menu, cfg, rcounter, it))
         {
            std::vector<int> firstVector = VectorOpenHlt1PhotonPassed(
                  18,
                  0,
                  3,
                  5,
                  3,
                  3,
                  0.05,
                  0.98,
                  0.014,
                  0.035);
            if (firstVector.size()>=1)
            {
               std::vector<int> secondVector =
                     VectorOpenHlt1PhotonPassedR9ID(
                           20,
                           0.8,
                           0,
                           999,
                           999,
                           999,
                           999,
                           0.05,
                           0.98);
               if (secondVector.size()>=1)
               {
                  if ((firstVector.size() == 1)&&(secondVector.size()==1))
                  {
                     if (firstVector.front() != secondVector.front())
                        triggerBit[it] = true;
                  }
                  else
                     triggerBit[it] = true;
               }
            }
         }
      }
   }
   else
   {
      if (nMissingTriggerWarnings < 100)
         cout << "Warning: the requested trigger " << triggerName
               << " is not implemented in OHltTreeOpen. No rate will be calculated."
               << endl;
      nMissingTriggerWarnings++;
   }
}

// functions


void OHltTree::PrintOhltVariables(int level, int type)
{
   cout << "Run " << Run <<", Event " << Event << endl;
   switch (type)
   {
   case muon:

      if (level == 3)
      {

         cout << "Level 3: number of muons = " << NohMuL3 << endl;

         for (int i=0; i<NohMuL3; i++)
         {
            cout << "ohMuL3Pt["<<i<<"] = " << ohMuL3Pt[i] << endl;
            cout << "ohMuL3PtErr["<<i<<"] = " << ohMuL3PtErr[i] << endl;
            cout << "ohMuL3Pt+Err["<<i<<"] = " << ohMuL3Pt[i]+2.2
                  *ohMuL3PtErr[i]*ohMuL3Pt[i] << endl;
            cout << "ohMuL3Phi["<<i<<"] = " << ohMuL3Phi[i] << endl;
            cout << "ohMuL3Eta["<<i<<"] = " << ohMuL3Eta[i] << endl;
            cout << "ohMuL3Chg["<<i<<"] = " << ohMuL3Chg[i] << endl;
            cout << "ohMuL3Iso["<<i<<"] = " << ohMuL3Iso[i] << endl;
            cout << "ohMuL3Dr["<<i<<"] = " << ohMuL3Dr[i] << endl;
            cout << "ohMuL3Dz["<<i<<"] = " << ohMuL3Dz[i] << endl;
            cout << "ohMuL3L2idx["<<i<<"] = " << ohMuL3L2idx[i] << endl;
         }
      }
      else if (level == 2)
      {
         cout << "Level 2: number of muons = " << NohMuL2 << endl;
         for (int i=0; i<NohMuL2; i++)
         {
            cout << "ohMuL2Pt["<<i<<"] = " << ohMuL2Pt[i] << endl;
            cout << "ohMuL2PtErr["<<i<<"] = " << ohMuL2PtErr[i] << endl;
            cout << "ohMuL2Pt+Err["<<i<<"] = " << ohMuL2Pt[i]+3.9
                  *ohMuL2PtErr[i]*ohMuL2Pt[i] << endl;
            cout << "ohMuL2Phi["<<i<<"] = " << ohMuL2Phi[i] << endl;
            cout << "ohMuL2Eta["<<i<<"] = " << ohMuL2Eta[i] << endl;
            cout << "ohMuL2Chg["<<i<<"] = " << ohMuL2Chg[i] << endl;
            cout << "ohMuL2Iso["<<i<<"] = " << ohMuL2Iso[i] << endl;
            cout << "ohMuL2Dr["<<i<<"] = " << ohMuL2Dr[i] << endl;
            cout << "ohMuL2Dz["<<i<<"] = " << ohMuL2Dz[i] << endl;
         }
      }
      else if (level == 1)
      {
         for (int i=0; i<NL1Mu; i++)
         {
            cout << "L1MuPt["<<i<<"] = " << L1MuPt[i] << endl;
            cout << "L1MuEta["<<i<<"] = " << L1MuEta[i] << endl;
            cout << "L1MuPhi["<<i<<"] = " << L1MuPhi[i] << endl;
            cout << "L1MuIsol["<<i<<"] = " << L1MuIsol[i] << endl;
            cout << "L1MuQal["<<i<<"] = " << L1MuQal[i] << endl;
         }
      }
      else
      {
         cout
               << "PrintOhltVariables: Ohlt has Muon variables only for L1, 2, and 3. Must provide one."
               << endl;
      }
      break;

   case electron:
      cout << "oh: number of electrons = " << NohEle << endl;
      for (int i=0; i<NohEle; i++)
      {
         float ohElePt = ohEleP[i] * TMath::Sin(2*TMath::ATan(TMath::Exp(-1
               *ohEleEta[i])));
         cout << "ohEleEt["<<i<<"] = " << ohEleEt[i] << endl;
         cout << "ohElePhi["<<i<<"] = " << ohElePhi[i] << endl;
         cout << "ohEleEta["<<i<<"] = " << ohEleEta[i] << endl;
         cout << "ohEleE["<<i<<"] = " << ohEleE[i] << endl;
         cout << "ohEleP["<<i<<"] = " << ohEleP[i] << endl;
         cout << "ohElePt["<<i<<"] =" << ohElePt << endl;
         cout << "ohEleHiso["<<i<<"] = " << ohEleHiso[i] << endl;
         cout << "ohEleTiso["<<i<<"] = " << ohEleTiso[i] << endl;
         cout << "ohEleL1iso["<<i<<"] = " << ohEleL1iso[i] << endl;
         cout << "ohEleHiso["<<i<<"]/ohEleEt["<<i<<"] = " << ohEleHiso[i]
               /ohEleEt[i] << endl;
         cout << "ohEleEiso["<<i<<"]/ohEleEt["<<i<<"] = " << ohEleEiso[i]
               /ohEleEt[i] << endl;
         cout << "ohEleTiso["<<i<<"]/ohEleEt["<<i<<"] = " << ohEleTiso[i]
               /ohEleEt[i] << endl;
         cout << "ohEleHforHoverE["<<i<<"] = " << ohEleHforHoverE[i] << endl;
         cout << "ohEleHforHoverE["<<i<<"]/ohEleE["<<i<<"] = "
               << ohEleHforHoverE[i]/ohEleE[i] << endl;
         cout << "ohEleNewSC["<<i<<"] = " << ohEleNewSC[i] << endl;
         cout << "ohElePixelSeeds["<<i<<"] = " << ohElePixelSeeds[i] << endl;
         cout << "ohEleClusShap["<<i<<"] = " << ohEleClusShap[i] << endl;
         cout << "ohEleR9["<<i<<"] = " << ohEleR9[i] << endl;
         cout << "ohEleDeta["<<i<<"] = " << ohEleDeta[i] << endl;
         cout << "ohEleDphi["<<i<<"] = " << ohEleDphi[i] << endl;
      }

      for (int i=0; i<NL1IsolEm; i++)
      {
         cout << "L1IsolEmEt["<<i<<"] = " << L1IsolEmEt[i] << endl;
         cout << "L1IsolEmE["<<i<<"] = " << L1IsolEmE[i] << endl;
         cout << "L1IsolEmEta["<<i<<"] = " << L1IsolEmEta[i] << endl;
         cout << "L1IsolEmPhi["<<i<<"] = " << L1IsolEmPhi[i] << endl;
      }
      for (int i=0; i<NL1NIsolEm; i++)
      {
         cout << "L1NIsolEmEt["<<i<<"] = " << L1NIsolEmEt[i] << endl;
         cout << "L1NIsolEmE["<<i<<"] = " << L1NIsolEmE[i] << endl;
         cout << "L1NIsolEmEta["<<i<<"] = " << L1NIsolEmEta[i] << endl;
         cout << "L1NIsolEmPhi["<<i<<"] = " << L1NIsolEmPhi[i] << endl;
      }

      break;

   case photon:

      cout << "oh: number of photons = " << NohPhot << endl;
      for (int i=0; i<NohPhot; i++)
      {
         cout << "ohPhotEt["<<i<<"] = " << ohPhotEt[i] << endl;
         cout << "ohPhotPhi["<<i<<"] = " << ohPhotPhi[i] << endl;
         cout << "ohPhotEta["<<i<<"] = " << ohPhotEta[i] << endl;
         cout << "ohPhotEiso["<<i<<"] = " << ohPhotEiso[i] << endl;
         cout << "ohPhotHiso["<<i<<"] = " << ohPhotHiso[i] << endl;
         cout << "ohPhotTiso["<<i<<"] = " << ohPhotTiso[i] << endl;
         cout << "ohPhotL1iso["<<i<<"] = " << ohPhotL1iso[i] << endl;
         cout << "ohPhotHiso["<<i<<"]/ohPhotEt["<<i<<"] = " << ohPhotHiso[i]
               /ohPhotEt[i] << endl;
         cout << "recoPhotE["<<i<<"] = " << recoPhotE[i] << endl;
         cout << "recoPhotEt["<<i<<"] = " << recoPhotEt[i] << endl;
         cout << "recoPhotPt["<<i<<"] = " << recoPhotPt[i] << endl;
         cout << "recoPhotPhi["<<i<<"] = " << recoPhotPhi[i] << endl;
         cout << "recoPhotEta["<<i<<"] = " << recoPhotEta[i] << endl;

      }
      break;

   case jet:
      cout << "oh: number of recoJetCal = " << NrecoJetCal << endl;
      for (int i=0; i<NrecoJetCal; i++)
      {
         cout << "recoJetCalE["<<i<<"] = " << recoJetCalE[i] << endl;
         cout << "recoJetCalEt["<<i<<"] = " << recoJetCalEt[i] << endl;
         cout << "recoJetCalPt["<<i<<"] = " << recoJetCalPt[i] << endl;
         cout << "recoJetCalPhi["<<i<<"] = " << recoJetCalPhi[i] << endl;
         cout << "recoJetCalEta["<<i<<"] = " << recoJetCalEta[i] << endl;
      }
      break;

   case tau:
      cout << "oh: number of taus = " << NohTau << endl;
      for (int i=0; i<NohTau; i++)
      {
         cout<<"ohTauEt["<<i<<"] = " <<ohTauPt[i]<<endl;
         cout<<"ohTauEiso["<<i<<"] = " <<ohTauEiso[i]<<endl;
         cout<<"ohTauL25Tpt["<<i<<"] = " <<ohTauL25Tpt[i]<<endl;
         cout<<"ohTauL25Tiso["<<i<<"] = " <<ohTauL25Tiso[i]<<endl;
         cout<<"ohTauL3Tpt["<<i<<"] = " <<ohTauL3Tpt[i]<<endl;
         cout<<"ohTauL3Tiso["<<i<<"] = " <<ohTauL3Tiso[i]<<endl;
      }
      break;

   default:

      cout << "PrintOhltVariables: You did not provide correct object type."
            <<endl;
      break;
   }
}

// PFTau Leg
int OHltTree::OpenHltPFTauPassedNoMuon(
      float Et,
      float L25TrkPt,
      float L3TrkIso,
      float L3GammaIso)
{
   int rc = 0;
   // Loop over all oh ohpfTaus
   for (int i=0; i < NohpfTau; i++)
   {
      if (ohpfTauPt[i] >= Et && fabs(ohpfTauEta[i]) < 2.5
            && ohpfTauLeadTrackPt[i] >= L25TrkPt && ohpfTauTrkIso[i] < L3TrkIso
            && ohpfTauGammaIso[i] < L3GammaIso && OpenHltTauMuonMatching(
            ohpfTauEta[i],
            ohpfTauPhi[i]) == 0)
         rc++;
   }
   return rc;
}

int OHltTree::OpenHltPFTauPassedNoEle(
      float Et,
      float L25TrkPt,
      int L3TrkIso,
      int L3GammaIso)
{

   int rc = 0;
   // Loop over all oh ohpfTaus
   for (int i=0; i < NohpfTau; i++)
   {
      //        if (ohpfTauJetPt[i] >= Et) 
      if (ohpfTauPt[i] >= Et)
         if (abs(ohpfTauEta[i])<2.5)
            if (ohpfTauLeadTrackPt[i] >= L25TrkPt)
               if (ohpfTauTrkIso[i] < L3TrkIso)
                  if (ohpfTauGammaIso[i] < L3GammaIso)
                     if (OpenHltTauPFToCaloMatching(
                           ohpfTauEta[i],
                           ohpfTauPhi[i]) == 1)
                        if (OpenHltTauEleMatching(ohpfTauEta[i], ohpfTauPhi[i])
                              == 0)
                           rc++;

   }

   return rc;
}

int OHltTree::OpenHltTauMuonMatching(float eta, float phi)
{
   for (int j=0; j<NohMuL2; j++)
   {
      double deltaphi = fabs(phi-ohMuL2Phi[j]);
      if (deltaphi > 3.14159)
         deltaphi = (2.0 * 3.14159) - deltaphi;
      double deltaeta = fabs(eta-ohMuL2Eta[j]);

      if (sqrt(deltaeta*deltaeta + deltaphi*deltaphi) < 0.3)
         return 1;
   }
   return 0;
}

int OHltTree::OpenHltTauEleMatching(float eta, float phi)
{
   for (int j=0; j<NohEle; j++)
   {
      double deltaphi = fabs(phi-ohElePhi[j]);
      if (deltaphi > 3.14159)
         deltaphi = (2.0 * 3.14159) - deltaphi;
      double deltaeta = fabs(eta-ohEleEta[j]);

      if (deltaeta<0.3 && deltaphi<0.3)
         return 1;
   }
   return 0;

}

int OHltTree::OpenHltTauPFToCaloMatching(float eta, float phi)
{
   for (int j=0; j<NrecoJetCal; j++)
   {
      if (recoJetCalPt[j]<8)
         continue;
      double deltaphi = fabs(phi-recoJetCalPhi[j]);
      if (deltaphi > 3.14159)
         deltaphi = (2.0 * 3.14159) - deltaphi;
      double deltaeta = fabs(eta-recoJetCalEta[j]);

      if (deltaeta<0.3 && deltaphi<0.3)
         return 1;
   }
   return 0;

}

int OHltTree::OpenHltL1L2TauMatching(
      float eta,
      float phi,
      float tauThr,
      float jetThr)
{
   for (int j=0; j<NL1Tau; j++)
   {
      double deltaphi = fabs(phi-L1TauPhi[j]);
      if (deltaphi > 3.14159)
         deltaphi = (2.0 * 3.14159) - deltaphi;
      double deltaeta = fabs(eta-L1TauEta[j]);

      if (deltaeta<0.3 && deltaphi<0.3 && L1TauEt[j]>tauThr)
      return 1;
   }
   for (int j=0; j<NL1CenJet; j++)
   {
      double deltaphi = fabs(phi-L1CenJetPhi[j]);
      if (deltaphi > 3.14159)
         deltaphi = (2.0 * 3.14159) - deltaphi;
      double deltaeta = fabs(eta-L1CenJetEta[j]);

      if (deltaeta<0.3 && deltaphi<0.3 && L1CenJetEt[j]>jetThr)
      return 1;
   }
   return 0;
}

int OHltTree::OpenHltTauPassed(
      float Et,
      float Eiso,
      float L25Tpt,
      int L25Tiso,
      float L3Tpt,
      int L3Tiso,
      float L1TauThr,
      float L1CenJetThr)
{
   int rc = 0;
   // Loop over all oh electrons
   for (int i=0; i<NohTau; i++)
   {
      if (ohTauPt[i] >= Et)
      {
         if (ohTauEiso[i] <= Eiso)
            if (ohTauL25Tpt[i] >= L25Tpt)
               if (ohTauL25Tiso[i] >= L25Tiso)
                  if (ohTauL3Tpt[i] >= L3Tpt)
                     if (ohTauL3Tiso[i] >= L3Tiso)
                        if (OpenHltL1L2TauMatching(
                              ohTauEta[i],
                              ohTauPhi[i],
                              L1TauThr,
                              L1CenJetThr) == 1)
                           rc++;
      }
   }

   return rc;
}

// L2 Ecal sliding cut isolation
int OHltTree::OpenHltTauL2SCPassed(
      float Et,
      float L25Tpt,
      int L25Tiso,
      float L3Tpt,
      int L3Tiso,
      float L1TauThr,
      float L1CenJetThr)
{
   int rc = 0;

   // Loop over all oh electrons
   for (int i=0; i<NohTau; i++)
   {
      if (ohTauPt[i] >= Et)
      {
         if (ohTauEiso[i] < (5 + 0.025*ohTauPt[i] + 0.0015*ohTauPt[i]
               *ohTauPt[i])) // sliding cut
            if (ohTauL25Tpt[i] >= L25Tpt)
               if (ohTauL25Tiso[i] >= L25Tiso)
                  if (ohTauL3Tpt[i] >= L3Tpt)
                     if (ohTauL3Tiso[i] >= L3Tiso)
                        if (OpenHltL1L2TauMatching(
                              ohTauEta[i],
                              ohTauPhi[i],
                              L1TauThr,
                              L1CenJetThr) == 1)
                           rc++;
      }
   }

   return rc;
}

int OHltTree::OpenHltTauL2SCMETPassed(
      float Et,
      float L25Tpt,
      int L25Tiso,
      float L3Tpt,
      int L3Tiso,
      float met,
      float L1TauThr,
      float L1CenJetThr)
{
   int rc = 0;

   // Loop over all oh electrons
   for (int i=0; i<NohTau; i++)
   {
      if (ohTauPt[i]>= Et)
      {
         if (ohTauEiso[i]
               < (5 + 0.025*ohTauPt[i] + 0.0015*ohTauPt[i]*ohTauPt[i])) // sliding cut
            if (ohTauL25Tpt[i]>= L25Tpt)
               if (ohTauL25Tiso[i]>= L25Tiso)
                  if (ohTauL3Tpt[i]>= L3Tpt)
                     if (ohTauL3Tiso[i]>= L3Tiso)
                        if (OpenHltL1L2TauMatching(
                              ohTauEta[i],
                              ohTauPhi[i],
                              L1TauThr,
                              L1CenJetThr) == 1)
                           if (recoMetCal> met)
                              rc++;
      }
   }
   return rc;
}

int OHltTree::OpenHlt2Tau1LegL3IsoPassed(
      float Et,
      float L25Tpt,
      int L25Tiso,
      float L3Tpt,
      float L1TauThr,
      float L1CenJetThr)
{
   int rc = 0;
   int l3iso = 0;

   // Loop over all oh taus
   for (int i=0; i<NohTau; i++)
   {
      if (ohTauPt[i] >= Et)
      {
         if (ohTauEiso[i] < (5 + 0.025*ohTauPt[i] + 0.0015*ohTauPt[i]
               *ohTauPt[i])) // sliding cut
            if (ohTauL25Tpt[i] >= L25Tpt)
               if (ohTauL25Tiso[i] >= L25Tiso)
                  if (ohTauL3Tpt[i] >= L3Tpt)
                     if (OpenHltL1L2TauMatching(
                           ohTauEta[i],
                           ohTauPhi[i],
                           L1TauThr,
                           L1CenJetThr) == 1)
                     {
                        rc++;
                        if (ohTauL3Tiso[i] >= 1)
                           l3iso++;
                     }
      }
   }

   if (rc>=2)
      return l3iso;
   return 0;
}

int OHltTree::OpenHlt1Ele1PFTauPassed(
				      float Et, int L1iso,
				      float Tisobarrel, float Tisoendcap,
				      float Tisoratiobarrel, float Tisoratioendcap,
				      float HisooverETbarrel, float HisooverETendcap,
				      float EisooverETbarrel, float EisooverETendcap,
				      float hoverebarrel, float hovereendcap,
				      float clusshapebarrel, float clusshapeendcap,
				      float r9barrel, float r9endcap,
				      float detabarrel, float detaendcap,
				      float dphibarrel, float dphiendcap,
				      float TauEt, float TauEta, float L25TrkPt,
				      float L3TrkIso, float L3GammaIso, float PFMHTCut)
{
  int rc=0;
  // Loop over all oh electrons
  for (int i=0; i<NohEle; i++)
    {
      float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      bool isBarrel, isEndCap, doTaus;
      isBarrel = isEndCap = doTaus = false;

      if (fabs(ohEleEta[i]) < 1.479)
	isBarrel = true;
      if (fabs(ohEleEta[i]) > 1.479 && fabs(ohEleEta[i]) < 2.65)
	isEndCap = true;

      if (ohEleEt[i] > Et && 
          (isBarrel || isEndCap) && 
          ohEleNewSC[i] == 1 &&
          ohElePixelSeeds[i] > 0 &&
          ohEleL1iso[i] >= L1iso && // L1iso is 0 or 1
          !ohEleL1Dupl[i]) { // remove double-counted L1 SCs
	if (isBarrel) {
	  if (ohEleHiso[i]/ohEleEt[i] < HisooverETbarrel &&
	      ohEleEiso[i]/ohEleEt[i] < EisooverETbarrel && 
	      ohEleHoverE < hoverebarrel &&
	      ((ohEleTiso[i] < Tisobarrel && ohEleTiso[i] != -999.) || Tisobarrel == 999.) && 
	      ohEleTiso[i]/ohEleEt[i] < Tisoratiobarrel && 
	      ohEleClusShap[i] < clusshapebarrel && 
	      ohEleR9[i] < r9barrel && 
	      fabs(ohEleDeta[i]) < detabarrel && 
	      ohEleDphi[i] < dphibarrel)
	    doTaus = true;
	}
	if (isEndCap) {
	  if (ohEleHiso[i]/ohEleEt[i] < HisooverETendcap && 
	      ohEleEiso[i]/ohEleEt[i] < EisooverETendcap && 
	      ohEleHoverE < hovereendcap && 
	      ((ohEleTiso[i] < Tisoendcap && ohEleTiso[i] != -999.) || (Tisoendcap == 999.)) && 
	      ohEleTiso[i]/ohEleEt[i] < Tisoratioendcap && 
	      ohEleClusShap[i] < clusshapeendcap &&
	      ohEleR9[i] < r9endcap && fabs(ohEleDeta[i]) < detabarrel && 
	      ohEleDphi[i] < detaendcap)
	    doTaus = true;
	}
      }
      if (doTaus)
	{
	  for (int j=0; j < NohpfTau; j++)
	    {
	      if (ohpfTauPt[j] >= TauEt && 
		  fabs(ohpfTauEta[j]) < TauEta && 
		  ohpfTauLeadTrackPt[j] >= L25TrkPt && 
		  ohpfTauTrkIso[j] < L3TrkIso && 
		  ohpfTauGammaIso[j] < L3GammaIso && 
		  pfMHT >= PFMHTCut)
		{
		  float dphi = fabs(ohElePhi[i] - ohpfTauPhi[j]);
		  float deta = fabs(ohEleEta[i] - ohpfTauEta[j]);
		  if (dphi > 3.14159)
		    dphi = (2.0 * 3.14159) - dphi;
		  if (sqrt(dphi*dphi + deta*deta) > 0.3)
		    ++rc; 
		}
	    }
	}
    }
  return rc;
}

int OHltTree::OpenHlt1PhotonSamHarperPassed(
      float Et,
      int L1iso,
      float Tisobarrel,
      float Tisoendcap,
      float Tisoratiobarrel,
      float Tisoratioendcap,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float clusshapebarrel,
      float clusshapeendcap,
      float r9barrel,
      float r9endcap,
      float detabarrel,
      float detaendcap,
      float dphibarrel,
      float dphiendcap)
{
   float barreleta = 1.479;
   float endcapeta = 2.65;

   int rc = 0;
   // Loop over all oh electrons
   for (int i=0; i<NohPhot; i++)
   {
      float ohPhotE = ohPhotEt[i] / (sin(2*atan(exp(-1.0*ohPhotEta[i]))));
      float ohPhotHoverE = ohPhotHforHoverE[i]/ohPhotE;
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohPhotEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i])
            < endcapeta)
         isendcap = 1;

      float quadraticEcalIsol = ohPhotEiso[i] + (0.012 * ohPhotEt[i]);
      float quadraticHcalIsol = ohPhotHiso[i] + (0.005 * ohPhotEt[i]);
      float quadraticTrackIsol = ohPhotTiso[i] + (0.002 * ohPhotEt[i]);

      if (ohPhotEt[i] > Et)
      {
         if (TMath::Abs(ohPhotEta[i]) < endcapeta)
         {
            if (ohPhotL1iso[i] >= L1iso)
            { // L1iso is 0 or 1 
               if (ohPhotL1Dupl[i] == false)
               { // remove double-counted L1 SCs 
                  if ( (isbarrel && (quadraticHcalIsol < HisooverETbarrel))
                        || (isendcap && (quadraticHcalIsol < HisooverETendcap)))
                  {
                     if ( (isbarrel && (quadraticEcalIsol < EisooverETbarrel))
                           || (isendcap && (quadraticEcalIsol
                                 < EisooverETendcap)))
                     {
                        if ( ((isbarrel) && (ohPhotHoverE < hoverebarrel))
                              || ((isendcap) && (ohPhotHoverE < hovereendcap)))
                        {
                           if (((isbarrel) && (quadraticTrackIsol < Tisobarrel))
                                 || ((isendcap) && (quadraticTrackIsol
                                       < Tisoendcap)))
                           {
                              if ( (isbarrel && ohPhotClusShap[i]
                                    < clusshapebarrel) || (isendcap
                                    && ohPhotClusShap[i] < clusshapeendcap))
                              {
                                 if ( (isbarrel && ohPhotR9[i] < r9barrel)
                                       || (isendcap && ohPhotR9[i] < r9endcap))
                                 {
                                    rc++;
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}

vector<int> OHltTree::VectorOpenHlt1PhotonSamHarperPassed(
      float Et,
      int L1iso,
      float Tisobarrel,
      float Tisoendcap,
      float Tisoratiobarrel,
      float Tisoratioendcap,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float clusshapebarrel,
      float clusshapeendcap,
      float r9barrel,
      float r9endcap,
      float detabarrel,
      float detaendcap,
      float dphibarrel,
      float dphiendcap)
{
   float barreleta = 1.479;
   float endcapeta = 2.65;

   vector<int> rc;
   // Loop over all oh electrons
   for (int i=0; i<NohPhot; i++)
   {
      float ohPhotE = ohPhotEt[i] / (sin(2*atan(exp(-1.0*ohPhotEta[i]))));
      float ohPhotHoverE = ohPhotHforHoverE[i]/ohPhotE;
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohPhotEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i])
            < endcapeta)
         isendcap = 1;

      float quadraticEcalIsol = ohPhotEiso[i] + (0.012 * ohPhotEt[i]);
      float quadraticHcalIsol = ohPhotHiso[i] + (0.005 * ohPhotEt[i]);
      float quadraticTrackIsol = ohPhotTiso[i] + (0.002 * ohPhotEt[i]);

      if (ohPhotEt[i] > Et)
      {
         if (TMath::Abs(ohPhotEta[i]) < endcapeta)
         {
            if (ohPhotL1iso[i] >= L1iso)
            { // L1iso is 0 or 1 
               if (ohPhotL1Dupl[i] == false)
               { // remove double-counted L1 SCs 
                  if ( (isbarrel && (quadraticHcalIsol < HisooverETbarrel))
                        || (isendcap && (quadraticHcalIsol < HisooverETendcap)))
                  {
                     if ( (isbarrel && (quadraticEcalIsol < EisooverETbarrel))
                           || (isendcap && (quadraticEcalIsol
                                 < EisooverETendcap)))
                     {
                        if ( ((isbarrel) && (ohPhotHoverE < hoverebarrel))
                              || ((isendcap) && (ohPhotHoverE < hovereendcap)))
                        {
                           if (((isbarrel) && (quadraticTrackIsol < Tisobarrel))
                                 || ((isendcap) && (quadraticTrackIsol
                                       < Tisoendcap)))
                           {
                              if ( (isbarrel && ohPhotClusShap[i]
                                    < clusshapebarrel) || (isendcap
                                    && ohPhotClusShap[i] < clusshapeendcap))
                              {
                                 if ( (isbarrel && ohPhotR9[i] < r9barrel)
                                       || (isendcap && ohPhotR9[i] < r9endcap))
                                 {
				   rc.push_back(i);
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}

int OHltTree::OpenHltPhoCuts(
      float e_t,
      float he_eb,
      float he_ee,
      float shape_eb,
      float shape_ee,
      float cal_iso,
      float trk_iso)
{

   // Default cleaning cuts for all photon paths
   float r9barrel = 0.98;
   float r9endcap = 1.0;
   float EtaGap = 1.479;
   float EtaMax = 2.65;

   int rc = 0;

   // Loop over all oh photons
   for (int i=0; i<NohPhot; i++)
   {
      if (ohPhotL1Dupl[i])
         continue;

      float eta = TMath::Abs(ohPhotEta[i]);
      if (eta > EtaMax)
         continue;

      if (ohPhotEt[i] < e_t)
         continue;

      bool isBarrel = (eta < EtaGap);
      bool isEndcap = (eta >= EtaGap);

      bool passSpikeCleaning = (isBarrel && ohPhotR9[i] < r9barrel)
            || (isEndcap && ohPhotR9[i] < r9endcap);
      if (!passSpikeCleaning)
         continue;

      float ohPhotE = ohPhotEt[i] / (sin(2*atan(exp(-1.0*ohPhotEta[i]))));
      float ohPhotHoverE = ohPhotHforHoverE[i]/ohPhotE;

      bool passHoverE = (isBarrel && ohPhotHoverE < he_eb) || (isEndcap
            && ohPhotHoverE < he_ee);
      if (!passHoverE)
         continue;

      bool passShape = (isBarrel && ohPhotClusShap[i] < shape_eb) || (isEndcap
            && ohPhotClusShap[i] < shape_ee);
      if (!passShape)
         continue;

      if (ohPhotEiso[i]/ohPhotEt[i] > cal_iso)
         continue;
      if (ohPhotHiso[i]/ohPhotEt[i] > cal_iso)
         continue;
      if (ohPhotTiso[i]/ohPhotEt[i] > trk_iso)
         continue;

      rc++;
   }// for

   return rc;
}

int OHltTree::OpenHlt1PhotonPassedRA3(
      float Et,
      int L1iso,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float r9barrel,
      float r9endcap)
{
   float barreleta = 1.479;
   float endcapeta = 2.65;

   int rc = 0;
   // Loop over all oh photons
   for (int i=0; i<NohPhot; i++)
   {
      float ohPhotE = ohPhotEt[i] / (sin(2*atan(exp(-1.0*ohPhotEta[i]))));
      float ohPhotHoverE = ohPhotHforHoverE[i]/ohPhotE;
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohPhotEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i])
            < endcapeta)
         isendcap = 1;

      if (ohPhotEt[i] > Et)
      {
         if (TMath::Abs(ohPhotEta[i]) < endcapeta)
         {
            if (ohPhotL1iso[i] >= L1iso)
            { // L1iso is 0 or 1 
               if (ohPhotL1Dupl[i] == false)
               { // remove double-counted L1 SCs 
                  if ( (isbarrel && ((ohPhotHiso[i]/ohPhotEt[i])
                        < HisooverETbarrel)) || (isendcap && ((ohPhotHiso[i]
                        /ohPhotEt[i]) < HisooverETendcap)))
                  {
                     if ( (isbarrel && ((ohPhotEiso[i]/ohPhotEt[i])
                           < EisooverETbarrel)) || (isendcap && ((ohPhotEiso[i]
                           /ohPhotEt[i]) < EisooverETendcap)))
                     {
                        if ( ((isbarrel) && (ohPhotHoverE < hoverebarrel))
                              || ((isendcap) && (ohPhotHoverE < hovereendcap)))
                        {
                           if ( (isbarrel && ohPhotR9[i] < r9barrel)
                                 || (isendcap && ohPhotR9[i] < r9endcap))
                           {
                              rc++;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   return rc;
}

bool OHltTree::OpenHLT_EleX_R9cut(const float& Et, const float& r9barrel)
{
   return OpenHlt1ElectronSamHarperPassed(Et, 0, // ET, L1isolation 
         999.,
         999., // Track iso barrel, Track iso endcap 
         999.,
         999., // Track/pT iso barrel, Track/pT iso endcap 
         999.,
         999., // H/ET iso barrel, H/ET iso endcap 
         999.,
         999., // E/ET iso barrel, E/ET iso endcap 
         0.15,
         0.15, // H/E barrel, H/E endcap 
         999.,
         999., // cluster shape barrel, cluster shape endcap 
         r9barrel,
         1.0, // R9 barrel, R9 endcap 
         999.,
         999., // Deta barrel, Deta endcap 
         999.,
         999. // Dphi barrel, Dphi endcap
   ) >= 1;
}

int OHltTree::OpenHlt1ElectronSamHarperPassed(
      float Et,
      int L1iso,
      float Tisobarrel,
      float Tisoendcap,
      float Tisoratiobarrel,
      float Tisoratioendcap,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float clusshapebarrel,
      float clusshapeendcap,
      float r9barrel,
      float r9endcap,
      float detabarrel,
      float detaendcap,
      float dphibarrel,
      float dphiendcap)
{
   float barreleta = 1.479;
   float endcapeta = 2.65;

   int rc = 0;
   // Loop over all oh electrons
   for (int i=0; i<NohEle; i++)
   {
      float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohEleEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
            < endcapeta)
         isendcap = 1;

      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < endcapeta)
         {
            if (ohEleNewSC[i]<=1)
            {
               if (ohElePixelSeeds[i]>0)
               {
                  if (ohEleL1iso[i] >= L1iso)
                  { // L1iso is 0 or 1 
                     if (ohEleL1Dupl[i] == false)
                     { // remove double-counted L1 SCs 
                        if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                              < HisooverETbarrel))
                              || (isendcap && ((ohEleHiso[i]/ohEleEt[i])
                                    < HisooverETendcap)))
                        {
                           if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                                 < EisooverETbarrel)) || (isendcap
                                 && ((ohEleEiso[i]/ohEleEt[i])
                                       < EisooverETendcap)))
                           {
                              if ( ((isbarrel) && (ohEleHoverE < hoverebarrel))
                                    || ((isendcap) && (ohEleHoverE
                                          < hovereendcap)))
                              {
                                 if ( (isbarrel && (((ohEleTiso[i] < Tisobarrel
                                       && ohEleTiso[i] != -999.) || (Tisobarrel
                                       == 999.)))) || (isendcap
                                       && (((ohEleTiso[i] < Tisoendcap
                                             && ohEleTiso[i] != -999.)
                                             || (Tisoendcap == 999.)))))
                                 {
                                    if (((isbarrel) && (ohEleTiso[i]/ohEleEt[i]
                                          < Tisoratiobarrel)) || ((isendcap)
                                          && (ohEleTiso[i]/ohEleEt[i]
                                                < Tisoratioendcap)))
                                    {
                                       if ( (isbarrel && ohEleClusShap[i]
                                             < clusshapebarrel) || (isendcap
                                             && ohEleClusShap[i]
                                                   < clusshapeendcap))
                                       {
                                          if ( (isbarrel && ohEleR9[i]
                                                < r9barrel) || (isendcap
                                                && ohEleR9[i] < r9endcap))
                                          {
                                             if ( (isbarrel
                                                   && TMath::Abs(ohEleDeta[i])
                                                         < detabarrel)
                                                   || (isendcap
                                                         && TMath::Abs(ohEleDeta[i])
                                                               < detaendcap))
                                             {
                                                if ( (isbarrel && ohEleDphi[i]
                                                      < dphibarrel)
                                                      || (isendcap
                                                            && ohEleDphi[i]
                                                                  < dphiendcap))
                                                {
                                                   rc++;
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}

int OHltTree::OpenHlt2ElectronsSamHarperPassed(
      float Et,
      int L1iso,
      float Tisobarrel,
      float Tisoendcap,
      float Tisoratiobarrel,
      float Tisoratioendcap,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float clusshapebarrel,
      float clusshapeendcap,
      float r9barrel,
      float r9endcap,
      float detabarrel,
      float detaendcap,
      float dphibarrel,
      float dphiendcap)
{
   float barreleta = 1.479;
   float endcapeta = 2.65;

   int rc = 0;
   int rcsconly = 0;

   // Loop over all oh electrons
   for (int i=0; i<NohEle; i++)
   {
      float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohEleEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
            < endcapeta)
         isendcap = 1;

      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < endcapeta)
         {
            if (ohEleNewSC[i]==1)
            {
               if (ohEleL1iso[i] >= L1iso)
               { // L1iso is 0 or 1 
                  if (ohEleL1Dupl[i] == false)
                  { // remove double-counted L1 SCs 
                     if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                           < EisooverETbarrel)) || (isendcap && ((ohEleEiso[i]
                           /ohEleEt[i]) < EisooverETendcap)))
                     {
                        if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                              < HisooverETbarrel))
                              || (isendcap && ((ohEleHiso[i]/ohEleEt[i])
                                    < HisooverETendcap)))
                        {
                           if ( ((isbarrel) && (ohEleHoverE < hoverebarrel))
                                 || ((isendcap) && (ohEleHoverE < hovereendcap)))
                           {
                              if ( (isbarrel && ohEleClusShap[i]
                                    < clusshapebarrel) || (isendcap
                                    && ohEleClusShap[i] < clusshapeendcap))
                              {
                                 if ( (isbarrel && ohEleR9[i] < r9barrel)
                                       || (isendcap && ohEleR9[i] < r9endcap))
                                 {
                                    if (ohElePixelSeeds[i]>0)
                                    {
                                       rcsconly++;
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   if (rcsconly >= 2)
   {
      for (int i=0; i<NohEle; i++)
      {
         float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
         int isbarrel = 0;
         int isendcap = 0;
         if (TMath::Abs(ohEleEta[i]) < barreleta)
            isbarrel = 1;
         if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
               < endcapeta)
            isendcap = 1;

         if (ohEleEt[i] > Et)
         {
            if (TMath::Abs(ohEleEta[i]) < endcapeta)
            {
               if (ohEleNewSC[i]<=1)
               {
                  if (ohElePixelSeeds[i]>0)
                  {
                     if (ohEleL1iso[i] >= L1iso)
                     { // L1iso is 0 or 1 
                        if (ohEleL1Dupl[i] == false)
                        { // remove double-counted L1 SCs 
                           if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                                 < HisooverETbarrel)) || (isendcap
                                 && ((ohEleHiso[i]/ohEleEt[i])
                                       < HisooverETendcap)))
                           {
                              if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                                    < EisooverETbarrel)) || (isendcap
                                    && ((ohEleEiso[i]/ohEleEt[i])
                                          < EisooverETendcap)))
                              {
                                 if ( ((isbarrel) && (ohEleHoverE
                                       < hoverebarrel)) || ((isendcap)
                                       && (ohEleHoverE < hovereendcap)))
                                 {
                                    if ( (isbarrel
                                          && (((ohEleTiso[i] < Tisobarrel
                                                && ohEleTiso[i] != -999.)
                                                || (Tisobarrel == 999.))))
                                          || (isendcap && (((ohEleTiso[i]
                                                < Tisoendcap && ohEleTiso[i]
                                                != -999.) || (Tisoendcap
                                                == 999.)))))
                                    {
                                       if (((isbarrel) && (ohEleTiso[i]
                                             /ohEleEt[i] < Tisoratiobarrel))
                                             || ((isendcap) && (ohEleTiso[i]
                                                   /ohEleEt[i]
                                                   < Tisoratioendcap)))
                                       {
                                          if ( (isbarrel && ohEleClusShap[i]
                                                < clusshapebarrel) || (isendcap
                                                && ohEleClusShap[i]
                                                      < clusshapeendcap))
                                          {
                                             if ( (isbarrel && ohEleR9[i]
                                                   < r9barrel) || (isendcap
                                                   && ohEleR9[i] < r9endcap))
                                             {
                                                if ( (isbarrel
                                                      && TMath::Abs(ohEleDeta[i])
                                                            < detabarrel)
                                                      || (isendcap
                                                            && TMath::Abs(ohEleDeta[i])
                                                                  < detaendcap))
                                                {
                                                   if ( (isbarrel
                                                         && ohEleDphi[i]
                                                               < dphibarrel)
                                                         || (isendcap
                                                               && ohEleDphi[i]
                                                                     < dphiendcap))
                                                   {
                                                      rc++;
                                                   }
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}

int OHltTree::OpenHlt3ElectronsSamHarperPassed(
      float Et,
      int L1iso,
      float Tisobarrel,
      float Tisoendcap,
      float Tisoratiobarrel,
      float Tisoratioendcap,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float clusshapebarrel,
      float clusshapeendcap,
      float r9barrel,
      float r9endcap,
      float detabarrel,
      float detaendcap,
      float dphibarrel,
      float dphiendcap)
{
   float barreleta = 1.479;
   float endcapeta = 2.65;

   int rc = 0;
   int rcsconly = 0;

   // Loop over all oh electrons
   for (int i=0; i<NohEle; i++)
   {
      // bug fix
      float ohEleHoverE;
      float ohEleR9v;
      //if(ohEleL1iso[i] == 1){
      ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      ohEleR9v = ohEleR9[i];
      //}

      //    float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohEleEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
            < endcapeta)
         isendcap = 1;
      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < endcapeta)
         {
            if (ohEleNewSC[i]==1)
            {
               if (ohEleL1iso[i] >= L1iso)
               { // L1iso is 0 or 1 
                  if (ohEleL1Dupl[i] == false)
                  { // remove double-counted L1 SCs 
                     if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                           < EisooverETbarrel)) || (isendcap && ((ohEleEiso[i]
                           /ohEleEt[i]) < EisooverETendcap)))
                     {
                        if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                              < HisooverETbarrel))
                              || (isendcap && ((ohEleHiso[i]/ohEleEt[i])
                                    < HisooverETendcap)))
                        {
                           if ( ((isbarrel) && (ohEleHoverE < hoverebarrel))
                                 || ((isendcap) && (ohEleHoverE < hovereendcap)))
                           {
                              if ( (isbarrel && ohEleClusShap[i]
                                    < clusshapebarrel) || (isendcap
                                    && ohEleClusShap[i] < clusshapeendcap))
                              {
                                 //                      if ( (isbarrel && ohEleR9[i] < r9barrel) ||
                                 //                         (isendcap && ohEleR9[i] < r9endcap) ) {
                                 if ( (isbarrel && ohEleR9v < r9barrel)
                                       || (isendcap && ohEleR9v < r9endcap))
                                 {
                                    if (ohElePixelSeeds[i]>0)
                                    {
                                       rcsconly++;
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   if (rcsconly >= 3)
   {
      for (int i=0; i<NohEle; i++)
      {
         // bug fix
         float ohEleHoverE;
         float ohEleR9v;
         //if(ohEleL1iso[i] == 1){
         ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
         ohEleR9v = ohEleR9[i];

         int isbarrel = 0;
         int isendcap = 0;
         if (TMath::Abs(ohEleEta[i]) < barreleta)
            isbarrel = 1;
         if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
               < endcapeta)
            isendcap = 1;
         if (ohEleEt[i] > Et)
         {
            if (TMath::Abs(ohEleEta[i]) < endcapeta)
            {
               if (ohEleNewSC[i]<=1)
               {
                  if (ohElePixelSeeds[i]>0)
                  {
                     if (ohEleL1iso[i] >= L1iso)
                     { // L1iso is 0 or 1 
                        if (ohEleL1Dupl[i] == false)
                        { // remove double-counted L1 SCs 
                           if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                                 < HisooverETbarrel)) || (isendcap
                                 && ((ohEleHiso[i]/ohEleEt[i])
                                       < HisooverETendcap)))
                           {
                              if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                                    < EisooverETbarrel)) || (isendcap
                                    && ((ohEleEiso[i]/ohEleEt[i])
                                          < EisooverETendcap)))
                              {
                                 if ( ((isbarrel) && (ohEleHoverE
                                       < hoverebarrel)) || ((isendcap)
                                       && (ohEleHoverE < hovereendcap)))
                                 {
                                    if ( (isbarrel
                                          && (((ohEleTiso[i] < Tisobarrel
                                                && ohEleTiso[i] != -999.)
                                                || (Tisobarrel == 999.))))
                                          || (isendcap && (((ohEleTiso[i]
                                                < Tisoendcap && ohEleTiso[i]
                                                != -999.) || (Tisoendcap
                                                == 999.)))))
                                    {
                                       if (((isbarrel) && (ohEleTiso[i]
                                             /ohEleEt[i] < Tisoratiobarrel))
                                             || ((isendcap) && (ohEleTiso[i]
                                                   /ohEleEt[i]
                                                   < Tisoratioendcap)))
                                       {
                                          if ( (isbarrel && ohEleClusShap[i]
                                                < clusshapebarrel) || (isendcap
                                                && ohEleClusShap[i]
                                                      < clusshapeendcap))
                                          {
                                             //                              if ( (isbarrel && ohEleR9[i] < r9barrel) ||
                                             //                                 (isendcap && ohEleR9[i] < r9endcap) ) {
                                             if ( (isbarrel && ohEleR9v
                                                   < r9barrel) || (isendcap
                                                   && ohEleR9v < r9endcap))
                                             {
                                                if ( (isbarrel
                                                      && TMath::Abs(ohEleDeta[i])
                                                            < detabarrel)
                                                      || (isendcap
                                                            && TMath::Abs(ohEleDeta[i])
                                                                  < detaendcap))
                                                {
                                                   if ( (isbarrel
                                                         && ohEleDphi[i]
                                                               < dphibarrel)
                                                         || (isendcap
                                                               && ohEleDphi[i]
                                                                     < dphiendcap))
                                                   {
                                                      rc++;
                                                   }
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}

int OHltTree::OpenHltGetElectronsSamHarperPassed(
      int *Passed,
      float Et,
      int L1iso,
      float Tisobarrel,
      float Tisoendcap,
      float Tisoratiobarrel,
      float Tisoratioendcap,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float clusshapebarrel,
      float clusshapeendcap,
      float r9barrel,
      float r9endcap,
      float detabarrel,
      float detaendcap,
      float dphibarrel,
      float dphiendcap)
{
   int NPassed = 0;

   float barreleta = 1.479;
   float endcapeta = 2.65;

   // First check if only one electron is going to pass the cuts using the one electron code
   // we use the one electron code to look to see if 0 or 1 electrons pass this condition
   int rc = 0;
   int tmpindex=-999;
   // Loop over all oh electrons
   for (int i=0; i<NohEle; i++)
   {
      //    float ohEleHoverE = ohEleHiso[i]/ohEleEt[i];
      float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohEleEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
            < endcapeta)
         isendcap = 1;

      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < endcapeta)
         {
            if (ohEleNewSC[i]<=1)
            {
               if (ohElePixelSeeds[i]>0)
               {
                  if (ohEleL1iso[i] >= L1iso)
                  { // L1iso is 0 or 1 
                     if (ohEleL1Dupl[i] == false)
                     { // remove double-counted L1 SCs 
                        if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                              < HisooverETbarrel))
                              || (isendcap && ((ohEleHiso[i]/ohEleEt[i])
                                    < HisooverETendcap)))
                        {
                           if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                                 < EisooverETbarrel)) || (isendcap
                                 && ((ohEleEiso[i]/ohEleEt[i])
                                       < EisooverETendcap)))
                           {
                              if ( ((isbarrel) && (ohEleHoverE < hoverebarrel))
                                    || ((isendcap) && (ohEleHoverE
                                          < hovereendcap)))
                              {
                                 if ( (isbarrel && (((ohEleTiso[i] < Tisobarrel
                                       && ohEleTiso[i] != -999.) || (Tisobarrel
                                       == 999.)))) || (isendcap
                                       && (((ohEleTiso[i] < Tisoendcap
                                             && ohEleTiso[i] != -999.)
                                             || (Tisoendcap == 999.)))))
                                 {
                                    if (((isbarrel) && (ohEleTiso[i]/ohEleEt[i]
                                          < Tisoratiobarrel)) || ((isendcap)
                                          && (ohEleTiso[i]/ohEleEt[i]
                                                < Tisoratioendcap)))
                                    {
                                       if ( (isbarrel && ohEleClusShap[i]
                                             < clusshapebarrel) || (isendcap
                                             && ohEleClusShap[i]
                                                   < clusshapeendcap))
                                       {
                                          if ( (isbarrel && ohEleR9[i]
                                                < r9barrel) || (isendcap
                                                && ohEleR9[i] < r9endcap))
                                          {
                                             if ( (isbarrel
                                                   && TMath::Abs(ohEleDeta[i])
                                                         < detabarrel)
                                                   || (isendcap
                                                         && TMath::Abs(ohEleDeta[i])
                                                               < detaendcap))
                                             {
                                                if ( (isbarrel && ohEleDphi[i]
                                                      < dphibarrel)
                                                      || (isendcap
                                                            && ohEleDphi[i]
                                                                  < dphiendcap))
                                                {
                                                   rc++;
                                                   tmpindex=i; // temporarily store the index of this event
                                                   if (rc >1)
                                                      break; // If we have 2 electrons passing, we need the 2 ele code
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   //  cout << "rc = " << rc << endl;;
   if (rc == 0)
   {
      return 0;
   }
   if (rc == 1)
   {
      Passed[NPassed++]=tmpindex; // if only 1 ele matched we can use this result without looping on the 2 ele code
      return NPassed;
   }
   // otherwise, we use the 2 ele code:

   int rcsconly=0;
   std::vector<int> csPassedEle;
   // Loop over all oh electrons
   for (int i=0; i<NohEle; i++)
   {
      //float ohEleHoverE = ohEleHiso[i]/ohEleEt[i];
      float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohEleEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
            < endcapeta)
         isendcap = 1;

      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < endcapeta)
         {
            if (ohEleNewSC[i]==1)
            {
               if (ohEleL1iso[i] >= L1iso)
               { // L1iso is 0 or 1 
                  if (ohEleL1Dupl[i] == false)
                  { // remove double-counted L1 SCs 
                     if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                           < EisooverETbarrel)) || (isendcap && ((ohEleEiso[i]
                           /ohEleEt[i]) < EisooverETendcap)))
                     {
                        if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                              < HisooverETbarrel))
                              || (isendcap && ((ohEleHiso[i]/ohEleEt[i])
                                    < HisooverETendcap)))
                        {
                           if ( ((isbarrel) && (ohEleHoverE < hoverebarrel))
                                 || ((isendcap) && (ohEleHoverE < hovereendcap)))
                           {
                              if ( (isbarrel && ohEleClusShap[i]
                                    < clusshapebarrel) || (isendcap
                                    && ohEleClusShap[i] < clusshapeendcap))
                              {
                                 if ( (isbarrel && ohEleR9[i] < r9barrel)
                                       || (isendcap && ohEleR9[i] < r9endcap))
                                 {
                                    if (ohElePixelSeeds[i]>0)
                                    {
                                       rcsconly++;
                                       csPassedEle.push_back(i);
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   //  cout << "rcsconly = " << rcsconly << endl;
   if (rcsconly == 0)
   { // This really shouldn't happen, but included for safety
      return NPassed;
   }
   if (rcsconly == 1)
   { // ok, we only had 1 cs, but 2 eles were assigned to it
      Passed[NPassed++] = tmpindex;
      return NPassed;
   }

   if (rcsconly >= 2)
   {
      for (int i=0; i<NohEle; i++)
      {
         float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
         int isbarrel = 0;
         int isendcap = 0;
         if (TMath::Abs(ohEleEta[i]) < barreleta)
            isbarrel = 1;
         if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
               < endcapeta)
            isendcap = 1;

         if (ohEleEt[i] > Et)
         {
            if (TMath::Abs(ohEleEta[i]) < endcapeta)
            {
               if (ohEleNewSC[i]<=1)
               {
                  if (ohElePixelSeeds[i]>0)
                  {
                     if (ohEleL1iso[i] >= L1iso)
                     { // L1iso is 0 or 1 
                        if (ohEleL1Dupl[i] == false)
                        { // remove double-counted L1 SCs 
                           if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                                 < HisooverETbarrel)) || (isendcap
                                 && ((ohEleHiso[i]/ohEleEt[i])
                                       < HisooverETendcap)))
                           {
                              if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                                    < EisooverETbarrel)) || (isendcap
                                    && ((ohEleEiso[i]/ohEleEt[i])
                                          < EisooverETendcap)))
                              {
                                 if ( ((isbarrel) && (ohEleHoverE
                                       < hoverebarrel)) || ((isendcap)
                                       && (ohEleHoverE < hovereendcap)))
                                 {
                                    if ( (isbarrel
                                          && (((ohEleTiso[i] < Tisobarrel
                                                && ohEleTiso[i] != -999.)
                                                || (Tisobarrel == 999.))))
                                          || (isendcap && (((ohEleTiso[i]
                                                < Tisoendcap && ohEleTiso[i]
                                                != -999.) || (Tisoendcap
                                                == 999.)))))
                                    {
                                       if (((isbarrel) && (ohEleTiso[i]
                                             /ohEleEt[i] < Tisoratiobarrel))
                                             || ((isendcap) && (ohEleTiso[i]
                                                   /ohEleEt[i]
                                                   < Tisoratioendcap)))
                                       {
                                          if ( (isbarrel && ohEleClusShap[i]
                                                < clusshapebarrel) || (isendcap
                                                && ohEleClusShap[i]
                                                      < clusshapeendcap))
                                          {
                                             if ( (isbarrel && ohEleR9[i]
                                                   < r9barrel) || (isendcap
                                                   && ohEleR9[i] < r9endcap))
                                             {
                                                if ( (isbarrel
                                                      && TMath::Abs(ohEleDeta[i])
                                                            < detabarrel)
                                                      || (isendcap
                                                            && TMath::Abs(ohEleDeta[i])
                                                                  < detaendcap))
                                                {
                                                   if ( (isbarrel
                                                         && ohEleDphi[i]
                                                               < dphibarrel)
                                                         || (isendcap
                                                               && ohEleDphi[i]
                                                                     < dphiendcap))
                                                   {
                                                      for (unsigned int j=0; j
                                                            <csPassedEle.size(); j++)
                                                      {
                                                         if (i
                                                               == csPassedEle.at(j))
                                                         { // check if the electron is in the cs matching list
                                                            Passed[NPassed++]
                                                                  = i;
                                                            rc++; // ok, don't really need this, but keeping for debugging
                                                            break;
                                                         }
                                                      }
                                                   }
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return NPassed;
}

int OHltTree::OpenHlt2ElectronsAsymSamHarperPassed(
      float Et1,
      int L1iso1,
      float Tisobarrel1,
      float Tisoendcap1,
      float Tisoratiobarrel1,
      float Tisoratioendcap1,
      float HisooverETbarrel1,
      float HisooverETendcap1,
      float EisooverETbarrel1,
      float EisooverETendcap1,
      float hoverebarrel1,
      float hovereendcap1,
      float clusshapebarrel1,
      float clusshapeendcap1,
      float r9barrel1,
      float r9endcap1,
      float detabarrel1,
      float detaendcap1,
      float dphibarrel1,
      float dphiendcap1,
      float Et2,
      int L1iso2,
      float Tisobarrel2,
      float Tisoendcap2,
      float Tisoratiobarrel2,
      float Tisoratioendcap2,
      float HisooverETbarrel2,
      float HisooverETendcap2,
      float EisooverETbarrel2,
      float EisooverETendcap2,
      float hoverebarrel2,
      float hovereendcap2,
      float clusshapebarrel2,
      float clusshapeendcap2,
      float r9barrel2,
      float r9endcap2,
      float detabarrel2,
      float detaendcap2,
      float dphibarrel2,
      float dphiendcap2)
{
   //  cout << "AB" << endl;
   int FirstEle[8000], SecondEle[8000];
   // cout << "BA" << endl;  
   int NFirst = OpenHltGetElectronsSamHarperPassed(
         FirstEle,
         Et1,
         L1iso1,
         Tisobarrel1,
         Tisoendcap1,
         Tisoratiobarrel1,
         Tisoratioendcap1,
         HisooverETbarrel1,
         HisooverETendcap1,
         EisooverETbarrel1,
         EisooverETendcap1,
         hoverebarrel1,
         hovereendcap1,
         clusshapebarrel1,
         clusshapeendcap1,
         r9barrel1,
         r9endcap1,
         detabarrel1,
         detaendcap1,
         dphibarrel1,
         dphiendcap1);
   int NSecond = OpenHltGetElectronsSamHarperPassed(
         SecondEle,
         Et2,
         L1iso2,
         Tisobarrel2,
         Tisoendcap2,
         Tisoratiobarrel2,
         Tisoratioendcap2,
         HisooverETbarrel2,
         HisooverETendcap2,
         EisooverETbarrel2,
         EisooverETendcap2,
         hoverebarrel2,
         hovereendcap2,
         clusshapebarrel2,
         clusshapeendcap2,
         r9barrel2,
         r9endcap2,
         detabarrel2,
         detaendcap2,
         dphibarrel2,
         dphiendcap2);
   // std::cout << "ABBA " << NFirst << "  " << NSecond << endl;
   if (NFirst == 0 || NSecond == 0)
      return 0; // if no eles passed one condition, fail
   if (NFirst == 1 && NSecond == 1 && FirstEle[0] == SecondEle[0])
      return 0; //only 1 electron passed both conditions
   return 1; // in any other case, at least 1 unique electron passed each condition, so pass the event

}

int OHltTree::OpenHlt1ElectronPassed(float Et, int L1iso, float Tiso, float Hiso)
{
   int rc = 0;
   // Loop over all oh electrons
   for (int i=0; i<NohEle; i++)
   {
      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < 2.65)
            //	if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
            if (ohEleHiso[i]/ohEleEt[i] < 0.15)
               if (ohEleNewSC[i]==1)
                  if (ohElePixelSeeds[i]>0)
                     if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.)
                           || (Tiso == 9999.))
                        if (ohEleL1iso[i] >= L1iso) // L1iso is 0 or 1
                           if (ohEleL1Dupl[i] == false) // remove double-counted L1 SCs  
                              rc++;
      }
   }

   return rc;
}

int OHltTree::OpenHlt1PhotonPassed(
      float Et,
      int L1iso,
      float Tiso,
      float Eiso,
      float HisoBR,
      float HisoEC)
{
   float barreleta = 1.479;
   float endcapeta = 2.65;

   // Default cleaning cuts for all photon paths
   float r9barrel = 0.98;
   float r9endcap = 1.0;
   float hoverebarrel = 0.15;
   float hovereendcap = 0.15;
   int rc = 0;

   // Loop over all oh photons 
   for (int i=0; i<NohPhot; i++)
   {
      float ohPhotE = ohPhotEt[i] / (sin(2*atan(exp(-1.0*ohPhotEta[i]))));
      float ohPhotHoverE = ohPhotHforHoverE[i]/ohPhotE;
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohPhotEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i])
            < endcapeta)
         isendcap = 1;

      if (ohPhotEt[i] > Et)
      {
         if ( ((isbarrel) && (ohPhotHoverE < hoverebarrel)) || ((isendcap)
               && (ohPhotHoverE < hovereendcap)))
         {
            if ( (isbarrel && ohPhotR9[i] < r9barrel) || (isendcap
                  && ohPhotR9[i] < r9endcap))
            {
               if (ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs    
                  rc++;
            }
         }
      }
   }

   return rc;
}

int OHltTree::OpenHlt2ElectronMassWinPassed(
      float Et,
      int L1iso,
      float Hiso,
      float massLow,
      float massHigh)
{
   TLorentzVector e1;
   TLorentzVector e2;
   TLorentzVector meson;

   int rc = 0;

   for (int i=0; i<NohEle; i++)
   {
      for (int j=0; j<NohEle && j != i; j++)
      {
         if (ohEleEt[i] > Et && ohEleEt[j] > Et)
         {
            if (TMath::Abs(ohEleEta[i]) < 2.65 && TMath::Abs(ohEleEta[j])
                  < 2.65)
            {
               if ( ((ohEleHiso[i] < Hiso) || (ohEleHiso[i]/ohEleEt[i] < 0.2))
                     && ((ohEleHiso[j] < Hiso) || (ohEleHiso[j]/ohEleEt[j]
                           < 0.2)))
               {
                  if (ohEleNewSC[i]==1 && ohEleNewSC[j]==1)
                  {
                     if (ohElePixelSeeds[i]>0 && ohElePixelSeeds[j]>0)
                     {
                        if (ohEleL1iso[i] >= L1iso && ohEleL1iso[j] >= L1iso)
                        { // L1iso is 0 or 1  
                           if (ohEleL1Dupl[i] == false && ohEleL1Dupl[j]
                                 == false)
                           { // remove double-counted L1 SCs    
                              e1.SetPtEtaPhiM(
                                    ohEleEt[i],
                                    ohEleEta[i],
                                    ohElePhi[i],
                                    0.0);
                              e2.SetPtEtaPhiM(
                                    ohEleEt[j],
                                    ohEleEta[j],
                                    ohElePhi[j],
                                    0.0);
                              meson = e1 + e2;

                              float mesonmass = meson.M();
                              if (mesonmass > massLow && mesonmass < massHigh)
                                 rc++;
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   if (rc>0)
      return 1;
   else
      return 0;
}

int OHltTree::OpenHlt2Electron1LegIdPassed(
      float Et,
      int L1iso,
      float Tiso,
      float Hiso)
{
   int rc = 0;
   // Loop over all oh electrons
   for (int i=0; i<NohEle; i++)
   {
      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < 2.65)
         {
            if (ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
            {
               if (ohEleNewSC[i]==1)
               {
                  if (ohElePixelSeeds[i]>0)
                  {
                     if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.)
                           || (Tiso == 9999.))
                     {
                        if (ohEleL1iso[i] >= L1iso)
                        { // L1iso is 0 or 1
                           if (ohEleL1Dupl[i] == false)
                           { // remove double-counted L1 SCs
                              // Loop over all oh electrons
                              for (int i=0; i<NohEle; i++)
                              {
                                 if (ohEleEt[i] > Et)
                                 {
                                    if (TMath::Abs(ohEleEta[i]) < 2.65)
                                       if (ohEleHiso[i] < Hiso || ohEleHiso[i]
                                             /ohEleEt[i] < 0.05)
                                          if (ohEleNewSC[i]==1)
                                             if (ohElePixelSeeds[i]>0)
                                                if ( (ohEleTiso[i] < Tiso
                                                      && ohEleTiso[i] != -999.)
                                                      || (Tiso == 9999.))
                                                   if (ohEleL1iso[i] >= L1iso) // L1iso is 0 or 1
                                                      if ( (TMath::Abs(ohEleEta[i])
                                                            < 1.479
                                                            && ohEleClusShap[i]
                                                                  < 0.015)
                                                            || (1.479
                                                                  < TMath::Abs(ohEleEta[i])
                                                                  && TMath::Abs(ohEleEta[i])
                                                                        < 2.65
                                                                  && ohEleClusShap[i]
                                                                        < 0.04))
                                                         if ( (ohEleDeta[i]
                                                               < 0.008)
                                                               && (ohEleDphi[i]
                                                                     < 0.1))
                                                            if (ohEleL1Dupl[i]
                                                                  == false) // remove double-counted L1 SCs
                                                               rc++;
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}

int OHltTree::OpenHlt1ElectronHTPassed(
      float Et,
      float HT,
      float jetThreshold,
      int L1iso,
      float Tiso,
      float Hiso,
      float dr)
{
   vector<int> PassedElectrons;
   int NPassedEle=0;
   for (int i=0; i<NohEle; i++)
   {
      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < 2.65)
            //if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
            if (ohEleHiso[i]/ohEleEt[i] < 0.15)
               if (ohEleNewSC[i]==1)
                  if (ohElePixelSeeds[i]>0)
                     if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.)
                           || (Tiso == 9999.))
                        if (ohEleL1iso[i] >= L1iso) // L1iso is 0 or 1
                           if (ohEleL1Dupl[i] == false)
                           { // remove double-counted L1 SCs  
                              PassedElectrons.push_back(i);
                              NPassedEle++;
                           }

      }
   }
   if (NPassedEle==0)
      return 0;

   float sumHT=0;
   for (int i=0; i<NrecoJetCal; i++)
   {
      if (recoJetCalPt[i] < jetThreshold)
         continue;
      bool jetPass=true;
      for (unsigned int iEle = 0; iEle<PassedElectrons.size(); iEle++)
      {
         float dphi = ohElePhi[PassedElectrons.at(iEle)] - recoJetCalPhi[i];
         float deta = ohEleEta[PassedElectrons.at(iEle)] - recoJetCalEta[i];
         if (dphi*dphi+deta*deta<dr*dr) // require electron not in any jet
            jetPass=false;
      }
      if (jetPass)
         sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
   }
   if (sumHT>HT)
      return 1;
   return 0;
}

int OHltTree::OpenHlt1ElectronEleIDHTPassed(
      float Et,
      float HT,
      float jetThreshold,
      int L1iso,
      float Tiso,
      float Hiso,
      float dr)
{
   vector<int> PassedElectrons;
   int NPassedEle=0;
   for (int i=0; i<NohEle; i++)
   {
      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < 2.65)
            //if ( ohEleHiso[i] < Hiso || ohEleHiso[i]/ohEleEt[i] < 0.05)
            if (ohEleHiso[i]/ohEleEt[i] < 0.15)
               if (ohEleNewSC[i]==1)
                  if (ohElePixelSeeds[i]>0)
                     if ( (ohEleTiso[i] < Tiso && ohEleTiso[i] != -999.)
                           || (Tiso == 9999.))
                        if (ohEleL1iso[i] >= L1iso) // L1iso is 0 or 1
                           if ( (TMath::Abs(ohEleEta[i]) < 1.479
                                 && ohEleClusShap[i] < 0.015) || (1.479
                                 < TMath::Abs(ohEleEta[i])
                                 && TMath::Abs(ohEleEta[i]) < 2.65
                                 && ohEleClusShap[i] < 0.04))
                              if ( (ohEleDeta[i] < 0.008) && (ohEleDphi[i]
                                    < 0.1))
                                 if (ohEleL1Dupl[i] == false)
                                 { // remove double-counted L1 SCs  
                                    PassedElectrons.push_back(i);
                                    NPassedEle++;
                                 }

      }
   }
   if (NPassedEle==0)
      return 0;

   float sumHT=0;
   for (int i=0; i<NrecoJetCal; i++)
   {
      if (recoJetCalPt[i] < jetThreshold)
         continue;
      bool jetPass=true;
      for (unsigned int iEle = 0; iEle<PassedElectrons.size(); iEle++)
      {
         float dphi = ohElePhi[PassedElectrons.at(iEle)] - recoJetCalPhi[i];
         float deta = ohEleEta[PassedElectrons.at(iEle)] - recoJetCalEta[i];
         if (dphi*dphi+deta*deta<dr*dr) // require electron not in any jet
            jetPass=false;
      }
      if (jetPass)
         sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
   }
   if (sumHT>HT)
      return 1;
   return 0;
}

int OHltTree::OpenHlt1BJetPassedEleRemoval(
      float jetEt,
      float jetEta,
      float drcut,
      float discL25,
      float discL3,
      float Et,
      int L1iso,
      float Tisobarrel,
      float Tisoendcap,
      float Tisoratiobarrel,
      float Tisoratioendcap,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float clusshapebarrel,
      float clusshapeendcap,
      float r9barrel,
      float r9endcap,
      float detabarrel,
      float detaendcap,
      float dphibarrel,
      float dphiendcap)
{

   int rc = 0;

   //Loop over uncorrected oh b-jets
   for (int j = 0; j < NohBJetL2; j++)
   {

      if (ohBJetL2Et[j] > jetEt && fabs(ohBJetL2Eta[j]) < jetEta)
      { // ET and eta cuts

         bool isOverlapping = false;

         // ****************************************************
         // Exclude jets which are matched to electrons
         // ****************************************************
         float barreleta = 1.479;
         float endcapeta = 2.65;

         // Loop over all oh electrons
         for (int i=0; i<NohEle; i++)
         {
            // ****************************************************
            // Bug fix
            // To be removed once the new ntuples are produced
            // ****************************************************
            float ohEleHoverE;
            float ohEleR9value;
            if (ohEleL1iso[i] == 1)
            {
               ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
               ohEleR9value = ohEleR9[i];
            }
            if (ohEleL1iso[i] == 0)
            {
               ohEleHoverE = ohEleR9[i]/ohEleE[i];
               ohEleR9value = ohEleHforHoverE[i];
            }
            // ****************************************************
            // ****************************************************
            int isbarrel = 0;
            int isendcap = 0;
            if (TMath::Abs(ohEleEta[i]) < barreleta)
               isbarrel = 1;
            if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
                  < endcapeta)
               isendcap = 1;

            if (ohEleEt[i] > Et)
            {
               if (TMath::Abs(ohEleEta[i]) < endcapeta)
               {
                  if (ohEleNewSC[i]<=1)
                  {
                     if (ohElePixelSeeds[i]>0)
                     {
                        if (ohEleL1iso[i] >= L1iso)
                        { // L1iso is 0 or 1 
                           if (ohEleL1Dupl[i] == false)
                           { // remove double-counted L1 SCs 
                              if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                                    < HisooverETbarrel)) || (isendcap
                                    && ((ohEleHiso[i]/ohEleEt[i])
                                          < HisooverETendcap)))
                              {
                                 if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                                       < EisooverETbarrel)) || (isendcap
                                       && ((ohEleEiso[i]/ohEleEt[i])
                                             < EisooverETendcap)))
                                 {
                                    if ( ((isbarrel) && (ohEleHoverE
                                          < hoverebarrel)) || ((isendcap)
                                          && (ohEleHoverE < hovereendcap)))
                                    {
                                       if ( (isbarrel
                                             && (((ohEleTiso[i] < Tisobarrel
                                                   && ohEleTiso[i] != -999.)
                                                   || (Tisobarrel == 999.))))
                                             || (isendcap && (((ohEleTiso[i]
                                                   < Tisoendcap && ohEleTiso[i]
                                                   != -999.) || (Tisoendcap
                                                   == 999.)))))
                                       {
                                          if (((isbarrel) && (ohEleTiso[i]
                                                /ohEleEt[i] < Tisoratiobarrel))
                                                || ((isendcap) && (ohEleTiso[i]
                                                      /ohEleEt[i]
                                                      < Tisoratioendcap)))
                                          {
                                             if ( (isbarrel && ohEleClusShap[i]
                                                   < clusshapebarrel)
                                                   || (isendcap
                                                         && ohEleClusShap[i]
                                                               < clusshapeendcap))
                                             {
                                                if ( (isbarrel && ohEleR9value
                                                      < r9barrel) || (isendcap
                                                      && ohEleR9value
                                                            < r9endcap))
                                                {
                                                   if ( (isbarrel
                                                         && TMath::Abs(ohEleDeta[i])
                                                               < detabarrel)
                                                         || (isendcap
                                                               && TMath::Abs(ohEleDeta[i])
                                                                     < detaendcap))
                                                   {
                                                      if ( (isbarrel
                                                            && ohEleDphi[i]
                                                                  < dphibarrel)
                                                            || (isendcap
                                                                  && ohEleDphi[i]
                                                                        < dphiendcap))
                                                      {

                                                         double
                                                               deltaphi =
                                                                     fabs(ohBJetL2Phi[j]
                                                                           -ohElePhi[i]);
                                                         if (deltaphi > 3.14159)
                                                            deltaphi = (2.0
                                                                  * 3.14159)
                                                                  - deltaphi;

                                                         double
                                                               deltaRJetEle =
                                                                     sqrt((ohBJetL2Eta[j]
                                                                           -ohEleEta[i])
                                                                           *(ohBJetL2Eta[j]
                                                                                 -ohEleEta[i])
                                                                           + (deltaphi
                                                                                 *deltaphi));

                                                         if (deltaRJetEle
                                                               < drcut)
                                                         {
                                                            isOverlapping
                                                                  = true;
                                                            break;
                                                         }
                                                      }
                                                   }
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }

         if (!isOverlapping)
         {//overlap
            if (ohBJetIPL25Tag[j] >= discL25)
            { // Level 2.5 b tag  
               if (ohBJetIPL3Tag[j] >= discL3)
               { // Level 3 b tag  
                  rc++;
               }
            }
         }//overlap  
      }
   }//loop over jets

   return rc;
}

int OHltTree::OpenHltRUPassed(
      float Rmin,
      float MRmin,
      bool MRP,
      int NJmax,
      float jetPt)
{
   //make a list of the vectors
   vector<TLorentzVector*> JETS;

   for (int i=0; i<NrecoJetCal; i++)
   {
      if (fabs(recoJetCalEta[i])>=3 || recoJetCalPt[i] < jetPt)
         continue; // require jets with eta<3
      TLorentzVector* tmp = new TLorentzVector();
      tmp->SetPtEtaPhiE(
            recoJetCalPt[i],
            recoJetCalEta[i],
            recoJetCalPhi[i],
            recoJetCalE[i]);

      JETS.push_back(tmp);
   }

   int jetsSize = 0;
   jetsSize = JETS.size();

   //Now make the hemispheres
   //for this simulation, we will used TLorentzVectors, although this is probably not
   //possible online
   if (jetsSize<2)
      return 0;
   if (NJmax!=-1 && jetsSize > NJmax)
      return 1;
   int N_comb = 1; // compute the number of combinations of jets possible
   for (int i = 0; i < jetsSize; i++)
   { // this is code is kept as close as possible
      N_comb *= 2; //to Chris' code for validation
   }
   TLorentzVector j1, j2;
   double M_min = 9999999999.0;
   double dHT_min = 99999999.0;
   int j_count;
   for (int i=0; i<N_comb; i++)
   {
      TLorentzVector j_temp1, j_temp2;
      int itemp = i;
      j_count = N_comb/2;
      int count = 0;
      while (j_count > 0)
      {
         if (itemp/j_count == 1)
         {
            j_temp1 += *(JETS.at(count));
         }
         else
         {
            j_temp2 += *(JETS.at(count));
         }
         itemp -= j_count*(itemp/j_count);
         j_count /= 2;
         count++;
      }
      double M_temp = j_temp1.M2()+j_temp2.M2();
      if (M_temp < M_min)
      {
         M_min = M_temp;
         j1= j_temp1;
         j2= j_temp2;
      }
      double dHT_temp = fabs(j_temp1.E()-j_temp2.E());
      if (dHT_temp < dHT_min)
      {
         dHT_min = dHT_temp;
         //deltaHT = dHT_temp;
      }
   }

   j1.SetPtEtaPhiM(j1.Pt(), j1.Eta(), j1.Phi(), 0.0);
   j2.SetPtEtaPhiM(j2.Pt(), j2.Eta(), j2.Phi(), 0.0);

   if (j2.Pt() > j1.Pt())
   {
      TLorentzVector temp = j1;
      j1 = j2;
      j2 = temp;
   }
   //Done Calculating Hemispheres
   //Now we can check if the event is of type R or R'

   double num = j1.P()-j2.P();
   double den = j1.Pz()-j2.Pz();
   if (fabs(num)==fabs(den))
      return 0; //ignore if beta=1
   if (fabs(num)<fabs(den) && MRP)
      return 0; //num<den ==> R event
   if (fabs(num)>fabs(den) && !MRP)
      return 0; // num>den ==> R' event

   //now we can calculate MTR
   TVector3 met;
   met.SetPtEtaPhi(recoMetCal, 0, recoMetCalPhi);
   double MTR = sqrt(0.5*(met.Mag()*(j1.Pt()+j2.Pt()) - met.Dot(j1.Vect()
         +j2.Vect())));

   //calculate MR or MRP
   double MR=0;
   if (!MRP)
   { //CALCULATE MR
      double temp = (j1.P()*j2.Pz()-j2.P()*j1.Pz())*(j1.P()*j2.Pz()-j2.P()
            *j1.Pz());
      temp /= (j1.Pz()-j2.Pz())*(j1.Pz()-j2.Pz())-(j1.P()-j2.P())*(j1.P()
            -j2.P());
      MR = 2.*sqrt(temp);
   }
   else
   { //CALCULATE MRP   
      double jaP = j1.Pt()*j1.Pt() +j1.Pz()*j2.Pz()-j1.P()*j2.P();
      double jbP = j2.Pt()*j2.Pt() +j1.Pz()*j2.Pz()-j1.P()*j2.P();
      jbP *= -1.;
      double den = sqrt((j1.P()-j2.P())*(j1.P()-j2.P())-(j1.Pz()-j2.Pz())
            *(j1.Pz()-j2.Pz()));

      jaP /= den;
      jbP /= den;

      double temp = jaP*met.Dot(j2.Vect())/met.Mag() + jbP*met.Dot(j1.Vect())
            /met.Mag();
      temp = temp*temp;

      den = (met.Dot(j1.Vect()+j2.Vect())/met.Mag())*(met.Dot(j1.Vect()
            +j2.Vect())/met.Mag())-(jaP-jbP)*(jaP-jbP);

      if (den <= 0.0)
         return 0.;

      temp /= den;
      temp = 2.*sqrt(temp);

      double bR = (jaP-jbP)/(met.Dot(j1.Vect()+j2.Vect())/met.Mag());
      double gR = 1./sqrt(1.-bR*bR);

      temp *= gR;

      MR = temp;
   }
   if (MR<MRmin || float(MTR)/float(MR)<Rmin)
      return 0;

   return 1;
}

int OHltTree::OpenHltRPassed(
      float Rmin,
      float MRmin,
      bool MRP,
      int NJmax,
      float jetPt)
{
   //make a list of the vectors
   vector<TLorentzVector*> JETS;

   for (int i=0; i<NrecoJetCorCal; i++)
   {
      if (fabs(recoJetCorCalEta[i])>=3 || recoJetCorCalPt[i] < jetPt)
         continue; // require jets with eta<3
      TLorentzVector* tmp = new TLorentzVector();
      tmp->SetPtEtaPhiE(
            recoJetCorCalPt[i],
            recoJetCorCalEta[i],
            recoJetCorCalPhi[i],
            recoJetCorCalE[i]);

      JETS.push_back(tmp);
   }

   int jetsSize = 0;
   jetsSize = JETS.size();

   //Now make the hemispheres
   //for this simulation, we will used TLorentzVectors, although this is probably not
   //possible online
   if (jetsSize<2)
      return 0;
   if (NJmax!=-1 && jetsSize > NJmax)
      return 1;
   int N_comb = 1; // compute the number of combinations of jets possible
   for (int i = 0; i < jetsSize; i++)
   { // this is code is kept as close as possible
      N_comb *= 2; //to Chris' code for validation
   }
   TLorentzVector j1, j2;
   double M_min = 9999999999.0;
   double dHT_min = 99999999.0;
   int j_count;
   for (int i=0; i<N_comb; i++)
   {
      TLorentzVector j_temp1, j_temp2;
      int itemp = i;
      j_count = N_comb/2;
      int count = 0;
      while (j_count > 0)
      {
         if (itemp/j_count == 1)
         {
            j_temp1 += *(JETS.at(count));
         }
         else
         {
            j_temp2 += *(JETS.at(count));
         }
         itemp -= j_count*(itemp/j_count);
         j_count /= 2;
         count++;
      }
      double M_temp = j_temp1.M2()+j_temp2.M2();
      if (M_temp < M_min)
      {
         M_min = M_temp;
         j1= j_temp1;
         j2= j_temp2;
      }
      double dHT_temp = fabs(j_temp1.E()-j_temp2.E());
      if (dHT_temp < dHT_min)
      {
         dHT_min = dHT_temp;
         //deltaHT = dHT_temp;
      }
   }

   j1.SetPtEtaPhiM(j1.Pt(), j1.Eta(), j1.Phi(), 0.0);
   j2.SetPtEtaPhiM(j2.Pt(), j2.Eta(), j2.Phi(), 0.0);

   if (j2.Pt() > j1.Pt())
   {
      TLorentzVector temp = j1;
      j1 = j2;
      j2 = temp;
   }
   //Done Calculating Hemispheres
   //Now we can check if the event is of type R or R'

   double num = j1.P()-j2.P();
   double den = j1.Pz()-j2.Pz();
   if (fabs(num)==fabs(den))
      return 0; //ignore if beta=1
   if (fabs(num)<fabs(den) && MRP)
      return 0; //num<den ==> R event
   if (fabs(num)>fabs(den) && !MRP)
      return 0; // num>den ==> R' event

   //now we can calculate MTR
   TVector3 met;
   met.SetPtEtaPhi(recoMetCal, 0, recoMetCalPhi);
   double MTR = sqrt(0.5*(met.Mag()*(j1.Pt()+j2.Pt()) - met.Dot(j1.Vect()
         +j2.Vect())));

   //calculate MR or MRP
   double MR=0;
   if (!MRP)
   { //CALCULATE MR
      double temp = (j1.P()*j2.Pz()-j2.P()*j1.Pz())*(j1.P()*j2.Pz()-j2.P()
            *j1.Pz());
      temp /= (j1.Pz()-j2.Pz())*(j1.Pz()-j2.Pz())-(j1.P()-j2.P())*(j1.P()
            -j2.P());
      MR = 2.*sqrt(temp);
   }
   else
   { //CALCULATE MRP   
      double jaP = j1.Pt()*j1.Pt() +j1.Pz()*j2.Pz()-j1.P()*j2.P();
      double jbP = j2.Pt()*j2.Pt() +j1.Pz()*j2.Pz()-j1.P()*j2.P();
      jbP *= -1.;
      double den = sqrt((j1.P()-j2.P())*(j1.P()-j2.P())-(j1.Pz()-j2.Pz())
            *(j1.Pz()-j2.Pz()));

      jaP /= den;
      jbP /= den;

      double temp = jaP*met.Dot(j2.Vect())/met.Mag() + jbP*met.Dot(j1.Vect())
            /met.Mag();
      temp = temp*temp;

      den = (met.Dot(j1.Vect()+j2.Vect())/met.Mag())*(met.Dot(j1.Vect()
            +j2.Vect())/met.Mag())-(jaP-jbP)*(jaP-jbP);

      if (den <= 0.0)
         return 0.;

      temp /= den;
      temp = 2.*sqrt(temp);

      double bR = (jaP-jbP)/(met.Dot(j1.Vect()+j2.Vect())/met.Mag());
      double gR = 1./sqrt(1.-bR*bR);

      temp *= gR;

      MR = temp;
   }
   if (MR<MRmin || float(MTR)/float(MR)<Rmin)
      return 0;

   return 1;
}


int OHltTree::OpenHlt1MuonPassed(
      double ptl1,
      double ptl2,
      double ptl3,
      double dr,
      int iso)
{
   // This example implements the new (CMSSW_2_X) flat muon pT cuts.
   // To emulate the old behavior, the cuts should be written
   // L2:        ohMuL2Pt[i]+3.9*ohMuL2PtErr[i]*ohMuL2Pt[i]
   // L3:        ohMuL3Pt[i]+2.2*ohMuL3PtErr[i]*ohMuL3Pt[i]

   int rcL1 = 0;
   int rcL2 = 0;
   int rcL3 = 0;
   int rcL1L2L3 = 0;
   int NL1Mu = 8;
   int L1MinimalQuality = 4;
   int L1MaximalQuality = 7;
   int doL1L2matching = 0;

   for (int ic = 0; ic < 10; ic++)
      L3MuCandIDForOnia[ic] = -1;

   // Loop over all oh L3 muons and apply cuts
   for (int i=0; i<NohMuL3; i++)
   {
      int bestl1l2drmatchind = -1;
      double bestl1l2drmatch = 999.0;

      if (fabs(ohMuL3Eta[i]) < 2.5)
      { // L3 eta cut  
         if (ohMuL3Pt[i] > ptl3)
         { // L3 pT cut        
            if (ohMuL3Dr[i] < dr)
            { // L3 DR cut
               if (ohMuL3Iso[i] >= iso)
               { // L3 isolation
                  rcL3++;

                  // Begin L2 muons here. 
                  // Get best L2<->L3 match, then 
                  // begin applying cuts to L2
                  int j = ohMuL3L2idx[i]; // Get best L2<->L3 match

                  if ( (fabs(ohMuL2Eta[j])<2.5))
                  { // L2 eta cut
                     if (ohMuL2Pt[j] > ptl2)
                     { // L2 pT cut
                        if (ohMuL2Iso[j] >= iso)
                        { // L2 isolation
                           rcL2++;

                           // Begin L1 muons here.
                           // Require there be an L1Extra muon Delta-R
                           // matched to the L2 candidate, and that it have 
                           // good quality and pass nominal L1 pT cuts 
                           for (int k = 0; k < NL1Mu; k++)
                           {
                              if ( (L1MuPt[k] < ptl1)) // L1 pT cut
                                 continue;

                              double deltaphi = fabs(ohMuL2Phi[j]-L1MuPhi[k]);
                              if (deltaphi > 3.14159)
                                 deltaphi = (2.0 * 3.14159) - deltaphi;

                              double deltarl1l2 =
                                    sqrt((ohMuL2Eta[j]-L1MuEta[k])
                                          *(ohMuL2Eta[j]-L1MuEta[k])
                                          + (deltaphi*deltaphi));
                              if (deltarl1l2 < bestl1l2drmatch)
                              {
                                 bestl1l2drmatchind = k;
                                 bestl1l2drmatch = deltarl1l2;
                              }
                           } // End loop over L1Extra muons

                           if (doL1L2matching == 1)
                           {
                              // Cut on L1<->L2 matching and L1 quality
                              if ((bestl1l2drmatch > 0.3)
                                    || (L1MuQal[bestl1l2drmatchind]
                                          < L1MinimalQuality)
                                    || (L1MuQal[bestl1l2drmatchind]
                                          > L1MaximalQuality))
                              {
                                 rcL1 = 0;
                                 cout << "Failed L1-L2 match/quality" << endl;
                                 cout << "L1-L2 delta-eta = "
                                       << L1MuEta[bestl1l2drmatchind] << ", "
                                       << ohMuL2Eta[j] << endl;
                                 cout << "L1-L2 delta-pho = "
                                       << L1MuPhi[bestl1l2drmatchind] << ", "
                                       << ohMuL2Phi[j] << endl;
                                 cout << "L1-L2 delta-R = " << bestl1l2drmatch
                                       << endl;
                              }
                              else
                              {
                                 cout << "Passed L1-L2 match/quality" << endl;
                                 L3MuCandIDForOnia[rcL1L2L3] = i;
                                 rcL1++;
                                 rcL1L2L3++;
                              } // End L1 matching and quality cuts	      
                           }
                           else
                           {
                              L3MuCandIDForOnia[rcL1L2L3] = i;
                              rcL1L2L3++;
                           }
                        } // End L2 isolation cut 
                     } // End L2 eta cut
                  } // End L2 pT cut
               } // End L3 isolation cut
            } // End L3 DR cut
         } // End L3 pT cut
      } // End L3 eta cut
   } // End loop over L3 muons		      

   return rcL1L2L3;
}

int OHltTree::OpenHlt2MuonOSMassPassed(
      double ptl1,
      double ptl2,
      double ptl3,
      double dr,
      int iso,
      double masslow,
      double masshigh)
{
  int rc = 0;
  TLorentzVector mu1;
  TLorentzVector mu2;
  TLorentzVector diMu;
  const double muMass = 0.105658367;

  for (int i=0; i<NohMuL3; i++)
    {
      for (int j=i+1; j<NohMuL3; j++)
	{
	  if (ohMuL3Pt[i] > ptl3 && ohMuL3Pt[j] > ptl3)
	    { // L3 pT cut
	      if (ohMuL3Dr[i] < dr && ohMuL3Dr[j] < dr)
		{ // L3 DR cut
		  if ((ohMuL3Chg[i] * ohMuL3Chg[j]) < 0)
		    { // opposite charge

		      mu1.SetPtEtaPhiM(
				       ohMuL3Pt[i],
				       ohMuL3Eta[i],
				       ohMuL3Phi[i],
				       muMass);
		      mu2.SetPtEtaPhiM(
				       ohMuL3Pt[j],
				       ohMuL3Eta[j],
				       ohMuL3Phi[j],
				       muMass);
		      diMu = mu1 + mu2;
		      float diMuMass = diMu.M();

		      if ((diMuMass <= masshigh) && (diMuMass >= masslow))
			{
			  int l2match1 = ohMuL3L2idx[i];
			  int l2match2 = ohMuL3L2idx[j];
			  
			  if ( (ohMuL2Pt[l2match1] > ptl2)
			       && (ohMuL2Pt[l2match2] > ptl2))
			    { // L2 pT cut
			      rc++;
			    }
			}
		    }
		}
	    }
	}
    }
  return rc;
}

//// Separating between Jets and Muon.....
int OHltTree::OpenHlt1MuonIsoJetPassed(
      double ptl1,
      double ptl2,
      double ptl3,
      double dr,
      int iso,
      double JetPt,
      double JetEta)
{
   // This example implements the new (CMSSW_2_X) flat muon pT cuts.
   // To emulate the old behavior, the cuts should be written
   // L2:        ohMuL2Pt[i]+3.9*ohMuL2PtErr[i]*ohMuL2Pt[i]
   // L3:        ohMuL3Pt[i]+2.2*ohMuL3PtErr[i]*ohMuL3Pt[i]

   int rcL1 = 0;
   int rcL2 = 0;
   int rcL3 = 0;
   int rcL1L2L3 = 0;
   int NL1Mu = 8;
   int L1MinimalQuality = 4;
   int L1MaximalQuality = 7;
   int doL1L2matching = 0;

   for (int ic = 0; ic < 10; ic++)
      L3MuCandIDForOnia[ic] = -1;

   // Loop over all oh L3 muons and apply cuts
   for (int i=0; i<NohMuL3; i++)
   {
      int bestl1l2drmatchind = -1;
      double bestl1l2drmatch = 999.0;

      if (fabs(ohMuL3Eta[i]) < 2.5)
      { // L3 eta cut  
         if (ohMuL3Pt[i] > ptl3)
         { // L3 pT cut        
            if (ohMuL3Dr[i] < dr)
            { // L3 DR cut
               if (ohMuL3Iso[i] >= iso)
               { // L3 isolation

                  // Loop over all oh corrected jets    
                  float minDR = 100.;
                  for (int j=0; j <NrecoJetCorCal; j++)
                  {
                     if (recoJetCorCalPt[j]>JetPt && fabs(recoJetCorCalEta[j])
                           <JetEta)
                     { // Jet pT cut
                        double deltaphi =
                              fabs(recoJetCorCalPhi[j]-ohMuL3Phi[i]);
                        if (deltaphi > 3.14159)
                           deltaphi = (2.0 * 3.14159) - deltaphi;
                        float deltaRMuJet = sqrt((recoJetCorCalEta[j]
                              -ohMuL3Eta[i])*(recoJetCorCalEta[j]-ohMuL3Eta[i])
                              + (deltaphi*deltaphi));
                        if (deltaRMuJet < minDR)
                        {
                           minDR = deltaRMuJet;
                        }
                     }
                  }
                  if (minDR < 0.3)
                     break;

                  rcL3++;

                  // Begin L2 muons here. 
                  // Get best L2<->L3 match, then 
                  // begin applying cuts to L2
                  int j = ohMuL3L2idx[i]; // Get best L2<->L3 match

                  if ( (fabs(ohMuL2Eta[j])<2.5))
                  { // L2 eta cut
                     if (ohMuL2Pt[j] > ptl2)
                     { // L2 pT cut
                        if (ohMuL2Iso[j] >= iso)
                        { // L2 isolation
                           rcL2++;

                           // Begin L1 muons here.
                           // Require there be an L1Extra muon Delta-R
                           // matched to the L2 candidate, and that it have 
                           // good quality and pass nominal L1 pT cuts 
                           for (int k = 0; k < NL1Mu; k++)
                           {
                              if ( (L1MuPt[k] < ptl1)) // L1 pT cut
                                 continue;

                              double deltaphi = fabs(ohMuL2Phi[j]-L1MuPhi[k]);
                              if (deltaphi > 3.14159)
                                 deltaphi = (2.0 * 3.14159) - deltaphi;

                              double deltarl1l2 =
                                    sqrt((ohMuL2Eta[j]-L1MuEta[k])
                                          *(ohMuL2Eta[j]-L1MuEta[k])
                                          + (deltaphi*deltaphi));
                              if (deltarl1l2 < bestl1l2drmatch)
                              {
                                 bestl1l2drmatchind = k;
                                 bestl1l2drmatch = deltarl1l2;
                              }
                           } // End loop over L1Extra muons

                           if (doL1L2matching == 1)
                           {
                              // Cut on L1<->L2 matching and L1 quality
                              if ((bestl1l2drmatch > 0.3)
                                    || (L1MuQal[bestl1l2drmatchind]
                                          < L1MinimalQuality)
                                    || (L1MuQal[bestl1l2drmatchind]
                                          > L1MaximalQuality))
                              {
                                 rcL1 = 0;
                                 cout << "Failed L1-L2 match/quality" << endl;
                                 cout << "L1-L2 delta-eta = "
                                       << L1MuEta[bestl1l2drmatchind] << ", "
                                       << ohMuL2Eta[j] << endl;
                                 cout << "L1-L2 delta-pho = "
                                       << L1MuPhi[bestl1l2drmatchind] << ", "
                                       << ohMuL2Phi[j] << endl;
                                 cout << "L1-L2 delta-R = " << bestl1l2drmatch
                                       << endl;
                              }
                              else
                              {
                                 cout << "Passed L1-L2 match/quality" << endl;
                                 L3MuCandIDForOnia[rcL1L2L3] = i;
                                 rcL1++;
                                 rcL1L2L3++;
                              } // End L1 matching and quality cuts      
                           }
                           else
                           {
                              L3MuCandIDForOnia[rcL1L2L3] = i;
                              rcL1L2L3++;
                           }
                        } // End L2 isolation cut 
                     } // End L2 eta cut
                  } // End L2 pT cut
               } // End L3 isolation cut
            } // End L3 DR cut
         } // End L3 pT cut
      } // End L3 eta cut
   } // End loop over L3 muons      
   return rcL1L2L3;

}

//++++++++++++++++++++++++++++++++
// For Overlap removal of muons in jets
//Added by J.Rani
int OHltTree::OpenHlt1BJetPassedMuRemoval(
      float jetEt,
      float jetEta,
      float drcut,
      float discL25,
      float discL3,
      double ptl1,
      double ptl2,
      double ptl3,
      double dr,
      int iso)

{

   int rc = 0;

   //Loop over uncorrected oh b-jets
   for (int kk = 0; kk < NohBJetL2; kk++)
   {

      if (ohBJetL2Et[kk] > jetEt && fabs(ohBJetL2Eta[kk]) < jetEta)
      { // ET and eta cuts

         bool isOverlapping = false;

         int rcL2 = 0;
         int rcL3 = 0;
         int NL1Mu = 8;

         // Loop over all oh L3 muons and apply cuts
         for (int i=0; i<NohMuL3; i++)
         {
            int bestl1l2drmatchind = -1;
            double bestl1l2drmatch = 999.0;

            if (fabs(ohMuL3Eta[i]) < 2.5)
            { // L3 eta cut  
               if (ohMuL3Pt[i] > ptl3)
               { // L3 pT cut        
                  if (ohMuL3Dr[i] < dr)
                  { // L3 DR cut
                     if (ohMuL3Iso[i] >= iso)
                     { // L3 isolation
                        rcL3++;

                        // Begin L2 muons here. 
                        // Get best L2<->L3 match, then 
                        // begin applying cuts to L2
                        int j = ohMuL3L2idx[i]; // Get best L2<->L3 match

                        if ( (fabs(ohMuL2Eta[j])<2.5))
                        { // L2 eta cut
                           if (ohMuL2Pt[j] > ptl2)
                           { // L2 pT cut
                              if (ohMuL2Iso[j] >= iso)
                              { // L2 isolation
                                 rcL2++;

                                 // Begin L1 muons here.
                                 // Require there be an L1Extra muon Delta-R
                                 // matched to the L2 candidate, and that it have 
                                 // good quality and pass nominal L1 pT cuts 
                                 for (int k = 0; k < NL1Mu; k++)
                                 {
                                    if ( (L1MuPt[k] < ptl1)) // L1 pT cut
                                       continue;

                                    double deltaphi = fabs(ohMuL2Phi[j]
                                          -L1MuPhi[k]);
                                    if (deltaphi > 3.14159)
                                       deltaphi = (2.0 * 3.14159) - deltaphi;

                                    double deltarl1l2 = sqrt((ohMuL2Eta[j]
                                          -L1MuEta[k])
                                          *(ohMuL2Eta[j]-L1MuEta[k])
                                          + (deltaphi*deltaphi));
                                    if (deltarl1l2 < bestl1l2drmatch)
                                    {
                                       bestl1l2drmatchind = k;
                                       bestl1l2drmatch = deltarl1l2;
                                    }
                                 } // End loop over L1Extra muons


                                 double deltaphi = fabs(ohBJetL2Phi[kk]
                                       -ohMuL2Phi[j]);
                                 if (deltaphi > 3.14159)
                                    deltaphi = (2.0 * 3.14159) - deltaphi;

                                 double deltaRJetMu = sqrt((ohBJetL2Eta[kk]
                                       -ohMuL2Eta[j])*(ohBJetL2Eta[kk]
                                       -ohMuL2Eta[j]) + (deltaphi*deltaphi));

                                 if (deltaRJetMu < drcut)
                                 {
                                    isOverlapping = true;
                                    break;
                                 }
                              } // End L2 isolation cut 
                           } // End L2 eta cut
                        } // End L2 pT cut
                     } // End L3 isolation cut
                  } // End L3 DR cut
               } // End L3 pT cut
            } // End L3 eta cut
         } // End loop over L3 muons      

         if (!isOverlapping)
         {//overlap
            if (ohBJetIPL25Tag[kk] >= discL25)
            { // Level 2.5 b tag  
               if (ohBJetIPL3Tag[kk] >= discL3)
               { // Level 3 b tag  
                  rc++;
               }
            }
         }//overlap  
      }
   }//loop over jets

   return rc;
}
int OHltTree::OpenHlt2MuonPassed(
      double ptl1,
      double ptl2,
      double ptl3,
      double dr,
      int iso)
{
   // Note that the dimuon paths generally have different L1 requirements than 
   // the single muon paths. Therefore this example is implemented in a separate
   // function.
   //
   // This example implements the new (CMSSW_2_X) flat muon pT cuts. 
   // To emulate the old behavior, the cuts should be written 
   // L2:        ohMuL2Pt[i]+3.9*ohMuL2PtErr[i]*ohMuL2Pt[i] 
   // L3:        ohMuL3Pt[i]+2.2*ohMuL3PtErr[i]*ohMuL3Pt[i] 

   int rcL1 = 0;
   int rcL2 = 0;
   int rcL3 = 0;
   int rcL1L2L3 = 0;
   int NL1Mu = 8;
   int L1MinimalQuality = 3;
   int L1MaximalQuality = 7;
   int doL1L2matching = 0;

   // Loop over all oh L3 muons and apply cuts 
   for (int i=0; i<NohMuL3; i++)
   {
      int bestl1l2drmatchind = -1;
      double bestl1l2drmatch = 999.0;

      if (fabs(ohMuL3Eta[i]) < 2.5)
      { // L3 eta cut   
         if (ohMuL3Pt[i] > ptl3)
         { // L3 pT cut         
            if (ohMuL3Dr[i] < dr)
            { // L3 DR cut 
               if (ohMuL3Iso[i] >= iso)
               { // L3 isolation 
                  rcL3++;

                  // Begin L2 muons here.  
                  // Get best L2<->L3 match, then  
                  // begin applying cuts to L2 
                  int j = ohMuL3L2idx[i]; // Get best L2<->L3 match 

                  if ( (fabs(ohMuL2Eta[j])<2.5))
                  { // L2 eta cut 
                     if (ohMuL2Pt[j] > ptl2)
                     { // L2 pT cut 
                        if (ohMuL2Iso[i] >= iso)
                        { // L2 isolation  
                           rcL2++;

                           // Begin L1 muons here. 
                           // Require there be an L1Extra muon Delta-R 
                           // matched to the L2 candidate, and that it have  
                           // good quality and pass nominal L1 pT cuts  
                           for (int k = 0; k < NL1Mu; k++)
                           {
                              if ( (L1MuPt[k] < ptl1)) // L1 pT cut 
                                 continue;

                              double deltaphi = fabs(ohMuL2Phi[j]-L1MuPhi[k]);
                              if (deltaphi > 3.14159)
                                 deltaphi = (2.0 * 3.14159) - deltaphi;

                              double deltarl1l2 =
                                    sqrt((ohMuL2Eta[j]-L1MuEta[k])
                                          *(ohMuL2Eta[j]-L1MuEta[k])
                                          + (deltaphi*deltaphi));
                              if (deltarl1l2 < bestl1l2drmatch)
                              {
                                 bestl1l2drmatchind = k;
                                 bestl1l2drmatch = deltarl1l2;
                              }
                           } // End loop over L1Extra muons 

                           if (doL1L2matching == 1)
                           {
                              // Cut on L1<->L2 matching and L1 quality 
                              if ((bestl1l2drmatch > 0.3)
                                    || (L1MuQal[bestl1l2drmatchind]
                                          < L1MinimalQuality)
                                    || (L1MuQal[bestl1l2drmatchind]
                                          > L1MaximalQuality))
                              {
                                 rcL1 = 0;
                              }
                              else
                              {
                                 rcL1++;
                                 rcL1L2L3++;
                              } // End L1 matching and quality cuts        
                           }
                           else
                           {
                              rcL1L2L3++;
                           }
                        } // End L2 isolation cut 
                     } // End L2 eta cut
                  } // End L2 pT cut 
               } // End L3 isolation cut 
            } // End L3 DR cut 
         } // End L3 pT cut 
      } // End L3 eta cut 
   } // End loop over L3 muons                  

   return rcL1L2L3;
}

int OHltTree::OpenHlt1L2MuonPassed(double ptl1, double ptl2, double dr)
{
   // This is a modification of the standard Hlt1Muon code, which does not consider L3 information 

   int rcL1 = 0;
   int rcL2 = 0;
   int rcL1L2L3 = 0;
   int NL1Mu = 8;
   int L1MinimalQuality = 3;
   int L1MaximalQuality = 7;
   int doL1L2matching = 0;

   // Loop over all oh L2 muons and apply cuts 
   for (int j=0; j<NohMuL2; j++)
   {
      int bestl1l2drmatchind = -1;
      double bestl1l2drmatch = 999.0;

      if (fabs(ohMuL2Eta[j])>=2.5)
         continue; // L2 eta cut 
      if (ohMuL2Pt[j] <= ptl2)
         continue; // L2 pT cut 
      rcL2++;

      // Begin L1 muons here. 
      // Require there be an L1Extra muon Delta-R 
      // matched to the L2 candidate, and that it have  
      // good quality and pass nominal L1 pT cuts  
      for (int k = 0; k < NL1Mu; k++)
      {
         if ( (L1MuPt[k] < ptl1))
            continue; // L1 pT cut 

         double deltaphi = fabs(ohMuL2Phi[j]-L1MuPhi[k]);
         if (deltaphi > 3.14159)
            deltaphi = (2.0 * 3.14159) - deltaphi;
         double deltarl1l2 = sqrt((ohMuL2Eta[j]-L1MuEta[k])*(ohMuL2Eta[j]
               -L1MuEta[k]) + (deltaphi*deltaphi));
         if (deltarl1l2 < bestl1l2drmatch)
         {
            bestl1l2drmatchind = k;
            bestl1l2drmatch = deltarl1l2;
         }
      } // End loop over L1Extra muons 
      if (doL1L2matching == 1)
      {
         // Cut on L1<->L2 matching and L1 quality 
         if ((bestl1l2drmatch > 0.3) || (L1MuQal[bestl1l2drmatchind]
               < L1MinimalQuality) || (L1MuQal[bestl1l2drmatchind]
               > L1MaximalQuality))
         {
            rcL1 = 0;
            cout << "Failed L1-L2 match/quality" << endl;
            cout << "L1-L2 delta-eta = " << L1MuEta[bestl1l2drmatchind] << ", "
                  << ohMuL2Eta[j] << endl;
            cout << "L1-L2 delta-pho = " << L1MuPhi[bestl1l2drmatchind] << ", "
                  << ohMuL2Phi[j] << endl;
            cout << "L1-L2 delta-R = " << bestl1l2drmatch << endl;
         }
         else
         {
            cout << "Passed L1-L2 match/quality" << endl;
            rcL1++;
            rcL1L2L3++;
         } // End L1 matching and quality cuts            
      }
      else
      {
         rcL1L2L3++;
      }
   } // End L2 loop over muons 
   return rcL1L2L3;
}

int OHltTree::OpenHlt1L2MuonNoVertexPassed(double ptl1, double ptl2, double dr) 
{ 
  // This is a modification of the standard Hlt1Muon code, which does not consider L3 information  
 
  int rcL1 = 0; 
  int rcL2 = 0; 
  int rcL1L2L3 = 0; 
  int NL1Mu = 8; 
  int L1MinimalQuality = 3; 
  int L1MaximalQuality = 7; 
  int doL1L2NoVtxmatching = 0; 
 
  // Loop over all oh L2NoVtx muons and apply cuts  
  for (int j=0; j<NohMuL2NoVtx; j++) 
    { 
      int bestl1l2drmatchind = -1; 
      double bestl1l2drmatch = 999.0; 
 
      if (fabs(ohMuL2NoVtxEta[j])>=2.5) 
	continue; // L2NoVtx eta cut  
      if (ohMuL2NoVtxPt[j] <= ptl2) 
	continue; // L2NoVtx pT cut  
      rcL2++; 
 
      // Begin L1 muons here.  
      // Require there be an L1Extra muon Delta-R  
      // matched to the L2NoVtx candidate, and that it have   
      // good quality and pass nominal L1 pT cuts   
      for (int k = 0; k < NL1Mu; k++) 
	{ 
	  if ( (L1MuPt[k] < ptl1)) 
            continue; // L1 pT cut  
 
	  double deltaphi = fabs(ohMuL2NoVtxPhi[j]-L1MuPhi[k]); 
	  if (deltaphi > 3.14159) 
            deltaphi = (2.0 * 3.14159) - deltaphi; 
	  double deltarl1l2 = sqrt((ohMuL2NoVtxEta[j]-L1MuEta[k])*(ohMuL2NoVtxEta[j] 
							      -L1MuEta[k]) + (deltaphi*deltaphi)); 
	  if (deltarl1l2 < bestl1l2drmatch) 
	    { 
	      bestl1l2drmatchind = k; 
	      bestl1l2drmatch = deltarl1l2; 
	    } 
	} // End loop over L1Extra muons  
      if (doL1L2NoVtxmatching == 1) 
	{ 
	  // Cut on L1<->L2NoVtx matching and L1 quality  
	  if ((bestl1l2drmatch > 0.3) || (L1MuQal[bestl1l2drmatchind] 
					  < L1MinimalQuality) || (L1MuQal[bestl1l2drmatchind] 
								  > L1MaximalQuality)) 
	    { 
	      rcL1 = 0; 
	    } 
	  else 
	    { 
	      cout << "Passed L1-L2NoVtx match/quality" << endl; 
	      rcL1++; 
	      rcL1L2L3++; 
	    } // End L1 matching and quality cuts             
	} 
      else 
	{ 
	  rcL1L2L3++; 
	} 
    } // End L2 loop over muons  
  return rcL1L2L3; 
} 


int OHltTree::OpenHltMuPixelPassed(
      double ptPix,
      double pPix,
      double etaPix,
      double DxyPix,
      double DzPix,
      int NHitsPix,
      double normChi2Pix,
      double *massMinPix,
      double *massMaxPix,
      double DzMuonPix,
      bool checkChargePix)
{

   //   printf("\n\n");
   const double muMass = 0.105658367;
   TLorentzVector pix4Mom, mu4Mom, onia4Mom;
   int iNPix = 0;
   //reset counter variables:
   for (int iMu = 0; iMu < 10; iMu++)
   {
      L3PixelCandIDForOnia[iMu] = -1;
      L3MuPixCandIDForOnia[iMu] = -1;
   }

   //0.) check how many L3 muons there are:
   int nMuons = 0;
   for (int iMu = 0; iMu < 10; iMu++)
      if (L3MuCandIDForOnia[iMu] > -1)
         nMuons++;

   //1.) loop over the Pixel tracks
   for (int iP = 0; iP < NohOniaPixel; iP++)
   {

      //select those that survive the kinematical and
      //topological selection cuts
      if (fabs(ohOniaPixelEta[iP]) > etaPix)
         continue; //eta cut
      if (ohOniaPixelPt[iP] < ptPix)
         continue; //pT cut
      double momThisPix = ohOniaPixelPt[iP] * cosh(ohOniaPixelEta[iP]);
      if (momThisPix < pPix)
         continue; //momentum cut
      if (ohOniaPixelHits[iP] < NHitsPix)
         continue; //min. nb. of hits
      if (ohOniaPixelNormChi2[iP] > normChi2Pix)
         continue; //chi2 cut
      if (fabs(ohOniaPixelDr[iP]) > DxyPix)
         continue; //Dr cut
      if (fabs(ohOniaPixelDz[iP]) > DzPix)
         continue;

      pix4Mom.SetPtEtaPhiM(
            ohOniaPixelPt[iP],
            ohOniaPixelEta[iP],
            ohOniaPixelPhi[iP],
            muMass);
      //2.) loop now over all L3 muons and check if they would give a
      //Onia (J/psi or upsilon) pair:
      for (int iMu = 0; iMu < nMuons; iMu++)
      {
         mu4Mom.SetPtEtaPhiM(
               ohMuL3Pt[L3MuCandIDForOnia[iMu]],
               ohMuL3Eta[L3MuCandIDForOnia[iMu]],
               ohMuL3Phi[L3MuCandIDForOnia[iMu]],
               muMass);
         onia4Mom = pix4Mom + mu4Mom;

         double oniaMass = onia4Mom.M();
         if (oniaMass < massMinPix[0] || oniaMass> massMaxPix[1]) continue; //mass cut
         if(oniaMass> massMaxPix[0] && oniaMass < massMinPix[1]) continue; //mass cut
         if(checkChargePix)
         if(ohMuL3Chg[iMu] == ohOniaPixelChg[iP]) continue; //charge cut
         if(fabs(ohMuL3Dz[iMu] - ohOniaPixelDz[iP])> DzMuonPix) continue;

         //store the surviving pixel-muon combinations:
         if(iNPix < 10)
         {
            L3PixelCandIDForOnia[iNPix] = iP;
            L3MuPixCandIDForOnia[iNPix] = iMu;
            iNPix++;
         }
         //       printf("mu[%d]-pixel[%d] inv. mass %f\n",
         //           L3MuCandIDForOnia[iMu], iP, oniaMass);
      }
   }

   //   hNPixelCand->Fill(iNPix);
   return iNPix;
}

int OHltTree::OpenHltMuTrackPassed(
      double ptTrack,
      double pTrack,
      double etaTrack,
      double DxyTrack,
      double DzTrack,
      int NHitsTrack,
      double normChi2Track,
      double *massMinTrack,
      double *massMaxTrack,
      double DzMuonTrack,
      bool checkChargeTrack)
{

   double pixMatchingDeltaR = 0.03;
   const double muMass = 0.105658367;
   TLorentzVector track4Mom, mu4Mom, onia4Mom;
   int iNTrack = 0;

   //0.) check how many pixel-muon combinations there are:
   int nComb = 0;
   for (int iMu = 0; iMu < 10; iMu++)
      if (L3MuPixCandIDForOnia[iMu] > -1)
         nComb++;

   //   printf("OpenHltMuTrackPassed: %d incoming pixels and %d tracks\n", nComb, NohOniaTrack);

   //1.) loop over the Tracker tracks
   for (int iT = 0; iT < NohOniaTrack; iT++)
   {

      //select those that survive the kinematical and
      //topological selection cuts
      if (fabs(ohOniaTrackEta[iT]) > etaTrack)
         continue; //eta cut
      if (ohOniaTrackPt[iT] < ptTrack)
         continue; //pT cut
      double momThisTrack = ohOniaTrackPt[iT] * cosh(ohOniaTrackEta[iT]);
      //     printf("track[%d] has eta %f, pT %f and mom %f\n",
      //         iT, ohOniaTrackEta[iT], ohOniaTrackPt[iT], momThisTrack);
      if (momThisTrack < pTrack)
         continue; //momentum cut
      if (ohOniaTrackHits[iT] < NHitsTrack)
         continue; //min. nb. of hits
      if (ohOniaTrackNormChi2[iT] > normChi2Track)
         continue; //chi2 cut
      if (fabs(ohOniaTrackDr[iT]) > DxyTrack)
         continue; //Dr cut
      if (fabs(ohOniaTrackDz[iT]) > DzTrack)
         continue;

      //2.) loop over the pixels candidates to see whether the track
      //under investigation has a match to the pixel track
      bool trackMatched = false;
      for (int iPix = 0; iPix < nComb; iPix++)
      {

         if (trackMatched)
            break; //in case the track was already matched
         if (L3PixelCandIDForOnia[iPix] < 0)
            continue; //in case the pixel has been matched to a previous track

         double deltaEta = ohOniaPixelEta[L3PixelCandIDForOnia[iPix]]
               - ohOniaTrackEta[iT];
         double deltaPhi = ohOniaPixelPhi[L3PixelCandIDForOnia[iPix]]
               - ohOniaTrackPhi[iT];
         double deltaR = sqrt(pow(deltaEta, 2) + pow(deltaPhi, 2));

         if (deltaR > pixMatchingDeltaR)
            continue;
         //       printf("track[%d], pixel[%d], delta R %f\n", iT, L3PixelCandIDForOnia[iPix], deltaR);

         trackMatched = true;
         L3PixelCandIDForOnia[iPix] = -1; //deactivate this candidate to not match it to any further track

         track4Mom.SetPtEtaPhiM(
               ohOniaTrackPt[iT],
               ohOniaTrackEta[iT],
               ohOniaTrackPhi[iT],
               muMass);
         //check if the matched tracker track combined with the
         //muon gives again an opposite sign onia:
         mu4Mom.SetPtEtaPhiM(
               ohMuL3Pt[L3MuPixCandIDForOnia[iPix]],
               ohMuL3Eta[L3MuPixCandIDForOnia[iPix]],
               ohMuL3Phi[L3MuPixCandIDForOnia[iPix]],
               muMass);
         onia4Mom = track4Mom + mu4Mom;

         double oniaMass = onia4Mom.M();
         //       printf("mu[%d]-track[%d] inv. mass %f\n",
         //           L3MuPixCandIDForOnia[iPix], iT, oniaMass);

         if (oniaMass < massMinTrack[0] || oniaMass> massMaxTrack[1]) continue; //mass cut
         if(oniaMass> massMaxTrack[0] && oniaMass < massMinTrack[1]) continue; //mass cut

         //       printf("surviving: mu[%d]-track[%d] inv. mass %f\n",
         //           L3MuPixCandIDForOnia[iPix], iT, oniaMass);

         if(checkChargeTrack)
         if(ohMuL3Chg[L3MuPixCandIDForOnia[iPix]] == ohOniaTrackChg[iT]) continue; //charge cut
         if(fabs(ohMuL3Dz[L3MuPixCandIDForOnia[iPix]] - ohOniaTrackDz[iT])> DzMuonTrack) continue; //deltaZ cut

         //store the surviving track-muon combinations:
         if(iNTrack < 10)
         iNTrack++;

         break; //don't check further pixels... go to next track
      }
   }

   //   if(iNTrack > 0)
   //     printf("found %d final candidates!!!\n", iNTrack);
   return iNTrack;
}

int OHltTree::OpenHltMuPixelPassed_JPsi(
      double ptPix,
      double pPix,
      double etaPix,
      double DxyPix,
      double DzPix,
      int NHitsPix,
      double normChi2Pix,
      double *massMinPix,
      double *massMaxPix,
      double DzMuonPix,
      bool checkChargePix,
      int histIndex)
{
   //   printf("in OpenHltMuPixelPassed_JPsi \n\n");
   const double muMass = 0.105658367;
   TLorentzVector pix4Mom, mu4Mom, onia4Mom;
   int iNPix = 0;
   //reset counter variables:
   for (int iMu = 0; iMu < 10; iMu++)
   {
      L3PixelCandIDForJPsi[iMu] = -1;
      L3MuPixCandIDForJPsi[iMu] = -1;
   }

   //0.) check how many L3 muons there are:
   int nMuons = 0;
   for (int iMu = 0; iMu < 10; iMu++)
      if (L3MuCandIDForOnia[iMu] > -1)
         nMuons++;

   //   Int_t countCut = 0, countOniaCut = 0;
   //1.) loop over the Pixel tracks
   for (int iP = 0; iP < NohOniaPixel; iP++)
   {

      //     countCut = 0;
      //     hEta[histIndex][0][countCut]->Fill(ohOniaPixelEta[iP]);
      //     hPt[histIndex][0][countCut]->Fill(ohOniaPixelPt[iP]);
      //     hHits[histIndex][0][countCut]->Fill(ohOniaPixelHits[iP]);
      //     hNormChi2[histIndex][0][countCut]->Fill(ohOniaPixelNormChi2[iP]);
      //     hDxy[histIndex][0][countCut]->Fill(ohOniaPixelDr[iP]);
      //     hDz[histIndex][0][countCut]->Fill(ohOniaPixelDz[iP]);

      //select those that survive the kinematical and
      //topological selection cuts
      if (fabs(ohOniaPixelEta[iP]) > etaPix)
         continue; //eta cut
      if (ohOniaPixelPt[iP] < ptPix)
         continue; //pT cut

      double momThisPix = ohOniaPixelPt[iP] * cosh(ohOniaPixelEta[iP]);
      //     hP[histIndex][0][countCut]->Fill(momThisPix);
      //     countCut++;

      if (momThisPix < pPix)
         continue; //momentum cut
      if (ohOniaPixelHits[iP] < NHitsPix)
         continue; //min. nb. of hits
      if (ohOniaPixelNormChi2[iP] > normChi2Pix)
         continue; //chi2 cut
      if (fabs(ohOniaPixelDr[iP]) > DxyPix)
         continue; //Dr cut
      if (fabs(ohOniaPixelDz[iP]) > DzPix)
         continue;

      //     hEta[histIndex][0][countCut]->Fill(ohOniaPixelEta[iP]);
      //     hPt[histIndex][0][countCut]->Fill(ohOniaPixelPt[iP]);
      //     hHits[histIndex][0][countCut]->Fill(ohOniaPixelHits[iP]);
      //     hNormChi2[histIndex][0][countCut]->Fill(ohOniaPixelNormChi2[iP]);
      //     hDxy[histIndex][0][countCut]->Fill(ohOniaPixelDr[iP]);
      //     hDz[histIndex][0][countCut]->Fill(ohOniaPixelDz[iP]);
      //     hP[histIndex][0][countCut]->Fill(momThisPix);
      //     countCut++;

      pix4Mom.SetPtEtaPhiM(
            ohOniaPixelPt[iP],
            ohOniaPixelEta[iP],
            ohOniaPixelPhi[iP],
            muMass);
      //2.) loop now over all L3 muons and check if they would give a
      //JPsi pair:
      for (int iMu = 0; iMu < nMuons; iMu++)
      {
         mu4Mom.SetPtEtaPhiM(
               ohMuL3Pt[L3MuCandIDForOnia[iMu]],
               ohMuL3Eta[L3MuCandIDForOnia[iMu]],
               ohMuL3Phi[L3MuCandIDForOnia[iMu]],
               muMass);
         onia4Mom = pix4Mom + mu4Mom;

         double oniaMass = onia4Mom.M();
         //       printf("mu[%d]-pixel[%d] inv. mass %f\n",
         //           L3MuCandIDForOnia[iMu], iP, oniaMass);
         //       countOniaCut = 0;
         //       if(oniaMass > 5.0) continue; //Only JPsi 
         //       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if (oniaMass < massMinPix[0] || oniaMass> massMaxPix[0]) continue; //mass cut
         //       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if(checkChargePix)
         if(ohMuL3Chg[iMu] == ohOniaPixelChg[iP]) continue; //charge cut

         //       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if(fabs(ohMuL3Dz[iMu] - ohOniaPixelDz[iP])> DzMuonPix) continue;

         //       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));

         //store the surviving pixel-muon combinations:
         if(iNPix < 10)
         {
            L3PixelCandIDForJPsi[iNPix] = iP;
            L3MuPixCandIDForJPsi[iNPix] = iMu;
            iNPix++;
         }
         //       printf("surviving: mu[%d]-pixel[%d] inv. mass %f\n",
         //           L3MuCandIDForOnia[iMu], iP, oniaMass);
      }
   }

   //   hNCand[histIndex][0]->Fill(iNPix);

   //Pixel Eta, Pt, P, DR
   if(iNPix!=0)
   {
      for(int inP=0;inP<iNPix;inP++)
      {

         //       hPixCandEta[histIndex]->Fill(ohOniaPixelEta[L3PixelCandIDForJPsi[inP]]);
         //       hPixCandPt[histIndex]->Fill(ohOniaPixelPt[L3PixelCandIDForJPsi[inP]]);
         //       hPixCandP[histIndex]->Fill(ohOniaPixelPt[L3PixelCandIDForJPsi[inP]] * cosh(ohOniaPixelEta[L3PixelCandIDForJPsi[inP]]));

         //       if(iNPix>=2){
         //         for(int jnP=inP+1;jnP<iNPix;jnP++){
         //           if(inP!=jnP){
         //              double dEta = fabs(ohOniaPixelEta[L3PixelCandIDForJPsi[inP]]-ohOniaPixelEta[L3PixelCandIDForJPsi[jnP]]);
         //              double dPhi = fabs(ohOniaPixelPhi[L3PixelCandIDForJPsi[inP]]-ohOniaPixelPhi[L3PixelCandIDForJPsi[jnP]]);
         //              if(dPhi>TMath::Pi()) dPhi = 2.0*TMath::Pi()-dPhi;
         //              hPixCanddr[histIndex]->Fill(sqrt(pow(dEta,2)+pow(dPhi,2)));
         //           }
         //         }
         //       }
      }
   }

   return iNPix;
}

int OHltTree::OpenHltMuTrackPassed_JPsi(
      double ptTrack,
      double pTrack,
      double etaTrack,
      double DxyTrack,
      double DzTrack,
      int NHitsTrack,
      double normChi2Track,
      double *massMinTrack,
      double *massMaxTrack,
      double DzMuonTrack,
      bool checkChargeTrack,
      int histIndex)
{

   double pixMatchingDeltaR = 0.01;
   const double muMass = 0.105658367;
   TLorentzVector track4Mom, mu4Mom, onia4Mom;
   int iNTrack = 0;

   //0.) check how many pixel-muon combinations there are:
   int nComb = 0;
   for (int iMu = 0; iMu < 10; iMu++)
      if (L3MuPixCandIDForJPsi[iMu] > -1)
         nComb++;

   //   printf("OpenHltMuTrackPassed_JPsi: %d incoming pixels and %d tracks\n", nComb, NohOniaTrack);
   //   Int_t countCut = 0, countOniaCut = 0;
   //1.) loop over the Tracker tracks
   for (int iT = 0; iT < NohOniaTrack; iT++)
   {

      //select those that survive the kinematical and
      //topological selection cuts
      //     countCut = 0;
      //     hEta[histIndex][1][countCut]->Fill(ohOniaTrackEta[iT]);
      //     hPt[histIndex][1][countCut]->Fill(ohOniaTrackPt[iT]);
      //     hHits[histIndex][1][countCut]->Fill(ohOniaTrackHits[iT]);
      //     hNormChi2[histIndex][1][countCut]->Fill(ohOniaTrackNormChi2[iT]);
      //     hDxy[histIndex][1][countCut]->Fill(ohOniaTrackDr[iT]);
      //     hDz[histIndex][1][countCut]->Fill(ohOniaTrackDz[iT]);

      if (fabs(ohOniaTrackEta[iT]) > etaTrack)
         continue; //eta cut
      if (ohOniaTrackPt[iT] < ptTrack)
         continue; //pT cut
      double momThisTrack = ohOniaTrackPt[iT] * cosh(ohOniaTrackEta[iT]);
      //     printf("track[%d] has eta %f, pT %f and mom %f\n",
      //         iT, ohOniaTrackEta[iT], ohOniaTrackPt[iT], momThisTrack);
      //     hP[histIndex][1][countCut]->Fill(momThisTrack);
      //     countCut++;

      if (momThisTrack < pTrack)
         continue; //momentum cut
      if (ohOniaTrackHits[iT] < NHitsTrack)
         continue; //min. nb. of hits
      if (ohOniaTrackNormChi2[iT] > normChi2Track)
         continue; //chi2 cut
      if (fabs(ohOniaTrackDr[iT]) > DxyTrack)
         continue; //Dr cut
      if (fabs(ohOniaTrackDz[iT]) > DzTrack)
         continue;

      //     hEta[histIndex][1][countCut]->Fill(ohOniaTrackEta[iT]);
      //     hPt[histIndex][1][countCut]->Fill(ohOniaTrackPt[iT]);
      //     hHits[histIndex][1][countCut]->Fill(ohOniaTrackHits[iT]);
      //     hNormChi2[histIndex][1][countCut]->Fill(ohOniaTrackNormChi2[iT]);
      //     hDxy[histIndex][1][countCut]->Fill(ohOniaTrackDr[iT]);
      //     hDz[histIndex][1][countCut]->Fill(ohOniaTrackDz[iT]);
      //     hP[histIndex][1][countCut]->Fill(momThisTrack);
      //     countCut++;

      //     printf("track %d surviving kinematical pre-selection\n", iT);
      //2.) loop over the pixels candidates to see whether the track
      //under investigation has a match to the pixel track
      bool trackMatched = false;
      for (int iPix = 0; iPix < nComb; iPix++)
      {
         if (trackMatched)
            break; //in case the track was already matched
         if (L3PixelCandIDForJPsi[iPix] < 0)
            continue; //in case the pixel has been matched to a previous track

         double deltaEta = ohOniaPixelEta[L3PixelCandIDForJPsi[iPix]]
               - ohOniaTrackEta[iT];
         double deltaPhi = ohOniaPixelPhi[L3PixelCandIDForJPsi[iPix]]
               - ohOniaTrackPhi[iT];
         if (deltaPhi>TMath::Pi())
            deltaPhi = 2.0*TMath::Pi()-deltaPhi;
         double deltaR = sqrt(pow(deltaEta, 2) + pow(deltaPhi, 2));

         //       printf("delta R = %f\n", deltaR);
         if (deltaR > pixMatchingDeltaR)
            continue;
         //       printf("track[%d] and pixel[%d] are compatible (deltaR %f)\n", iT, L3PixelCandIDForJPsi[iPix], deltaR);

         trackMatched = true;
         L3PixelCandIDForJPsi[iPix] = -1; //deactivate this candidate to not match it to any further track

         track4Mom.SetPtEtaPhiM(
               ohOniaTrackPt[iT],
               ohOniaTrackEta[iT],
               ohOniaTrackPhi[iT],
               muMass);
         //check if the matched tracker track combined with the
         //muon gives again an opposite sign onia:
         mu4Mom.SetPtEtaPhiM(
               ohMuL3Pt[L3MuPixCandIDForJPsi[iPix]],
               ohMuL3Eta[L3MuPixCandIDForJPsi[iPix]],
               ohMuL3Phi[L3MuPixCandIDForJPsi[iPix]],
               muMass);
         onia4Mom = track4Mom + mu4Mom;

         double oniaMass = onia4Mom.M();
         //       printf("mu[%d]-track[%d] inv. mass %f\n",
         //           L3MuPixCandIDForJPsi[iPix], iT, oniaMass);

         //       countOniaCut = 0;
         //       if(oniaMass>5.0) continue; //Only JPsi
         //       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if (oniaMass < massMinTrack[0] || oniaMass> massMaxTrack[0]) continue; //mass cut
         //       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if(checkChargeTrack)
         if(ohMuL3Chg[L3MuPixCandIDForJPsi[iPix]] == ohOniaTrackChg[iT]) continue; //charge cut

         //       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if(fabs(ohMuL3Dz[L3MuPixCandIDForJPsi[iPix]] - ohOniaTrackDz[iT])> DzMuonTrack) continue; //deltaZ cut
         //       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));

         //store the surviving track-muon combinations:
         if(iNTrack < 10)
         iNTrack++;
         break; //don't check further pixels... go to next track
      }
   }

   //   hNCand[histIndex][1]->Fill(iNTrack);
   return iNTrack;
}

int OHltTree::OpenHltMuPixelPassed_Ups(
      double ptPix,
      double pPix,
      double etaPix,
      double DxyPix,
      double DzPix,
      int NHitsPix,
      double normChi2Pix,
      double *massMinPix,
      double *massMaxPix,
      double DzMuonPix,
      bool checkChargePix,
      int histIndex)
{

   const double muMass = 0.105658367;
   TLorentzVector pix4Mom, mu4Mom, onia4Mom;
   int iNPix = 0;
   //reset counter variables:
   for (int iMu = 0; iMu < 10; iMu++)
   {
      L3PixelCandIDForUps[iMu] = -1;
      L3MuPixCandIDForUps[iMu] = -1;
   }

   //0.) check how many L3 muons there are:
   int nMuons = 0;
   for (int iMu = 0; iMu < 10; iMu++)
      if (L3MuCandIDForOnia[iMu] > -1)
         nMuons++;

   //   Int_t countCut = 0, countOniaCut = 0;
   //1.) loop over the Pixel tracks
   for (int iP = 0; iP < NohOniaPixel; iP++)
   {

      //     countCut = 0;
      //     hEta[histIndex][0][countCut]->Fill(ohOniaPixelEta[iP]);
      //     hPt[histIndex][0][countCut]->Fill(ohOniaPixelPt[iP]);
      //     hHits[histIndex][0][countCut]->Fill(ohOniaPixelHits[iP]);
      //     hNormChi2[histIndex][0][countCut]->Fill(ohOniaPixelNormChi2[iP]);
      //     hDxy[histIndex][0][countCut]->Fill(ohOniaPixelDr[iP]);
      //     hDz[histIndex][0][countCut]->Fill(ohOniaPixelDz[iP]);

      //select those that survive the kinematical and
      //topological selection cuts
      if (fabs(ohOniaPixelEta[iP]) > etaPix)
         continue; //eta cut
      if (ohOniaPixelPt[iP] < ptPix)
         continue; //pT cut
      double momThisPix = ohOniaPixelPt[iP] * cosh(ohOniaPixelEta[iP]);

      //     hP[histIndex][0][countCut]->Fill(momThisPix);
      //     countCut++;

      if (momThisPix < pPix)
         continue; //momentum cut
      if (ohOniaPixelHits[iP] < NHitsPix)
         continue; //min. nb. of hits
      if (ohOniaPixelNormChi2[iP] > normChi2Pix)
         continue; //chi2 cut
      if (fabs(ohOniaPixelDr[iP]) > DxyPix)
         continue; //Dr cut
      if (fabs(ohOniaPixelDz[iP]) > DzPix)
         continue;

      //     hEta[histIndex][0][countCut]->Fill(ohOniaPixelEta[iP]);
      //     hPt[histIndex][0][countCut]->Fill(ohOniaPixelPt[iP]);
      //     hHits[histIndex][0][countCut]->Fill(ohOniaPixelHits[iP]);
      //     hNormChi2[histIndex][0][countCut]->Fill(ohOniaPixelNormChi2[iP]);
      //     hDxy[histIndex][0][countCut]->Fill(ohOniaPixelDr[iP]);
      //     hDz[histIndex][0][countCut]->Fill(ohOniaPixelDz[iP]);
      //     hP[histIndex][0][countCut]->Fill(momThisPix);
      //     countCut++;

      pix4Mom.SetPtEtaPhiM(
            ohOniaPixelPt[iP],
            ohOniaPixelEta[iP],
            ohOniaPixelPhi[iP],
            muMass);
      //2.) loop now over all L3 muons and check if they would give a
      //Ups pair:
      for (int iMu = 0; iMu < nMuons; iMu++)
      {
         mu4Mom.SetPtEtaPhiM(
               ohMuL3Pt[L3MuCandIDForOnia[iMu]],
               ohMuL3Eta[L3MuCandIDForOnia[iMu]],
               ohMuL3Phi[L3MuCandIDForOnia[iMu]],
               muMass);
         onia4Mom = pix4Mom + mu4Mom;
         double oniaMass = onia4Mom.M();

         //       countOniaCut = 0;
         //       if(oniaMass < 8.0) continue; //Only Ups
         //       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if (oniaMass < massMinPix[0] || oniaMass> massMaxPix[0]) continue; //mass cut

         //       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if(checkChargePix)
         if(ohMuL3Chg[iMu] == ohOniaPixelChg[iP]) continue; //charge cut

         //       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if(fabs(ohMuL3Dz[iMu] - ohOniaPixelDz[iP])> DzMuonPix) continue;
         //       hOniaEta[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][0][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][0][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][0][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][0][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));

         //store the surviving pixel-muon combinations:
         if(iNPix < 10)
         {
            L3PixelCandIDForUps[iNPix] = iP;
            L3MuPixCandIDForUps[iNPix] = iMu;
            iNPix++;
         }
         //       printf("mu[%d]-pixel[%d] inv. mass %f\n",
         //           L3MuCandIDForOnia[iMu], iP, oniaMass);
      }
   }

   //   hNCand[histIndex][0]->Fill(iNPix);

   if(iNPix!=0)
   {
      for(int inP=0;inP<iNPix;inP++)
      {

         //       hPixCandEta[histIndex]->Fill(ohOniaPixelEta[L3PixelCandIDForUps[inP]]);
         //       hPixCandPt[histIndex]->Fill(ohOniaPixelPt[L3PixelCandIDForUps[inP]]);
         //       hPixCandP[histIndex]->Fill(ohOniaPixelPt[L3PixelCandIDForUps[inP]] * cosh(ohOniaPixelEta[L3PixelCandIDForUps[inP]]));

         //       if(iNPix>=2){
         //         for(int jnP=inP+1;jnP<iNPix;jnP++){
         //            if(inP!=jnP){
         //              double dEta = fabs(ohOniaPixelEta[L3PixelCandIDForUps[inP]]-ohOniaPixelEta[L3PixelCandIDForUps[jnP]]);
         //              double dPhi = fabs(ohOniaPixelPhi[L3PixelCandIDForUps[inP]]-ohOniaPixelPhi[L3PixelCandIDForUps[jnP]]);
         //              if(dPhi>TMath::Pi()) dPhi = 2.0*TMath::Pi()-dPhi;
         // //              hPixCanddr[histIndex]->Fill(sqrt(pow(dEta,2)+pow(dPhi,2)));
         //            }
         //         }
         //       }
      }
   }

   return iNPix;
}

int OHltTree::OpenHltMuTrackPassed_Ups(
      double ptTrack,
      double pTrack,
      double etaTrack,
      double DxyTrack,
      double DzTrack,
      int NHitsTrack,
      double normChi2Track,
      double *massMinTrack,
      double *massMaxTrack,
      double DzMuonTrack,
      bool checkChargeTrack,
      int histIndex)
{

   double pixMatchingDeltaR = 0.01;
   const double muMass = 0.105658367;
   TLorentzVector track4Mom, mu4Mom, onia4Mom;
   int iNTrack = 0;

   //0.) check how many pixel-muon combinations there are:
   int nComb = 0;
   for (int iMu = 0; iMu < 10; iMu++)
      if (L3MuPixCandIDForUps[iMu] > -1)
         nComb++;
   //   Int_t countCut = 0, countOniaCut = 0;
   //1.) loop over the Tracker tracks
   for (int iT = 0; iT < NohOniaTrack; iT++)
   {

      //select those that survive the kinematical and
      //topological selection cuts
      //     countCut++;
      //     hEta[histIndex][1][countCut]->Fill(ohOniaTrackEta[iT]);
      //     hPt[histIndex][1][countCut]->Fill(ohOniaTrackPt[iT]);
      //     hHits[histIndex][1][countCut]->Fill(ohOniaTrackHits[iT]);
      //     hNormChi2[histIndex][1][countCut]->Fill(ohOniaTrackNormChi2[iT]);
      //     hDxy[histIndex][1][countCut]->Fill(ohOniaTrackDr[iT]);
      //     hDz[histIndex][1][countCut]->Fill(ohOniaTrackDz[iT]);

      if (fabs(ohOniaTrackEta[iT]) > etaTrack)
         continue; //eta cut
      if (ohOniaTrackPt[iT] < ptTrack)
         continue; //pT cut
      double momThisTrack = ohOniaTrackPt[iT] * cosh(ohOniaTrackEta[iT]);
      //     printf("track[%d] has eta %f, pT %f and mom %f\n",
      //         iT, ohOniaTrackEta[iT], ohOniaTrackPt[iT], momThisTrack);

      //     hP[histIndex][1][countCut]->Fill(momThisTrack);
      //     countCut++;

      if (momThisTrack < pTrack)
         continue; //momentum cut
      if (ohOniaTrackHits[iT] < NHitsTrack)
         continue; //min. nb. of hits
      if (ohOniaTrackNormChi2[iT] > normChi2Track)
         continue; //chi2 cut
      if (fabs(ohOniaTrackDr[iT]) > DxyTrack)
         continue; //Dr cut
      if (fabs(ohOniaTrackDz[iT]) > DzTrack)
         continue;

      //     hEta[histIndex][1][countCut]->Fill(ohOniaTrackEta[iT]);
      //     hPt[histIndex][1][countCut]->Fill(ohOniaTrackPt[iT]);
      //     hHits[histIndex][1][countCut]->Fill(ohOniaTrackHits[iT]);
      //     hNormChi2[histIndex][1][countCut]->Fill(ohOniaTrackNormChi2[iT]);
      //     hDxy[histIndex][1][countCut]->Fill(ohOniaTrackDr[iT]);
      //     hDz[histIndex][1][countCut]->Fill(ohOniaTrackDz[iT]);
      //     hP[histIndex][1][countCut]->Fill(momThisTrack);

      //     printf("track %d surviving kinematical pre-selection\n", iT);
      //2.) loop over the pixels candidates to see whether the track
      //under investigation has a match to the pixel track
      bool trackMatched = false;
      for (int iPix = 0; iPix < nComb; iPix++)
      {

         if (trackMatched)
            break; //in case the track was already matched
         if (L3PixelCandIDForUps[iPix] < 0)
            continue; //in case the pixel has been matched to a previous track

         double deltaEta = ohOniaPixelEta[L3PixelCandIDForUps[iPix]]
               - ohOniaTrackEta[iT];
         double deltaPhi = ohOniaPixelPhi[L3PixelCandIDForUps[iPix]]
               - ohOniaTrackPhi[iT];
         if (deltaPhi>TMath::Pi())
            deltaPhi = 2.0*TMath::Pi()-deltaPhi;
         double deltaR = sqrt(pow(deltaEta, 2) + pow(deltaPhi, 2));

         //       printf("delta R = %f\n", deltaR);
         if (deltaR > pixMatchingDeltaR)
            continue;
         //       printf("track[%d] and pixel[%d] are compatible (deltaR %f)\n", iT, L3PixelCandIDForUps[iPix], deltaR);

         trackMatched = true;
         L3PixelCandIDForUps[iPix] = -1; //deactivate this candidate to not match it to any further track

         track4Mom.SetPtEtaPhiM(
               ohOniaTrackPt[iT],
               ohOniaTrackEta[iT],
               ohOniaTrackPhi[iT],
               muMass);
         //check if the matched tracker track combined with the
         //muon gives again an opposite sign onia:
         mu4Mom.SetPtEtaPhiM(
               ohMuL3Pt[L3MuPixCandIDForUps[iPix]],
               ohMuL3Eta[L3MuPixCandIDForUps[iPix]],
               ohMuL3Phi[L3MuPixCandIDForUps[iPix]],
               muMass);
         onia4Mom = track4Mom + mu4Mom;

         double oniaMass = onia4Mom.M();
         //       printf("mu[%d]-track[%d] inv. mass %f\n",
         //           L3MuPixCandIDForUps[iPix], iT, oniaMass);

         //       countOniaCut = 0;
         //       if(oniaMass < 8.0) continue; //Only Ups
         //       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if (oniaMass < massMinTrack[0] || oniaMass> massMaxTrack[0]) continue; //mass cut

         //       printf("surviving: mu[%d]-track[%d] inv. mass %f\n",
         //           L3MuPixCandIDForUps[iPix], iT, oniaMass);

         //       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if(checkChargeTrack)
         if(ohMuL3Chg[L3MuPixCandIDForUps[iPix]] == ohOniaTrackChg[iT]) continue; //charge cut

         //       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       countOniaCut++;

         if(fabs(ohMuL3Dz[L3MuPixCandIDForUps[iPix]] - ohOniaTrackDz[iT])> DzMuonTrack) continue; //deltaZ cut

         //       hOniaEta[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta());
         //       hOniaRap[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity());
         //       hOniaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Pt());
         //       hOniaMass[histIndex][1][countOniaCut]->Fill(onia4Mom.M());
         //       hOniaP[histIndex][1][countOniaCut]->Fill(sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));
         //       hOniaEtaPt[histIndex][1][countOniaCut]->Fill(onia4Mom.Eta(),onia4Mom.Pt());
         //       hOniaRapP[histIndex][1][countOniaCut]->Fill(onia4Mom.Rapidity(),sqrt(pow(onia4Mom.Px(),2)+pow(onia4Mom.Py(),2)+pow(onia4Mom.Pz(),2)));

         //store the surviving track-muon combinations:
         if(iNTrack < 10)
         iNTrack++;

         break; //don't check further pixels... go to next track
      }
   }

   //   if(iNTrack > 0)
   //     printf("found %d final candidates!!!\n", iNTrack);

   //   hNCand[histIndex][1]->Fill(iNTrack);
   return iNTrack;
}

bool OHltTree::OpenJetID(int jetindex)
{
	if (jetindex>=NrecoJetCorCal) return false;
	bool jetID = true ; //jetID is true by default
	if (fabs(recoJetCorCalEta[jetindex])< 2.6) {//jetID might be changed to false only for central jets : jetID is a cut only meant for central jets
       jetID =  (recoJetCorCalEMF[jetindex] > 1.0E-6) && (recoJetCorCalEMF[jetindex] < 999.0) && recoJetCorCalN90[jetindex]>=2;
     }
	return jetID;

}


int OHltTree::OpenHlt1JetPassed(double pt)
{
   int rc = 0;

   // Loop over all oh jets 
   for (int i=0; i<NrecoJetCal; i++)
   {
      if (recoJetCalPt[i]>pt)
      { // Jet pT cut
         rc++;
      }
   }

   return rc;
}

int OHltTree::OpenHlt1JetPassed(double pt, double etamax)
{
   int rc = 0;
   //ccla
   // Loop over all oh jets 
   for (int i=0; i<NrecoJetCal; i++)
   {
      if (recoJetCalPt[i]>pt && fabs(recoJetCalEta[i])<etamax)
      { // Jet pT cut
         rc++;
      }
   }

   return rc;
}

int OHltTree::OpenHlt1JetPassed(
      double pt,
      double etamax,
      double emfmin,
      double emfmax)
{
   int rc = 0;
   //ccla
   // Loop over all oh jets 
   for (int i=0; i<NrecoJetCal; i++)
   {
      if (recoJetCalPt[i]>pt && fabs(recoJetCalEta[i])<etamax
            && recoJetCalEMF[i] > emfmin && recoJetCalEMF[i] < emfmax)
      { // Jet pT cut
         rc++;
      }
   }

   return rc;
}

int OHltTree::OpenHlt1CorJetPassedNoJetID(double pt)
{
   int rc = 0;

   // Loop over all oh corrected jets
   for (int i=0; i<NrecoJetCorCal; i++)
   {
     if ( recoJetCorCalPt[i]>pt)
      { // Jet pT cut
         rc++;
      }
   }

   return rc;
}

int OHltTree::OpenHlt1CorJetPassed(double pt)
{
   int rc = 0;

   // Loop over all oh corrected jets
   for (int i=0; i<NrecoJetCorCal; i++)
   {
     if (OpenJetID(i) && recoJetCorCalPt[i]>pt)
      { // Jet pT cut
         rc++;
      }
    }

   return rc;
}


int OHltTree::OpenHlt1CorJetPassed(double pt, double etamax)
{
  int rc = 0;

  // Loop over all oh corrected jets
  for (int i=0; i<NrecoJetCorCal; i++)
    {
      if (OpenJetID(i) && recoJetCorCalPt[i]>pt && fabs(recoJetCorCalEta[i])<etamax)
	  { // Jet pT cut
	    rc++;
	  }
    }
  return rc;
}

int OHltTree::OpenHltDiJetAvePassed(double pt)
{
  int rc = 0;

  // Loop over all oh jets, select events where the *average* pT of a pair is above threshold
  //std::cout << "FL: NrecoJetCal = " << NrecoJetCal << std::endl;
  for (int i=0; i<NrecoJetCal; i++)
    {
      bool jetID0 = true ; //jetID is true by default
      if (fabs(recoJetCalEta[i])< 2.6) {//jetID might be changed to false only for central jets : jetID is a cut only meant for central jets
	jetID0 =  (recoJetCalEMF[i] > 1.0E-6) && (recoJetCalEMF[i] < 999.0) && recoJetCalN90[i]>=2;
      }
      if (!jetID0) continue;
      for (int j=0; j<NrecoJetCal && j!=i; j++)
	{
	  bool jetID1 = true ; //jetID is true by default
	  if (fabs(recoJetCalEta[i])< 2.6) {//jetID might be changed to false only for central jets : jetID is a cut only meant for central jets
	    jetID1 =  (recoJetCalEMF[i] > 1.0E-6) && (recoJetCalEMF[i] < 999.0) && recoJetCalN90[i]>=2;
	  }
	  if (jetID1 && (recoJetCalPt[i]+recoJetCalPt[j])/2.0 > pt)
	    { // Jet pT cut 
	      //      if((recoJetCalE[i]/cosh(recoJetCalEta[i])+recoJetCalE[j]/cosh(recoJetCalEta[j]))/2.0 > pt) {
	      rc++;
	    }
	}
    }
  return rc;
}


int OHltTree::OpenHltCorDiJetAvePassed(double pt)
{
  int rc = 0;

  // Loop over all oh jets, select events where the *average* pT of a pair is above threshold
  //std::cout << "FL: NrecoJetCal = " << NrecoJetCal << std::endl;
  for (int i=0; i<NrecoJetCorCal; i++)
    {
      bool jetID0 = true ; //jetID is true by default
      if (fabs(recoJetCorCalEta[i])< 2.6) {//jetID might be changed to false only for central jets : jetID is a cut only meant for central jets
	jetID0 =  (recoJetCorCalEMF[i] > 1.0E-6) && (recoJetCorCalEMF[i] < 999.0) && recoJetCorCalN90[i]>=2;
      }
      if (!jetID0) continue;
      for (int j=0; j<NrecoJetCal && j!=i; j++)
	{
	  bool jetID1 = true ; //jetID is true by default
	  if (fabs(recoJetCorCalEta[i])< 2.6) {//jetID might be changed to false only for central jets : jetID is a cut only meant for central jets
	    jetID1 =  (recoJetCorCalEMF[i] > 1.0E-6) && (recoJetCorCalEMF[i] < 999.0) && recoJetCorCalN90[i]>=2;
	  }
	  if (jetID1 && (recoJetCorCalPt[i]+recoJetCorCalPt[j])/2.0 > pt)
	    { // Jet pT cut 
	      //      if((recoJetCalE[i]/cosh(recoJetCalEta[i])+recoJetCalE[j]/cosh(recoJetCalEta[j]))/2.0 > pt) {
	      rc++;
	    }
	}
    }
  return rc;
}


int OHltTree::OpenHltQuadCorJetPassed(double pt)
{
   int njet = 0;
   int rc = 0;

   // Loop over all oh jets
   for (int i=0; i<NrecoJetCorCal; i++)
   {
     if (recoJetCorCalPt[i] > pt && fabs(recoJetCorCalEta[i]) < 5.0)
      { // Jet pT cut
         //std::cout << "FL: fires the jet pt cut" << std::endl;
         njet++;
      }
   }

   if (njet >= 4)
   {
      rc = 1;
   }

   return rc;
}

int OHltTree::OpenHltQuadJetPassed(double pt)
{
   int njet = 0;
   int rc = 0;

   // Loop over all oh jets
   for (int i=0; i<NrecoJetCal; i++)
   {
     if (recoJetCalPt[i] > pt && fabs(recoJetCalEta[i]) < 5.0)
      { // Jet pT cut
         njet++;
      }
   }

   if (njet >= 4)
      rc = 1;

   return rc;
}

int OHltTree::OpenHltQuadJetPassedPlusTauPFId(
      double pt,
      double etaJet,
      double ptTau)
{
   int njet = 0;
   int rc = 0;
   bool foundPFTau = false;
   for (int i=0; i<NrecoJetCorCal; i++)
   {
      if (recoJetCorCalPt[i] > pt && fabs(recoJetCorCalEta[i]) < etaJet)
      { // Jet pT cut 
         njet++;
         for (int j=0; j<NohpfTau; j++)
         {

            if (ohpfTauPt[j] > ptTau && ohpfTauLeadTrackPt[j]>= 5
                  && fabs(ohpfTauEta[j]) <2.5 && ohpfTauTrkIso[j] <1
                  && ohpfTauGammaIso[j] <1)
            {

               float deltaEta = ohpfTauEta[j] - recoJetCorCalEta[i];
               float deltaPhi = ohpfTauPhi[j] - recoJetCorCalPhi[i];

               if (fabs(deltaPhi)>3.141592654)
                  deltaPhi = 6.283185308-fabs(deltaPhi);

               float deltaR = sqrt(pow(deltaEta, 2) + pow(deltaPhi, 2));

               if (deltaR<0.3)
               {
                  foundPFTau = true;
               }
            }
         }
      }
   }
   if (njet >= 4 && foundPFTau == true)
      rc = 1;
   return rc;
}

int OHltTree::OpenL1QuadJet8(double jetPt, double jetEta)
{
   int rc = 0;

   for (int i=0; i<NL1CenJet; i++)
      if (L1CenJetEt[i] >= jetPt && fabs(L1CenJetEta[i])<jetEta)
         rc++;
   for (int i=0; i<NL1ForJet; i++)
      if (L1ForJetEt[i] >= jetPt && fabs(L1ForJetEta[i])<jetEta)
         rc++;
   for (int i=0; i<NL1Tau; i++)
      if (L1TauEt [i] >= jetPt && fabs(L1TauEta[i])<jetEta)
         rc++;

   return rc;
}

int OHltTree::OpenHltFwdCorJetPassed(double esum)
{
   int rc = 0;
   double gap = 0.;

   // Loop over all oh jets, count the sum of energy deposited in HF 
   for (int i=0; i<NrecoJetCorCal; i++)
   {
      if (((recoJetCorCalEta[i] > 3.0 && recoJetCorCalEta[i] < 5.0)
            || (recoJetCorCalEta[i] < -3.0 && recoJetCorCalEta[i] > -5.0)))
      {
         gap+=recoJetCorCalE[i];
      }
   }

   // Backward FWD physics logic - we want to select the events *without* large jet energy in HF 
   if (gap < esum)
      rc = 1;
   else
      rc = 0;

   return rc;
}

int OHltTree::OpenHltFwdJetPassed(double esum)
{
   int rc = 0;
   double gap = 0.;

   // Loop over all oh jets, count the sum of energy deposited in HF 
   for (int i=0; i<NrecoJetCal; i++)
   {
      if (((recoJetCalEta[i] > 3.0 && recoJetCalEta[i] < 5.0)
            || (recoJetCalEta[i] < -3.0 && recoJetCalEta[i] > -5.0)))
      {
         gap+=recoJetCalE[i];
      }
   }

   // Backward FWD physics logic - we want to select the events *without* large jet energy in HF 
   if (gap < esum)
      rc = 1;
   else
      rc = 0;

   return rc;
}

int OHltTree::OpenHltHTJetNJPassed(
      double HTthreshold,
      double jetthreshold,
      double etamax,
      int nj)
{
   int rc = 0, njets=0;
   double sumHT = 0.;

   // Loop over all oh jets, sum up the energy  
   for (int i=0; i<NrecoJetCal; ++i)
   {
      if (recoJetCalPt[i] >= jetthreshold && fabs(recoJetCalEta[i])<etamax)
      {
         //sumHT+=recoJetCalPt[i];
         njets++;
         sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
      }
   }

   if (sumHT >= HTthreshold && njets>=nj)
      rc = 1;

   return rc;
}

int OHltTree::OpenHltMHT(double MHTthreshold, double jetthreshold, double etathreshold)
{
   int rc = 0;
   double mhtx=0., mhty=0.;
   for (int i=0; i<NrecoJetCorCal; ++i)
   {
      if (OpenJetID(i) && recoJetCorCalPt[i] >= jetthreshold && fabs(recoJetCorCalEta[i]) < etathreshold)
      {
         mhtx-=recoJetCorCalPt[i]*cos(recoJetCorCalPhi[i]);
         mhty-=recoJetCorCalPt[i]*sin(recoJetCorCalPhi[i]);
      }
   }
   if (sqrt(mhtx*mhtx+mhty*mhty)>MHTthreshold)
      rc = 1;
   else
      rc = 0;
   //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
   return rc;
}
int OHltTree::OpenHltMHTU(double MHTthreshold, double jetthreshold, double etathreshold)
{
   int rc = 0;
   double mhtx=0., mhty=0.;
   for (int i=0; i<NrecoJetCal; ++i)
   {
      if (recoJetCalPt[i] >= jetthreshold && fabs(recoJetCalEta[i]) < etathreshold)
      {
         mhtx-=recoJetCalPt[i]*cos(recoJetCalPhi[i]);
         mhty-=recoJetCalPt[i]*sin(recoJetCalPhi[i]);
      }
   }
   if (sqrt(mhtx*mhtx+mhty*mhty)>MHTthreshold)
      rc = 1;
   else
      rc = 0;
   //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
   return rc;
}

int OHltTree::OpenHltPT12U(double PT12threshold, double jetthreshold)
{
   int rc = 0;
   int njets = 0;
   double pt12tx=0., pt12ty=0.;
   for (int i=0; i<NrecoJetCal; ++i)
   {
      if ((recoJetCalPt[i] >= jetthreshold) && (fabs(recoJetCalEta[i]) <3))
      {
         njets++;
         if (njets<3)
         {
            pt12tx-=recoJetCalPt[i]*cos(recoJetCalPhi[i]);
            pt12ty-=recoJetCalPt[i]*sin(recoJetCalPhi[i]);
         }
      }
   }
   if ((njets >= 2) && (sqrt(pt12tx*pt12tx+pt12ty*pt12ty)>PT12threshold))
      rc = 1;
   else
      rc = 0;
   //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
   return rc;
}

int OHltTree::OpenHltPT12(double PT12threshold, double jetthreshold)
{
   int rc = 0;
   int njets = 0;
   double pt12tx=0., pt12ty=0.;
   for (int i=0; i<NrecoJetCorCal; ++i)
   {
      if ((recoJetCorCalPt[i] >= jetthreshold)
            && (fabs(recoJetCorCalEta[i]) <3))
      {
         njets++;
         if (njets<3)
         {
            pt12tx-=recoJetCorCalPt[i]*cos(recoJetCorCalPhi[i]);
            pt12ty-=recoJetCorCalPt[i]*sin(recoJetCorCalPhi[i]);
         }
      }
   }
   if ((njets >= 2) && (sqrt(pt12tx*pt12tx+pt12ty*pt12ty)>PT12threshold))
      rc = 1;
   else
      rc = 0;
   //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
   return rc;
}

int OHltTree::OpenHltSumHTPassed(double sumHTthreshold, double jetthreshold)
{
   int rc = 0;
   double sumHT = 0.;

   // Loop over all oh jets, sum up the energy  
   for (int i=0; i<NrecoJetCal; ++i)
   {
      if (recoJetCalPt[i] >= jetthreshold)
      {
         //sumHT+=recoJetCorCalPt[i];

         sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
      }
   }

   if (sumHT >= sumHTthreshold)
      rc = 1;

   return rc;
}

int OHltTree::OpenHltSumCorHTPassed(double sumHTthreshold, double jetthreshold, double etathreshold)
{
   int rc = 0;
   double sumHT = 0.;

   // Loop over all oh jets, sum up the energy  
   for (int i=0; i<NrecoJetCorCal; ++i)
   {
		 if (OpenJetID(i) && recoJetCorCalPt[i] >= jetthreshold && fabs(recoJetCorCalEta[i]) < etathreshold)
      {
         //sumHT+=recoJetCorCorCalPt[i];

         sumHT+=(recoJetCorCalE[i]/cosh(recoJetCorCalEta[i]));
      }
   }

   if (sumHT >= sumHTthreshold)
      rc = 1;

   return rc;
}

int OHltTree::OpenHltMeffU(double Meffthreshold, double jetthreshold, double etathreshold)
{
   int rc = 0;
   //MHT
   double mhtx=0., mhty=0.;
   //HT
   double sumHT = 0.;  
   
   for (int i=0; i<NrecoJetCal; ++i)
     {
       if (recoJetCalPt[i] >= jetthreshold && fabs(recoJetCalEta[i]) < etathreshold)
	 {
	   mhtx-=recoJetCalPt[i]*cos(recoJetCalPhi[i]);
	   mhty-=recoJetCalPt[i]*sin(recoJetCalPhi[i]);
	   sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
	 }
     }

   if (sqrt(mhtx*mhtx+mhty*mhty)+sumHT>Meffthreshold)
     rc = 1;
   else
     rc = 0;
   //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
   return rc;
}

int OHltTree::OpenHltMeff(double Meffthreshold, double jetthreshold, double etathreshold)
{
   int rc = 0;
   //MHT
   double mhtx=0., mhty=0.;
   //HT
   double sumHT = 0.;   
   for (int i=0; i<NrecoJetCorCal; ++i)
     {
       if (OpenJetID(i) && recoJetCorCalPt[i] >= jetthreshold && fabs(recoJetCorCalEta[i]) < etathreshold)
	 {
	   mhtx-=recoJetCorCalPt[i]*cos(recoJetCorCalPhi[i]);
	   mhty-=recoJetCorCalPt[i]*sin(recoJetCorCalPhi[i]);
	   sumHT+=recoJetCorCalPt[i];
	 }
     }

   if (sqrt(mhtx*mhtx+mhty*mhty)+sumHT>Meffthreshold)
      rc = 1;
   else
      rc = 0;
   //std::cout << "sqrt(mhtx*mhtx+mhty*mhty) = " << sqrt(mhtx*mhtx+mhty*mhty) << std::endl;
   return rc;
}

int OHltTree::OpenHltSumHTPassed(
      double sumHTthreshold,
      double jetthreshold,
      double etajetthreshold)
{
   int rc = 0;
   double sumHT = 0.;

   // Loop over all oh jets, sum up the energy  
   for (int i=0; i<NrecoJetCal; ++i)
   {
      if (recoJetCalPt[i] >= jetthreshold && fabs(recoJetCalEta[i])
            < etajetthreshold)
      {
         //sumHT+=recoJetCorCalPt[i];

         sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
      }
   }

   if (sumHT >= sumHTthreshold)
      rc = 1;

   return rc;
}

int OHltTree::OpenHltSumHTPassed(
      double sumHTthreshold,
      double jetthreshold,
      double etajetthreshold,
      int Njetthreshold)
{
   int rc = 0;
   double sumHT = 0.;
   int Njet = 0.;

   // Loop over all oh jets, sum up the energy  
   for (int i=0; i<NrecoJetCal; ++i)
   {
      if (recoJetCalPt[i] >= jetthreshold && fabs(recoJetCalEta[i])
            < etajetthreshold)
      {
         //sumHT+=recoJetCorCalPt[i];
         Njet++;

         sumHT+=(recoJetCalE[i]/cosh(recoJetCalEta[i]));
      }
   }

   if (sumHT >= sumHTthreshold && Njet >= Njetthreshold)
      rc = 1;

   return rc;
}

int OHltTree::OpenHlt1PixelTrackPassed(float minpt, float minsep, float miniso)
{
   int rc = 0;

   // Loop over all oh pixel tracks, check threshold and separation
   for (int i = 0; i < NohPixelTracksL3; i++)
   {
      for (int i = 0; i < NohPixelTracksL3; i++)
      {
         if (ohPixelTracksL3Pt[i] > minpt)
         {
            float closestdr = 999.;

            // Calculate separation from other tracks above threshold
            for (int j = 0; j < NohPixelTracksL3 && j != i; j++)
            {
               if (ohPixelTracksL3Pt[j] > minpt)
               {
                  float dphi = ohPixelTracksL3Phi[i]-ohPixelTracksL3Phi[j];
                  float deta = ohPixelTracksL3Eta[i]-ohPixelTracksL3Eta[j];
                  float dr = sqrt((deta*deta) + (dphi*dphi));
                  if (dr < closestdr)
                     closestdr = dr;
               }
            }
            if (closestdr > minsep)
            {
               // Calculate isolation from *all* other tracks without threshold.
               if (miniso > 0)
               {
                  int tracksincone = 0;
                  for (int k = 0; k < NohPixelTracksL3 && k != i; k++)
                  {
                     float dphi = ohPixelTracksL3Phi[i]-ohPixelTracksL3Phi[k];
                     float deta = ohPixelTracksL3Eta[i]-ohPixelTracksL3Eta[k];
                     float dr = sqrt((deta*deta) + (dphi*dphi));
                     if (dr < miniso)
                        tracksincone++;
                  }
                  if (tracksincone == 0)
                     rc++;
               }
               else
                  rc++;
            }
         }
      }
   }

   return rc;
}

int OHltTree::OpenHlt1L3MuonPassed(double pt, double eta)
{
   //for BTagMu trigger 

   int rcL3 = 0;
   // Loop over all oh L3 muons and apply cuts 
   for (int i=0; i<NohMuL3; i++)
   {
      if (ohMuL3Pt[i] > pt && fabs(ohMuL3Eta[i]) < eta)
      { // L3 pT and eta cut  
         rcL3++;
      }
   }

   return rcL3;

}

int OHltTree::OpenHltMhtOverHTPassed(
      double HardJetThreshold,
      double HtJetThreshold,
      double MhtJetThreshold,
      double MHTovHT,
      int NoJets)
{

   int rc = 0;
   double newHT = 0.;
   double mhtx = 0., mhty = 0.;
   int nJets = 0;
   for (int i=0; i<NrecoJetCal; i++)
   {
      if (recoJetCalPt[i]>HtJetThreshold)
      {
         nJets++;
      }
   }
   if ( (recoJetCalE[0]/cosh(recoJetCalEta[0])) > HardJetThreshold
         && (recoJetCalE[1]/cosh(recoJetCalEta[1])) > HardJetThreshold)
   {
      if (nJets > NoJets)
      {
         //loop over NrecoJetCal to calculate a new HT for jets above inputJetPt
         for (int i=0; i<NrecoJetCal; i++)
         {
            if (fabs(recoJetCalEta[i]) > 3.)
            {
               continue;
            }
            if ((recoJetCalE[i]/cosh(recoJetCalEta[i])) > MhtJetThreshold)
            {
               mhtx-=((recoJetCalE[i]/cosh(recoJetCalEta[i]))
                     *cos(recoJetCalPhi[i]));
               mhty-=((recoJetCalE[i]/cosh(recoJetCalEta[i]))
                     *sin(recoJetCalPhi[i]));

            }
            if ( (recoJetCalE[i]/cosh(recoJetCalEta[i])) > HtJetThreshold)
            {
               newHT+=((recoJetCalE[i]/cosh(recoJetCalEta[i])));
            }
         }
      }
      //end calculation of new HT
      if (nJets > 2)
      {
         if (newHT > 0. && sqrt(mhtx*mhtx+mhty*mhty)/newHT > MHTovHT)
         {
            rc++;
         }
      }
      if (nJets == 2)
      {
         if ( (newHT - fabs( (recoJetCalE[0]/cosh(recoJetCalEta[0])
               - (recoJetCalE[1]/cosh(recoJetCalEta[1])))))/(2*sqrt(newHT*newHT
               - (mhtx*mhtx+mhty*mhty))) > sqrt(1.
               / (4.*(1.-(MHTovHT*MHTovHT)))))
         {
            rc++;
         }
      }

   }

   // std::cout << "MHT/HT from Jets is " << sqrt(mhtx*mhtx+mhty*mhty)  /  newHT << " HT is: " << newHT << " MHT is: "<<
   // sqrt(mhtx*mhtx+mhty*mhty) << " The Jet threshold is: " << HtJetThreshold << std::endl;
   return rc;
}

int OHltTree::OpenHltMhtOverHTPassedHTthresh(double HT, double MHTovHT)
{

   int rc = 0;
   double newHT = 0.;
   double mhtx = 0., mhty = 0.;
   int nJets = 0;
   for (int i=0; i<NrecoJetCal; i++)
   {
      if ((recoJetCalE[i]/cosh(recoJetCalEta[i]))>30.)
      {
         nJets++;
      }
   }
   //loop over NrecoJetCal to calculate a new HT for jets above inputJetPt
   for (int i=0; i<NrecoJetCal; i++)
   {
      if ((recoJetCalE[i]/cosh(recoJetCalEta[i])) > 20.)
      {
         mhtx-=((recoJetCalE[i]/cosh(recoJetCalEta[i]))*cos(recoJetCalPhi[i]));
         mhty-=((recoJetCalE[i]/cosh(recoJetCalEta[i]))*sin(recoJetCalPhi[i]));
      }
      if ((recoJetCalE[i]/cosh(recoJetCalEta[i])) > 30.)
      {
         newHT+=((recoJetCalE[i]/cosh(recoJetCalEta[i])));
      }
   }
   // }//end calculation of new HT
   if (nJets > 2)
   {
      if (newHT > 0. && sqrt(mhtx*mhtx+mhty*mhty)/newHT > MHTovHT)
      {
         rc++;
      }
   }
   if (nJets == 2)
   {
      if ( (newHT - fabs( (recoJetCalE[0]/cosh(recoJetCalEta[0])
            - (recoJetCalE[1]/cosh(recoJetCalEta[1])))))/(2*sqrt(newHT*newHT
            - (mhtx*mhtx+mhty*mhty))) > sqrt(1. / (4.*(1.-(MHTovHT*MHTovHT)))))
      {
         rc++;
      }
   }

   return rc;
}

vector<int> OHltTree::VectorOpenHlt1PhotonPassed(
      float Et,
      int L1iso,
      float Tiso,
      float Eiso,
      float HisoBR,
      float HisoEC,
      float HoverEEB,
      float HoverEEC,
      float R9,
      float ClusShapEB,
      float ClusShapEC)
{
   vector<int> rc;
   // Loop over all oh photons 
   for (int i=0; i<NohPhot; i++)
   {
      if (ohPhotEt[i] > Et)
      {
         if (TMath::Abs(ohPhotEta[i]) < 2.65)
         {
            if (ohPhotL1iso[i] >= L1iso)
            {
               if (ohPhotTiso[i] < Tiso + 0.001*ohPhotEt[i])
               {
                  if (ohPhotEiso[i] < Eiso + 0.006*ohPhotEt[i])
                  {
                     if ( (TMath::Abs(ohPhotEta[i]) < 1.479
                           && ohPhotHiso[i] < HisoBR + 0.0025*ohPhotEt[i] 
                           && ohEleClusShap[i] < ClusShapEB
                           && ohPhotR9[i] < R9)
                           ||
                           (1.479 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.65 
                           && ohPhotHiso[i] < HisoEC + 0.0025*ohPhotEt[i]
                           && ohEleClusShap[i] < ClusShapEC))
                     {
                        float EcalEnergy = ohPhotEt[i]/(sin(2*atan(exp(0-ohPhotEta[i]))));
                        if ( (TMath::Abs(ohPhotEta[i]) < 1.479
                              && ohPhotHforHoverE[i]/EcalEnergy < HoverEEB)
                              ||
                              ((1.479 < TMath::Abs(ohPhotEta[i]) && TMath::Abs(ohPhotEta[i]) < 2.65) 
                              && ohPhotHforHoverE[i]/EcalEnergy < HoverEEC))
                           if (ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs   
                              rc.push_back(i);
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}


vector<int> OHltTree::VectorOpenHlt1ElectronSamHarperPassed(
      float Et,
      int L1iso,
      float Tisobarrel,
      float Tisoendcap,
      float Tisoratiobarrel,
      float Tisoratioendcap,
      float HisooverETbarrel,
      float HisooverETendcap,
      float EisooverETbarrel,
      float EisooverETendcap,
      float hoverebarrel,
      float hovereendcap,
      float clusshapebarrel,
      float clusshapeendcap,
      float r9barrel,
      float r9endcap,
      float detabarrel,
      float detaendcap,
      float dphibarrel,
      float dphiendcap)
{
   float barreleta = 1.479;
   float endcapeta = 2.65;

   vector<int> rc;
  // Loop over all oh electrons
   for (int i=0; i<NohEle; i++)
   {
      float ohEleHoverE = ohEleHforHoverE[i]/ohEleE[i];
      int isbarrel = 0;
      int isendcap = 0;
      if (TMath::Abs(ohEleEta[i]) < barreleta)
         isbarrel = 1;
      if (barreleta < TMath::Abs(ohEleEta[i]) && TMath::Abs(ohEleEta[i])
            < endcapeta)
         isendcap = 1;

      if (ohEleEt[i] > Et)
      {
         if (TMath::Abs(ohEleEta[i]) < endcapeta)
         {
            if (ohEleNewSC[i]<=1)
            {
               if (ohElePixelSeeds[i]>0)
               {
                  if (ohEleL1iso[i] >= L1iso)
                  { // L1iso is 0 or 1 
                     if (ohEleL1Dupl[i] == false)
                     { // remove double-counted L1 SCs 
                        if ( (isbarrel && ((ohEleHiso[i]/ohEleEt[i])
                              < HisooverETbarrel))
                              || (isendcap && ((ohEleHiso[i]/ohEleEt[i])
                                    < HisooverETendcap)))
                        {
                           if ( (isbarrel && ((ohEleEiso[i]/ohEleEt[i])
                                 < EisooverETbarrel)) || (isendcap
                                 && ((ohEleEiso[i]/ohEleEt[i])
                                       < EisooverETendcap)))
                           {
                              if ( ((isbarrel) && (ohEleHoverE < hoverebarrel))
                                    || ((isendcap) && (ohEleHoverE
                                          < hovereendcap)))
                              {
                                 if ( (isbarrel && (((ohEleTiso[i] < Tisobarrel
                                       && ohEleTiso[i] != -999.) || (Tisobarrel
                                       == 999.)))) || (isendcap
                                       && (((ohEleTiso[i] < Tisoendcap
                                             && ohEleTiso[i] != -999.)
                                             || (Tisoendcap == 999.)))))
                                 {
                                    if (((isbarrel) && (ohEleTiso[i]/ohEleEt[i]
                                          < Tisoratiobarrel)) || ((isendcap)
                                          && (ohEleTiso[i]/ohEleEt[i]
                                                < Tisoratioendcap)))
                                    {
                                       if ( (isbarrel && ohEleClusShap[i]
                                             < clusshapebarrel) || (isendcap
                                             && ohEleClusShap[i]
                                                   < clusshapeendcap))
                                       {
                                          if ( (isbarrel && ohEleR9[i]
                                                < r9barrel) || (isendcap
                                                && ohEleR9[i] < r9endcap))
                                          {
                                             if ( (isbarrel
                                                   && TMath::Abs(ohEleDeta[i])
                                                         < detabarrel)
                                                   || (isendcap
                                                         && TMath::Abs(ohEleDeta[i])
                                                               < detaendcap))
                                             {
                                                if ( (isbarrel && ohEleDphi[i]
                                                      < dphibarrel)
                                                      || (isendcap
                                                            && ohEleDphi[i]
                                                                  < dphiendcap))
                                                {
						  rc.push_back(i);
                                                }
                                             }
                                          }
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}

vector<int> OHltTree::VectorOpenHlt1PhotonPassedR9ID(
      float Et,
      float R9ID,
      int L1iso,
      float Tiso,
      float Eiso,
      float HisoBR,
      float HisoEC,
      float HoverE,
      float R9,
      float ClusShapEB,
      float ClusShapEC)
{
   vector<int> rc;
   for (int i=0; i<NohPhot; i++)
   {
      if (ohPhotEt[i] > Et)
      {
         if (TMath::Abs(ohPhotEta[i]) < 2.65)
         {
            if (ohPhotL1iso[i] >= L1iso)
            {
               if (ohPhotTiso[i] < Tiso + 0.001*ohPhotEt[i])
               {
                  if (ohPhotEiso[i] < Eiso + 0.012*ohPhotEt[i])
                  {
                     if ((TMath::Abs(ohPhotEta[i]) < 1.479
                           && ohPhotHiso[i] < HisoBR + 0.005*ohPhotEt[i]
                           && ohEleClusShap[i] < ClusShapEB
                           && ohPhotR9[i] < R9) 
                           || 
                           (1.479 < TMath::Abs(ohPhotEta[i]) 
                           && TMath::Abs(ohPhotEta[i]) < 2.65 
                           && ohPhotHiso[i] < HisoEC + 0.005*ohPhotEt[i] 
                           && ohEleClusShap[i] < ClusShapEC))
                     {
                        float EcalEnergy = ohPhotEt[i]/(sin(2*atan(exp(0-ohPhotEta[i]))));
                        if (ohPhotHforHoverE[i]/EcalEnergy < HoverE)
                           if (ohPhotR9ID[i] > R9ID)
                              if (ohPhotL1Dupl[i] == false) // remove double-counted L1 SCs  
                                 rc.push_back(i);
                     }
                  }
               }
            }
         }
      }
   }

   return rc;
}


int OHltTree::OpenL1SetSingleJetBit(const float& thresh)
{

   // count number of L1 central, forward and tau jets above threshold

   int rc=0;

   bool CenJet=false, ForJet=false, TauJet=false;

   //size_t size = sizeof(L1CenJetEt)/sizeof(*L1CenJetEt);
   const size_t size = 4;
   // cout << thresh << "\t" << size << endl;

   int ncenjet=0;
   for (unsigned int i=0; i<size; ++i)
   {
      if (L1CenJetEt[i] >= thresh)
         ++ncenjet;
   }
   CenJet=ncenjet>=1;

   int nforjet=0;
   for (unsigned int i=0; i<size; ++i)
   {
      if (L1ForJetEt[i] >= thresh)
         ++nforjet;
   }
   ForJet=nforjet>=1;

   int ntaujet=0;
   for (unsigned int i=0; i<size; ++i)
   {
      if (L1TauEt[i] >= thresh)
         ++ntaujet;
   }
   TauJet=ntaujet>=1;

   bool L1SingleJet=(CenJet || ForJet || TauJet );

   if (L1SingleJet)
      rc=1;

   return (rc );

}

int OHltTree::OpenHltCleanedDiJetPassed(
      float Et1,
      float Et2,
      bool cor,
      const std::string& algo,
      float Deta,
      float Mjj,
      bool etaOpposite,
      bool jetID,
      std::vector<int> ohEleIts)
{
   // fill the jet collection
   int NohJet = 0;
   float ohJetEta[200];
   float ohJetPhi[200];
   float ohJetEt[200];
   float ohJetE[200];
   float ohJetEMF[200];
   float ohJetN90[200];

   if ( (cor == false) && (algo == "Calo"))
   {
      NohJet = NrecoJetCal;
      for (int ohJetIt = 0; ohJetIt < NohJet; ++ohJetIt)
      {
         ohJetEta[ohJetIt] = recoJetCalEta[ohJetIt];
         ohJetPhi[ohJetIt] = recoJetCalPhi[ohJetIt];
         ohJetEt[ohJetIt] = recoJetCalE[ohJetIt] * sin(2.*atan(exp(-1.
               *recoJetCalEta[ohJetIt])));
         ohJetE[ohJetIt] = recoJetCalE[ohJetIt];
         ohJetEMF[ohJetIt] = recoJetCalEMF[ohJetIt];
         ohJetN90[ohJetIt] = recoJetCalN90[ohJetIt];
      }
   }

   if ( (cor == true) && (algo == "Calo"))
   {
      NohJet = NrecoJetCorCal;
      for (int ohJetIt = 0; ohJetIt < NohJet; ++ohJetIt)
      {
         ohJetEta[ohJetIt] = recoJetCorCalEta[ohJetIt];
         ohJetPhi[ohJetIt] = recoJetCorCalPhi[ohJetIt];
         ohJetEt[ohJetIt] = recoJetCorCalE[ohJetIt] * sin(2.*atan(exp(-1.
               *recoJetCorCalEta[ohJetIt])));
         ohJetE[ohJetIt] = recoJetCorCalE[ohJetIt];
         ohJetEMF[ohJetIt] = recoJetCorCalEMF[ohJetIt];
         ohJetN90[ohJetIt] = recoJetCorCalN90[ohJetIt];
      }
   }

   if ( (cor == false) && (algo == "PF"))
   {
      NohJet = NohPFJet;
      for (int ohJetIt = 0; ohJetIt < NohJet; ++ohJetIt)
      {
         ohJetEta[ohJetIt] = pfJetEta[ohJetIt];
         ohJetPhi[ohJetIt] = pfJetPhi[ohJetIt];
         ohJetEt[ohJetIt] = pfJetPt[ohJetIt];
         ohJetEMF[ohJetIt] = -1.;
         ohJetN90[ohJetIt] = -1.;
      }
   }

   // clean the jet collection from electrons
   int NohCleanedJet = 0;
   float ohCleanedJetEta[200];
   float ohCleanedJetPhi[200];
   float ohCleanedJetEt[200];
   float ohCleanedJetE[200];
   float ohCleanedJetEMF[200];
   float ohCleanedJetN90[200];

   for (int ohJetIt = 0; ohJetIt < NohJet; ++ohJetIt)
   {
      bool isMatching = false;
      for (unsigned int ohEleIt = 0; ohEleIt < ohEleIts.size(); ++ohEleIt)
         if (deltaR(ohEleEta[ohEleIts.at(ohEleIt)], ohElePhi[ohEleIts.at(ohEleIt)], ohJetEta[ohJetIt], ohJetPhi[ohJetIt]) < 0.5)
            isMatching = true;

      if (isMatching == true)
         continue;

      ohCleanedJetEta[NohCleanedJet] = ohJetEta[ohJetIt];
      ohCleanedJetPhi[NohCleanedJet] = ohJetPhi[ohJetIt];
      ohCleanedJetEt[NohCleanedJet] = ohJetEt[ohJetIt];
      ohCleanedJetE[NohCleanedJet] = ohJetE[ohJetIt];
      ohCleanedJetEMF[NohCleanedJet] = ohJetEMF[ohJetIt];
      ohCleanedJetN90[NohCleanedJet] = ohJetN90[ohJetIt];
      ++NohCleanedJet;
   }

   // do the selection
   int rc = 0;
   if (NohCleanedJet < 2)
      return rc;

   // loop on jets
   for (int i = 0; i < NohCleanedJet; ++i)
   {
      if ( (jetID == true) && (TMath::Abs(ohCleanedJetEta[i]) < 2.6)
            && (ohCleanedJetEMF[i] < 0.01))
         continue;
      if ( (jetID == true) && (TMath::Abs(ohCleanedJetEta[i]) < 2.6)
            && (ohCleanedJetN90[i] < 2))
         continue;

      for (int j = i+1; j < NohCleanedJet; j++)
      {
         if ( (jetID == true) && (TMath::Abs(ohCleanedJetEta[j]) < 2.6)
               && (ohCleanedJetEMF[j] < 0.01))
            continue;
         if ( (jetID == true) && (TMath::Abs(ohCleanedJetEta[j]) < 2.6)
               && (ohCleanedJetN90[j] < 2))
            continue;

         PtEtaPhiELorentzVector j1(
               ohCleanedJetEt[i],
               ohCleanedJetEta[i],
               ohCleanedJetPhi[i],
               ohCleanedJetE[i]);
         PtEtaPhiELorentzVector j2(
               ohCleanedJetEt[j],
               ohCleanedJetEta[j],
               ohCleanedJetPhi[j],
               ohCleanedJetE[j]);

         if ( (std::max(ohCleanedJetEt[i], ohCleanedJetEt[j]) > Et1)
               && (std::min(ohCleanedJetEt[i], ohCleanedJetEt[j]) > Et2)
               && (deltaEta(ohCleanedJetEta[i], ohCleanedJetEta[j]) > Deta)
               && ((j1+j2).mass() > Mjj) && ( (etaOpposite == true
               && ohCleanedJetEta[i]*ohCleanedJetEta[j] < 0.) || (etaOpposite
               == false) ))
            ++rc;
      }
   }

   return rc;
}

int OHltTree::OpenHlt1ElectronVbfEleIDPassed(
      float Et,
      float L1SeedEt,
      bool iso,
      int& EtMaxIt,
      std::vector<int>* it)
{
   int rc = 0;
   float EtMax = -9999.;
   if (it != NULL)
      it->clear();

   // Loop over all oh electrons  
   for (int i = 0; i < NohEle; ++i)
   {
      // ET/eta/pixelMatch cuts
      if (ohEleEt[i] < Et)
         continue;
      if (TMath::Abs(ohEleEta[i]) > 2.65)
         continue;
      if (ohElePixelSeeds[i] <= 0)
         continue;
      if (ohEleP[i] < 0.)
         continue;

      // EleID
      // R9
      if ( (TMath::Abs(ohEleEta[i]) < 1.479) && (ohEleR9[i] > 0.98))
         continue;
      if ( (TMath::Abs(ohEleEta[i]) > 1.479) && (ohEleR9[i] > 999.))
         continue;
      // sigmaietaieta
      if ( (TMath::Abs(ohEleEta[i]) < 1.479) && (ohEleClusShap[i] > 0.011))
         continue;
      if ( (TMath::Abs(ohEleEta[i]) > 1.479) && (ohEleClusShap[i] > 0.031))
         continue;

      // deta
      if ( (TMath::Abs(ohEleEta[i]) < 1.479) && (TMath::Abs(ohEleDeta[i])
            > 0.008))
         continue;
      if ( (TMath::Abs(ohEleEta[i]) > 1.479) && (TMath::Abs(ohEleDeta[i])
            > 0.008))
         continue;

      // dphi
      if ( (TMath::Abs(ohEleEta[i]) < 1.479) && (TMath::Abs(ohEleDphi[i])
            > 0.070))
         continue;
      if ( (TMath::Abs(ohEleEta[i]) > 1.479) && (TMath::Abs(ohEleDphi[i])
            > 0.050))
         continue;

      // H/E
      if (ohEleHforHoverE[i]/ohEleE[i] > 0.05)
         continue;

      // isolation

      // tracker iso
      if ( (iso == true) && (TMath::Abs(ohEleEta[i]) < 1.479) && (ohEleTiso[i]
            /ohEleEt[i] > 0.125))
         continue;
      if ( (iso == true) && (TMath::Abs(ohEleEta[i]) > 1.479) && (ohEleTiso[i]
            /ohEleEt[i] > 0.075))
         continue;

      // ecal iso
      if ( (iso == true) && (TMath::Abs(ohEleEta[i]) < 1.479) && (ohEleEiso[i]
            /ohEleEt[i] > 0.125))
         continue;
      if ( (iso == true) && (TMath::Abs(ohEleEta[i]) > 1.479) && (ohEleEiso[i]
            /ohEleEt[i] > 0.075))
         continue;

      // hcal iso
      if ( (iso == true) && (TMath::Abs(ohEleEta[i]) < 1.479) && (ohEleHiso[i]
            /ohEleEt[i] > 0.125))
         continue;
      if ( (iso == true) && (TMath::Abs(ohEleEta[i]) > 1.479) && (ohEleHiso[i]
            /ohEleEt[i] > 0.075))
         continue;

      // L1 matching - isolated

      bool isL1IsolMatched = false;
      for (int j = 0; j < NL1IsolEm; ++j)
      {
         if (L1IsolEmEt[j] > L1SeedEt)
         {
            float etaLow = 0.;
            float etaHigh = 0.;

            if (fabs(ohEleEta[i]) < 1.479)
            {
               etaLow = L1IsolEmEta[j] - 0.522/2.;
               etaHigh = L1IsolEmEta[j] + 0.522/2.;
            }
            else
            {
               etaLow = L1IsolEmEta[j] - 1.0/2.;
               etaHigh = L1IsolEmEta[j] + 1.0/2.;
            }

            float Dphi = deltaPhi(L1IsolEmPhi[j], ohElePhi[i]);

            if (ohEleEta[i] > etaLow && ohEleEta[i] < etaHigh && Dphi < 1.044
                  /2.)
               isL1IsolMatched = true;
         }
      }

      // L1 matching - non isolated
      bool isL1NIsolMatched = false;
      for (int j = 0; j < NL1NIsolEm; ++j)
      {
         if (L1NIsolEmEt[j] > L1SeedEt)
         {
            float etaLow = 0.;
            float etaHigh = 0.;

            if (fabs(ohEleEta[i]) < 1.479)
            {
               etaLow = L1NIsolEmEta[j] - 0.522/2.;
               etaHigh = L1NIsolEmEta[j] + 0.522/2.;
            }
            else
            {
               etaLow = L1NIsolEmEta[j] - 1.0/2.;
               etaHigh = L1NIsolEmEta[j] + 1.0/2.;
            }

            float Dphi = deltaPhi(L1NIsolEmPhi[j], ohElePhi[i]);

            if (ohEleEta[i] > etaLow && ohEleEta[i] < etaHigh && Dphi < 1.044
                  /2.)
               isL1NIsolMatched = true;
         }
      }

      if ( (L1SeedEt > 0.) && (isL1IsolMatched == false) && (isL1NIsolMatched
            == false))
         continue;

      ++rc;

      if (ohEleEt[i] > EtMax)
      {
         EtMaxIt = i;
         EtMax = ohEleEt[i];
      }

      if (it != NULL)
         it->push_back(i);
   }

   return rc;
}

float OHltTree::deltaPhi(const float& phi1, const float& phi2)
{
   float deltaphi = TMath::Abs(phi1 - phi2);
   if (deltaphi > 6.283185308)
      deltaphi -= 6.283185308;
   if (deltaphi > 3.141592654)
      deltaphi = 6.283185308 - deltaphi;
   return deltaphi;
}

float OHltTree::deltaEta(const float& eta1, const float& eta2)
{
   return TMath::Abs(eta1 - eta2);
}

float OHltTree::deltaR(
      const float& eta1,
      const float& phi1,
      const float& eta2,
      const float& phi2)
{
   return sqrt(deltaEta(eta1, eta2)*deltaEta(eta1, eta2) + deltaPhi(phi1, phi2)
         *deltaPhi(phi1, phi2));
}
