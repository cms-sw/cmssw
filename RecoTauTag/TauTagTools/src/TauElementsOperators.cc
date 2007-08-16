#include "RecoTauTag/TauTagTools/interface/TauElementsOperators.h"

double TauElementsOperators::computeConeSize(const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax){
  double x=Tau_.energy();
  double y=Tau_.et();
  double ConeSize=ConeSizeTFormula.Eval(x,y);
  if (ConeSize<ConeSizeMin)ConeSize=ConeSizeMin;
  if (ConeSize>ConeSizeMax)ConeSize=ConeSizeMax;
  return ConeSize;
}
TFormula TauElementsOperators::computeConeSizeTFormula(const string& ConeSizeFormula,const char* errorMessage){
  //--- check functional form 
  //    given as configuration parameter for matching and signal cone sizes;
  //
  //    The size of a cone may depend on the energy "E" and/or transverse energy "ET" of the tau-jet candidate.
  //    Any functional form that is supported by ROOT's TFormula class can be used (e.g. "3.0/E", "0.25/sqrt(ET)")
  //
  //    replace "E"  by TFormula variable "x"
  //            "ET"                      "y"
  string ConeSizeFormulaStr = ConeSizeFormula;
  replaceSubStr(ConeSizeFormulaStr,"E","x");
  replaceSubStr(ConeSizeFormulaStr,"ET","y");
  TFormula ConeSizeTFormula;
  ConeSizeTFormula.SetName("ConeSize");
  ConeSizeTFormula.SetTitle(ConeSizeFormulaStr.data()); // the function definition is actually stored in the "Title" data-member of the TFormula object
  int errorFlag = ConeSizeTFormula.Compile();
  if (errorFlag!= 0) {
    throw cms::Exception("") << "\n unsupported functional Form for " << errorMessage << " " << ConeSizeFormula << endl
			     << "Please check that the Definition in \"" << ConeSizeTFormula.GetName() << "\" only contains the variables \"E\" or \"ET\""
			     << " and Functions that are supported by ROOT's TFormular Class." << endl;
  }else return ConeSizeTFormula;
}
void TauElementsOperators::replaceSubStr(string& s,const string& oldSubStr,const string& newSubStr){
  //--- protect replacement algorithm
  //    from case that oldSubStr and newSubStr are equal
  //    (nothing to be done anyway)
  if ( oldSubStr == newSubStr ) return;
  
  //--- protect replacement algorithm
  //    from case that oldSubStr contains no characters
  //    (i.e. matches everything)
  if ( oldSubStr.empty() ) return;
  
  const string::size_type lengthOldSubStr = oldSubStr.size();
  const string::size_type lengthNewSubStr = newSubStr.size();
  
  string::size_type positionPreviousMatch = 0;
  string::size_type positionNextMatch = 0;
  
  //--- consecutively replace all occurences of oldSubStr by newSubStr;
  //    keep iterating until no occurence of oldSubStr left
  while ( (positionNextMatch = s.find(oldSubStr, positionPreviousMatch)) != string::npos ) {
    s.replace(positionNextMatch, lengthOldSubStr, newSubStr);
    positionPreviousMatch = positionNextMatch + lengthNewSubStr;
  } 
}


