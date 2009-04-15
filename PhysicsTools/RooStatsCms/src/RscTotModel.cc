#include "RooAddPdf.h"
#include "RooExponential.h"
#include "RooGaussian.h"

#include "PhysicsTools/RooStatsCms/interface/RscTotModel.h"


/// The Constructor

/** 
It takes the name of the datacard where the model is described.
**/
RscTotModel::RscTotModel(TString theName)
  : variablesString("variables",0,"")
{
  if (verbose())
  cout << "[RscTotModel::RscTotModel] Instanciating a " << ClassName()
       << " object named " << theName << "\n";
  _name = theName;

  theConstraints=NULL;

  readDataCard();
  if (variablesString.getVal()=="") 
    cout << "ERROR, you forgot to specify the discriminating variables\n";
}

/*----------------------------------------------------------------------------*/

///   Destructor

RscTotModel::~RscTotModel()
{
  //cout << "[RscTotModel::~RscTotModel] Destructing the " 
  //     << _name << " object\n";

  // to do: iterate and delete all objects in the 'theVars' list
  if (theVars!=NULL)
    delete theVars;
  //delete theConstraints;
  //if (sigExtendedPdf!=NULL)
  //  delete sigExtendedPdf;
  //if (bkgExtendedPdf!=NULL)
  //  delete bkgExtendedPdf;
//   if (sigModel!=NULL)
//     delete sigModel;
//   if (bkgModel!=NULL)
//     delete bkgModel;
}

/*----------------------------------------------------------------------------*/


/// Builds the RooArgList of the discriminating variables

void RscTotModel::buildVars()
{
  if (verbose())
  cout << "[RscTotModel::buildVars] " 
       << "Build the RooRealVar objects for the discriminating variables\n";
  //variablesString.Print("v");

  theVars = new RooArgList();

  // temporary objects for the loop
  TString tmpVarName("");
  char tmpLetter=' ';
  TString tmpVarString(variablesString.getVal());
  tmpVarString+=" #";

  // split at spaces, create RooRealVar objects and append them to the list of 
  // variables
  for (int i=0;i<(int)tmpVarString.Length();++i) {
    if (tmpLetter==' ') {
      if (i!=0) {
    // do not forget to delete!
	RooRealVar* newVar = new RooRealVar(tmpVarName,"",0); 
	//newVar->Print("v");
    //DP we change the status to owned so to be able to read the RooFormulaVar
	theVars->add(*newVar);
      } tmpVarName="";
    }
    else tmpVarName+=TString(tmpLetter); // move to beginning of the loop!

    // retrieve the next letter of the string
    tmpLetter=tmpVarString[i]; 
  }
}

/*----------------------------------------------------------------------------*/

/// Build the model

void RscTotModel::buildPdf()
{
  if (verbose())
  cout << "Building the " << _name << " model\n";

  // create the sig component of the analysis model
  sigModel = new RscCompModel(_name+"_sig",*theVars,*theVars);

  // create the bkg component of the analysis model
  bkgModel = new RscCompModel(_name+"_bkg",*theVars,*theVars);

  // retrieve some pointers for the RooAddPdf construction
  RooAbsPdf* sigPdf = sigModel->getPdf();
  RooAbsPdf* bkgPdf = bkgModel->getPdf();
  //RooFormulaVar* sigYield = sigModel->getYield();
  //RooFormulaVar* bkgYield = bkgModel->getYield();

  // add the sig/bkg model together into an extended PDF

  /* 
    Not necessary anymore: rscmultimodel (and therefore compmodel) provides extended pdfs
  */
//   _thePdf = new RooAddPdf(_name,"total sig+bkg PDF",
//                    RooArgList(*sigPdf,*bkgPdf),RooArgList(*sigYield,*bkgYield));
  _thePdf = new RooAddPdf(_name,"total sig+bkg PDF",RooArgList(*sigPdf,*bkgPdf));



  // make the signal extended PDF: required by some of the statistics classes
  //sigExtendedPdf = new RooExtendPdf(_name+"_sig_extended","",*sigPdf,*sigYield);
  sigExtendedPdf = static_cast<RooExtendPdf*> (sigPdf);

  //make the background extended PDF: required by some of the statistics classes
  //bkgExtendedPdf = new RooExtendPdf(_name+"_bkg_extended","",*bkgPdf,*bkgYield);
  bkgExtendedPdf = static_cast<RooExtendPdf*> (bkgPdf);

  // read and define the constraints from the configuration file
  //buildConstraints();
  // keep that here or somewhere else? in RscConstaint?
  // more general constraints/across analysis correlations? -> not here!
}

/*----------------------------------------------------------------------------*/

/// Return a point to the RooRealVar for the signal yield

RooFormulaVar* RscTotModel::getSigYield()
{
  if (_thePdf==NULL)
    getPdf();
  return sigModel->getYield();
}

/*----------------------------------------------------------------------------*/

/// Return a point to the RooRealVar for the background yield

RooFormulaVar* RscTotModel::getBkgYield()
{
  if (_thePdf==NULL)
    getPdf();
  return bkgModel->getYield();
}

/*----------------------------------------------------------------------------*/

/// Return a pointer to the discriminating variable RooRealVar specified by name

RooRealVar* RscTotModel::getVar(TString varName)
{
  return (RooRealVar*) theVars->find(varName);
}

/*----------------------------------------------------------------------------*/

/// Return a pointer to the RooArgList of all discriminating variables 

RooArgList* RscTotModel::getVars()
{
  return theVars;
}

/*----------------------------------------------------------------------------*/

void RscTotModel::buildConstraints(){

    theConstraints=new RooArgList("The list of constraints");

    if (_thePdf==NULL)
        getPdf();
    RooArgList var_list(*(_thePdf->getVariables()));

    for(int i=0;i<var_list.getSize();++i){
        var_list[i].Print();
        if (TString("Constraint") == var_list[i].ClassName())
           theConstraints->add(var_list[i]);
        }

    }
/*----------------------------------------------------------------------------*/

/// Returns the Constraints present in the model in a list
RooArgList* RscTotModel::getConstraints(){
    if (theConstraints==NULL)
        buildConstraints();
    return theConstraints;

    }

/*----------------------------------------------------------------------------*/

/// Return a pointer to the signal (extended) PDF

RooAbsPdf* RscTotModel::getExtendedSigPdf()
{
  if (!_thePdf) buildPdf();
  return sigExtendedPdf;
}

/*----------------------------------------------------------------------------*/

/// Return a pointer to the background (extended) PDF

RooAbsPdf* RscTotModel::getExtendedBkgPdf()
{
  if (!_thePdf) buildPdf();
  return bkgExtendedPdf;
}

/*----------------------------------------------------------------------------*/

/// Read parametrisation from the configuration file

void RscTotModel::readDataCard()
{
  if (getDataCard()) {    
    RooArgSet(variablesString).readFromFile(getDataCard(), 0, _name);
  }
  if (this->verbose) std::cout << "variables = " 
                               << variablesString.getVal() << "\n";

  // created the discriminating variables
  buildVars();

  //cout << "before dataCard read out\n";
  //theVars->Print("v");

  // read variables from the dataCard
  RooArgSet(*theVars).readFromFile(getDataCard(), 0, _name);

  //cout << "after dataCard read out\n";
  if (verbose())
    theVars->Print("v");
}

/*----------------------------------------------------------------------------*/

/// Write parametrisation to a stream

void RscTotModel::writeDataCard(ostream& out)
{
  out << "[" << _name << "]" << endl;
  RooArgSet(variablesString).writeToStream(out,false);
}

/*----------------------------------------------------------------------------*/
// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
