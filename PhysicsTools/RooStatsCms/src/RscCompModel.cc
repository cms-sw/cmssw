
#include "assert.h"

#include "RooAddPdf.h"
#include "RooExponential.h"
#include "RooGaussian.h"

#include "PhysicsTools/RooStatsCms/interface/RscCompModel.h"
#include "PhysicsTools/RooStatsCms/interface/RscBaseModel.h"
#include "PhysicsTools/RooStatsCms/interface/RscMultiModel.h"


//ClassImp(RscCompModel);

/*----------------------------------------------------------------------------*/

RscCompModel::RscCompModel(TString theName, RooArgList& theVars)
  : number_components("number_components","number of sub-components to the PDF",1)
{
  if (verbose())
  cout << "Instanciating a " << ClassName() << " object named " << theName << "\n";
  _name = theName;
  _vars = &theVars;
  _caller_vars=NULL;
  /*
  DP here we choose if we want to use a RooFormulaVar instead of a RooRealVar.
  */
  //_yield = new RooRealVar(_name+"_yield","yield of "+_name,0);

  pdfs = 0; 
  ownPDFs = kFALSE;
  _yield = 0;
  readDataCard();

}

/*----------------------------------------------------------------------------*/

RscCompModel::RscCompModel(TString theName, 
                           RooArgList& theVars, 
                           RooArgList& caller_vars)
: number_components("number_components","number of sub-components to the PDF",1)
{
  if (verbose())
  cout << "Instanciating a " << ClassName() << " object named " << theName << "\n";
  _name = theName;
  _vars = &theVars;
  _caller_vars=&caller_vars;
  
  /*
  DP here we choose if we want to use a RooFormulaVar instead of a RooRealVar.
  */
  //_yield = new RooRealVar(_name+"_yield","yield of "+_name,0);

  pdfs = 0; 
  ownPDFs = kFALSE;
  _yield = 0;
  readDataCard();

}

/*----------------------------------------------------------------------------*/

RscCompModel::~RscCompModel() {
  // destructor
  if (verbose())
  cout << "Destructing the " << _name << " object\n";
  if ( ownPDFs ) delete pdfs;
}

/*----------------------------------------------------------------------------*/

void RscCompModel::buildPdf()
{
  // build the PDF
  RooArgList pdfList;
  RooArgList coefList;

  if (verbose())
  cout << "Building composite PDF " << _name << endl;

  if (number_components.getVal()==1) {
    RscMultiModel* thisPdf = new RscMultiModel(_name,*_vars);
    _thePdf = thisPdf->getPdf();
    _yield = thisPdf->getYield();

  } else {
    ownPDFs = true;
    pdfs = new TList();
    pdfs->SetOwner();
    int nComp = int(number_components.getVal());
    for (int i=1; i<=nComp; i++) {
      if (verbose())
        cout << "Building component " << i << " of " << _name << endl;
      TString aname = _name;
      aname += i;
      RscMultiModel* bld = new RscMultiModel(aname,*_vars);
      pdfs->Add(bld); 
    }
    /*
    Here we prepare the list of pdfs that we get from the RscMultiModel.
    At the same time a list of all the components of the yields in a RooArgList.
    If the name is already present the parameter is not stored (in line with
    the "same name-same pointer" principle of the modelling part of Rsc).
    At the end all the single formulas are summed together.
    */
    RooArgList tot_yield_components;
    TString* yield_formulas = new TString[(int)number_components.getVal()];
    TIter next(pdfs);
    int index=0;
    while (TObject *obj = next()) {
      RscMultiModel* bld = dynamic_cast<RscMultiModel*>(obj);
      assert(bld);
      pdfList.add(*bld->getPdf());

      // Build the yield
      RooArgList yield_vars(*bld->getYield()->getVariables());
      for (int i=0;i< yield_vars.getSize();++i){
        yield_formulas[index]+=yield_vars[i].GetName();
        yield_formulas[index]+= " * ";

        if (tot_yield_components.find(yield_vars[i].GetName())==NULL)
            tot_yield_components.add(yield_vars[i]);
        }
      // Remove the last "* "
      yield_formulas[index].Chop();
      yield_formulas[index].Chop();
      yield_formulas[index].Chop();
      //DBG
      RooAbsPdf* f=bld->getPdf();
      if (verbose())
        std::cout <<  "DEBUG\n " <<" - Pdf " << f->GetName() << " is extended? " 
                    << f->canBeExtended() << std::endl
                    << "Single Yield formula is: " <<yield_formulas[index].Data() 
                    << std::endl;

      index++;
      }
    TString tot_yield_formula("");
    for (int i=0;i< index ;++i){
        tot_yield_formula += yield_formulas[i] + " + ";
        }
    tot_yield_formula.Chop();
    tot_yield_formula.Chop();
    tot_yield_formula.Chop();
    if (verbose())
        std::cout <<  "DEBUG\n " << "Total Yield formula is: " 
                << tot_yield_formula.Data() << std::endl;

    tot_yield_components.Print("v");

    _yield = new RooFormulaVar ((_name+"_yield").Data(),
                                ("Total yield for "+_name).Data(),
                                tot_yield_formula.Data(),
                                tot_yield_components);
    /*
    Since in Multimodel the pdfs are already extended, no need for coefficients
    */
    //_thePdf = new RooAddPdf(_name,"composite PDF",pdfList,coefList);
    _thePdf = new RooAddPdf(_name,"composite PDF",pdfList);

    delete yield_formulas;
  }
}

/*----------------------------------------------------------------------------*/

RooFormulaVar* RscCompModel::getYield() {
  return _yield;
}

/*----------------------------------------------------------------------------*/

void RscCompModel::readDataCard()
{
  // read the data card
  if (getDataCard())
    //RooArgSet(number_components,*_yield/*,extended_pdf*/).readFromFile(getDataCard(), 0, _name);
    RooArgSet(number_components).readFromFile(getDataCard(), 0, _name);
  else
    cout << "Unable to read datacard!\n";

}

/*----------------------------------------------------------------------------*/

void RscCompModel::writeDataCard(ostream& out)
{
  // write the data card
  out << "[" << _name << "]" << endl;
  RooArgSet(number_components,*_yield/*,extended_pdf*/).writeToStream(out,false);
 }


/*----------------------------------------------------------------------------*/
