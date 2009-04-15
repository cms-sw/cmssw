#include "RooAddPdf.h"
#include "RooExponential.h"
#include "RooGaussian.h"

#include "PhysicsTools/RooStatsCms/interface/RscCatModel.h"
#include "PhysicsTools/RooStatsCms/interface/RscBaseModel.h"


RscCatModel::RscCatModel(TString theName, RooRealVar& theVar, RooArgSet* discVars)
  : number_categories("number_categories","number of sub-categories to the PDF",1),
    extended_pdf("extended_pdf","tells whether sub-categories own separate yields (or fractions)"),
    yield(_name+"_yield","yield of this PDF",0)
{
  if (verbose())
    cout << "Instanciating a " << ClassName() << " object named " << theName << "\n";
  _name = theName;
  _discVars = discVars;
  x = &theVar;
  ownCategories = false;

  extended_pdf.defineType("true");
  extended_pdf.defineType("false");
  extended_pdf.setLabel("true");

  readDataCard();

  if (extended_pdf=="true") extendedPdf = true; else extendedPdf = false; 
}

RscCatModel::~RscCatModel() {
  // destructor
  if (verbose())
    cout << "Destructing the " << _name << " object\n";
  if (ownCategories) delete pdfs;
}

void RscCatModel::buildPdf()
{
  // build the PDF
  RooArgList pdfList;
  RooArgList coefList;

  if (verbose())
    cout << "Build composite PDF\n";
  if (number_categories.getVal()==1) {
    RscBaseModel* thisPdf = new RscBaseModel(_name,*x,_discVars);
    _thePdf = thisPdf->getPdf();
  } else {
    ownCategories = true;

    int nCat = int(number_categories.getVal());
    pdfs = new TList();
    pdfs->SetOwner();

    for (int i=1; i<=nCat; i++) {
      if (verbose())
        cout << "Building category " << i << endl;
      TString aname = _name;
      aname += "_";
      aname += i;
      RscBaseModel* bld = new RscBaseModel(aname,*x,_discVars);
      pdfs->Add(bld); 
    }

    TIter next(pdfs);
    while (TObject *obj = next()) {
      RscBaseModel* bld = dynamic_cast<RscBaseModel*>(obj);
      assert(bld);
      pdfList.add(*bld->getPdf());
//       if (extendedPdf) {
// 	coefList.add(bld->yield);
//       }
    }

    _thePdf = new RooAddPdf(_name,"composite PDF",pdfList,coefList);
  }
}


void RscCatModel::readDataCard()
{
  // read the data card
  if (getDataCard()) {
    RooArgSet(number_categories,extended_pdf,yield).readFromFile(getDataCard(), 0, _name);
   }
}

void RscCatModel::writeDataCard(ostream& out)
{
  // write the data card
  out << "[" << _name << "]" << endl;
  RooArgSet(number_categories,extended_pdf,yield).writeToStream(out,false);
 }

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
