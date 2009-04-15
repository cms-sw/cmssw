// This class is the abstract class that serves as the base of all the
// Pdf builders of the RhoPiTools/RhoRhoTools packages. To write a Pdf 
// builder, one should :
//
// - inherit from RscAbsPdfBuilder
//
// - define a constructor that is meant to be initialised with the 
// user's discriminating variables. At the end of the constructor, 
// readDataCard() should be called.
//
// - implement the buildPdf() method that will build the Pdf pointed to by
// _thePdf. During the building, it is possible to read the flag saying 
// wether the variables should be blind by testing isData().
//
// - implement readDataCard() and writeDataCard(ostream& out) that will 
// read and write the parameters contained by the builder in a section
// specific to it.

#include <assert.h>
#include <string.h>

//DP Rsc banner
#include <iostream>

//DP Python config management
#include <fstream>

#include "RooStringVar.h"

#include "PhysicsTools/RooStatsCms/interface/RscAbsPdfBuilder.h"
#include "PhysicsTools/RooStatsCms/interface/Constraint.h"


RooArgSet RscAbsPdfBuilder::globalConfig;

/*----------------------------------------------------------------------------*/

RscAbsPdfBuilder::RscAbsPdfBuilder(){
  // Constructor
  _thePdf = 0;
  _discVars = 0;
  _deleteThePdf = true;
}

/*----------------------------------------------------------------------------*/

RscAbsPdfBuilder::~RscAbsPdfBuilder() {
  // Destructor
  if (_deleteThePdf) delete _thePdf;
}

/*----------------------------------------------------------------------------*/

RooAbsPdf* RscAbsPdfBuilder::getPdf() {
  // Return the built Pdf
  if (!_thePdf) {
    if (verbose())
        cout << "Building Pdf for a " << ClassName() << endl;
    buildPdf();
  }
  assert(_thePdf);
  return _thePdf;
}

/*----------------------------------------------------------------------------*/

char* RscAbsPdfBuilder::dataCard = 0;

void RscAbsPdfBuilder::setDataCard(const char* init) {
  // Static method to set the name of the file to be read for the data card.
  // Before the user does anything, he should make a call such as :
  // RscAbsPdfBuilder::setDataCard("myDataCard.txt");
  if (init) {
    //DP here the code to exec the python file, dump the content must be placed.
    // Handling configuration
    dataCard=ConfigHandler(init);
    //dataCard = new char[strlen(init)+1];
    //strcpy(dataCard, init);
  } else {
    dataCard = 0;
  }

}

/*----------------------------------------------------------------------------*/

const char* RscAbsPdfBuilder::getDataCard() {
  // Return the name of the file containing the data card.
  return dataCard;
}

/*----------------------------------------------------------------------------*/

Bool_t RscAbsPdfBuilder::isDataFlag = false;
Bool_t RscAbsPdfBuilder::isVerboseFlag = false;

RooRealVar* RscAbsPdfBuilder::makeRealVar(const RooRealVar& var) {
  RooRealVar* res;
  RooAbsArg * arg = globalConfig.find(var.GetName());
  if (!arg) {
    res = new RooRealVar(var);
    globalConfig.add(*res);
    return res;
  } else {
    if (res = dynamic_cast<RooRealVar*>(arg)) return res;
    else {
      if (verbose()){
        cout << "RscAbsPdfBuilder::makeRealVar: Found argument of name "
	    << var.GetName()
	    << " which is not a RooRealVar*" << endl;
        arg->Print("v");
        }
      abort();
    }
  }
}

/*----------------------------------------------------------------------------*/

void RscAbsPdfBuilder::fixAllParams(Bool_t fixParams) {
  /*
   * Function to switch between fixing and floating all of the
   * parameters in a PDF associated with this model.
   */
  RooAbsPdf * pdf = getPdf();
  RooArgSet* compos = pdf->getComponents();
  RooArgSet* pdfpar = pdf->getParameters(compos);
 
  TIterator * iter = pdfpar->createIterator();
  TIter next(iter);
  TObject * obj=0;
  while (obj = (TObject *)next())  {
    RooRealVar * var = dynamic_cast<RooRealVar*>(obj);
    if(var) var->setConstant(fixParams);
  }
}

/*----------------------------------------------------------------------------*/

/**
Read a parameter from the card and decide whether it is a constraint or not.
The decision is made this way:
if in the same block there is a string whose name is <par_name>_cosntraint, 
the pointer will point to a Constraint instead of a RooRealVar.
**/

bool RscAbsPdfBuilder::readParameter(TString name, 
                                     TString title,
                                     TString block_name, 
                                     TString datacard_name, 
                                     RooRealVar*& par_ptr,
                                     Double_t rangeMin,
                                     Double_t rangeMax, Double_t defaultValue){

  bool theBool = readParameter(name,title,block_name,datacard_name,par_ptr,defaultValue);
  par_ptr->setRange(rangeMin,rangeMax);
  return theBool;
}

bool RscAbsPdfBuilder::readParameter(TString name, 
                                     TString title,
                                     TString block_name, 
                                     TString datacard_name, 
                                     RooRealVar*& par_ptr, Double_t defaultValue){
  
  TString constr_info_name=name;
  constr_info_name+="_constraint";
  RooStringVar constr_checker (constr_info_name.Data(), "", "");
  RooArgSet(constr_checker).readFromFile(datacard_name, 0, block_name);
  
  bool is_constraint=false;
  
  if (TString(constr_checker.getVal())!=""){
    is_constraint=true;
    if (verbose())
      std::cout << "[RscAbsPdfBuilder::readParameter] "
		<< " Found constraint: " << constr_info_name.Data() 
		<< " = " << constr_checker.getVal() << std::endl;
  }
  
  // Read the RooRealvar anyway.
  RooRealVar* temp_param= new RooRealVar(name,title,defaultValue);
  bool status=RooArgSet(*temp_param).readFromFile(datacard_name,
						  0,
						  block_name);
  
  // now decide what to allocate:
  char constr_checker_clone[100];
  strcpy(constr_checker_clone, constr_checker.getVal());
  if (is_constraint)
    par_ptr = new Constraint(*temp_param,
			     constr_checker_clone);
  else{
    par_ptr = new RooRealVar(*temp_param);
    if (verbose())
      par_ptr->Print("v");
  }

  return status;
}

/*----------------------------------------------------------------------------*/

/**
Handle the configurationfile name. Manage it if it is python or simple ASCII.
**/
char* ConfigHandler(const char* init){

    char py_suffix [] = ".py";
    char* datacard_name;
  
    // No python config
    if (strstr(init,py_suffix)==NULL){
        datacard_name = new char[strlen(init)+1];
        strcpy(datacard_name, init);
        }
    /* Python config. This section might be replaced by a proper interpreter 
       integration..*/
    else {
        
        // a temporary file is created and executed to obtain the plain cfg
        std::string config_translator("");
        config_translator+="import sys\n";
        config_translator+="execfile(sys.argv[1])\n";
        config_translator+="file=open(sys.argv[2],'w')\n";
        config_translator+="file.write('# card automatically generated!\\n'+analysis._dump_to_str())\n";
        config_translator+="file.close()\n";

        // create the file with a random name

        char cfg_interpreter_name[] = "Rsc_cfgint_XXXXXX";
        mkstemp(cfg_interpreter_name);

        ofstream cfg_interpreter;
        cfg_interpreter.open (cfg_interpreter_name);
        cfg_interpreter << config_translator;
        cfg_interpreter.close(); 

        // execute it with the proper arguments
  
        // template for the random configfilename
        char cfg_name[] = "Rsc_cfg_XXXXXX";
        mkstemp(cfg_name);


        std::string command("python ");
        command += cfg_interpreter_name;
        command += " ";
        command += init;
        command += " ";
        command += cfg_name;
        
        std::cout << "Inspecting the analysis object ..." << std::endl;
        std::cout << "  -> Executing: " << command << " ...)" << std::endl;

        system(command.c_str());

        // delete it
        std::string delete_command ("rm ");
        delete_command += cfg_interpreter_name;
        
        std::cout << "Cleaning ..." << std::endl;
        std::cout<<"  -> (Executing: "<<delete_command<<"...)"<< std::endl;
        
        system(delete_command.c_str());

        datacard_name = new char[strlen(cfg_name)+1];
        strcpy(datacard_name, cfg_name);

        }

    return datacard_name;

}

/*----------------------------------------------------------------------------*/
// Removed in CMSSW
// Display on screen the logo of Rsc when the library is loaded.
// bool RscLogo(){
//     std::cout << "\n"
//               << "\033[1;2mRooStatsCms:\033[1;0m A CMS Modelling, combination and Statistics package \n" 
//               << "             www-ekp.physik.uni-karlsruhe.de/~RooStatsCms\n"
//               << "             Universitaet Karlsruhe (2008).\n"
//               << std::endl;
//     return true;
// }
// 
// 
// static bool RscLogo_flag=RscLogo();

/*----------------------------------------------------------------------------*/

// Automatically converted from the standalone version Wed Apr 15 11:36:34 2009
