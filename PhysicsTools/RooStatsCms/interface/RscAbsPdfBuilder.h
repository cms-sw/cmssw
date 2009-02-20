#ifndef __RSCABSPDFBUILDER__
#define __RSCABSPDFBUILDER__

#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "TObject.h"

#ifdef __GNUC__
#if __GNUC__ > 2 
#include <ostream>
#else
#include <ostream.h>
#endif
#else
#include <ostream.h>
#endif

using namespace std;

class RscAbsPdfBuilder : public TObject {
public:
  RscAbsPdfBuilder();
  ~RscAbsPdfBuilder();
  RooAbsPdf* getPdf();

  static void setDataCard(const char* init);
  static const char* getDataCard();

  static Bool_t isData() { return isDataFlag; }
  static void setIsData(Bool_t newIsData = true) { isDataFlag = newIsData; }

  virtual void readDataCard() = 0;
  virtual void writeDataCard(ostream& out) = 0;

  static RooArgSet globalConfig;

  void fixAllParams(Bool_t fixParams=kTRUE);

  static void setVerbose(Bool_t verbosity) { isVerboseFlag = verbosity; }
  static Bool_t verbose() { return isVerboseFlag; }

  bool readParameter(TString name, 
                     TString title,
                     TString block_name, 
                     TString datacard_name, 
                     RooRealVar*& par_ptr,
                     Double_t rangeMin,
                     Double_t rangeMax,
                     Double_t defaultValue=0);

  bool readParameter(TString name, 
                     TString title,
                     TString block_name, 
                     TString datacard_name, 
                     RooRealVar*& par_ptr,
                     Double_t defaultValue=0);

protected:
  virtual void buildPdf() = 0;
  RooAbsPdf* _thePdf; // Pointer to the built Pdf
  RooArgSet* _discVars; // Pointer to a list of discriminating variables
  Bool_t _deleteThePdf;

  RooRealVar* makeRealVar(const RooRealVar& var);

private:

  static char* dataCard; // Name of the file containing the data card
  static Bool_t isDataFlag; // Should we initiate Pdfs with blind variables ?
  static Bool_t isVerboseFlag;

  //ClassDef(RscAbsPdfBuilder,1) // Abstract Pdf Builder mechanism
};

#endif
char* ConfigHandler(const char*);

//------------------------------------------------------------------------------
// old title \mainpage RooStatsCms: A CMS Statistics/Combination/Modelling Package
// Doxygen Main Page
/**

\mainpage .

\image html logobello.png 

\section intro_sec Introduction
RooStatsCms (RSC), based on the 
<a href="http://roofit.sourceforge.net/RooFit" target="blank">RooFit</a> 
technology, is a software framework whose scope is to allow the modelling and 
combination of analysis channels, the accomplishment of statistical studies 
performed through a variety of techniques described in the literature and the 
production of plots by means of sophisticated formatting, drawing and graphics 
manipulation routines (StatisticalPlot). One of the key features of RSC is the 
consistent and complete treatment of the constraints and correlations on and 
among the variables.Each analysis is described in a configuration file, 
separating thus physics inputs from the C++ code. This feature eases the 
sharing of the inputs among the groups and provides an automatic bookkeeping of 
what was done (RscCombinedModel). RSC is therefore meant to complement the 
existing analyses by means of their combination therewith obtaining sharper 
limits and more refined measurements of relevant quantities. The tool is born 
to fulfil the need of the Higgs searches at LHC. Therefore natural quantities 
that can be studied are the Higgs mass or cross section and their limits. For 
what concerns these aspects we were able already to reproduce the results of 
the combinations of analysis channels present in the published CMS Physics 
Technical Design and Report (PTDR). With RSC it is possible to perform detailed 
studies using different statistical methods. For example the profile-likelihood, 
the modified frequentist analysis of search results (CLs technique), the 
unified approach to classical statistical analysis of small signals 
(Feldman-Cousins) and likelihood marginalisation techniques which can be 
performed using the classes provided within the framework.

\section dependencies_sec Dependencies
RooStatsCms depends only on 
<a href="http://root.cern.ch/" target="blank">ROOT</a> 5.22. 
This version of the package has been 
developed and tested with Scientific Linux 4 (Intel(R) Xeon(R) CPU 5150, gcc 
3.1.4 ) and Debian 4.0 Etch (amd64 gcc 4.1.2).

\section getting_started_sec Getting started

\subsection getting_sec Obtain the package
The package can be found in the Cms PhysicsTools CVS 
<a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/
CMSSW/PhysicsTools/RooStatsCms" target="blank">here</a>. 
To check it out type (bash users):
\verbatim
export CVSROOT=<yourusername>@cmscvs.cern.ch:/cvs_server/repositories/CMSSW
export CVS_RSH=ssh
cvs co -r V00-05-02 PhysicsTools/RooStatsCms
\endverbatim
You can have the package as a tarball <a href="http://cmssw.cvs.cern.ch/
cgi-bin/cmssw.cgi/CMSSW/PhysicsTools/
RooStatsCms.tar.gz?view=tar&pathrev=V00-05-02">here</a>.

The head is available also as a tarball <a href="http://cmssw.cvs.cern.ch/
cgi-bin/cmssw.cgi/CMSSW/PhysicsTools/RooStatsCms.tar.gz?view=tar" 
target="blank">here</a>. \n \b Warning: this latter is a developement 
version. It might be unstable. For the latest stable version, 
please refer to the last tag!

\subsection root_afs_subsec Root 5.22 on lxplus
Setting Root on the lxplus machines is easy and works out of the box just copy
and paste these lines (bash shell).

\b Bash \b shell:
\verbatim
export ROOTSYS=/afs/cern.ch/sw/lcg/app/releases/ROOT/5.22.00/slc4_ia32_gcc34/root/
export PATH=$PATH:$ROOTSYS/bin
export LD_LIBRARY_PATH=$ROOTSYS/lib
\endverbatim

\b Tcsh \b shell:
\verbatim
setenv ROOTSYS /afs/cern.ch/sw/lcg/app/releases/ROOT/5.22.00/slc4_ia32_gcc34/root/
setenv LD_LIBRARY_PATH $ROOTSYS/lib
set path=($path $ROOTSYS/bin)
\endverbatim


\subsection compile_sec Compile the package
Just enter the directory RooStatsCms after the checkout or unrolling the 
tarball and type \b make. This will create a library dynamic called 
\e libRooStatsCms.so in the \e lib directory.

\subsection kick_off Kick off
So far, so good: how can we start?
Let's move before each example to the root dir of the RSC installation
\verbatim
cd [yourdir]/PhysicsTools/RooStatsCms
\endverbatim
Here we go.

\subsubsection model_diagram Model Diagram
This program will use a feature of roofit that needs the program \e dot.
Moreover, to inspect the png which is dumped one of these three programs: 
\e display, \e gwenview or \e firefox .\n

Let's start building the diagram of a model (in the style of the one displayed 
in the RscCombinedModel page) using the program 
\e create_diagram.exe. To compile it let's use a pleasant feature of RSC: type 
in the root of the installation 
\verbatim
make exe
\endverbatim
This will turn all the \e *.cpp files in the progs directory in executables 
\e *exe in the bin directory. For more info read the readme in the progs 
directory.\n
And now:
\verbatim
bin/create_diagram.exe macros/examples/example_qqhtt.rsc qqhtt
\endverbatim

\subsubsection model_html Model Html
This program creates for you a small website to let you browse your combined 
model. All the information will be tidily stored with hyper links 
to cross reference the objects created and ease the navigation. Also plots 
using dot will be created. Just run it with:

\verbatim
bin/model_html.exe macros/examples/example_qqhtt.rsc qqhtt
\endverbatim

You could also produce the combined model of the H to ZZ to 4 leptons with the 
\e TDR_HZZ_card_maker.py script in the scripts directory (see the 
\link datacard \endlink Datacards section).

\subsubsection PL Profile Likelihood
Now, after the warm up is time for a statistical method. Let's try to produce 
something with Profile Likelihood.
Change your dir to the examples directory:
\verbatim
cd macros/examples
root profilelikelihood_htt.cxx
\endverbatim
Done. To know more about the plot that popped up, see the codeof the macro you 
have just run: it is fully commented and explains step by step what is done.
You might also see the datacard that describes the model which is studied:
\e example_qqhtt.rsc.
You might have noticed that 2 files were created on disk: 
\e PLScanPlot_68CL_95CL.png and \e PLScanPlot_68CL_95CL.root. This a powerful 
feature of RSC: the statistical plots can be directly dumped to image or their 
entire content saved on rootfiles as single entities.

\subsubsection m2lnQ -2lnQ distributions
Why not producing some -2lnQ distributions starting from the model described in 
\e example_qqhtt.rsc ?
\verbatim
cd macros/examples
root qqhtt_-2lnQ_distributions.cxx
\endverbatim
Done. Again see the code of the macro to know the details!

\subsection datacard Datacards
The syntax of the datacard is very simple. Anyway a "manual" on how to create 
cards is being written. For the time being look at the previous example card: 
you can find there a lot of explainations and comments.
If you feel like having snippets ready for cut/paste and modification, the
script \e create_card_skeleton.py in the scripts directory can be helpful: 
from command line it produces the snippets for your card, according to your 
requests. Run it without arguments to get some help.
Another useful tool is the script \e TDR_HZZ_card_maker.py which is located in 
the scripts directory. This Python script dumps the datacards for the single 
H->ZZ->4l models described in our TDR for the different mass hypotheses. 
The cards that import these single-model cards to produce the combined models 
are created as well. With 99% confidence level, your usecase will be covered!

\section statistical_methods_sec Statistical Methods
- \link LimitCalculator The CLs Method \endlink 
- \link PLScan The Profile Likelihood Method \endlink 
- \link FCCalculator The Feldman Cousins unified approach \endlink (being tested for validation)

\section plotting_routines Plotting Routines
- The plot of the CLs method results: LimitPlot
- The plot of the Profile likelihood scan method / Feldman-Cousins results: PLScanPlot
- The -2lnQ band plot "a la LEP": LEPBandPlot
- The SM exclusion plot "a la TEVATRON": ExclusionBandPlot

\section advanced_topics Advanced topics

\subsection CLs_exclusion Exclusion calculating the CLs quantity
A very general program, ratio_finder.exe, 
scans a variable in the model inside a max and a 
min value so to obtain a desired CLs value. 
The class used inside the program is RatioFinder.
At the end of the procedure the RatioFinderResults object is written to a root 
file and a png image of the evolution of the scan is dumped.
Useful to answer questions like:
 - I want to do a green/yellow plot "a la Tevatron" where I exclude a certain R factor on the SM cross section
 - I want to know how much lumi I need for a CLs of X percent


\section more_material More material
More documentaion and material can be found in the Twiki of RooStatsCms:
https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWGRooStatsCms
Many presentations are on Indico as well (bottom of the link above).


Authors:

- Gregory Schott: gregory.schott<at>cern.ch
- Danilo Piparo: danilo.piparo<at>cern.ch
- Guenter Quast

**/
