~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~

Here are described analyzers and programs used by UE-charged jet analysis to
produce validation plots and understanding of datasets and runs inside datasets.

Three logical groups of programs are used; the product of each group is given as
input to the subsequent one:

1) NTUPLE CREATION FROM ORIGINAL CMSSW FILES
2) CREATION OF VALIDATION HISTOGRAM, EXPLORATION OF NTUPLE
3) PRESENTATION PLOTS, UTILITIES FOR HISTOS COMPARISONS

~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~

____________________________________________
1) NTUPLE CREATION FROM ORIGINAL CMSSW FILES
____________________________________________

  a) analyzer:           Event_Ntuplizer.cc
       description:
       a file is created containing a Tree filled by event. For each Event some
       interesting variables are filled:
       --> of the event: runNumber, LumisectionBlock, bunchCrossing number
       --> of primary vertices (PV) in the event;
       --> of tracks in the event;  
    PV and tracks properties are filled as arrays: for these reason just Drawing
    by line (or through a TTreeViewer) may result in strange things. Proper
    histogrammation is rendered running programs in chapter 2).
    
  b)configuration file: eventntuplizer_cfg.py
      description:
      b.1) mind the kind of collections you want to take: in the early data taking lot
      of different objects have been created for the same class
      (RECO,EXPRESS,TOBONLY,...).
  
      b.2)OnlyRECO: running on data you have to set this to True, running on MC you
      can set this to False if needed: it will access generator level informations.
  
      b.3)the process process.L1T1 is filtering teh dataset for the specified L1
      trigger strea chosen (process.L1T1.L1SeedsLogicalExpression)
  
__________________________________________________________
2) CREATION OF VALIDATION HISTOGRAM, EXPLORATION OF NTUPLE
__________________________________________________________

!!! Here we perform OUR STANDARD CUTS ON TRACK AND EVENT SELECTION, and collect information previously and
after applying cuts. !!!

a)The template program: 
>>template_ntupleViewer.C
 is prepared, which makes histograms from the Tree previously produced.
The program is configurables in some parameters here listed (if something will
miss will be recognizable maintainig the same "syntax") tha t need to be tuned
on desire of the runner:
STRINGrealdata   : true=>real data, false=>MC;
STRINGbeamspot   : the value of the BeamSpot set in the dataset 
STRINGhistname   : a prefix to put in front of histogram name (like DATA_ or MC_)
STRINGinputfile  : the ntuple previously produced
STRINGmarkerstyle: well, you know what I mean...
STRINGoutfile    : clear, no?

A part of adding/changing histograms, YOU DON"T NEED TO TOUCH THE TEMPLATE!!
All these parameters are configurable in the script
>>plotScript.csh
which you have to run: ./plotScript.csh <nameOfFolderToCreate>

You will obtain the folder "nameOfFolderToCreate_dateAndTimeOfCreation"
containing a copy of the ntuple used and the output histogram file.

b.1) ntupleViewer_Chain_RealData.C 

  in case you have submitted the ntuple creation under CRAB and you have obtained
  a list of root files (and not just a single file) run this program. It will
  "TChain" the list of *.root in which the ntuple TTree is fragmented: you just
  have to specify the path where the .root files are located (typically the /res
  folder of the CRAB job if you have getoutput-ed the job).
  The parametrs previously specifyed (STRINGsomething) are now set directly in the
  program: don't forget to modify them as you need.

For run:
bash$: g++ -pthread -m32 
-I/afs/cern.ch/cms/sw/slc4_ia32_gcc345/lcg/root/5.22.00d-cms4/include 
-L/afs/cern.ch/cms/sw/slc4_ia32_gcc345/lcg/root/5.22.00d-cms4/lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint
-lPostscript -lMatrix -lPhysics -lMathCore -lThread -lz -pthread -lm
-ldl -rdynamic ntupleViewer_Chain_RealData.C -o ntplueViewer

bash$: ./ntplueViewer RunNumber

RunNumber is Run number that you want to analyze. (bash$: ./ntplueViewer 
125296) 


b.2) ntupleViewer_Chain_MC.C:
  As b.1, but in addition teh reference 'validation plots' are divided also for different kind processes (HC, SD, DD)

b.1.1) dati_RealData.h
b.2.1) dati_MC.h

included in the "_Chain_" programs, they contain the structures needed to open
and retrieve the Tree's informations.

_______________________________________________________
3) PRESENTATION PLOTS, UTILITIES FOR HISTOS COMPARISONS
_______________________________________________________

a) finalPots.C
input: the histograms files previously produced, both MC and RealData;
output: images files (.png, .gif, .eps, .pdf)

Four functions:
>>plotHistlog(TH1D* hist1,TH1D* hist2,TH1D* hist3,TH1D* hist4,std::string name);
>>void plotHist(TH1D* hist1,TH1D* hist2,TH1D* hist3,TH1D* hist4,std::string name);
>>void plotHistlog(TH1D* hist1,TH1D* hist2,std::string name);
>>void plotHist(TH1D* hist1,TH1D* hist2,std::string name);
perform the final ploting, comparing RealData without selection and with All
selections (both in linear and log scale): analogous plots are performed comparing
the same histos with MC predictions (both with and without selections, linear
and log scale)
