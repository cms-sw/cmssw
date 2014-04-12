#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <boost/tokenizer.hpp>


#include <TChain.h>
#include <TFile.h>
#include <TProfile2D.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>



struct EcalTPGVariables
{
  // event variables
  unsigned int runNb ;
  unsigned int evtNb ;
  unsigned int bxNb ;
  unsigned int orbitNb ;
  unsigned int nbOfActiveTriggers ;
  int activeTriggers[128] ;
  
  // tower variables
  unsigned int nbOfTowers ; //max 4032 EB+EE
  int ieta[4032] ;
  int iphi[4032] ;
  int nbOfXtals[4032] ;
  int rawTPData[4032] ;
  int rawTPEmul1[4032] ;
  int rawTPEmul2[4032] ;
  int rawTPEmul3[4032] ;
  int rawTPEmul4[4032] ;
  int rawTPEmul5[4032] ;
  float eRec[4032] ;
} ;

//! branch addresses settings
void setBranchAddresses (TChain * chain, EcalTPGVariables & treeVars) {
   chain->SetBranchAddress ("runNb",&treeVars.runNb) ; 
   chain->SetBranchAddress ("evtNb",&treeVars.evtNb) ; 
   chain->SetBranchAddress ("bxNb",&treeVars.bxNb) ; 
   chain->SetBranchAddress ("orbitNb",&treeVars.orbitNb) ; 
   chain->SetBranchAddress ("nbOfActiveTriggers",&treeVars.nbOfActiveTriggers) ; 
   chain->SetBranchAddress ("activeTriggers",treeVars.activeTriggers) ; 

   chain->SetBranchAddress ("nbOfTowers",&treeVars.nbOfTowers) ; 
   chain->SetBranchAddress ("ieta",treeVars.ieta) ; 
   chain->SetBranchAddress ("iphi",treeVars.iphi) ; 
   chain->SetBranchAddress ("nbOfXtals",treeVars.nbOfXtals) ; 
   chain->SetBranchAddress ("rawTPData",treeVars.rawTPData) ; 
   chain->SetBranchAddress ("rawTPEmul1",treeVars.rawTPEmul1) ; 
   chain->SetBranchAddress ("rawTPEmul2",treeVars.rawTPEmul2) ; 
   chain->SetBranchAddress ("rawTPEmul3",treeVars.rawTPEmul3) ; 
   chain->SetBranchAddress ("rawTPEmul4",treeVars.rawTPEmul4) ; 
   chain->SetBranchAddress ("rawTPEmul5",treeVars.rawTPEmul5) ; 
   chain->SetBranchAddress ("eRec",treeVars.eRec) ; 
}

void printHelp()
{
  std::cout << "\n Help" << std::endl ;
  std::cout << " -h : display help" << std::endl ;
  std::cout << " -i <input root files containing the TPG tree>" << std::endl ;     
  std::cout << "    2 possible formats when more than 1 file:" << std::endl ;
  std::cout << "    - files names separated by comma without blanks: file1.root,file2.root,file3.root" << std::endl ;
  std::cout << "    - files names using the wildcard \":\" file:1:3:.root" << std::endl ;
  std::cout << "      with this last format, the loop from file1.root to file3.root is performed (see examples below)" << std::endl ;
  std::cout << " -d <directory containing the input root files, default=ignored>" << std::endl ;     
  std::cout << " -o <output root file name, default=histoTPG.root>" << std::endl ;
  std::cout << " -v <verbosity level(int), default=0 (minimal)>" << std::endl ;
  std::cout << " -l1 <L1 algo bits required. If several, use a comma with no blank. Default=select all>" << std::endl ;
  std::cout << " Additional long options are:" << std::endl ;
  std::cout << " --cutTPOccup <minimal value of TP to fill the occupancy plot, default=0>" << std::endl ;

  std::cout << "\n Example:" << std::endl ;
  std::cout << "1) TPGTreeReder -o myfile -l1 16,46 -i file1,file2" << std::endl ;
  std::cout << "will produce histo in file myfile, selecting only events triggered "<< std::endl ;
  std::cout << "by algo bit 16 and 46 and using as inputs the file1 and file2" << std::endl ;
  std::cout << "2) TPGTreeReder -d rfio:/castor/cern.ch/user/p/paganini/67977 -i run67977_:1:18:.root" << std::endl ;
  std::cout << "will produce histo in the default file histoTPG.root, selecting all events"<< std::endl ;
  std::cout << "and using as inputs the 18 files run67977_1.root to run67977_18.root" << std::endl ;
  std::cout << "located in the castor directory /castor/cern.ch/user/p/paganini " << std::endl ;

}

int getEt(int val) {return (val&0xff) ;}

unsigned int getFg(unsigned int val) {return ((val&0x100)!=0) ;}

unsigned int getTtf(unsigned int val) {return ((val>>9)&0x7) ;}

std::vector<std::string> split(std::string msg, std::string separator)
{
  boost::char_separator<char> sep(separator.c_str());
  boost::tokenizer<boost::char_separator<char> > tok(msg, sep );
  std::vector<std::string> token ;
  for ( boost::tokenizer<boost::char_separator<char> >::const_iterator i = tok.begin(); i != tok.end(); ++i ) {
    token.push_back(std::string(*i)) ;
  }
  return token ;
}

double getEta(int ietaTower) 
{
  // Paga: to be confirmed, specially in EE:
  return 0.0174*fabs(ietaTower) ;
}



///////  Main program /////////

int main (int argc, char** argv)
{

  if (argc < 3) {
    printHelp() ;
    exit (1) ;
  }

  std::string inputfiles, inputdir ;
  std::string outputRootName = "histoTPG.root" ;
  int verbose = 0 ;
  int occupancyCut = 0 ;
  std::string l1algo ; 

  bool ok(false) ;
  for (int i=0 ; i<argc ; i++) {
    if (argv[i] == std::string("-h") ) {
      printHelp() ;
      exit(1);
    }
    if (argv[i] == std::string("-i") && argc>i+1) {
      ok = true ;
      inputfiles = argv[i+1] ;
    }
    if (argv[i] == std::string("-d") && argc>i+1) inputdir = argv[i+1] ;
    if (argv[i] == std::string("-o") && argc>i+1) outputRootName = argv[i+1] ;
    if (argv[i] == std::string("-v") && argc>i+1) verbose = atoi(argv[i+1]) ;
    if (argv[i] == std::string("-l1") && argc>i+1) l1algo =  std::string(argv[i+1]) ;
    if (argv[i] == std::string("--cutTPOccup") && argc>i+1) occupancyCut = atoi(argv[i+1]) ;
  }
  if (!ok) {
    std::cout<<"No input files have been given: nothing to do!"<<std::endl ;
    printHelp() ;
    exit(1);
  }
  
  std::vector<int> algobits ;
  std::vector<std::string> algos = split(l1algo,",") ;
  for (unsigned int i=0 ; i<algos.size() ; i++) algobits.push_back(atoi(algos[i].c_str())) ;


  unsigned int ref = 2 ;



  ///////////////////////
  // book the histograms
  ///////////////////////

  TH2F * occupancyTP = new TH2F("occupancyTP", "Occupancy TP data", 72, 1, 73, 38, -19, 19) ;
  occupancyTP->GetYaxis()->SetTitle("eta index") ;
  occupancyTP->GetXaxis()->SetTitle("phi index") ;
  TH2F * occupancyTPEmul = new TH2F("occupancyTPEmul", "Occupancy TP emulator", 72, 1, 73, 38, -19, 19) ;
  occupancyTPEmul->GetYaxis()->SetTitle("eta index") ;
  occupancyTPEmul->GetXaxis()->SetTitle("phi index") ;

  TH1F * TP = new TH1F("TP", "TP", 256, 0., 256.) ;
  TP->GetXaxis()->SetTitle("TP (ADC)") ;
  TH1F * TPEmul = new TH1F("TPEmul", "TP Emulator", 256, 0., 256.) ;
  TPEmul->GetXaxis()->SetTitle("TP (ADC)") ;
  TH1F * TPEmulMax = new TH1F("TPEmulMax", "TP Emulator max", 256, 0., 256.) ;
  TPEmulMax->GetXaxis()->SetTitle("TP (ADC)") ;
  TH3F * TPspectrumMap3D = new TH3F("TPspectrumMap3D", "TP data spectrum map", 72, 1, 73, 38, -19, 19, 256, 0., 256.) ;
  TPspectrumMap3D->GetYaxis()->SetTitle("eta index") ;
  TPspectrumMap3D->GetXaxis()->SetTitle("phi index") ;

  TH1F * TPMatchEmul = new TH1F("TPMatchEmul", "TP data matching Emulator", 7, -1., 6.) ;
  TH1F * TPEmulMaxIndex = new TH1F("TPEmulMaxIndex", "Index of the max TP from Emulator", 7, -1., 6.) ;
  TH3I * TPMatchEmul3D = new TH3I("TPMatchEmul3D", "TP data matching Emulator", 72, 1, 73, 38, -19, 19, 7, -1, 6) ;
  TPMatchEmul3D->GetYaxis()->SetTitle("eta index") ;
  TPMatchEmul3D->GetXaxis()->SetTitle("phi index") ;

  TH2I * ttfMismatch = new TH2I("ttfMismatch", "TTF mismatch map",  72, 1, 73, 38, -19, 19) ;
  ttfMismatch->GetYaxis()->SetTitle("eta index") ;
  ttfMismatch->GetXaxis()->SetTitle("phi index") ;

  ///////////////////////
  // Chain the trees:
  ///////////////////////

  TChain * chain = new TChain ("EcalTPGAnalysis") ;
  std::vector<std::string> files ;
  if (inputfiles.find(std::string(",")) != std::string::npos) files = split(inputfiles,",") ;
  if (inputfiles.find(std::string(":")) != std::string::npos) {
    std::vector<std::string> filesbase = split(inputfiles,":") ;
    if (filesbase.size() == 4) {
      int first = atoi(filesbase[1].c_str()) ;
      int last = atoi(filesbase[2].c_str()) ;
      for (int i=first ; i<=last ; i++) {
	std::stringstream name ;
	name<<filesbase[0]<<i<<filesbase[3] ;
	files.push_back(name.str()) ;
      }
    }
  }
  for (unsigned int i=0 ; i<files.size() ; i++) {
    files[i] = inputdir+"/"+files[i] ;
    std::cout<<"Input file: "<<files[i]<<std::endl ;
    chain->Add (files[i].c_str()) ;
  }

  EcalTPGVariables treeVars ;
  setBranchAddresses (chain, treeVars) ;

  int nEntries = chain->GetEntries () ;
  std::cout << "Number of entries: " << nEntries <<std::endl ;    



  ///////////////////////
  // Main loop over entries
  ///////////////////////

  for (int entry = 0 ; entry < nEntries ; ++entry) {
    chain->GetEntry (entry) ;
    if (entry%1000==0) std::cout <<"------> "<< entry+1 <<" entries processed" << " <------\n" ; 
    if (verbose>0) std::cout<<"Run="<<treeVars.runNb<<" Evt="<<treeVars.runNb<<std::endl ;

    // trigger selection if any
    bool keep(false) ;
    if (!algobits.size()) keep = true ; // keep all events when no trigger selection
    for (unsigned int algo = 0 ; algo<algobits.size() ; algo++)
      for (unsigned int ntrig = 0 ; ntrig < treeVars.nbOfActiveTriggers ; ntrig++)
	if (algobits[algo] == treeVars.activeTriggers[ntrig]) keep = true ;
    if (!keep) continue ;
    
             
    // loop on towers
    for (unsigned int tower = 0 ; tower < treeVars.nbOfTowers ; tower++) {

      int tp = getEt(treeVars.rawTPData[tower]) ;
      int emul[5] = {getEt(treeVars.rawTPEmul1[tower]),
		     getEt(treeVars.rawTPEmul2[tower]),
		     getEt(treeVars.rawTPEmul3[tower]),
		     getEt(treeVars.rawTPEmul4[tower]),
		     getEt(treeVars.rawTPEmul5[tower])} ;
      int maxOfTPEmul = 0 ;
      int indexOfTPEmulMax = -1 ;
      for (int i=0 ; i<5 ; i++) if (emul[i]>maxOfTPEmul) {
	maxOfTPEmul = emul[i] ; 
	indexOfTPEmulMax = i ;
      }
      int ieta = treeVars.ieta[tower] ;
      int iphi = treeVars.iphi[tower] ;
      int nbXtals = treeVars.nbOfXtals[tower] ;
      int ttf = getTtf(treeVars.rawTPData[tower]) ;


      if (verbose>9 && (tp>0 || maxOfTPEmul>0)) {
	std::cout<<"(phi,eta, Nbxtals)="<<std::dec<<iphi<<" "<<ieta<<" "<<nbXtals<<std::endl ;
	std::cout<<"Data Et, TTF: "<<tp<<" "<<ttf<<std::endl ;
	std::cout<<"Emulator: " ;
	for (int i=0 ; i<5 ; i++) std::cout<<emul[i]<<" " ;
	std::cout<<std::endl ;
      }


      // Fill TP spctrum
      TP->Fill(tp) ;
      TPEmul->Fill(emul[ref]) ;
      TPEmulMax->Fill(maxOfTPEmul) ;
      TPspectrumMap3D->Fill(iphi, ieta, tp) ;


      // Fill TP occupancy
      if (tp>occupancyCut) occupancyTP->Fill(iphi, ieta) ;
      if (emul[ref]>occupancyCut) occupancyTPEmul->Fill(iphi, ieta) ;


      // Fill TP-Emulator matching
      // comparison is meaningful when:
      if (tp>0 && nbXtals == 25) {
	bool match(false) ;
	for (int i=0 ; i<5 ; i++) {
	  if (tp == emul[i]) {
	    TPMatchEmul->Fill(i+1) ;
	    TPMatchEmul3D->Fill(iphi, ieta, i+1) ;
	    match = true ;
	  }
	}
	if (!match) {
	  TPMatchEmul->Fill(-1) ;
	  TPMatchEmul3D->Fill(iphi, ieta, -1) ;
	  if (verbose>5) {
	    std::cout<<"MISMATCH"<<std::endl ;
	    std::cout<<"(phi,eta, Nbxtals)="<<std::dec<<iphi<<" "<<ieta<<" "<<nbXtals<<std::endl ;
	    std::cout<<"Data Et, TTF: "<<tp<<" "<<ttf<<std::endl ;
	    std::cout<<"Emulator: " ;
	    for (int i=0 ; i<5 ; i++) std::cout<<emul[i]<<" " ;
	    std::cout<<std::endl ;
	  }
	}
      }
      if (maxOfTPEmul>0) TPEmulMaxIndex->Fill(indexOfTPEmulMax+1) ;


      // Fill TTF mismatch
      if ((ttf==1 || ttf==3) && nbXtals != 25) ttfMismatch->Fill(iphi, ieta) ;


    } // end loop towers


  } // endloop entries

  

  ///////////////////////
  // Format & write histos
  ///////////////////////


  // 1. TP Spectrum  
  TProfile2D * TPspectrumMap = TPspectrumMap3D->Project3DProfile("yx") ;
  TPspectrumMap->SetName("TPspectrumMap") ;

  // 2. TP Timing
  TH2F * TPMatchEmul2D = new TH2F("TPMatchEmul2D", "TP data matching Emulator", 72, 1, 73, 38, -19, 19) ;
  TH2F * TPMatchFraction2D = new TH2F("TPMatchFraction2D", "TP data: fraction of non-single timing", 72, 1, 73, 38, -19, 19) ;
  TPMatchEmul2D->GetYaxis()->SetTitle("eta index") ; 
  TPMatchEmul2D->GetXaxis()->SetTitle("phi index") ;
  TPMatchEmul2D->GetZaxis()->SetRangeUser(-1,6) ;
  TPMatchFraction2D->GetYaxis()->SetTitle("eta index") ; 
  TPMatchFraction2D->GetXaxis()->SetTitle("phi index") ;
  for (int binx=1 ; binx<=72 ; binx++)    
    for (int biny=1 ; biny<=38 ; biny++) {
      int maxBinz = 5 ;
      double maxCell = TPMatchEmul3D->GetBinContent(binx, biny, maxBinz) ;
      double totalCell(0) ;
      for (int binz=1; binz<=7 ; binz++) {
	double content = TPMatchEmul3D->GetBinContent(binx, biny, binz) ;
	if (content>maxCell) {
	  maxCell = content ;
	  maxBinz = binz ;
	}
	totalCell += content ;
      }
      if (maxCell <=0) maxBinz = 2 ; // empty cell
      TPMatchEmul2D->SetBinContent(binx, biny, float(maxBinz)-2.) ; //z must be in [-1,5] 
      double fraction = 0 ;
      if (totalCell>0) fraction = 1.- maxCell/totalCell ;
      TPMatchFraction2D->SetBinContent(binx, biny, fraction) ;
      if (totalCell > maxCell && verbose>9) {
	std::cout<<"--->"<<std::endl ;	
	for (int binz=1; binz<=7 ; binz++) {	  
	  std::cout<< "(phi,eta, z): (" 
		   << TPMatchEmul3D->GetXaxis()->GetBinLowEdge(binx) 
		   << ", " << TPMatchEmul3D->GetYaxis()->GetBinLowEdge(biny) 
		   << ", " << TPMatchEmul3D->GetZaxis()->GetBinLowEdge(binz)		   
		   << ") Content="<<TPMatchEmul3D->GetBinContent(binx, biny, binz)		   
		   << ", erro="<<TPMatchEmul3D->GetBinContent(binx, biny, binz)	   
		   << std::endl ;	
	}
      }
    }



  TFile saving (outputRootName.c_str (),"recreate") ;
  saving.cd () ;
  
  occupancyTP->Write() ;
  occupancyTPEmul->Write() ;
  
  TP->Write() ;
  TPEmul->Write() ;
  TPEmulMax->Write() ;
  TPspectrumMap->Write() ;

  TPMatchEmul->Write() ; 
  TPMatchEmul3D->Write() ; 
  TPEmulMaxIndex->Write() ;
  TPMatchEmul2D->Write() ; 
  TPMatchFraction2D->Write() ; 

  ttfMismatch->Write() ; 

     
  saving.Close () ;
  delete chain ;

  return 0 ;
}


