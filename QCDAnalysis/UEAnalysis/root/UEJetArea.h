#ifndef UEJetArea_h
#define UEJetArea_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TFile.h>

#include<TLorentzVector.h>

#include <TH1F.h>
#include <TH2D.h>
#include <TProfile.h>

#include <TClonesArray.h>

// FastJet includes
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/ClusterSequenceActiveArea.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/SISConePlugin.hh"

using namespace std;

///
///___________________________________________________________________
///
class UEJetWithArea {
  public :

  UEJetWithArea();
  UEJetWithArea( TLorentzVector&, double, unsigned int ); 
  ~UEJetWithArea() 
    {
      //      cout << "~UEJetWithArea()" << endl;
      //      delete _momentum;
    }
  TLorentzVector* GetMomentum()        { return _momentum          ; }
  double          GetArea()            { return _area              ; }
  unsigned int    GetNConstituents()   { return _nconstituents     ; }

  private :

  TLorentzVector* _momentum;
  double       _area;
  unsigned int _nconstituents;   
};

///
///___________________________________________________________________
///
class UEJetAreaFinder {
 public :
   
   /// input:
   /// eta-range
   /// pT-threshold
   /// choice of jet-algorithm
  UEJetAreaFinder( float, float, string );
  ~UEJetAreaFinder()
    {
      //      cout << "~UEJetAreaFinder()" << endl;

      //delete mPlugin;
      delete mJetDefinition;
      //delete mActiveArea;
      delete theGhostedAreaSpec;
      delete theAreaDefinition;
    }
  
 float etaRegionInput;

 float etaRegion;
 float ptThreshold;

 /// findJets :
 ///
 /// input-collection (e.g. tracks, particles, ...) (TClonesArray)
 Bool_t find( TClonesArray&, vector<UEJetWithArea>& );

 ///_______________________________________________________________________
 ///
 /// FastJet package
 ///
 /// written by Matteo Cacciari, Gavin Salam and Gregory Soyez
 /// http://www.lpthe.jussieu.fr/~salam/fastjet/
 ///
 /// Phys. Lett. B 641 (2006) 57 (FastJet package)
 /// arXiv:0802.1188 (jet area)
 ///

 /// define jet algorithm
 double coneRadius;           // SISCone
 double coneOverlapThreshold; // SISCone
 int    maxPasses;            // SISCone
 double protojetPtMin;        // SISCone
 bool   caching;              // SISCone
 double rParam;               // kT
 fastjet::Strategy       fjStrategy;
 fastjet::SISConePlugin::SplitMergeScale scale;
 fastjet::SISConePlugin* mPlugin;
 fastjet::JetDefinition* mJetDefinition;

 /// jet areas
 double ghostEtaMax;
 int activeAreaRepeats;
 double ghostArea;
 fastjet::GhostedAreaSpec* theGhostedAreaSpec;
 fastjet::AreaDefinition*  theAreaDefinition;
};

///
///___________________________________________________________________
///
class UEJetAreaHistograms {
  public :

  UEJetAreaHistograms( const char*, string* );
  ~UEJetAreaHistograms() 
    { 
      file->Write(); 
      file->Close(); 

      delete[] h_pTAllJets           ; // all jets
      delete[] h_etaAllJets          ;
      delete[] h_areaAllJets         ;
      delete[] h_ptByAreaAllJets     ;
      delete[] h_nConstituentsAllJets;
      delete[] h2d_pTAllJets_vs_pTjet           ;
      delete[] h2d_areaAllJets_vs_pTjet         ;
      delete[] h2d_ptByAreaAllJets_vs_pTjet     ;
      delete[] h2d_nConstituentsAllJets_vs_pTjet;

      delete[] h_pTJet           ; // leading jet
      delete[] h_etaJet          ;
      delete[] h_areaJet         ;
      delete[] h_ptByAreaJet     ;
      delete[] h_nConstituentsJet;
      delete[] h2d_areaJet_vs_pTjet         ;
      delete[] h2d_ptByAreaJet_vs_pTjet     ;
      delete[] h2d_nConstituentsJet_vs_pTjet;

      delete[] h_medianPt      ; // event-by-event medians
      delete[] h_medianArea    ;
      delete[] h_medianPtByArea;
      delete[] h2d_medianPt_vs_pTjet      ;
      delete[] h2d_medianArea_vs_pTjet    ;
      delete[] h2d_medianPtByArea_vs_pTjet;

      delete[] subdir;
    };

  void fill( vector<UEJetWithArea>& );
  void fill( vector<UEJetWithArea>&, TClonesArray& );

  private :

  TFile* file;
  TDirectory** subdir;

  string HLTBitNames[11];

  TH1D** h_pTAllJets           ; // all jets
  TH1D** h_etaAllJets          ;
  TH1D** h_areaAllJets         ;
  TH1D** h_ptByAreaAllJets     ;
  TH1D** h_nConstituentsAllJets;
  TH2D** h2d_pTAllJets_vs_pTjet           ;
  TH2D** h2d_areaAllJets_vs_pTjet         ;
  TH2D** h2d_ptByAreaAllJets_vs_pTjet     ;
  TH2D** h2d_nConstituentsAllJets_vs_pTjet;

  TH1D** h_pTJet           ; // leading jet
  TH1D** h_etaJet          ;
  TH1D** h_areaJet         ;
  TH1D** h_ptByAreaJet     ;
  TH1D** h_nConstituentsJet;
  TH2D** h2d_areaJet_vs_pTjet         ;
  TH2D** h2d_ptByAreaJet_vs_pTjet     ;
  TH2D** h2d_nConstituentsJet_vs_pTjet;

  TH1D** h_medianPt      ; // event-by-event medians
  TH1D** h_medianArea    ;
  TH1D** h_medianPtByArea;
  TH2D** h2d_medianPt_vs_pTjet      ;
  TH2D** h2d_medianArea_vs_pTjet    ;
  TH2D** h2d_medianPtByArea_vs_pTjet;


};



#endif
