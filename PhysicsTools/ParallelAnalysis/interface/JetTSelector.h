#ifndef JetTSelector_h
#define JetTSelector_h
/** \class JetTSelector
 *
 * Simple interactive analysis example based on TSelector
 * accessing EDM data
 *
 * \author Luca Lista, INFN
 *
 * $Id$
 */
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TSelector.h>
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Wrapper.h"

typedef reco::CandidateCollection JetCollection;

class JetTSelector : public TSelector {
public :
  /// constructor
  JetTSelector( TTree *tree = 0 ) { }
  /// destructor
  virtual ~JetTSelector() { }
  /// return version number
  virtual Int_t Version() const { return 1; }
  /// begin of processing: book histograms
  virtual void  Begin( TTree *tree );
  /// begin slave processing: call Init
  virtual void  SlaveBegin( TTree *tree );
  /// init: attach branches
  virtual void  Init(TTree *tree);
  /// notify: get branches
  virtual bool  Notify();
  /// process one event entry: fill histograms
  virtual bool  Process( long long entry );
  /// set options
  virtual void  SetOption( const char * option ) { fOption = option; }
  /// set object
  virtual void  SetObject( TObject *obj ) { fObject = obj; }
  /// set input list
  virtual void  SetInputList( TList *input ) { fInput = input; }
  /// get input list
  virtual TList * GetOutputList() const { return fOutput; }
  /// terminate slave processing: do nothing
  virtual void  SlaveTerminate();
  /// terminate processing: save histograms
  virtual void  Terminate();

private:
  /// root file chain 
  TTree * chain;
  /// jet collection
  JetCollection jets;
  /// jet branch
  TBranch * jetsBranch;
  /// histograms
  TH1F * h_et, * h_eta;

  JetTSelector(JetTSelector const&);
  JetTSelector operator=(JetTSelector const&);
};

#endif
