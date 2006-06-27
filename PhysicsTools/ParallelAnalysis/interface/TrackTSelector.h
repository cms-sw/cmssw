#ifndef TrackTSelector_h
#define TrackTSelector_h
/** \class TrackTSelector
 *
 * Simple interactive analysis example based on TSelector
 * accessing EDM data
 *
 * \author Luca Lista, INFN
 *
 * $Id: TrackTSelector.h,v 1.2 2006/03/20 11:09:21 llista Exp $
 */
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TSelector.h>
#include "DataFormats/TrackReco/interface/Track.h"

class TrackTSelector : public TSelector {
public :
  /// constructor
  TrackTSelector( TTree *tree = 0 ) { }
  /// destructor
  virtual ~TrackTSelector() { }
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
  /// track collection
  reco::TrackCollection tracks;
  /// track branch
  TBranch * tracksBranch;
  /// histograms
  TH1F * h_pt, * h_eta;

  TrackTSelector(TrackTSelector const&);
  TrackTSelector operator=(TrackTSelector const&);
};

#endif
