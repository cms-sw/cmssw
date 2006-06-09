#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

using namespace std;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( MonitorUserInterface* mui ) 
  : mui_(mui),
    cme_(),
    task_(sistrip::UNKNOWN_TASK) {
  cout << "[CommissioningHistograms::CommissioningHistograms]" 
       << " Created base object!" << endl;
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  cout << "[CommissioningHistograms::~CommissioningHistograms]" << endl;
  mui_->save("client.root");
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollateMEs() {
  cout << "[CommissioningHistograms::createCollateMEs]" << endl;
  
  mui_->subscribeNew( "*" ); //@@ temporary?
  vector<string> added_contents;
  mui_->getAddedContents( added_contents );
  
  if ( added_contents.empty() ) { 
    cout << "[CommissioningHistograms::createCollateMEs]"
	 << " No 'added contents' found!" << endl;
    return; 
  } else {  
    cout << "[CommissioningHistograms::createCollateMEs]"
	 << " Found " << added_contents.size () 
	 << " 'added contents' found!" << endl;
  }

  vector<string>::iterator idir;
  for ( idir = added_contents.begin(); idir != added_contents.end(); idir++ ) {
    
    // Extract directory paths
    string collector_dir = idir->substr( 0, idir->find(":") );
    const SiStripHistoNamingScheme::ControlPath& path = SiStripHistoNamingScheme::controlPath( collector_dir );
    string client_dir = SiStripHistoNamingScheme::controlPath( path.fecCrate_,
							       path.fecSlot_,
							       path.fecRing_,
							       path.ccuAddr_,
							       path.ccuChan_ );
    cout << "[CommissioningHistograms::createCollateMEs]"
	 << " FEC crate/ring/slot, CCU addr/chan: "
	 << path.fecCrate_ << "/"
	 << path.fecSlot_ << "/"
	 << path.fecRing_ << ", "
	 << path.ccuAddr_ << "/"
	 << path.ccuChan_ << endl;

    if ( path.fecCrate_ == sistrip::all_ ||
	 path.fecSlot_ == sistrip::all_ ||
	 path.fecRing_ == sistrip::all_ ||
	 path.ccuAddr_ == sistrip::all_ ||
	 path.ccuChan_ == sistrip::all_ ) { continue; } 
    
    // Retrieve MonitorElements from pwd directory
    mui_->setCurrentFolder( collector_dir );
    vector<string> me = mui_->getMEs();
    cout << "[CommissioningHistograms::createCollateMEs]"
	 << " Found " << me.size() << " MEs in " << collector_dir << endl;
    
    uint16_t n_cme = 0;
    vector<string>::iterator ime = me.begin(); 
    for ( ; ime != me.end(); ime++ ) {
      string cme_name = *ime;
      string cme_title = *ime;
      string cme_dir = client_dir;
      string search_str = "*/" + client_dir + *ime;
      if ( find( cme_.begin(), cme_.end(), search_str ) == cme_.end() ) {
 	CollateMonitorElement* cme = mui_->collate1D( cme_name, cme_title, cme_dir );
	mui_->add( cme, search_str );
	cme_.push_back( cme_dir );
	n_cme++;
	cout << "[CommissioningHistograms::createCollateMEs]"
	     << " CollateME number " << cme_.size() 
	     << " CollateME name " << cme_name
	     << " CollateME title " << cme_title
	     << " CollateME dir " << cme_dir
	     << " CollateME search string " << search_str
	     << endl;
      }
    }
    cout << "[CommissioningHistograms::createCollateMEs]"
	 << "  Created " << n_cme << " CollateMEs in " << client_dir << endl;
  }

}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createProfileHistos() {
  cout << "[CommissioningHistograms::createProfileHistos]" << endl;
  if ( mui_ ) { 
    vector<string> dirs;
    getListOfDirs( dirs );
    vector<string>::const_iterator idir = dirs.begin();
    for ( ; idir != dirs.end(); idir++ ) {
      mui_->setCurrentFolder( *idir );
      vector<string> me = mui_->getMEs();
      book(me); 
      update();
    }
  }
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createSummaryHistos() {
  cout << "[CommissioningHistograms::createSummaryHistos] Nothing done!" << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createTrackerMap() {
  cout << "[CommissioningHistograms::createTrackerMap] Nothing done!" << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::uploadToConfigDb() {
  cout << "[CommissioningHistograms::uploadToConfigDb] Nothing done!" << endl;
}


// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::getListOfDirs( vector<string>& dirs ) {
  dirs.clear();
  mui_->setCurrentFolder( sistrip::root_ );
  cdIntoDir( mui_->pwd(), dirs );
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::cdIntoDir( const string& dir, vector<string>& dirs ) {
  const vector<string> sub_dirs = mui_->getSubdirs();
  if ( sub_dirs.empty() ) { dirs.push_back( mui_->pwd() ); }
  else {
    vector<string>::const_iterator idir = sub_dirs.begin();
    for ( ; idir != sub_dirs.end(); idir++ ) {
      mui_->cd( *idir ); 
      cdIntoDir( *idir, dirs );
    }
  }
  mui_->goUp();
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::book( const std::vector<std::string>& me_list ) {
  cout << "[CommissioningHistograms::book] Nothing done!" << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::update() {
  cout << "[CommissioningHistograms::update] Nothing done!" << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::initHistoSet( const SiStripHistoNamingScheme::HistoTitle& title, 
					    HistoSet& histo_set, 
					    MonitorElement* me ) {
  // Set ME pointers for "sum2", "sum" and "num" histos
  if      ( title.contents_ == sistrip::COMBINED ) { histo_set.combined_ = me; }
  else if ( title.contents_ == sistrip::SUM2 )     { histo_set.sumOfSquares_ = me; }
  else if ( title.contents_ == sistrip::SUM )      { histo_set.sumOfContents_ = me; }
  else if ( title.contents_ == sistrip::NUM )      { histo_set.numOfEntries_ = me; }
  else {
    cerr << "[CommissioningHistograms::initHistoSet]"
	 << " Unexpected sistrip::Contents value!" << endl;
  }
  
  if ( histo_set.profile_ ) { return; } // TProfile already exists

  if ( !histo_set.sumOfSquares_ || //@@ what about combined histo?...
       !histo_set.sumOfContents_ ||
       !histo_set.numOfEntries_ ) { return; }
  
  TH1F* sum2 = ExtractTObject<TH1F>()( histo_set.sumOfSquares_ );
  TH1F* sum  = ExtractTObject<TH1F>()( histo_set.sumOfContents_ );
  TH1F* num  = ExtractTObject<TH1F>()( histo_set.numOfEntries_ );
  if ( !num || !sum || !sum2 ) { return; }
  
  if ( num->GetNbinsX() != sum->GetNbinsX() ||
       sum->GetNbinsX() != sum2->GetNbinsX() ) { 
    cerr << "[CommissioningHistograms::initHistoSet]"
	 << " Inconsistent number of bins for TObjects!" << endl;
    return;
  }
  
  histo_set.title_ = title;
  histo_set.title_.contents_ = sistrip::COMBINED;
  string name = SiStripHistoNamingScheme::histoTitle( histo_set.title_ );
  
  int bins = num->GetNbinsX();
  double low = num->GetXaxis()->GetXmin();
  double high = num->GetXaxis()->GetXmax();
  histo_set.profile_ = mui_->getBEInterface()->bookProfile( name, name, // title
							    bins, low, high, 
							    1024, 0., 1024. );
//   histo_set.profile_ = mui_->getBEInterface()->bookProfile( name, name, // title
//   							    num->GetNbinsX(),
//   							    num->GetXaxis()->GetXmin(),
//   							    num->GetXaxis()->GetXmax(),
//   							    1024, 0., 1024. );
  cout << "[CommissioningHistograms::initHistoSet]"
       << " Created TProfile called " << name 
       << " in " << mui_->pwd() << endl;
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::updateHistoSet( HistoSet& histo_set ) {

  if ( !histo_set.sumOfSquares_ || 
       !histo_set.sumOfContents_ ||
       !histo_set.numOfEntries_ ||
       !histo_set.profile_ ) { return; }
//     cerr << "[CommissioningHistograms::updateHistoSet]"
// 	 << " NULL pointer to MonitorElement!" << endl;
//     return;
//   }
  
  TH1F*     sum2 = ExtractTObject<TH1F>()( histo_set.sumOfSquares_ );
  TH1F*     sum  = ExtractTObject<TH1F>()( histo_set.sumOfContents_ );
  TH1F*     num  = ExtractTObject<TH1F>()( histo_set.numOfEntries_ );
  TProfile* prof = ExtractTObject<TProfile>()( histo_set.profile_ );
  if ( !num || !sum || !sum2 || !prof ) {
    cerr << "[CommissioningHistograms::updateHistoSet]"
	 << " NULL pointer to TObjects!" << endl;
    return;
  }
  
  if ( num->GetNbinsX() != sum->GetNbinsX() ||
       sum->GetNbinsX() != sum2->GetNbinsX() ||
       sum2->GetNbinsX() != prof->GetNbinsX() ) { 
    cerr << "[CommissioningHistograms::updateHistoSet]"
	 << " Inconsistent number of bins for TObjecets!" << endl;
    return;
  }
  
  cout << "[CommissioningHistograms::updateHistoSet]"
       << " Updating TProfile called " << histo_set.profile_->getName() 
       << " in " << mui_->pwd() << endl;

  for ( Int_t ibin = 1; ibin <= prof->GetNbinsX(); ibin++ ) {
    if ( num->GetBinContent(ibin) ) {
      // Calc entries, contents, errors
      Double_t entries = num->GetBinContent(ibin);
      Double_t content = sum->GetBinContent(ibin) / num->GetBinContent(ibin);
      Double_t mean_sum2 = sum2->GetBinContent(ibin) / num->GetBinContent(ibin);
      Double_t error = sqrt( fabs( mean_sum2 - content*content ) );
      // Set profile entries, contents, errors (order it important!)
      prof->SetBinEntries( ibin, entries );
      prof->SetBinContent( ibin, content*entries );
      Double_t weight = sqrt( entries*entries*error*error + content*content*entries );
      prof->SetBinError( ibin, weight );
    }
  }
  
}

