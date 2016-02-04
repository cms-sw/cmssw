// Last commit: $Id: SiStripTFile.cc,v 1.4 2009/02/10 21:45:55 lowette Exp $

#include "DQM/SiStripCommissioningClients/interface/SiStripTFile.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TDirectory.h"
#include "TH1.h"
#include <sstream>

using namespace sistrip;

//-----------------------------------------------------------------------------

SiStripTFile::SiStripTFile( const char* fname, 
			    Option_t* option, 
			    const char* ftitle, 
			    Int_t compress ) :
  TFile(fname,option,ftitle,compress),
  runType_(sistrip::UNKNOWN_RUN_TYPE),
  view_(sistrip::UNKNOWN_VIEW),
  top_(gDirectory),
  dqmTop_(0),
  sistripTop_(0),
  dqmFormat_(false)
{
  readDQMFormat();
}

//-----------------------------------------------------------------------------

SiStripTFile::~SiStripTFile() {;}

//-----------------------------------------------------------------------------

TDirectory* SiStripTFile::setDQMFormat( sistrip::RunType run_type, 
					sistrip::View view) {
  
  view_ = view;
  runType_ = run_type;
  
  if (view == sistrip::CONTROL_VIEW) {
    std::stringstream ss("");
    ss << sistrip::dqmRoot_ << sistrip::dir_ << sistrip::root_ << sistrip::dir_ << sistrip::controlView_;
    top_ = addPath( ss.str() );
    dqmTop_ = GetDirectory(sistrip::dqmRoot_);
    sistripTop_ = dqmTop_->GetDirectory(sistrip::root_);
    dqmFormat_ = true;
    
    //TNamed defining commissioning runType
    std::stringstream run_type_label;
    std::stringstream run_type_title;
    run_type_label << sistrip::taskId_ << sistrip::sep_ << SiStripEnumsAndStrings::runType(runType_);
    run_type_title << "s=" << SiStripEnumsAndStrings::runType(runType_);
    TNamed run_type_description(run_type_label.str().c_str(),run_type_title.str().c_str());
    sistripTop_->WriteTObject(&run_type_description);
  }

  else {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningFile::setDQMFormat]: Currently only implemented for Control View." 
      << std::endl; 
    return 0;
  }
  
  return top_;
}

//-----------------------------------------------------------------------------

TDirectory* SiStripTFile::readDQMFormat() {
  
  //check directory structure and find readout view
  dqmTop_ = GetDirectory(sistrip::dqmRoot_);
  if (dqmTop_) sistripTop_ = dqmTop_->GetDirectory(sistrip::root_);
  if (sistripTop_) top_ = sistripTop_->GetDirectory(sistrip::controlView_);
  if (top_!=gDirectory) view_ = sistrip::CONTROL_VIEW;
  
  //does file conform with DQM Format requirements?
  if (dqmTop_ && sistripTop_ && top_) {
    dqmFormat_ = true;}
  
  // Search for commissioning run_type
  if (sistripTop_) {
    TList* keylist = sistripTop_->GetListOfKeys();
    if (keylist) {
      TObject* obj = keylist->First(); //the object
      if (obj) {
        bool loop = true;
        while (loop) { 
          if (obj == keylist->Last()) {loop = false;}
          if ( std::string(obj->GetName()).find(sistrip::taskId_) != std::string::npos ) {
            runType_ = SiStripEnumsAndStrings::runType( std::string(obj->GetTitle()).substr(2,std::string::npos) );
	    // 	    cout << " name: " << std::string(obj->GetName())
	    // 		 << " title: " << std::string(obj->GetTitle()) 
	    // 		 << " runType: " << SiStripEnumsAndStrings::runType( runType_ )
	    // 		 << std::endl;
          }
          obj = keylist->After(obj);
        }
      }
    }
  } 

  return top_;
}

//-----------------------------------------------------------------------------

bool SiStripTFile::queryDQMFormat() {
  return dqmFormat_;
}

//-----------------------------------------------------------------------------

TDirectory* SiStripTFile::top() {
  return top_;}

//-----------------------------------------------------------------------------

TDirectory* SiStripTFile::dqmTop() {
  if (!dqmFormat_) {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripTFile::dqm]: Error requested dqm directory when not in dqm format." 
      << std::endl; 
    return 0;
  }

  return dqmTop_;}


//-----------------------------------------------------------------------------

TDirectory* SiStripTFile::sistripTop() {
  if (!dqmFormat_) {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripTFile::dqm]: Error requested dqm directory when not in dqm format."
      << std::endl; 
    return 0;
  }

  return sistripTop_;}

//-----------------------------------------------------------------------------

sistrip::RunType& SiStripTFile::runType() {
  return runType_;}

//-----------------------------------------------------------------------------

sistrip::View& SiStripTFile::View() {
  return view_;}

//-----------------------------------------------------------------------------

void SiStripTFile::addDevice(unsigned int key) {

  if (view_ == sistrip::CONTROL_VIEW) {
    if (!dqmFormat_) setDQMFormat(sistrip::UNKNOWN_RUN_TYPE, sistrip::CONTROL_VIEW);
    SiStripFecKey control_path(key);
    std::string directory_path = control_path.path();
    cd(sistrip::dqmRoot_);
    addPath(directory_path);
  }

  else {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningFile::addDevice]: Currently only implemented for Control View." 
      << std::endl; 
  }

}

//-----------------------------------------------------------------------------

TDirectory* SiStripTFile::addPath( const std::string& path ) {
  
//   std::string path = dir;
//   std::string root = sistrip::dqmRoot_+"/"+sistrip::root_+"/";
//   if ( path.find( root ) == std::string::npos ) {
//     cerr << "Did not find \"" << root << "\" root in path: " << dir;
//     path = root + dir;
//   }
  
  std::vector<std::string> directories; directories.reserve(10);

  //fill std::vector
  std::string::const_iterator it, previous_dir, latest_dir;
  if (*(path.begin()) == std::string(sistrip::dir_)) {
    it = previous_dir = latest_dir = path.begin();}
  else {it = previous_dir = latest_dir = path.begin()-1;}

  while (it != path.end()) {
    it++;
    if (*it == std::string(sistrip::dir_)) {
      previous_dir = latest_dir; 
      latest_dir = it;
      directories.push_back(std::string(previous_dir+1, latest_dir));
    }
  }

  if (latest_dir != (path.end()-1)) {
    directories.push_back(std::string(latest_dir+1, path.end()));}
 
  //update file
  TDirectory* child = gDirectory;
  for (std::vector<std::string>::const_iterator dir = directories.begin(); dir != directories.end(); dir++) {
    if (!dynamic_cast<TDirectory*>(child->Get(dir->c_str()))) {
      child = child->mkdir(dir->c_str());
      child->cd();}
    else {child->Cd(dir->c_str()); child = gDirectory;}
  }
  return child;
}

//-----------------------------------------------------------------------------

void SiStripTFile::findHistos( TDirectory* dir, std::map< std::string, std::vector<TH1*> >* histos ) {

  std::vector< TDirectory* > dirs;
  dirs.reserve(20000);
  dirs.push_back(dir);

  //loop through all directories and record tprofiles (matching label taskId_) contained within them.

  while ( !dirs.empty() ) { 
    dirContent(dirs[0], &dirs, histos);
    dirs.erase(dirs.begin());
  }
}

//-----------------------------------------------------------------------------


void SiStripTFile::dirContent(TDirectory* dir, 
					  std::vector<TDirectory*>* dirs, 
					  std::map< std::string, std::vector<TH1*> >* histos ) {

  TList* keylist = dir->GetListOfKeys();
  if (keylist) {

    TObject* obj = keylist->First(); // the object (dir or histo)

    if ( obj ) {
      bool loop = true;
      while (loop) { 
	if (obj == keylist->Last()) {loop = false;}
 
	if (dynamic_cast<TDirectory*>(dir->Get(obj->GetName()))) {
	  TDirectory* child = dynamic_cast<TDirectory*>(dir->Get(obj->GetName()));

	  //update record of directories
	  dirs->push_back(child);
	}
	  
	TH1* his = dynamic_cast<TH1*>( dir->Get(obj->GetName()) );
	if ( his ) {
	  bool found = false;
	  std::vector<TH1*>::iterator ihis = (*histos)[std::string(dir->GetPath())].begin();
	  for ( ; ihis != (*histos)[std::string(dir->GetPath())].end(); ihis++ ) {
	    if ( (*ihis)->GetName() == his->GetName() ) { found = true; }
	  }
	  if ( !found ) { (*histos)[std::string(dir->GetPath())].push_back(his); }
	}
	obj = keylist->After(obj);
      }
    }
  }

}

