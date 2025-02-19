#include "DQMServices/Components/src/DQMFileReader.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include <iostream>

DQMFileReader::DQMFileReader(const edm::ParameterSet& ps)  
{   

  pset_ = ps;

  dbe_ = edm::Service<DQMStore>().operator->();

  filenames_.clear();
  filenames_=pset_.getUntrackedParameter<std::vector<std::string > >("FileNames");
  referenceFileName_=pset_.getUntrackedParameter<std::string>("referenceFileName","");
}

DQMFileReader::~DQMFileReader()
{}

void 
DQMFileReader::beginJob()
{
  
  if (referenceFileName_ != "") 
  {
    const std::string override = "";
    std::vector<std::string> in ; in.push_back(referenceFileName_);
    edm::InputFileCatalog catalog(in,override,true);

    std::string ff=catalog.fileNames()[0];
    std::cout << "DQMFileReader: reading reference file '" << ff << "'\n";

    // now open file, quietly continuing if it does not exist
    if (dbe_->open(ff, true, "", "Reference", DQMStore::StripRunDirs, false))
    {
      dbe_->cd(); dbe_->setCurrentFolder("Info/ProvInfo"); 
      dbe_->bookString("referenceFileName",ff);
      std::cout << "DQMFileReader: reference file '" << ff << "' successfully read in \n";
    }
    else      
    {
      dbe_->cd(); dbe_->setCurrentFolder("Info/ProvInfo"); 
      dbe_->bookString("referenceFileName","non-existent:"+ff);
      std::cout << "DQMFileReader: reference file '" << ff << "' does not exist \n";
    }
    dbe_->cd();
    return;
  }  

  dbe_->bookString("referenceFileName","no reference file specified");
  dbe_->cd();
  
  // read in files, stripping off Run Summary and Run <number> folders
  
  for (unsigned int i=0; i<filenames_.size(); i++)
  {
    std::cout << "DQMFileReader::beginJob: loading" << filenames_[i] << std::endl;
    if (dbe_) 
       dbe_->load(filenames_[i]);
  }
}

void 
DQMFileReader::endJob() 
{}

void
DQMFileReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{}

