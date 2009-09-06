#include "DQMServices/Components/src/DQMFileReader.h"

DQMFileReader::DQMFileReader(const edm::ParameterSet& iConfig)  
{   
  filenames_.clear();
  filenames_=iConfig.getUntrackedParameter<std::vector<std::string > >("FileNames");
}

DQMFileReader::~DQMFileReader()
{}

void 
DQMFileReader::beginJob(const edm::EventSetup& iSetup)
{

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  
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

