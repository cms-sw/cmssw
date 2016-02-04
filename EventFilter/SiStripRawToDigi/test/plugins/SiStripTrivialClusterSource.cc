#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialClusterSource.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

SiStripTrivialClusterSource::SiStripTrivialClusterSource(const edm::ParameterSet& pset) :

  minocc_(pset.getUntrackedParameter<double>("MinOccupancy",0.001)),
  maxocc_(pset.getUntrackedParameter<double>("MaxOccupancy",0.03)),
  mincluster_(pset.getUntrackedParameter<unsigned int>("MinCluster",4)),
  maxcluster_(pset.getUntrackedParameter<unsigned int>("MaxCluster",4)),
  separation_(pset.getUntrackedParameter<unsigned int>("Separation",2)),
  cabling_(),
  detids_(),
  nstrips_(0),
  random_()
{
  produces< edm::DetSetVector<SiStripDigi> >();
}

SiStripTrivialClusterSource::~SiStripTrivialClusterSource() {}

void SiStripTrivialClusterSource::beginRun( edm::Run&, const edm::EventSetup& setup ) {  

  setup.get<SiStripDetCablingRcd>().get(cabling_);
  cabling_->addAllDetectorsRawIds(detids_);
  for (unsigned int i=0;i<detids_.size();i++) {
    nstrips_+=cabling_->getConnections(detids_[i]).size()*256;
  }
}

void SiStripTrivialClusterSource::endJob() {}

void SiStripTrivialClusterSource::produce(edm::Event& iEvent,const edm::EventSetup& iSetup) {
  
  std::auto_ptr< edm::DetSetVector<SiStripDigi> > clusters(new edm::DetSetVector<SiStripDigi>());
  
  double occupancy = random_.Uniform(minocc_,maxocc_);
  double indexdigis = nstrips_ * occupancy;
  double indexcluster = random_.Uniform(mincluster_,maxcluster_);
  uint32_t ndigis = (indexdigis>0.) ? static_cast<uint32_t>(indexdigis) : 0;
  uint16_t clustersize = (indexcluster>0.) ? static_cast<uint16_t>(indexcluster) : 0;

  uint32_t counter = 0;
  while (counter < 10000) {
 
  if (clustersize && ndigis >= clustersize) ndigis-=clustersize;
  else break;

  while (counter < 10000) {

    double indexdet = random_.Uniform(0.,detids_.size());    
    uint32_t detid = detids_[static_cast<uint32_t>(indexdet)];
    uint32_t maxstrip = 256*cabling_->getConnections(detid).size();
    double indexstrip = random_.Uniform(0.,maxstrip-clustersize);
    uint16_t strip = static_cast<uint16_t>(indexstrip);
    
    edm::DetSet<SiStripDigi>& detset = clusters->find_or_insert(detid);
    detset.data.reserve(768);
    
    if (available(detset,strip,clustersize)) {
      addcluster(detset,strip,clustersize);
      counter = 0;
      break;
    }

    counter++;
  }
  }
  
  iEvent.put(clusters);
}

bool SiStripTrivialClusterSource::available(const edm::DetSet<SiStripDigi>& detset, const uint16_t firststrip, const uint32_t size) {

  for (edm::DetSet<SiStripDigi>::const_iterator idigi = detset.data.begin(); idigi != detset.data.end(); idigi++) {
    if (idigi->strip() >= (firststrip-separation_) && idigi->strip() < static_cast<int>(firststrip+size+separation_) && idigi->adc()) {
      return false; 
    }
  }
  return true;
}

void SiStripTrivialClusterSource::addcluster(edm::DetSet<SiStripDigi>& detset, const uint16_t firststrip, const uint16_t size) {

  for (unsigned int istrip=0;istrip<size;++istrip) { 
    detset.data.push_back(SiStripDigi(firststrip+istrip,0xFF));
  }
}
