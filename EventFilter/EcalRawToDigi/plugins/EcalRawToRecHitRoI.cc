
//#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>

#include "EventFilter/EcalRawToDigi/plugins/EcalRawToRecHitRoI.h"
#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"

#include "FWCore/Utilities/interface/Exception.h"

// Ecal Mapping 
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
//#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>


// Level 1 Trigger
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometry.h"
#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometryRecord.h"
                                                                                                                        
// EgammaCoreTools
//#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

// Muon stuff
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

// Jets stuff
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

//candidate stuff
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"


#include "EventFilter/EcalRawToDigi/interface/EcalRegionCablingRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"
#include "EventFilter/EcalRawToDigi/interface/MyWatcher.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitComparison.h"

#include <vector>

using namespace l1extra;




EcalRawToRecHitRoI::EcalRawToRecHitRoI(const edm::ParameterSet& pset) :
  EGamma_(false), Muon_(false), Jet_(false), Candidate_(false){

  sourceTag_=pset.getParameter<edm::InputTag>("sourceTag");

  std::string type = pset.getParameter<std::string>("type");

  if (type.find("candidate")!=std::string::npos){
    Candidate_ = true;
    std::vector<edm::ParameterSet> emPSet =  pset.getParameter<std::vector<edm::ParameterSet> >("CandJobPSet");
    for (std::vector<edm::ParameterSet>::iterator iepset = emPSet.begin(); iepset!=emPSet.end();++iepset){
      CandSource_.push_back(CandJobPSet(*iepset));
    }
  }
  
  if (type.find("muon")!=std::string::npos){
    Muon_ = true;
    edm::ParameterSet ps = pset.getParameter<edm::ParameterSet>("MuJobPSet");
    MuonSource_ = MuJobPSet(ps);
  }
  
  if (type.find("egamma")!=std::string::npos){
    EGamma_ = true;
    std::vector<edm::ParameterSet> emPSet =  pset.getParameter<std::vector<edm::ParameterSet> >("EmJobPSet");
    for (std::vector<edm::ParameterSet>::iterator iepset = emPSet.begin(); iepset!=emPSet.end();++iepset){
      EmSource_.push_back(EmJobPSet(*iepset));
    }
  }
  
  if (type.find("jet")!=std::string::npos){
    Jet_ =true;
    std::vector<edm::ParameterSet> jetPSet = pset.getParameter<std::vector<edm::ParameterSet> >("JetJobPSet");
    for (std::vector<edm::ParameterSet>::iterator ijpset = jetPSet.begin(); ijpset!=jetPSet.end();++ijpset){
      JetSource_.push_back(JetJobPSet(*ijpset));
    }
  }

  if (type.find("all")!=std::string::npos){
    All_ = true;
  }

  if (!All_ && !Muon_ && !EGamma_ && !Jet_){
    edm::LogError("EcalRawToRecHitRoI")<<"I have no specified type of work."
				       <<"\nI will produce empty list of FEDs."
				       <<"\nI will produce an empty EcalRecHitRefGetter."
				       <<"\n I am sure you don't want that.";
    //throw. this is a typo/omittion in a cfg.
  }

 produces<EcalListOfFEDS>();
 produces<EcalRecHitRefGetter>();
}



EcalRawToRecHitRoI::~EcalRawToRecHitRoI() {
}


void EcalRawToRecHitRoI::beginJob(const edm::EventSetup& c){
}

void EcalRawToRecHitRoI::endJob(){
}

#include <sstream>
std::string EcalRawToRecHitRoI::dumpFEDs(const std::vector<int> & FEDs){
  std::stringstream ss;
  for (int i=0; i < (int)FEDs.size(); i++) {
    ss<< "unpack FED " << FEDs[i] << std::endl;
  }
  ss<< "Number of FEDS is " << FEDs.size() << std::endl;
  return ss.str();
}

void EcalRawToRecHitRoI::produce(edm::Event & e, const edm::EventSetup& iSetup){
  const std::string category = "EcalRawToRecHit|RoI";
  MyWatcher watcher("RoI");
  LogDebug(category)<<watcher.lap();

  // retreive cabling
  edm::ESHandle<EcalRegionCabling> cabling;
  iSetup.get<EcalRegionCablingRecord>().get(cabling);
  LogDebug(category)<<"cabling retrieved."
		    <<watcher.lap();
  TheMapping = cabling->mapping();

  std::pair<int,int> ecalfeds = FEDNumbering::getEcalFEDIds();
  int first_fed = ecalfeds.first;
  
  std::auto_ptr<EcalListOfFEDS> productAddress(new EcalListOfFEDS);
  std::vector<int> feds;		// the list of FEDS produced by this module

 if (EGamma_) {   Egamma(e, iSetup, feds); }

 if (Muon_) {   Muon(e, iSetup, feds); }

 if (Jet_) {   Jet(e, iSetup, feds); }

 if (Candidate_) { Cand(e, iSetup, feds); }

 if ( !EGamma_ && !Muon_ && ! Jet_)  {   for (int i=1; i <= 54; feds.push_back(i++)){} }
 
 uint nf = feds.size();
 for (uint i=0; i <nf; feds[i++]+=first_fed) {}
 
 LogDebug(category)<< "Will unpack FED " <<dumpFEDs(feds)
		   <<watcher.lap();
 
 if (nf<1){edm::LogWarning(category)<<"no ECAL FED to unpack for Run " << e.id().run() << "  Event " << e.id().event() ;}
 
 productAddress->SetList(feds);
 e.put(productAddress);
 LogDebug(category)<< "list of fed put in the event."
		   <<watcher.lap();
 
 //now defined the Region of interest to be unpacked. from the feds list

 
 //get the lazy gettter
 edm::Handle<EcalRecHitLazyGetter> lgetter;
 e.getByLabel(sourceTag_, lgetter);
 LogDebug(category)<<"lazy getter retrieved from: "<<sourceTag_
		   <<watcher.lap();
 
 //prepare a refgetter
 std::auto_ptr<EcalRecHitRefGetter> rgetter(new EcalRecHitRefGetter);
 LogDebug(category)<<"ref getter ready to be updated."
				<<watcher.lap();
 
 for (uint i=0;i!=nf;i++){
   cabling->updateEcalRefGetterWithFedIndex(*rgetter, lgetter, feds[i]);  
 }
 
 //put the refgetter in the event  
 LogDebug(category)<<"refGetter to be put in the event."
		   << watcher.lap();
 e.put(rgetter);
 LogDebug(category)<<"refGetter loaded."
				<< watcher.lap();
}

void EcalRawToRecHitRoI::OneCandCollection(const edm::Handle<edm::View< reco::Candidate > >Coll,
					       const CandJobPSet & cjpset,
					       std::vector<int> & feds){
  const std::string category ="EcalRawToRecHit|Cand";
  edm::View< reco::Candidate >::const_iterator it = Coll->begin();
  for (; it!= Coll->end();++it){
    double pt    =  (*it).pt();
    double eta   =  (*it).eta();
    double phi   =  (*it).phi();
    
    LogDebug(category)<<" here is a L1 muon Seed  with (eta,phi) = " 
		      <<eta << " " << phi << " and pt " << pt;
    if (pt < cjpset.Ptmin) continue;
    
    ListOfFEDS(eta, eta, phi-cjpset.epsilon, phi+cjpset.epsilon, cjpset.regionEtaMargin, cjpset.regionPhiMargin,feds);
  }
}
 
void EcalRawToRecHitRoI::Cand(edm::Event& e, const edm::EventSetup& es , std::vector<int> & FEDs) {
  const std::string category ="EcalRawToRecHit|Cand";
  
  edm::Handle< edm::View< reco::Candidate > > candColl;
  uint nc=CandSource_.size();
  for (uint ic=0;ic!=nc;++ic){
    e.getByLabel(CandSource_[ic].Source, candColl);
    //    if (!candColl.isValid()){edm::LogError(category)<<" viewed candidate collection: "<<CandSource_[ic].Source<<" is not valid."; continue;}
    
    OneCandCollection(candColl, CandSource_[ic], FEDs);
  }
  
  unique(FEDs);
  LogDebug(category)<<"unpack FED\n"<<dumpFEDs(FEDs);
}

void EcalRawToRecHitRoI::Egamma_OneL1EmCollection(const edm::Handle< l1extra::L1EmParticleCollection > emColl,
						      const EmJobPSet &ejpset,
                                                      const edm::ESHandle<L1CaloGeometry> & l1CaloGeom,
                                                      std::vector<int> & FEDs){
  const   std::string category = "EcalRawToRecHit|Egamma";
  for( l1extra::L1EmParticleCollection::const_iterator emItr = emColl->begin();
       emItr != emColl->end() ;++emItr ){
    float pt = emItr -> pt();
    if (pt < ejpset.Ptmin ) continue;
    LogDebug(category)<<" Here is an L1 isoEM candidate of pt " << pt;
    // Access the GCT hardware object corresponding to the L1Extra EM object.
    int etaIndex = emItr->gctEmCand()->etaIndex() ;
    int phiIndex = emItr->gctEmCand()->phiIndex() ;
    // Use the L1CaloGeometry to find the eta, phi bin boundaries.
    double etaLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
    double etaHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
    double phiLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
    double phiHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;
    
    ListOfFEDS(etaLow, etaHigh, phiLow, phiHigh, ejpset.regionEtaMargin, ejpset.regionPhiMargin,FEDs);
  }
}

void EcalRawToRecHitRoI::Egamma(edm::Event& e, const edm::EventSetup& es , std::vector<int> & FEDs) {
  const std::string category = "EcalRawToRecHit|Egamma";
  
  LogDebug(category)<< " enter in EcalRawToRecHitRoI::Egamma";
  
  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  es.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;

  edm::Handle< l1extra::L1EmParticleCollection > emColl;
  uint ne=EmSource_.size();
  for (uint ie=0;ie!=ne;++ie){
    e.getByLabel(EmSource_[ie].Source,emColl);
    if (!emColl.isValid()){edm::LogError(category)<<"L1Em Collection: "<<EmSource_[ie].Source<<" is not valid.";continue;}

    Egamma_OneL1EmCollection(emColl,EmSource_[ie],l1CaloGeom,FEDs);
  }

  unique(FEDs);
  LogDebug(category)<<"end of get list of feds\n"<<dumpFEDs(FEDs);
  return;
}


void EcalRawToRecHitRoI::Muon(edm::Event& e, const edm::EventSetup& es, std::vector<int>& FEDs) {
  const std::string category = "EcalRawToRecHit|Muon";

  LogDebug(category)<< " enter in EcalRawToRecHitRoI::Muon";

  edm::Handle<L1MuonParticleCollection> muColl;
  e.getByLabel(MuonSource_.Source, muColl);

  for (L1MuonParticleCollection::const_iterator it=muColl->begin(); it != muColl->end(); it++) {

    const L1MuGMTExtendedCand muonCand = (*it).gmtMuonCand();
    double pt    =  (*it).pt();
    double eta   =  (*it).eta();
    double phi   =  (*it).phi();

    LogDebug(category)<<" here is a L1 muon Seed  with (eta,phi) = " 
		      <<eta << " " << phi << " and pt " << pt;
    if (pt < MuonSource_.Ptmin) continue;

    ListOfFEDS(eta, eta, phi-MuonSource_.epsilon, phi+MuonSource_.epsilon, MuonSource_.regionEtaMargin, MuonSource_.regionPhiMargin,FEDs);
  }
  
  unique(FEDs);
  LogDebug(category)<<"end of get list of feds\n"<<dumpFEDs(FEDs);
  
  return;
}
 
void EcalRawToRecHitRoI::Jet_OneL1JetCollection(const edm::Handle< l1extra::L1JetParticleCollection > jetColl,
						    const JetJobPSet & jjpset,
						    std::vector<int> & feds){
  const std::string category ="EcalRawToRecHit|Jet";
  for (L1JetParticleCollection::const_iterator it=jetColl->begin(); it != jetColl->end(); it++) {
    double pt    =  it -> pt();
    double eta   =  it -> eta();
    double phi   =  it -> phi();

    LogDebug(category) << " here is a L1 CentralJet Seed  with (eta,phi) = "
		       << eta << " " << phi << " and pt " << pt;

    if (pt < jjpset.Ptmin ) continue;

    ListOfFEDS(eta, eta, phi-jjpset.epsilon, phi+jjpset.epsilon, jjpset.regionEtaMargin, jjpset.regionPhiMargin,feds);
  }
}

void EcalRawToRecHitRoI::Jet(edm::Event& e, const edm::EventSetup& es, std::vector<int> & FEDs) {
  const std::string category = "EcalRawToRecHit|Jet";

  edm::Handle<L1JetParticleCollection> jetColl;
  uint nj=JetSource_.size();
  for (uint ij=0;ij!=nj;++ij){
    e.getByLabel(JetSource_[ij].Source,jetColl);
    if (!jetColl.isValid()){edm::LogError(category)<<"L1Jet collection: "<<JetSource_[ij].Source<<" is not valid.";continue;}

    Jet_OneL1JetCollection(jetColl,JetSource_[ij],FEDs);
  }

  unique(FEDs);
  LogDebug(category)<<"unpack FED\n"<<dumpFEDs(FEDs);
}




void EcalRawToRecHitRoI::ListOfFEDS(double etaLow, double etaHigh, double phiLow, 
					double phiHigh, double etamargin, double phimargin,
					std::vector<int> & FEDs)
{
  const std::string category = "EcalRawToRecHit|ListOfFEDS";
  
  if (phimargin > Geom::pi()) phimargin =  Geom::pi() ;

  
  LogDebug(category)<< " etaLow etaHigh phiLow phiHigh " << etaLow << " "
                    <<etaHigh << " " << phiLow << " " << phiHigh;

  
  etaLow -= etamargin;
  etaHigh += etamargin;
  double phiMinus = phiLow - phimargin;
  double phiPlus = phiHigh + phimargin;
  
  bool all = false;
  double dd = fabs(phiPlus-phiMinus);
  LogDebug(category)<< " dd = " << dd;
  if (dd > 2.*Geom::pi() ) all = true;

  while (phiPlus > Geom::pi()) { phiPlus -= 2.*Geom::pi() ; }
  while (phiMinus < 0) { phiMinus += 2.*Geom::pi() ; }
  if ( phiMinus > Geom::pi()) phiMinus -= 2.*Geom::pi() ;

  double dphi = phiPlus - phiMinus;
  if (dphi < 0) dphi += 2.*Geom::pi() ;
  LogDebug(category) << "dphi = " << dphi;
  if (dphi > Geom::pi()) {
    int fed_low1 = TheMapping -> GetFED(etaLow,phiMinus*180./Geom::pi());
    int fed_low2 = TheMapping -> GetFED(etaLow,phiPlus*180./Geom::pi());
    LogDebug(category) << "fed_low1 fed_low2 " << fed_low1 << " " << fed_low2;
    if (fed_low1 == fed_low2) all = true;
    int fed_hi1 = TheMapping -> GetFED(etaHigh,phiMinus*180./Geom::pi());
    int fed_hi2 = TheMapping -> GetFED(etaHigh,phiPlus*180./Geom::pi());
    LogDebug(category) << "fed_hi1 fed_hi2 " << fed_hi1 << " " << fed_hi2;
    if (fed_hi1 == fed_hi2) all = true;
  }


  if (all) {
    LogDebug(category)<< " unpack everything in phi ! ";
    phiMinus = -20 * Geom::pi() / 180.;  // -20 deg
    phiPlus = -40 * Geom::pi() / 180.;  // -20 deg
  }
  
  LogDebug(category) << " with margins : " << etaLow << " " << etaHigh << " "
                     << phiMinus << " " << phiPlus;


  const EcalEtaPhiRegion ecalregion(etaLow,etaHigh,phiMinus,phiPlus);

  std::vector<int> feds = TheMapping -> GetListofFEDs(ecalregion);
  LogDebug(category)<<"unpack fed:\n"<<dumpFEDs(feds);

  //insert the result into collection
  FEDs.insert(FEDs.end(), feds.begin(), feds.end());
  //FIXME, the mapping should do that for you !
}
