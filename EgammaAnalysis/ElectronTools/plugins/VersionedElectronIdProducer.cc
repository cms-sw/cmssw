// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"

#include "EgammaAnalysis/ElectronTools/interface/VersionedGsfElectronSelector.h"
#include "EgammaAnalysis/ElectronTools/interface/VersionedPatElectronSelector.h"

#include "EgammaAnalysis/ElectronTools/interface/EGammaMvaEleEstimator.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
//
// class declaration
//

#include <memory>
#include <sstream>

class VersionedElectronIdProducer : public edm::EDProducer {
public:
  typedef edm::View<reco::GsfElectron> Collection;

  explicit VersionedElectronIdProducer(const edm::ParameterSet&);
  ~VersionedElectronIdProducer() {}
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:  
  // ----------member data ---------------------------
  bool verbose_, isPAT_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > electronSrc_;
    
  std::vector<std::unique_ptr<VersionedGsfElectronSelector> > forGsf_;  
  std::vector<std::unique_ptr<VersionedPatElectronSelector> > forPat_;  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
VersionedElectronIdProducer::
VersionedElectronIdProducer(const edm::ParameterSet& iConfig) {
  verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);
  
  isPAT_ = iConfig.getParameter<bool>("electronsArePAT");
  electronSrc_ = 
    consumes<Collection>(iConfig.getParameter<edm::InputTag>("electronSrc"));
  
  const std::vector<edm::ParameterSet>& ids = 
    iConfig.getParameterSetVector("electronIDs");
  for(const auto& id : ids ) {    
    const std::string& idMD5 = 
      id.getParameter<std::string>("idMD5");
    const edm::ParameterSet& the_id = 
      id.getParameterSet("idDefinition");
    const std::string& idname =
      the_id.getParameter<std::string>("idName");
    std::string calculated_md5;
    if( isPAT_ ) { 
      forPat_.emplace_back( new VersionedPatElectronSelector(the_id) );
      calculated_md5 = forPat_.back()->md5String();
      forPat_.back()->setConsumes(consumesCollector());
      if( forPat_.back()->cutFlowSize() == 0 ) {
	throw cms::Exception("InvalidCutFlow")
	  << "Post-processing cutflow size is zero! You may have configured"
	  << " the python incorrectly!";
      }
    } else {
      forGsf_.emplace_back( new VersionedGsfElectronSelector(the_id) );
      calculated_md5 = forGsf_.back()->md5String();
      forGsf_.back()->setConsumes(consumesCollector());
      if( forGsf_.back()->cutFlowSize() == 0 ) {
	throw cms::Exception("InvalidCutFlow")
	  << "Post-processing cutflow size is zero! You may have configured"
	  << " the python incorrectly!";
      }
    }
    if( idMD5 != calculated_md5 ) {
      throw cms::Exception("IdConfigurationNotValidated")
	<< "The expected md5: " << idMD5 << " does not match the md5\n"
	<< "calculated by the ID: " << calculated_md5 << " please\n"
	<< "update your python configuration or determine the source\n"
	<< "of transcription error!";
    }

    std::stringstream idmsg;

    // dump whatever information about the ID we have     
    idmsg << "Instantiated ID: " << idname << std::endl
	  << "with MD5 hash: " << idMD5 << std::endl;
    const bool isPOGApproved = 
      id.getUntrackedParameter<bool>("isPOGApproved",false);
    if( isPOGApproved ) {
      idmsg << "This ID is POG approved!" << std::endl;
    } else {
      idmsg << "This ID is not POG approved and likely under development!!!\n"
	    << "Please make sure to report your progress with this ID "
	    << "at the next relevant POG meeting." << std::endl;
    }

    edm::LogWarning("IdInformation")
      << idmsg.str();

    produces<std::string>(idname);
    produces<edm::ValueMap<bool> >(idname);
    produces<edm::ValueMap<unsigned> >(idname);
  }
}

void VersionedElectronIdProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::GsfElectron> > electronsHandle;
  iEvent.getByToken(electronSrc_,electronsHandle);

  const edm::View<reco::GsfElectron>& electrons = *electronsHandle;

  if( isPAT_ ) {

    for( const auto& id : forPat_ ) {
       std::auto_ptr<edm::ValueMap<bool> > outPass(new edm::ValueMap<bool>() );
       std::auto_ptr<edm::ValueMap<unsigned> > outHowFar(new edm::ValueMap<unsigned>() );
       std::vector<bool> passfail;
       std::vector<unsigned> howfar;
       for(const auto& eleptr : electrons.refVector()) {	 
	 if( eleptr.isNonnull() ) {
	   passfail.push_back((*id)(eleptr.castTo<pat::ElectronRef>(),
				    iEvent));
	   howfar.push_back(id->howFarInCutFlow());
	 } else {
	   throw cms::Exception("InvalidCast")
	     << "Unable to cast GsfElectron pointer to pat::Electron pointer"
	     << " despite being in PAT mode!";
	 }
       }
       edm::ValueMap<bool>::Filler fillerpassfail(*outPass);
       fillerpassfail.insert(electronsHandle, passfail.begin(), passfail.end());
       fillerpassfail.fill();
       edm::ValueMap<unsigned>::Filler fillerhowfar(*outHowFar);
       fillerhowfar.insert(electronsHandle, howfar.begin(), howfar.end() );
       fillerhowfar.fill();
       iEvent.put(outPass,id->name());
       iEvent.put(outHowFar,id->name());
       iEvent.put(std::auto_ptr<std::string>(new std::string(id->md5String())),
					     id->name());
    }
  } else {
    for( const auto& id : forGsf_ ) {
      std::auto_ptr<edm::ValueMap<bool> > outPass(new edm::ValueMap<bool>() );
      std::auto_ptr<edm::ValueMap<unsigned> > outHowFar(new edm::ValueMap<unsigned>() );
      std::vector<bool> passfail;
      std::vector<unsigned> howfar;
      for(const auto& ele : electrons.refVector()) {
	passfail.push_back((*id)(ele.castTo<reco::GsfElectronRef>(),iEvent));
	howfar.push_back(id->howFarInCutFlow());
      }
      edm::ValueMap<bool>::Filler fillerpassfail(*outPass);
      fillerpassfail.insert(electronsHandle, passfail.begin(), passfail.end());
      fillerpassfail.fill();
      edm::ValueMap<unsigned>::Filler fillerhowfar(*outHowFar);
      fillerhowfar.insert(electronsHandle, howfar.begin(), howfar.end() );
      fillerhowfar.fill();      
      iEvent.put(outPass,id->name());
      iEvent.put(outHowFar,id->name());
      iEvent.put(std::auto_ptr<std::string>(new std::string(id->md5String())),
					     id->name());
    }
  }  
}

//define this as a plug-in
DEFINE_FWK_MODULE(VersionedElectronIdProducer);
