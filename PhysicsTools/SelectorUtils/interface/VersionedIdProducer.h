#ifndef PhysicsTools_SelectorUtils_VersionedIdProducer_h
#define PhysicsTools_SelectorUtils_VersionedIdProducer_h
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/VIDCutFlowResult.h"
#include "DataFormats/PatCandidates/interface/UserData.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"

#include <memory>
#include <sstream>

template< class PhysicsObjectPtr , class SelectorType=VersionedSelector<PhysicsObjectPtr> >
class VersionedIdProducer : public edm::stream::EDProducer<> {
public:
  using PhysicsObjectType =  typename PhysicsObjectPtr::value_type;

  using Collection =  edm::View<PhysicsObjectType>;
  using TokenType = edm::EDGetTokenT<Collection>;

  explicit VersionedIdProducer(const edm::ParameterSet&);
  ~VersionedIdProducer() {}
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:  
  // ----------member data ---------------------------
  bool verbose_;
  TokenType physicsObjectSrc_;
    
  std::vector<std::unique_ptr<SelectorType> > ids_;

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
template< class PhysicsObjectPtr , class SelectorType >
VersionedIdProducer<PhysicsObjectPtr,SelectorType>::
VersionedIdProducer(const edm::ParameterSet& iConfig) {
  constexpr char bitmap_label[] = "Bitmap";
  
  verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);
  
  physicsObjectSrc_ = 
    consumes<Collection>(iConfig.getParameter<edm::InputTag>("physicsObjectSrc"));
  
  const std::vector<edm::ParameterSet>& ids = 
    iConfig.getParameterSetVector("physicsObjectIDs");
  for(const auto& id : ids ) {    
    const std::string& idMD5 = 
      id.getParameter<std::string>("idMD5");
    const edm::ParameterSet& the_id = 
      id.getParameterSet("idDefinition");
    const std::string& idname =
      the_id.getParameter<std::string>("idName");
    std::string calculated_md5;
    ids_.emplace_back( new SelectorType(the_id) );
    calculated_md5 = ids_.back()->md5String();
    ids_.back()->setConsumes(consumesCollector());
    if( ids_.back()->cutFlowSize() == 0 ) {
      throw cms::Exception("InvalidCutFlow")
	<< "Post-processing cutflow size is zero! You may have configured"
	<< " the python incorrectly!";
    }

    if( idMD5 != calculated_md5 ) {
      edm::LogError("IdConfigurationNotValidated")
        << "ID: " << ids_.back()->name() << "\n"
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

    if( !isPOGApproved ) {
      edm::LogWarning("IdInformation")
        << idmsg.str();
    } else {
      edm::LogInfo("IdInformation")
        << idmsg.str();
    }    

    produces<std::string>(idname);
    produces<edm::ValueMap<bool> >(idname);
    produces<edm::ValueMap<float> >(idname); // for PAT
    produces<edm::ValueMap<unsigned> >(idname);  
    produces<edm::ValueMap<unsigned> >(idname+std::string(bitmap_label));
    produces<edm::ValueMap<vid::CutFlowResult> >(idname);
    produces<pat::UserDataCollection>(idname);
    produces<edm::ValueMap<edm::Ptr<pat::UserData> > >(idname);
  }
}

template< class PhysicsObjectPtr , class SelectorType >
void VersionedIdProducer<PhysicsObjectPtr,SelectorType>::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  constexpr char bitmap_label[] = "Bitmap";
  
  edm::Handle<Collection> physicsObjectsHandle;
  iEvent.getByToken(physicsObjectSrc_,physicsObjectsHandle);

  const Collection& physicsobjects = *physicsObjectsHandle;

  for( const auto& id : ids_ ) {
    std::auto_ptr<edm::ValueMap<bool> > outPass(new edm::ValueMap<bool>() );
    std::auto_ptr<edm::ValueMap<float> > outPassf(new edm::ValueMap<float>() );
    std::auto_ptr<edm::ValueMap<unsigned> > outHowFar(new edm::ValueMap<unsigned>() );
    std::auto_ptr<edm::ValueMap<unsigned> > outBitmap(new edm::ValueMap<unsigned>() );
    std::auto_ptr<edm::ValueMap<vid::CutFlowResult> > out_cfrs(new edm::ValueMap<vid::CutFlowResult>() );
    std::auto_ptr<pat::UserDataCollection> out_usrd(new pat::UserDataCollection);
        
    std::vector<bool> passfail;
    std::vector<float> passfailf;
    std::vector<unsigned> howfar;
    std::vector<unsigned> bitmap;
    std::vector<vid::CutFlowResult> cfrs;
    
    for(size_t i = 0; i < physicsobjects.size(); ++i) {
      auto po = physicsobjects.ptrAt(i);
      passfail.push_back((*id)(po,iEvent));
      passfailf.push_back(passfail.back());
      howfar.push_back(id->howFarInCutFlow());
      bitmap.push_back(id->bitMap());
      cfrs.push_back(id->cutFlowResult());
      out_usrd->push_back(pat::UserData::make(cfrs.back(),false));
    }
    
    edm::ValueMap<bool>::Filler fillerpassfail(*outPass);
    fillerpassfail.insert(physicsObjectsHandle, passfail.begin(), passfail.end());
    fillerpassfail.fill();
    
    edm::ValueMap<float>::Filler fillerpassfailf(*outPassf);
    fillerpassfailf.insert(physicsObjectsHandle, passfailf.begin(), passfailf.end());
    fillerpassfailf.fill();

    edm::ValueMap<unsigned>::Filler fillerhowfar(*outHowFar);
    fillerhowfar.insert(physicsObjectsHandle, howfar.begin(), howfar.end() );
    fillerhowfar.fill();  
    
    edm::ValueMap<unsigned>::Filler fillerbitmap(*outBitmap);
    fillerbitmap.insert(physicsObjectsHandle, bitmap.begin(), bitmap.end() );
    fillerbitmap.fill(); 

    edm::ValueMap<vid::CutFlowResult>::Filler fillercfr(*out_cfrs);
    fillercfr.insert(physicsObjectsHandle, cfrs.begin(), cfrs.end() );
    fillercfr.fill(); 

    iEvent.put(outPass,id->name());
    iEvent.put(outPassf,id->name());
    iEvent.put(outHowFar,id->name());
    iEvent.put(outBitmap,id->name()+std::string(bitmap_label));
    iEvent.put(out_cfrs,id->name());
    iEvent.put(std::auto_ptr<std::string>(new std::string(id->md5String())),
	       id->name());
    auto usrd_handle = iEvent.put(out_usrd,id->name());
    //now add the value map of ptrs to user datas
    std::auto_ptr<edm::ValueMap<edm::Ptr<pat::UserData> > > out_usrdptrs(new edm::ValueMap<edm::Ptr<pat::UserData> >);
    std::vector<edm::Ptr<pat::UserData> > usrdptrs;
    for( unsigned i = 0; i < usrd_handle->size(); ++i ){
      usrdptrs.push_back(edm::Ptr<pat::UserData>(usrd_handle,i));
    }

    edm::ValueMap<edm::Ptr<pat::UserData> >::Filler fillerusrdptrs(*out_usrdptrs);
    fillerusrdptrs.insert(physicsObjectsHandle, usrdptrs.begin(), usrdptrs.end());
    fillerusrdptrs.fill();

    iEvent.put(out_usrdptrs,id->name());        
  }   
}

#endif
