/** \class MuonProducer
 *  See header file.
 *
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/MuonIdentification/plugins/MuonProducer.h"

#include "RecoMuon/MuonIsolation/interface/MuPFIsoHelper.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"


#ifndef dout
#define dout if(debug_) std::cout
#endif

using std::endl;

typedef std::map<reco::MuonRef, unsigned int> MuToPFMap;


namespace reco {
  typedef edm::ValueMap<reco::MuonShower> MuonShowerMap;
}


/// Constructor
MuonProducer::MuonProducer(const edm::ParameterSet& pSet):debug_(pSet.getUntrackedParameter<bool>("ActivateDebug",false)){
  
  setAlias(pSet.getParameter<std::string>("@module_label"));
  
  fastLabelling_ = pSet.getUntrackedParameter<bool>("FastLabelling",true);

  
  theMuonsCollectionLabel = pSet.getParameter<edm::InputTag>("InputMuons");
  theMuonsCollectionToken_ = consumes<reco::MuonCollection>(theMuonsCollectionLabel);

  thePFCandLabel = pSet.getParameter<edm::InputTag>("PFCandidates");
  thePFCandToken_ = consumes<reco::PFCandidateCollection>(thePFCandLabel);
  
  // Variables to switch on-off the differnt parts
  fillSelectors_              = pSet.getParameter<bool>("FillSelectorMaps");
  fillCosmicsIdMap_           = pSet.getParameter<bool>("FillCosmicsIdMap");
  fillPFMomentum_             = pSet.getParameter<bool>("FillPFMomentumAndAssociation");
  fillPFIsolation_            = pSet.getParameter<bool>("FillPFIsolation");
  fillDetectorBasedIsolation_ = pSet.getParameter<bool>("FillDetectorBasedIsolation"); 
  fillShoweringInfo_          = pSet.getParameter<bool>("FillShoweringInfo");
  fillTimingInfo_             = pSet.getParameter<bool>("FillTimingInfo");
  computeStandardSelectors_   = pSet.getParameter<bool>("ComputeStandardSelectors");

  produces<reco::MuonCollection>();

  if(fillTimingInfo_){

    timeMapCmbToken_= consumes<reco::MuonTimeExtraMap>(edm::InputTag(theMuonsCollectionLabel.label(),"combined")) ;
    timeMapDTToken_ = consumes<reco::MuonTimeExtraMap>(edm::InputTag(theMuonsCollectionLabel.label(),"dt")) ;
    timeMapCSCToken_= consumes<reco::MuonTimeExtraMap>(edm::InputTag(theMuonsCollectionLabel.label(),"csc")) ;

    produces<reco::MuonTimeExtraMap>("combined");
    produces<reco::MuonTimeExtraMap>("dt");
    produces<reco::MuonTimeExtraMap>("csc");
  }
  
  if (fillDetectorBasedIsolation_){
    theTrackDepositName = pSet.getParameter<edm::InputTag>("TrackIsoDeposits");
    theTrackDepositToken_ = consumes<reco::IsoDepositMap>(theTrackDepositName);
    produces<reco::IsoDepositMap>(labelOrInstance(theTrackDepositName));

    theJetDepositName = pSet.getParameter<edm::InputTag>("JetIsoDeposits");
    theJetDepositToken_ = consumes<reco::IsoDepositMap>(theJetDepositName);
    produces<reco::IsoDepositMap>(labelOrInstance(theJetDepositName));

    theEcalDepositName = pSet.getParameter<edm::InputTag>("EcalIsoDeposits");
    theEcalDepositToken_ = consumes<reco::IsoDepositMap>(theEcalDepositName);
    produces<reco::IsoDepositMap>(theEcalDepositName.instance());

    theHcalDepositName = pSet.getParameter<edm::InputTag>("HcalIsoDeposits");
    theHcalDepositToken_ = consumes<reco::IsoDepositMap>(theHcalDepositName);
    produces<reco::IsoDepositMap>(theHcalDepositName.instance());

    theHoDepositName = pSet.getParameter<edm::InputTag>("HoIsoDeposits");
    theHoDepositToken_ = consumes<reco::IsoDepositMap>(theHoDepositName);

    produces<reco::IsoDepositMap>(theHoDepositName.instance());
  }
  
  if(fillSelectors_){
    theSelectorMapNames = pSet.getParameter<InputTags>("SelectorMaps");
    
    for(InputTags::const_iterator tag = theSelectorMapNames.begin(); tag != theSelectorMapNames.end(); ++tag) {
      theSelectorMapTokens_.push_back(consumes<edm::ValueMap<bool> >(*tag));
      produces<edm::ValueMap<bool> >(labelOrInstance(*tag));
    }
  
  }

  if(fillShoweringInfo_){
    theShowerMapName = pSet.getParameter<edm::InputTag>("ShowerInfoMap");
    theShowerMapToken_ = consumes<reco::MuonShowerMap>(theShowerMapName);
    produces<edm::ValueMap<reco::MuonShower> >(labelOrInstance(theShowerMapName));
  }
  
  if(fillCosmicsIdMap_){
    theCosmicCompMapName = pSet.getParameter<edm::InputTag>("CosmicIdMap");
    theCosmicIdMapToken_ = consumes<edm::ValueMap<unsigned int> >(theCosmicCompMapName);
    theCosmicCompMapToken_ = consumes<edm::ValueMap<reco::MuonCosmicCompatibility> >(theCosmicCompMapName);

    produces<edm::ValueMap<reco::MuonCosmicCompatibility> >(labelOrInstance(theCosmicCompMapName));
    produces<edm::ValueMap<unsigned int> >(labelOrInstance(theCosmicCompMapName));
  }
  
  theMuToMuMapName = theMuonsCollectionLabel.label()+"2"+theAlias+"sMap";
  produces<edm::ValueMap<reco::MuonRef> >(theMuToMuMapName);



  
  if(fillPFIsolation_){
    
    edm::ParameterSet pfIsoPSet = pSet.getParameter<edm::ParameterSet >("PFIsolation");

    //Define a map between the isolation and the PSet for the PFHelper
    std::map<std::string,edm::ParameterSet> psetMap;

    //First declare what isolation you are going to read
    std::vector<std::string> isolationLabels;
    isolationLabels.push_back("pfIsolationR03");
    isolationLabels.push_back("pfIsoMeanDRProfileR03");
    isolationLabels.push_back("pfIsoSumDRProfileR03");
    isolationLabels.push_back("pfIsolationR04");
    isolationLabels.push_back("pfIsoMeanDRProfileR04");
    isolationLabels.push_back("pfIsoSumDRProfileR04");

    //Fill the label,pet map and initialize MuPFIsoHelper
    for( std::vector<std::string>::const_iterator label = isolationLabels.begin();label != isolationLabels.end();++label)
      psetMap[*label] =pfIsoPSet.getParameter<edm::ParameterSet >(*label); 

    thePFIsoHelper = new MuPFIsoHelper(psetMap,consumesCollector());

    //Now loop on the mass read for each PSet the parameters and save them to the mapNames for later

    for(std::map<std::string,edm::ParameterSet>::const_iterator map = psetMap.begin();map!= psetMap.end();++map) {
      std::map<std::string,edm::InputTag> isoMap;
      isoMap["chargedParticle"]              = map->second.getParameter<edm::InputTag>("chargedParticle");
      isoMap["chargedHadron"]                = map->second.getParameter<edm::InputTag>("chargedHadron");
      isoMap["neutralHadron"]                = map->second.getParameter<edm::InputTag>("neutralHadron");
      isoMap["neutralHadronHighThreshold"]   = map->second.getParameter<edm::InputTag>("neutralHadronHighThreshold");
      isoMap["photon"]                       = map->second.getParameter<edm::InputTag>("photon");
      isoMap["photonHighThreshold"]          = map->second.getParameter<edm::InputTag>("photonHighThreshold");
      isoMap["pu"]                           = map->second.getParameter<edm::InputTag>("pu");

      std::map<std::string,edm::EDGetTokenT<edm::ValueMap<double> > > isoMapToken;
      isoMapToken["chargedParticle"]             = consumes<edm::ValueMap<double> >(isoMap["chargedParticle"]); 
      isoMapToken["chargedHadron"]               = consumes<edm::ValueMap<double> >(isoMap["chargedHadron"]);  
      isoMapToken["neutralHadron"]               = consumes<edm::ValueMap<double> >(isoMap["neutralHadron"]);  
      isoMapToken["neutralHadronHighThreshold"]  = consumes<edm::ValueMap<double> >(isoMap["neutralHadronHighThreshold"]);  
      isoMapToken["photon"]                      = consumes<edm::ValueMap<double> >(isoMap["photon"]);  
      isoMapToken["photonHighThreshold"]         = consumes<edm::ValueMap<double> >(isoMap["photonHighThreshold"]);  
      isoMapToken["pu"]                          = consumes<edm::ValueMap<double> >(isoMap["pu"]);  
      

      pfIsoMapNames.push_back(isoMap);
      pfIsoMapTokens_.push_back(isoMapToken);
    }


    for(unsigned int j=0;j<pfIsoMapNames.size();++j) {
      for(std::map<std::string,edm::InputTag>::const_iterator map = pfIsoMapNames.at(j).begin(); map != pfIsoMapNames.at(j).end(); ++map)
	produces<edm::ValueMap<double> >(labelOrInstance(map->second));

    }
    
  }
  
  if (computeStandardSelectors_){
    vertexes_ = consumes<reco::VertexCollection>(pSet.getParameter<edm::InputTag>("vertices"));
  }

}

/// Destructor
MuonProducer::~MuonProducer(){ 
  if (thePFIsoHelper && fillPFIsolation_) delete thePFIsoHelper;
}


/// reconstruct muons
void MuonProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup){

   const std::string metname = "Muon|RecoMuon|MuonIdentification|MuonProducer";

   // the muon collection, it will be loaded in the event
   auto outputMuons = std::make_unique<reco::MuonCollection>();
   reco::MuonRefProd outputMuonsRefProd = event.getRefBeforePut<reco::MuonCollection>();

   edm::Handle<reco::MuonCollection> inputMuons; 
   event.getByToken(theMuonsCollectionToken_, inputMuons);
   edm::OrphanHandle<reco::MuonCollection> inputMuonsOH(inputMuons.product(), inputMuons.id());
   
   edm::Handle<reco::PFCandidateCollection> pfCandidates; 
   event.getByToken(thePFCandToken_, pfCandidates);

   edm::Handle<reco::VertexCollection> primaryVertices;
   const reco::Vertex* vertex(nullptr);
   if (computeStandardSelectors_){
     event.getByToken(vertexes_, primaryVertices);
     if (!primaryVertices->empty())
       vertex = &(primaryVertices->front());
   }

   // fetch collections for PFIso
   if(fillPFIsolation_) thePFIsoHelper->beginEvent(event);

   
   // Fill timing information
   edm::Handle<reco::MuonTimeExtraMap> timeMapCmb;
   edm::Handle<reco::MuonTimeExtraMap> timeMapDT;
   edm::Handle<reco::MuonTimeExtraMap> timeMapCSC;
   
   int nMuons=inputMuons->size();

   std::vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
   std::vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
   std::vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);

   if(fillTimingInfo_){
     event.getByToken(timeMapCmbToken_,timeMapCmb);
     event.getByToken(timeMapDTToken_,timeMapDT);
     event.getByToken(timeMapCSCToken_,timeMapCSC);
   }

   std::vector<reco::IsoDeposit> trackDepColl(nMuons);
   std::vector<reco::IsoDeposit> ecalDepColl(nMuons);
   std::vector<reco::IsoDeposit> hcalDepColl(nMuons);
   std::vector<reco::IsoDeposit> hoDepColl(nMuons);
   std::vector<reco::IsoDeposit> jetDepColl(nMuons);


   edm::Handle<reco::IsoDepositMap> trackIsoDepMap;
   edm::Handle<reco::IsoDepositMap> ecalIsoDepMap;
   edm::Handle<reco::IsoDepositMap> hcalIsoDepMap;
   edm::Handle<reco::IsoDepositMap> hoIsoDepMap;
   edm::Handle<reco::IsoDepositMap> jetIsoDepMap;


   if(fillDetectorBasedIsolation_){
     event.getByToken(theTrackDepositToken_,trackIsoDepMap);
     event.getByToken(theEcalDepositToken_,ecalIsoDepMap);
     event.getByToken(theHcalDepositToken_,hcalIsoDepMap);
     event.getByToken(theHoDepositToken_,hoIsoDepMap);
     event.getByToken(theJetDepositToken_,jetIsoDepMap);
   }

   

   std::vector<std::map<std::string,edm::Handle<edm::ValueMap<double> > > > pfIsoMaps;
   std::vector<std::map<std::string,std::vector<double> > > pfIsoMapVals;

   if(fillPFIsolation_){
    for(unsigned int j=0;j<pfIsoMapNames.size();++j) {
	std::map<std::string,std::vector<double> > mapVals;
	std::map<std::string,edm::Handle<edm::ValueMap<double> > > maps;
	for(std::map<std::string,edm::InputTag>::const_iterator map = pfIsoMapNames.at(j).begin(); map != pfIsoMapNames.at(j).end(); ++map) {
	  edm::Handle<edm::ValueMap<double> > handleTmp;
	  event.getByToken(pfIsoMapTokens_.at(j)[map->first],handleTmp);
	  maps[map->first]=handleTmp;
	  mapVals[map->first].resize(nMuons);
	}
	pfIsoMapVals.push_back(mapVals);
	pfIsoMaps.push_back(maps);

    }
   }


   
   std::vector<edm::Handle<edm::ValueMap<bool> > >  selectorMaps(fillSelectors_ ? theSelectorMapNames.size() : 0); 
   std::vector<std::vector<bool> > selectorMapResults(fillSelectors_ ? theSelectorMapNames.size() : 0);
   if(fillSelectors_){
     unsigned int s=0;
     for(InputTags::const_iterator tag = theSelectorMapNames.begin(); tag != theSelectorMapNames.end(); ++tag, ++s){
       event.getByToken(theSelectorMapTokens_.at(s),selectorMaps[s]);
       selectorMapResults[s].resize(nMuons);
     }
   }

   edm::Handle<reco::MuonShowerMap> showerInfoMap;
   if(fillShoweringInfo_) event.getByToken(theShowerMapToken_,showerInfoMap);

   std::vector<reco::MuonShower> showerInfoColl(nMuons);

   edm::Handle<edm::ValueMap<unsigned int> > cosmicIdMap;
   if(fillCosmicsIdMap_) event.getByToken(theCosmicIdMapToken_,cosmicIdMap);
   std::vector<unsigned int> cosmicIdColl(fillCosmicsIdMap_ ? nMuons : 0);
   
   edm::Handle<edm::ValueMap<reco::MuonCosmicCompatibility> > cosmicCompMap;
   if(fillCosmicsIdMap_) event.getByToken(theCosmicCompMapToken_,cosmicCompMap);
   std::vector<reco::MuonCosmicCompatibility> cosmicCompColl(fillCosmicsIdMap_ ? nMuons : 0);


   std::vector<reco::MuonRef> muonRefColl(nMuons);



   if(inputMuons->empty()) {
     edm::OrphanHandle<reco::MuonCollection> muonHandle = event.put(std::move(outputMuons));
     
     if(fillTimingInfo_){
       fillMuonMap<reco::MuonTimeExtra>(event, muonHandle, combinedTimeColl,"combined");
       fillMuonMap<reco::MuonTimeExtra>(event, muonHandle, dtTimeColl,"dt");
       fillMuonMap<reco::MuonTimeExtra>(event, muonHandle, cscTimeColl,"csc");
     }
     
     if(fillDetectorBasedIsolation_){
       fillMuonMap<reco::IsoDeposit>(event, muonHandle, trackDepColl, labelOrInstance(theTrackDepositName));
       fillMuonMap<reco::IsoDeposit>(event, muonHandle, jetDepColl,   labelOrInstance(theJetDepositName));
       fillMuonMap<reco::IsoDeposit>(event, muonHandle, ecalDepColl,  theEcalDepositName.instance());
       fillMuonMap<reco::IsoDeposit>(event, muonHandle, hcalDepColl,  theHcalDepositName.instance());
       fillMuonMap<reco::IsoDeposit>(event, muonHandle, hoDepColl,    theHoDepositName.instance());
     }

     if(fillSelectors_){
       unsigned int s = 0;
       for(InputTags::const_iterator tag = theSelectorMapNames.begin(); 
	   tag != theSelectorMapNames.end(); ++tag, ++s){
	 fillMuonMap<bool>(event, muonHandle, selectorMapResults[s], labelOrInstance(*tag));
       }
     }

     if(fillShoweringInfo_) fillMuonMap<reco::MuonShower>(event, muonHandle, showerInfoColl, labelOrInstance(theShowerMapName));

     if(fillCosmicsIdMap_){
       fillMuonMap<unsigned int>(event, muonHandle, cosmicIdColl, labelOrInstance(theCosmicCompMapName));
       fillMuonMap<reco::MuonCosmicCompatibility>(event, muonHandle, cosmicCompColl, labelOrInstance(theCosmicCompMapName));
     }

     fillMuonMap<reco::MuonRef>(event,inputMuonsOH, muonRefColl, theMuToMuMapName);

     if(fillPFIsolation_){
       for(unsigned int j=0;j<pfIsoMapNames.size();++j) 
	 for(std::map<std::string,edm::InputTag>::const_iterator map = pfIsoMapNames.at(j).begin(); map != pfIsoMapNames.at(j).end(); ++map) {
	   fillMuonMap<double>(event, muonHandle, pfIsoMapVals.at(j)[map->first], labelOrInstance(map->second));
	}
     }
     return;
   }
   
   // FIXME: add the option to swith off the Muon-PF "info association".
   

   MuToPFMap muToPFMap;

   if(fillPFMomentum_){
     dout << "Number of PFCandidates: " << pfCandidates->size() << endl;
     for(unsigned int i=0;i< pfCandidates->size();++i)
       if(abs(pfCandidates->at(i).pdgId()) == 13){
	 muToPFMap[pfCandidates->at(i).muonRef()] = i;
	 dout << "MuonRef: " << pfCandidates->at(i).muonRef().id() << " " << pfCandidates->at(i).muonRef().key() << " PF p4: " << pfCandidates->at(i).p4() << endl;
       }
     dout << "Number of PFMuons: " << muToPFMap.size() << endl;
     dout << "Number of Muons in the original collection: " << inputMuons->size() << endl;
   }

   reco::MuonRef::key_type muIndex = 0;
   unsigned int i = 0;
   for(auto const& inMuon : *inputMuons){
     
     reco::MuonRef muRef(inputMuons, muIndex);
     muonRefColl[i] = reco::MuonRef(outputMuonsRefProd, muIndex++);

     // Copy the muon 
     reco::Muon outMuon = inMuon;
    
     if(fillPFMomentum_){ 
     // search for the corresponding pf candidate
       MuToPFMap::iterator iter =  muToPFMap.find(muRef);
       if(iter != muToPFMap.end()){
	 const auto& pfMu = pfCandidates->at(iter->second);
	 outMuon.setPFP4(pfMu.p4());
	 outMuon.setP4(pfMu.p4());//PF is the default
	 outMuon.setCharge(pfMu.charge());//PF is the default
	 outMuon.setPdgId(-13*pfMu.charge());
	 outMuon.setBestTrack(pfMu.bestMuonTrackType());
	 muToPFMap.erase(iter);
	 dout << "MuonRef: " << muRef.id() << " " << muRef.key() 
	      << " Is it PF? " << outMuon.isPFMuon() 
	      << " PF p4: " << outMuon.pfP4() << endl;
       }
       
       
       dout << "MuonRef: " << muRef.id() << " " << muRef.key() 
	    << " Is it PF? " << outMuon.isPFMuon() << endl;
       
       dout << "GLB "  << outMuon.isGlobalMuon()
	    << " TM "  << outMuon.isTrackerMuon()
	    << " STA " << outMuon.isStandAloneMuon() 
	    << " p4 "  << outMuon.p4() << endl;
     }

     // Add PF isolation info
     if(fillPFIsolation_){
       thePFIsoHelper->embedPFIsolation(outMuon,muRef);

       for(unsigned int j=0;j<pfIsoMapNames.size();++j) {
	 for(std::map<std::string,edm::InputTag>::const_iterator map = pfIsoMapNames[j].begin(); map != pfIsoMapNames[j].end(); ++map){
	   (pfIsoMapVals[j])[map->first][i] = (*pfIsoMaps[j][map->first])[muRef];
	 }
       }
     }
     
     // Fill timing information   
     if(fillTimingInfo_){
       combinedTimeColl[i] = (*timeMapCmb)[muRef];
       dtTimeColl[i] = (*timeMapDT)[muRef];
       cscTimeColl[i] = (*timeMapCSC)[muRef];
     }
     
     if(fillDetectorBasedIsolation_){
       trackDepColl[i] = (*trackIsoDepMap)[muRef];
       ecalDepColl[i]  = (*ecalIsoDepMap)[muRef];
       hcalDepColl[i]  = (*hcalIsoDepMap)[muRef];
       hoDepColl[i]    = (*hoIsoDepMap)[muRef];
       jetDepColl[i]   = (*jetIsoDepMap)[muRef];;
     }
     
     if(fillSelectors_){
       unsigned int s = 0;
       for(InputTags::const_iterator tag = theSelectorMapNames.begin(); 
	   tag != theSelectorMapNames.end(); ++tag, ++s)
	 selectorMapResults[s][i] = (*selectorMaps[s])[muRef];
     }

     // Fill the Showering Info
     if(fillShoweringInfo_) showerInfoColl[i] = (*showerInfoMap)[muRef];

     if(fillCosmicsIdMap_){
       cosmicIdColl[i] = (*cosmicIdMap)[muRef];
       cosmicCompColl[i] = (*cosmicCompMap)[muRef];
     }
     
     // Standard Selectors - keep it at the end so that all inputs are available
     if (computeStandardSelectors_){
       outMuon.setSelectors(0); // reset flags
       bool isRun2016BCDEF = (272728 <= event.run() && event.run() <= 278808);
       muon::setCutBasedSelectorFlags(outMuon, vertex, isRun2016BCDEF);
     }

     outputMuons->push_back(outMuon); 
     ++i;
   }
   
   dout << "Number of Muons in the new muon collection: " << outputMuons->size() << endl;
   edm::OrphanHandle<reco::MuonCollection> muonHandle = event.put(std::move(outputMuons));

   if(fillTimingInfo_){
     fillMuonMap<reco::MuonTimeExtra>(event, muonHandle, combinedTimeColl,"combined");
     fillMuonMap<reco::MuonTimeExtra>(event, muonHandle, dtTimeColl,"dt");
     fillMuonMap<reco::MuonTimeExtra>(event, muonHandle, cscTimeColl,"csc");
   }

   if(fillDetectorBasedIsolation_){
     fillMuonMap<reco::IsoDeposit>(event, muonHandle, trackDepColl, labelOrInstance(theTrackDepositName));
     fillMuonMap<reco::IsoDeposit>(event, muonHandle, jetDepColl,   labelOrInstance(theJetDepositName));
     fillMuonMap<reco::IsoDeposit>(event, muonHandle, ecalDepColl,  theEcalDepositName.instance());
     fillMuonMap<reco::IsoDeposit>(event, muonHandle, hcalDepColl,  theHcalDepositName.instance());
     fillMuonMap<reco::IsoDeposit>(event, muonHandle, hoDepColl,    theHoDepositName.instance());
   }
   
   if(fillPFIsolation_){

     for(unsigned int j=0;j<pfIsoMapNames.size();++j) {
       for(std::map<std::string,edm::InputTag>::const_iterator map = pfIsoMapNames[j].begin(); map != pfIsoMapNames[j].end(); ++map) 
	 fillMuonMap<double>(event, muonHandle, pfIsoMapVals[j][map->first], labelOrInstance(map->second));
     }
   }   

   if(fillSelectors_){
     unsigned int s = 0;
     for(InputTags::const_iterator tag = theSelectorMapNames.begin(); 
	 tag != theSelectorMapNames.end(); ++tag, ++s)
       fillMuonMap<bool>(event, muonHandle, selectorMapResults[s], labelOrInstance(*tag));
   }

   if(fillShoweringInfo_) fillMuonMap<reco::MuonShower>(event, muonHandle, showerInfoColl, labelOrInstance(theShowerMapName));

   if(fillCosmicsIdMap_){
     fillMuonMap<unsigned int>(event, muonHandle, cosmicIdColl, labelOrInstance(theCosmicCompMapName));
     fillMuonMap<reco::MuonCosmicCompatibility>(event, muonHandle, cosmicCompColl, labelOrInstance(theCosmicCompMapName));
   }

   fillMuonMap<reco::MuonRef>(event,inputMuonsOH, muonRefColl, theMuToMuMapName);
   
 
}



template<typename TYPE>
void MuonProducer::fillMuonMap(edm::Event& event,
			       const edm::OrphanHandle<reco::MuonCollection>& muonHandle,
			       const std::vector<TYPE>& muonExtra,
			       const std::string& label){
 
  typedef typename edm::ValueMap<TYPE>::Filler FILLER; 

  auto muonMap = std::make_unique<edm::ValueMap<TYPE>>();
  if(!muonExtra.empty()){
    FILLER filler(*muonMap);
    filler.insert(muonHandle, muonExtra.begin(), muonExtra.end());
    filler.fill();
  }
  event.put(std::move(muonMap),label);
}


std::string MuonProducer::labelOrInstance(const edm::InputTag &input) const{
  if(fastLabelling_) return input.label();

  return input.label() != theMuonsCollectionLabel.label() ? input.label() : input.instance();
}
