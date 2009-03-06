////////////////////////////////////////////////////////////////////////////////
//                    Header file for this                                    //
////////////////////////////////////////////////////////////////////////////////
#include "HLTriggerOffline/Egamma/interface/EmDQM.h"

////////////////////////////////////////////////////////////////////////////////
//                    Collaborating Class Header                              //
////////////////////////////////////////////////////////////////////////////////
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/Exception.h"

////////////////////////////////////////////////////////////////////////////////
//                           Root include files                               //
////////////////////////////////////////////////////////////////////////////////
#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"
#include <iostream>
#include <string>
#include <Math/VectorUtil.h>
using namespace ROOT::Math::VectorUtil ;


////////////////////////////////////////////////////////////////////////////////
//                             Constructor                                    //
////////////////////////////////////////////////////////////////////////////////
EmDQM::EmDQM(const edm::ParameterSet& pset)  
{



  dbe = edm::Service < DQMStore > ().operator->();
  dbe->setVerbose(0);


  ////////////////////////////////////////////////////////////
  //          Read from configuration file                  //
  ////////////////////////////////////////////////////////////
  dirname_="HLT/HLTEgammaValidation/"+pset.getParameter<std::string>("@module_label");
  dbe->setCurrentFolder(dirname_);

  // paramters for generator study 
  reqNum    = pset.getParameter<unsigned int>("reqNum");
  pdgGen    = pset.getParameter<int>("pdgGen");
  genEtaAcc = pset.getParameter<double>("genEtaAcc");
  genEtAcc  = pset.getParameter<double>("genEtAcc");
  // plotting paramters (untracked because they don't affect the physics)
  thePtMin  = pset.getUntrackedParameter<double>("PtMin",0.);
  thePtMax  = pset.getUntrackedParameter<double>("PtMax",1000.);
  theNbins  = pset.getUntrackedParameter<unsigned int>("Nbins",40);

  //preselction cuts 
  gencutCollection_= pset.getParameter<edm::InputTag>("cutcollection");
  gencut_          = pset.getParameter<int>("cutnum");

  ////////////////////////////////////////////////////////////
  //         Read in the Vector of Parameter Sets.          //
  //           Information for each filter-step             //
  ////////////////////////////////////////////////////////////
  std::vector<edm::ParameterSet> filters = 
       pset.getParameter<std::vector<edm::ParameterSet> >("filters");

  for(std::vector<edm::ParameterSet>::iterator filterconf = filters.begin() ; filterconf != filters.end() ; filterconf++)
  {

    theHLTCollectionLabels.push_back(filterconf->getParameter<edm::InputTag>("HLTCollectionLabels"));
    theHLTOutputTypes.push_back(filterconf->getParameter<unsigned int>("theHLTOutputTypes"));
    std::vector<double> bounds = filterconf->getParameter<std::vector<double> >("PlotBounds");
    // If the size of plot "bounds" vector != 2, abort
    assert(bounds.size() == 2);
    plotBounds.push_back(std::pair<double,double>(bounds[0],bounds[1]));
    isoNames.push_back(filterconf->getParameter<std::vector<edm::InputTag> >("IsoCollections"));
    // If the size of the isoNames vector is not greater than zero, abort
    assert(isoNames.back().size()>0);
    if (isoNames.back().at(0).label()=="none")
      plotiso.push_back(false);
    else
      plotiso.push_back(true);

  } // END of loop over parameter sets
}


////////////////////////////////////////////////////////////////////////////////
//       method called once each job just before starting event loop          //
////////////////////////////////////////////////////////////////////////////////
void 
EmDQM::beginJob(const edm::EventSetup&)
{
  //edm::Service<TFileService> fs;
  dbe->setCurrentFolder(dirname_);
  
  std::string histoname="total eff";

  total = dbe->book1D(histoname.c_str(),histoname.c_str(),theHLTCollectionLabels.size()+2,0,theHLTCollectionLabels.size()+2);
  total->setBinLabel(theHLTCollectionLabels.size()+1,"Total");
  total->setBinLabel(theHLTCollectionLabels.size()+2,"Gen");
  for (unsigned int u=0; u<theHLTCollectionLabels.size(); u++){total->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());}

  MonitorElement* tmphisto;
  MonitorElement* tmpiso;

  histoname = "gen et";
  etgen =  dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
  histoname = "gen eta";
  etagen = dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
 
  for(unsigned int i = 0; i< theHLTCollectionLabels.size() ; i++){
    histoname = theHLTCollectionLabels[i].label()+"et";
    tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
    ethist.push_back(tmphisto);
    
    histoname = theHLTCollectionLabels[i].label()+"eta";
    tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
    etahist.push_back(tmphisto);          

    histoname = theHLTCollectionLabels[i].label()+"et MC matched";
    tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax);
    ethistmatch.push_back(tmphisto);
    
    histoname = theHLTCollectionLabels[i].label()+"eta MC matched";
    tmphisto =  dbe->book1D(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7);
    etahistmatch.push_back(tmphisto);          
    
    if(plotiso[i]) {
      histoname = theHLTCollectionLabels[i].label()+"eta isolation";
      tmpiso = dbe->book2D(histoname.c_str(),histoname.c_str(),theNbins,-2.7,2.7,theNbins,plotBounds[i].first,plotBounds[i].second);
    }
    else {
      tmpiso = NULL;
    }
    etahistiso.push_back(tmpiso);

    if(plotiso[i]){
      histoname = theHLTCollectionLabels[i].label()+"et isolation";
      tmpiso = dbe->book2D(histoname.c_str(),histoname.c_str(),theNbins,thePtMin,thePtMax,theNbins,plotBounds[i].first,plotBounds[i].second);
    }
    else{
      tmpiso = NULL;
    }
    ethistiso.push_back(tmpiso);

  }
}


////////////////////////////////////////////////////////////////////////////////
//                                Destructor                                  //
////////////////////////////////////////////////////////////////////////////////
EmDQM::~EmDQM(){
}


////////////////////////////////////////////////////////////////////////////////
//                     method called to for each event                        //
////////////////////////////////////////////////////////////////////////////////
void 
EmDQM::analyze(const edm::Event & event , const edm::EventSetup& setup)
{
  
  ////////////////////////////////////////////////////////////
  //           Check if there's enough gen particles        //
  //             of interest                                //
  ////////////////////////////////////////////////////////////
  edm::Handle< edm::View<reco::Candidate> > cutCounter;
  event.getByLabel(gencutCollection_,cutCounter);
  if (cutCounter->size() < (unsigned int)gencut_) {
    //edm::LogWarning("EmDQM") << "Less than "<< reqNum <<" gen particles with pdgId=" << pdgGen;
    return;
  }


  // fill L1 and HLT info
  // get objects possed by each filter
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  event.getByLabel("hltTriggerSummaryRAW",triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("EmDQM") << "RAW-type HLT results not found, skipping event";
    return;
  }

  // total event number
  total->Fill(theHLTCollectionLabels.size()+0.5);


  ////////////////////////////////////////////////////////////
  //               Fill generator info                      //
  ////////////////////////////////////////////////////////////
  edm::Handle<edm::HepMCProduct> genEvt;
  event.getByLabel("source", genEvt);
  
  std::vector<HepMC::GenParticle> mcparts;
  const HepMC::GenEvent * myGenEvent = genEvt->GetEvent();
  unsigned int ncand = 0;
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) {

    // If the ID number is not what we're looking for or 
    //  it's status is !=1, go to the next particle
    if (  !( abs((*p)->pdg_id())==pdgGen  && (*p)->status()==1 )  )  continue;
    float eta   = (*p)->momentum().eta();
    float e     = (*p)->momentum().e();
    float theta = 2*atan(exp(-eta));
    float Et    = e*sin(theta);
    if ( fabs(eta)<genEtaAcc  &&  Et > genEtAcc ) 
    {
      ncand++;
      etgen->Fill(Et);
      etagen->Fill(eta);
      mcparts.push_back(*(*p));
    }
  } // END of loop over Generated particles
  if (ncand >= reqNum) total->Fill(theHLTCollectionLabels.size()+1.5);
	  


  ////////////////////////////////////////////////////////////
  //            Loop over filter modules                    //
  ////////////////////////////////////////////////////////////
  for(unsigned int n=0; n < theHLTCollectionLabels.size() ; n++) {
    // These numbers are from the Parameter Set, such as:
    //   theHLTOutputTypes = cms.uint32(100)
    switch(theHLTOutputTypes[n]) 
    {
      case 82: // Non-isolated Level 1
        fillHistos<l1extra::L1EmParticleCollection>(triggerObj,event,n,mcparts);break;
      case 83: // Isolated Level 1
        fillHistos<l1extra::L1EmParticleCollection>(triggerObj,event,n,mcparts);break;
      case 91: // Photon 
        fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,event,n,mcparts);break;
      case 92: // Electron 
        fillHistos<reco::ElectronCollection>(triggerObj,event,n,mcparts);break;
      case 100: // TriggerCluster
        fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,event,n,mcparts);break;
      default: 
        throw(cms::Exception("Release Validation Error") << "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
  } // END of loop over filter modules
}


////////////////////////////////////////////////////////////////////////////////
// fillHistos                                                                 //
//   Called by analyze method.                                                //
//   
////////////////////////////////////////////////////////////////////////////////
template <class T> void EmDQM::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent ,unsigned int n,std::vector<HepMC::GenParticle>& mcparts)
{
  std::vector<edm::Ref<T> > recoecalcands;
  if (!( triggerObj->filterIndex(theHLTCollectionLabels[n])>=triggerObj->size() )){ // only process if available
  
    ////////////////////////////////////////////////////////////
    //      Retrieve saved filter objects                     //
    ////////////////////////////////////////////////////////////
    triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),theHLTOutputTypes[n],recoecalcands);
    //Danger: special case, L1 non-isolated
    // needs to be merged with L1 iso
    if (theHLTOutputTypes[n] == 82)
    {
      std::vector<edm::Ref<T> > isocands;
      triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),83,isocands);
      if (isocands.size()>0) 
      {
        for (unsigned int i=0; i < isocands.size(); i++)
	  recoecalcands.push_back(isocands[i]);
      }
    } // END of if theHLTOutputTypes == 82

    ////////////////////////////////////////////////////////////
    //        Fill filter objects into histograms             //
    ////////////////////////////////////////////////////////////
    if (recoecalcands.size() != 0)
    {
      if (recoecalcands.size() >= reqNum ) 
	total->Fill(n+0.5);
      for (unsigned int i=0; i<recoecalcands.size(); i++) {
	//unmatched
	ethist[n]->Fill(recoecalcands[i]->et() );
	etahist[n]->Fill(recoecalcands[i]->eta() );
	//matched
	math::XYZVector candDir=recoecalcands[i]->momentum();
	std::vector<HepMC::GenParticle>::iterator closest = mcparts.end();
	double closestDr = 1000. ;
	for(std::vector<HepMC::GenParticle>::iterator mc = mcparts.begin(); mc !=  mcparts.end() ; mc++){
	  math::XYZVector mcDir( mc->momentum().px(),
				 mc->momentum().py(),
				 mc->momentum().pz());	  
	  double dr = DeltaR(mcDir,candDir);
	  if (dr < closestDr) {
	    closestDr = dr;
	    closest = mc;
	  }
	}
	if (closest == mcparts.end())  
          edm::LogWarning("EmDQM") << "no MC match, may skew efficieny";
	else {
	  float eta   = closest->momentum().eta();
	  float e     = closest->momentum().e();
	  float theta = 2*atan(exp(-eta));
	  float Et    = e*sin(theta);
	  ethistmatch[n]->Fill( Et );
	  etahistmatch[n]->Fill( eta );
	}

	////////////////////////////////////////////////////////////
	//  Plot isolation variables (show the not-yet-cut        //
        //  isolation, i.e. associated to next filter)            //
	////////////////////////////////////////////////////////////
	if (n+1 < theHLTCollectionLabels.size()) // can't plot beyond last
        {
	  if (plotiso[n+1] ){
	    for (unsigned int j =  0 ; j < isoNames[n+1].size() ;j++  ){
	      edm::Handle<edm::AssociationMap<edm::OneToValue< T , float > > > depMap; 
	      iEvent.getByLabel(isoNames[n+1].at(j).label(),depMap);
	      if (depMap.isValid()){ //Map may not exist if only one candidate passes a double filter
		typename edm::AssociationMap<edm::OneToValue< T , float > >::const_iterator mapi = depMap->find(recoecalcands[i]);
		if (mapi!=depMap->end()){  // found candidate in isolation map! 
		  etahistiso[n+1]->Fill(recoecalcands[i]->eta(),mapi->val);
		  ethistiso[n+1]->Fill(recoecalcands[i]->et(),mapi->val);
		  break; // to avoid multiple filling we only look until we found the candidate once.
		}
	      }
	    }
	  }
	} // END of if n+1 < then the number of hlt collections
      }
    }
  }
}


//////////////////////////////////////////////////////////////////////////////// 
//      method called once each job just after ending the event loop          //
//////////////////////////////////////////////////////////////////////////////// 
void EmDQM::endJob(){
  // Normalize the histograms
  //  total->Scale(1./total->GetBinContent(1));
  //for(unsigned int n= theHLTCollectionLabels.size()-1 ; n>0;n--){
  //  ethist[n]->Divide(ethist[n-1]);
  //  etahist[n]->Divide(etahist[n-1]);
  //}
}

DEFINE_FWK_MODULE(EmDQM);
