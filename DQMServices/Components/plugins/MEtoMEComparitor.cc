// -*- C++ -*-
//
// Package:    MEtoMEComparitor
// Class:      MEtoMEComparitor
// 
/**\class MEtoMEComparitor MEtoMEComparitor.cc DQMOffline/MEtoMEComparitor/src/MEtoMEComparitor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  jean-roch Vlimant,40 3-A28,+41227671209,
//         Created:  Tue Nov 30 18:55:50 CET 2010
// $Id: MEtoMEComparitor.cc,v 1.2 2010/12/03 18:30:03 vlimant Exp $
//
//

#include "MEtoMEComparitor.h"

MEtoMEComparitor::MEtoMEComparitor(const edm::ParameterSet& iConfig)

{
  _moduleLabel = iConfig.getParameter<std::string>("MEtoEDMLabel");
  
  _lumiInstance = iConfig.getParameter<std::string>("lumiInstance");
  _runInstance = iConfig.getParameter<std::string>("runInstance");

  _process_ref = iConfig.getParameter<std::string>("processRef");
  _process_new = iConfig.getParameter<std::string>("processNew");

  if (iConfig.getParameter<bool>("autoProcess")){
    //get the last two process from the provenance

  }
  _KSgoodness = iConfig.getParameter<double>("KSgoodness");

  _dbe = edm::Service<DQMStore>().operator->();


  /*
    produces<MEtoEDM<TH1F>, edm::InLumi>("name");

    product<TH1S,edm::InLumi>();
    product<TH1F,edm::InLumi>();
    product<TH1D,edm::InLumi>();

    product<TH1S,edm::InRun>();
    product<TH1F,edm::InRun>();
    product<TH1D,edm::InRun>();
  */
}

template<class T,class W>
void
MEtoMEComparitor::product(){
  /*
    typedef typename MEtoEDM<T> prod;
    produces<prod,W>("ref");
    produces<prod,W>("new");
    produces<prod,W>("diff");
  */
}

MEtoMEComparitor::~MEtoMEComparitor()
{

}

void 
MEtoMEComparitor::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup&)
{

  compare<edm::LuminosityBlock,TH1S>(iLumi,_lumiInstance);
  compare<edm::LuminosityBlock,TH1F>(iLumi,_lumiInstance);
  compare<edm::LuminosityBlock,TH1D>(iLumi,_lumiInstance);
}

void
MEtoMEComparitor::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  compare<edm::Run,TH1S>(iRun,_runInstance);
  compare<edm::Run,TH1F>(iRun,_runInstance);
  compare<edm::Run,TH1D>(iRun,_runInstance);

}

template <class T> 
void keepBadHistograms(const T * h_new, const T * h_ref){
  //put it in a collection rather.

  /*  
  _dbe->setCurrentFolder("Differences");
  
  std::string d_n(h_new->GetName());
  d_n+="_diff";
  T * difference = new T(d_n,
			 h_new->GetTitle(),
			 h_new->GetNbinsX(),
			 h_new->GetXaxis()->GetXmin(),
			 h_new->GetXaxis()->GetXmax());
  difference->Add(h_new);
  difference->Add(h_ref,-1.);
  */
  

}


template <class W,
	  //class Wto,
	  class T>
void MEtoMEComparitor::compare(const W& where,const std::string & instance){

  edm::Handle<MEtoEDM<T> > metoedm_ref;
  edm::Handle<MEtoEDM<T> > metoedm_new;
  where.getByLabel(edm::InputTag(_moduleLabel,
				 instance,
				 _process_ref),
		   metoedm_ref);
  where.getByLabel(edm::InputTag(_moduleLabel,
				 instance,
				 _process_new),
		   metoedm_new);

  if (metoedm_ref.failedToGet() || metoedm_new.failedToGet()){
    edm::LogError("ProductNotFound")<<"MEtoMEComparitor did not find his products.";
    return;
  }

  typedef typename MEtoEDM<T>::MEtoEDMObject MEtoEDMObject; 
  
  const std::vector<MEtoEDMObject> & metoedmobject_ref = metoedm_ref->getMEtoEdmObject();
  const std::vector<MEtoEDMObject> & metoedmobject_new = metoedm_new->getMEtoEdmObject();

  typedef std::map<std::string, std::pair<const MEtoEDMObject*, const MEtoEDMObject*> > Mapping;
  typedef typename std::map<std::string, std::pair<const MEtoEDMObject*, const MEtoEDMObject*> >::iterator Mapping_iterator;

  Mapping mapping;

  LogDebug("MEtoMEComparitor")<<"going to do the mapping from "<<metoedmobject_ref.size()<<" x "<<metoedmobject_new.size();
  unsigned int countMe=0;
  for (unsigned int i_new=0; i_new!= metoedmobject_new.size(); ++i_new){
    const std::string & pathname = metoedmobject_new[i_new].name;
    if (metoedmobject_new[i_new].object.GetEntries()==0 ||
        metoedmobject_new[i_new].object.Integral()==0){
      countMe--;
      continue;
    }
    mapping[pathname]=std::make_pair(&metoedmobject_new[i_new],(const MEtoEDMObject*)0);
  }
  for (unsigned int i_ref=0; i_ref!= metoedmobject_ref.size() ; ++i_ref){
    const std::string & pathname = metoedmobject_ref[i_ref].name;
    Mapping_iterator there = mapping.find(pathname);
    if (there != mapping.end()){
      there->second.second = &metoedmobject_ref[i_ref];
    }
  }


  /*
  for (unsigned int i_new=0; i_new!= metoedmobject_new.size() &&  countMe++<= 200; ++i_new){
    const std::string & pathname = metoedmobject_new[i_new].name;
    //const std::string name(metoedmobject_new[i_new].object.GetName());
    
    if (metoedmobject_new[i_new].object.GetEntries()==0 ||
	metoedmobject_new[i_new].object.Integral()==0){
      countMe--;
      continue;
    }

    bool there=false;
    for (unsigned int i_ref=0; i_ref!= metoedmobject_ref.size() ; ++i_ref){
      if (metoedmobject_ref[i_ref].name == pathname){
	mapping[pathname]=std::make_pair(&metoedmobject_new[i_new], &metoedmobject_ref[i_ref]);
	there=true;
	break;
      }
    }
    if (!there)
    LogDebug("MEtoMEComparitor")<<metoedmobject_new[i_new].object.GetName()<<"is unmatched in "<<pathname;
  }
  */


  LogDebug("MEtoMEComparitor")<<"found "<<mapping.size()<<" pairs of plots";
  countMe=0;

  unsigned int nNoMatch=0;
  unsigned int nEmpty=0;
  unsigned int nHollow=0;
  unsigned int nGoodKS=0;
  unsigned int nBadKS=0;

  for (Mapping_iterator it = mapping.begin();
       it!=mapping.end(); 
       ++it){
    if (!it->second.second){
      //this is expected by how the map was created
      nNoMatch++;
      continue;
    }
    const T * h_ref = &it->second.second->object;
    const T * h_new = &it->second.first->object;


    if (h_ref->GetEntries()!=0 && h_ref->Integral()!=0){
      double KS=0;
      try {
	KS = h_new->KolmogorovTest(h_ref);
      }
      catch( cms::Exception& exception ){
	edm::LogInfo("MEtoMEComparitor")<<" KolmogorovTest is not happy on : "<<h_new->GetName()<<std::endl;
      }
      if (KS<_KSgoodness){
	edm::LogInfo("MEtoMEComparitor")<<"for "<<h_new->GetName()
					<<" in "<<it->first    
					<<" the KS is "<<KS;
	
	//there you want to output the plots somewhere
	keepBadHistograms(h_new,h_ref);

	nBadKS++;
      }
      else
	nGoodKS++;
    }
    else{
      if (h_ref->GetEntries()==0)
	nEmpty++;
      else
	nHollow++;
      LogDebug("MEtoMEComparitor")<<h_new->GetName()   <<" in "<<it->first    <<" is empty";
      countMe--;
    }
    
  }

  if (mapping.size()!=0){
    edm::LogInfo("MEtoMEComparitor")<<" Summary :"
	     <<"\n not matched : "<<nNoMatch
	     <<"\n empty : "<<nEmpty
	     <<"\n integral zero : "<<nHollow
	     <<"\n good KS : "<<nGoodKS
	     <<"\n bad KS : "<<nBadKS;
  }
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
MEtoMEComparitor::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MEtoMEComparitor::endJob() {
}


