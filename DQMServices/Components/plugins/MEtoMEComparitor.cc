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
// $Id: MEtoMEComparitor.cc,v 1.6 2011/02/01 19:17:38 vlimant Exp $
//
//

#include "MEtoMEComparitor.h"

#include "classlib/utils/StringList.h"
#include "classlib/utils/StringOps.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"

MEtoMEComparitor::MEtoMEComparitor(const edm::ParameterSet& iConfig)

{
  _moduleLabel = iConfig.getParameter<std::string>("MEtoEDMLabel");
  
  _lumiInstance = iConfig.getParameter<std::string>("lumiInstance");
  _runInstance = iConfig.getParameter<std::string>("runInstance");

  _process_ref = iConfig.getParameter<std::string>("processRef");
  _process_new = iConfig.getParameter<std::string>("processNew");

  _autoProcess=iConfig.getParameter<bool>("autoProcess");

  _KSgoodness = iConfig.getParameter<double>("KSgoodness");
  _diffgoodness = iConfig.getParameter<double>("Diffgoodness");
  _dirDepth = iConfig.getParameter<unsigned int>("dirDepth");
  _overallgoodness = iConfig.getParameter<double>("OverAllgoodness");
  
  _dbe = edm::Service<DQMStore>().operator->();


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
MEtoMEComparitor::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  if (_autoProcess)
    {
      const edm::ProcessHistory& iHistory=iRun.processHistory();

      edm::ProcessHistory::const_reverse_iterator hi=iHistory.rbegin();
      _process_new=hi->processName();
      hi++;
      _process_ref=hi->processName();
      std::cout<<_process_ref<<" vs "<<_process_new<<std::endl;
    }
}

void
MEtoMEComparitor::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  compare<edm::Run,TH1S>(iRun,_runInstance);
  compare<edm::Run,TH1F>(iRun,_runInstance);
  compare<edm::Run,TH1D>(iRun,_runInstance);

}


template <class T>
void MEtoMEComparitor::book(const std::string & directory,const std::string & type, const T * h){
  _dbe->setCurrentFolder(type);
  std::type_info const & tp = typeid(*h);
  if (tp == typeid(TH1S))
    _dbe->book1S(h->GetName(),dynamic_cast<TH1S*>(const_cast<T*>(h)));
  else if (tp == typeid(TH1F))
    _dbe->book1D(h->GetName(),dynamic_cast<TH1F*>(const_cast<T*>(h)));
  else if (tp == typeid(TH1D))
    _dbe->book1DD(h->GetName(),dynamic_cast<TH1D*>(const_cast<T*>(h)));
}
template <class T> 
void MEtoMEComparitor::keepBadHistograms(const std::string & directory, const T * h_new, const T * h_ref){
  //put it in a collection rather.

  
  std::string d_n(h_new->GetName());
  d_n+="_diff";
  T * difference = new T(d_n.c_str(),
			 h_new->GetTitle(),
			 h_new->GetNbinsX(),
			 h_new->GetXaxis()->GetXmin(),
			 h_new->GetXaxis()->GetXmax());
  difference->Add(h_new);
  difference->Add(h_ref,-1.);

  book(directory,"Ref",h_ref);
  book(directory,"New",h_new);
  book(directory,"Diff",difference);
  delete difference;

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

  LogDebug("MEtoMEComparitor")<<"found "<<mapping.size()<<" pairs of plots";
  countMe=0;

  unsigned int nNoMatch=0;
  unsigned int nEmpty=0;
  unsigned int nHollow=0;
  unsigned int nGoodKS=0;
  unsigned int nBadKS=0;
  unsigned int nBadDiff=0;
  unsigned int nGoodDiff=0;

  typedef std::map<std::string, std::pair<unsigned int,unsigned int> > Subs;
  Subs subSystems;

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

    lat::StringList dir = lat::StringOps::split(it->second.second->name,"/");
    std::string subsystem = dir[0];
    if (dir.size()>=_dirDepth)
      for (unsigned int iD=1;iD!=_dirDepth;++iD) subsystem+="/"+dir[iD];
    subSystems[subsystem].first++;

    if (h_ref->GetEntries()!=0 && h_ref->Integral()!=0){
      double KS=0;
      bool cannotComputeKS=false;
      try {
	KS = h_new->KolmogorovTest(h_ref);
      }
      catch( cms::Exception& exception ){
	cannotComputeKS=true;
      }
      if (KS<_KSgoodness){

	unsigned int total_ref=0;
	unsigned int absdiff=0;
	for (unsigned int iBin=0;
	     iBin!=(unsigned int)h_new->GetNbinsX()+1 ;
	     ++iBin){
	  total_ref+=h_ref->GetBinContent(iBin);
	  absdiff=std::abs(h_new->GetBinContent(iBin) - h_ref->GetBinContent(iBin));
	}
	double relativediff=1;
	if (total_ref!=0){
	  relativediff=absdiff / (double) total_ref;
	}
	if (relativediff > _diffgoodness ){
	  edm::LogWarning("MEtoMEComparitor")<<"for "<<h_new->GetName()
					     <<" in "<<it->first    
					     <<" the KS is "<<KS*100.<<" %"
					     <<" and the relative diff is: "<<relativediff*100.<<" %"
					     <<" KS"<<((cannotComputeKS)?" not valid":" is valid");
	  //std::string(" KolmogorovTest is not happy on : ")+h_new->GetName() : "";
	  //there you want to output the plots somewhere
	  keepBadHistograms(subsystem,h_new,h_ref);
	  
	  nBadDiff++;
	  subSystems[subsystem].second++;
	}else{
	  nGoodDiff++;
	}
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
    std::stringstream summary;
    summary<<" Summary :"
	   <<"\n not matched : "<<nNoMatch
	   <<"\n empty : "<<nEmpty
	   <<"\n integral zero : "<<nHollow
	   <<"\n good KS : "<<nGoodKS
	   <<"\n bad KS : "<<nBadKS
	   <<"\n bad diff : "<<nBadDiff
	   <<"\n godd diff : "<<nGoodDiff;
    bool tell=false;
    for (Subs::iterator iSub=subSystems.begin();
	 iSub!=subSystems.end();++iSub){
      double fraction = 1-(iSub->second.second / (double)iSub->second.first);
      summary<<std::endl<<"Subsytem: "<<iSub->first<<" has "<< fraction*100<<" % goodness";
      if (fraction < _overallgoodness)
	tell=true;
    }
    if (tell)
      edm::LogWarning("MEtoMEComparitor")<<summary.str();
    else
      edm::LogInfo("MEtoMEComparitor")<<summary.str();
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


