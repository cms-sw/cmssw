// -*- C++ -*-
//
// Package:     FwkIO
// Class  :     DQMRootSource
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue May  3 11:13:47 CDT 2011
// $Id: DQMRootSource.cc,v 1.21 2011/07/07 16:46:53 chrjones Exp $
//

// system include files
#include <vector>
#include <string>
#include <map>
#include <list>
#include <set>
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

// user include files
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
//#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/FileBlock.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "FWCore/Utilities/interface/Digest.h"

#include "format.h"

namespace {
  //adapter functions
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TH1F* iHist) {
    //std::cout <<"create: hist size "<<iName <<" "<<iHist->GetEffectiveEntries()<<std::endl;
    return iStore.book1D(iName, iHist);
  }
  //NOTE: the merge logic comes from DataFormats/Histograms/interface/MEtoEDMFormat.h
  void mergeTogether(TH1* iOriginal,TH1* iToAdd) {
    if(iOriginal->TestBit(TH1::kCanRebin)==true && iToAdd->TestBit(TH1::kCanRebin) ==true) {
      TList list;
      list.Add(iToAdd);
      if( -1 == iOriginal->Merge(&list)) {
        edm::LogError("MergeFailure")<<"Failed to merge DQM element "<<iOriginal->GetName();
      }
    } else {
      if (iOriginal->GetNbinsX() == iToAdd->GetNbinsX() &&
          iOriginal->GetXaxis()->GetXmin() == iToAdd->GetXaxis()->GetXmin() &&
          iOriginal->GetXaxis()->GetXmax() == iToAdd->GetXaxis()->GetXmax() &&
          iOriginal->GetNbinsY() == iToAdd->GetNbinsY() &&
          iOriginal->GetYaxis()->GetXmin() == iToAdd->GetYaxis()->GetXmin() &&
          iOriginal->GetYaxis()->GetXmax() == iToAdd->GetYaxis()->GetXmax() &&
          iOriginal->GetNbinsZ() == iToAdd->GetNbinsZ() &&
          iOriginal->GetZaxis()->GetXmin() == iToAdd->GetZaxis()->GetXmin() &&
          iOriginal->GetZaxis()->GetXmax() == iToAdd->GetZaxis()->GetXmax()) {
          iOriginal->Add(iToAdd);
      } else {
        edm::LogError("MergeFailure")<<"Found histograms with different axis limitsm '"<<iOriginal->GetName()<<"' not merged.";
      }
    } 
  }
  void mergeWithElement(MonitorElement* iElement, TH1F* iHist) {
    //std::cout <<"merge: hist size "<<iElement->getName() <<" "<<iHist->GetEffectiveEntries()<<std::endl;
    mergeTogether(iElement->getTH1F(),iHist);
  }
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TH1S* iHist) {
    return iStore.book1S(iName, iHist);
  }
  void mergeWithElement(MonitorElement* iElement, TH1S* iHist) {
    mergeTogether(iElement->getTH1S(),iHist);
  }  
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TH1D* iHist) {
    return iStore.book1DD(iName, iHist);
  }
  void mergeWithElement(MonitorElement* iElement, TH1D* iHist) {
    mergeTogether(iElement->getTH1D(),iHist);
  }
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TH2F* iHist) {
    return iStore.book2D(iName, iHist);
  }
  void mergeWithElement(MonitorElement* iElement, TH2F* iHist) {
    mergeTogether(iElement->getTH2F(),iHist);
  }
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TH2S* iHist) {
    return iStore.book2S(iName, iHist);
  }
  void mergeWithElement(MonitorElement* iElement, TH2S* iHist) {
    mergeTogether(iElement->getTH2S(),iHist);
  }  
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TH2D* iHist) {
    return iStore.book2DD(iName, iHist);
  }
  void mergeWithElement(MonitorElement* iElement, TH2D* iHist) {
    mergeTogether(iElement->getTH2D(),iHist);
  }
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TH3F* iHist) {
    return iStore.book3D(iName, iHist);
  }
  void mergeWithElement(MonitorElement* iElement, TH3F* iHist) {
    mergeTogether(iElement->getTH3F(),iHist);
  }
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TProfile* iHist) {
    return iStore.bookProfile(iName, iHist);
  }
  void mergeWithElement(MonitorElement* iElement, TProfile* iHist) {
    mergeTogether(iElement->getTProfile(),iHist);
  }
  MonitorElement* createElement(DQMStore& iStore, const char* iName, TProfile2D* iHist) {
    return iStore.bookProfile2D(iName, iHist);
  }
  void mergeWithElement(MonitorElement* iElement, TProfile2D* iHist) {
    mergeTogether(iElement->getTProfile2D(),iHist);
  }

  MonitorElement* createElement(DQMStore& iStore, const char* iName, Long64_t& iValue) {
    MonitorElement* e = iStore.bookInt(iName);
    e->Fill(iValue);
    return e;
  }
  
  //NOTE: the merge logic comes from DataFormats/Histograms/interface/MEtoEDMFormat.h
  void mergeWithElement(MonitorElement* iElement, Long64_t& iValue) {
    const std::string& name = iElement->getFullname();
    if(name.find("EventInfo/processedEvents") != std::string::npos) {
      iElement->Fill(iValue+iElement->getIntValue());
    } else {
      if(name.find("EventInfo/iEvent") != std::string::npos ||
         name.find("EventInfo/iLumiSection") != std::string::npos) {
        if(iValue > iElement->getIntValue()) {
             iElement->Fill(iValue);
        }
      }
    }
  }

  MonitorElement* createElement(DQMStore& iStore, const char* iName, double& iValue) {
    MonitorElement* e = iStore.bookFloat(iName);
    e->Fill(iValue);
    return e;
  }
  void mergeWithElement(MonitorElement* iElement, double& iValue) {
    //no merging    
  }
  MonitorElement* createElement(DQMStore& iStore, const char* iName, std::string* iValue) {
    return iStore.bookString(iName,*iValue);
  }
  void mergeWithElement(MonitorElement* , std::string* ) {
    //no merging    
  }
  
  void splitName(const std::string& iFullName, std::string& oPath,const char*& oName) {
    oPath = iFullName;
    size_t index = oPath.find_last_of('/');
    if(index == std::string::npos) {
      oPath = std::string();
      oName = iFullName.c_str();
    } else {
      oPath.resize(index);
      oName = iFullName.c_str()+index+1;
    }
  }
  
  struct RunLumiToRange { 
    unsigned int m_run, m_lumi,m_historyIDIndex;
    ULong64_t m_beginTime;
    ULong64_t m_endTime;
    ULong64_t m_firstIndex, m_lastIndex; //last is inclusive
    unsigned int m_type; //A value in TypeIndex 
  };
  
  class TreeReaderBase {
  public:
    TreeReaderBase() {}
    virtual ~TreeReaderBase() {}
    
    MonitorElement* read(ULong64_t iIndex, DQMStore& iStore, bool iIsLumi){
      return doRead(iIndex,iStore,iIsLumi);
    }
    virtual void setTree(TTree* iTree) =0;
  protected:
    TTree* m_tree;
  private:
    virtual MonitorElement* doRead(ULong64_t iIndex, DQMStore& iStore, bool iIsLumi)=0;
  };
  
  template<class T>
  class TreeObjectReader: public TreeReaderBase {
  public:
    TreeObjectReader():m_tree(0),m_fullName(0),m_buffer(0),m_tag(0){
    }
    virtual MonitorElement* doRead(ULong64_t iIndex, DQMStore& iStore, bool iIsLumi) {
      m_tree->GetEntry(iIndex);
      MonitorElement* element = iStore.get(*m_fullName);
      if(0 == element) {
        std::string path;
        const char* name;
        splitName(*m_fullName, path,name);
        iStore.setCurrentFolder(path);
         element = createElement(iStore,name,m_buffer);
        if(iIsLumi) { element->setLumiFlag();}
      } else {
        mergeWithElement(element,m_buffer);
      }
      if(0!= m_tag) {
        iStore.tag(element,m_tag);
      }
      return element;
    }
    virtual void setTree(TTree* iTree)  {
      m_tree = iTree;
      m_tree->SetBranchAddress(kFullNameBranch,&m_fullName);
      m_tree->SetBranchAddress(kFlagBranch,&m_tag);
      m_tree->SetBranchAddress(kValueBranch,&m_buffer);
    }
  private:
    TTree* m_tree;
    std::string* m_fullName;
    T* m_buffer;
    uint32_t m_tag;
  };
  
  template<class T>
  class TreeSimpleReader : public TreeReaderBase {
  public:
    TreeSimpleReader():m_tree(0),m_fullName(0),m_buffer(),m_tag(0){
    }
    virtual MonitorElement* doRead(ULong64_t iIndex, DQMStore& iStore,bool iIsLumi) {
      m_tree->GetEntry(iIndex);
      MonitorElement* element = iStore.get(*m_fullName);
      if(0 == element) {
        std::string path;
        const char* name;
        splitName(*m_fullName, path,name);
        iStore.setCurrentFolder(path);
        element = createElement(iStore,name,m_buffer);
        if(iIsLumi) { element->setLumiFlag();}
      } else {
        mergeWithElement(element, m_buffer);
      }
      if(0!=m_tag) {
        iStore.tag(element,m_tag);
      }
      return element;
    }
    virtual void setTree(TTree* iTree)  {
      m_tree = iTree;
      m_tree->SetBranchAddress(kFullNameBranch,&m_fullName);
      m_tree->SetBranchAddress(kFlagBranch,&m_tag);
      m_tree->SetBranchAddress(kValueBranch,&m_buffer);
    }
  private:
    TTree* m_tree;
    std::string* m_fullName;
    T m_buffer;
    uint32_t m_tag;
  };
  
}

class DQMRootSource : public edm::InputSource
{

   public:
      DQMRootSource(edm::ParameterSet const&, const edm::InputSourceDescription&);
      virtual ~DQMRootSource();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
   protected:
     virtual void endLuminosityBlock(edm::LuminosityBlock&);
     virtual void endRun(edm::Run&);
     
   private:
      DQMRootSource(const DQMRootSource&); // stop default

      virtual edm::InputSource::ItemType getNextItemType();
      //NOTE: the following is really read next run auxiliary
      virtual boost::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() ;
      virtual boost::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() ;
      virtual boost::shared_ptr<edm::RunPrincipal> readRun_(boost::shared_ptr<edm::RunPrincipal> rpCache);
      virtual boost::shared_ptr<edm::LuminosityBlockPrincipal> readLuminosityBlock_( boost::shared_ptr<edm::LuminosityBlockPrincipal> lbCache);
      virtual edm::EventPrincipal* readEvent_() ;
      
      virtual boost::shared_ptr<edm::FileBlock> readFile_();
      virtual void closeFile_();
      
      void readNextItemType();
      void setupFile(unsigned int iIndex);
      void readElements();
      
      const DQMRootSource& operator=(const DQMRootSource&); // stop default

      // ---------- member data --------------------------------
      edm::InputFileCatalog m_catalog;
      edm::RunAuxiliary m_runAux;
      edm::LuminosityBlockAuxiliary m_lumiAux;
      edm::InputSource::ItemType m_nextItemType;

      size_t m_fileIndex;
      std::list<unsigned int>::iterator m_nextIndexItr;
      std::list<unsigned int>::iterator m_presentIndexItr;
      std::vector<RunLumiToRange> m_runlumiToRange;
      std::auto_ptr<TFile> m_file;
      std::vector<TTree*> m_trees;
      std::vector<boost::shared_ptr<TreeReaderBase> > m_treeReaders;
      
      std::list<unsigned int> m_orderedIndices;
      unsigned int m_lastSeenRun;
      bool m_justOpenedFileSoNeedToGenerateRunTransition;
      bool m_doNotReadRemainingPartsOfFileSinceFrameworkTerminating;
      std::set<MonitorElement*> m_lumiElements;
      std::set<MonitorElement*> m_runElements;
      std::vector<edm::ProcessHistoryID> m_historyIDs;
      
      edm::JobReport::Token m_jrToken;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

void 
DQMRootSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string> >("fileNames")
    ->setComment("Names of files to be processed.");
  desc.addUntracked<std::string>("overrideCatalog",std::string())
    ->setComment("An alternate file catalog to use instead of the standard site one.");
  descriptions.addDefault(desc);
}
//
// constructors and destructor
//
DQMRootSource::DQMRootSource(edm::ParameterSet const& iPSet, const edm::InputSourceDescription& iDesc):
edm::InputSource(iPSet,iDesc),
m_catalog(iPSet.getUntrackedParameter<std::vector<std::string> >("fileNames"), 
          iPSet.getUntrackedParameter<std::string>("overrideCatalog")),
m_nextItemType(edm::InputSource::IsFile),
m_fileIndex(0),
m_trees(kNIndicies,static_cast<TTree*>(0)),
m_treeReaders(kNIndicies,boost::shared_ptr<TreeReaderBase>()),
m_lastSeenRun(0),
m_justOpenedFileSoNeedToGenerateRunTransition(false),
m_doNotReadRemainingPartsOfFileSinceFrameworkTerminating(false)
{
  if(m_fileIndex ==m_catalog.fileNames().size()) {
    m_nextItemType=edm::InputSource::IsStop;
  } else{
    m_treeReaders[kIntIndex].reset(new TreeSimpleReader<Long64_t>());
    m_treeReaders[kFloatIndex].reset(new TreeSimpleReader<double>());
    m_treeReaders[kStringIndex].reset(new TreeObjectReader<std::string>());
    m_treeReaders[kTH1FIndex].reset(new TreeObjectReader<TH1F>());
    m_treeReaders[kTH1SIndex].reset(new TreeObjectReader<TH1S>());
    m_treeReaders[kTH1DIndex].reset(new TreeObjectReader<TH1D>());
    m_treeReaders[kTH2FIndex].reset(new TreeObjectReader<TH2F>());
    m_treeReaders[kTH2SIndex].reset(new TreeObjectReader<TH2S>());
    m_treeReaders[kTH2DIndex].reset(new TreeObjectReader<TH2D>());
    m_treeReaders[kTH3FIndex].reset(new TreeObjectReader<TH3F>());
    m_treeReaders[kTProfileIndex].reset(new TreeObjectReader<TProfile>());
    m_treeReaders[kTProfile2DIndex].reset(new TreeObjectReader<TProfile2D>());
  }
  
}

// DQMRootSource::DQMRootSource(const DQMRootSource& rhs)
// {
//    // do actual copying here;
// }

DQMRootSource::~DQMRootSource()
{
}

//
// assignment operators
//
// const DQMRootSource& DQMRootSource::operator=(const DQMRootSource& rhs)
// {
//   //An exception safe implementation is
//   DQMRootSource temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
namespace {
  struct no_deleter {
    void operator()(void*) {}
  };
}

edm::EventPrincipal*
DQMRootSource::readEvent_() 
{
  return 0;
}

edm::InputSource::ItemType 
DQMRootSource::getNextItemType()
{
  //std::cout <<"getNextItemType "<<m_nextItemType<<std::endl;
  return m_nextItemType;
}
boost::shared_ptr<edm::RunAuxiliary> 
DQMRootSource::readRunAuxiliary_()
{
  //std::cout <<"readRunAuxiliary_"<<std::endl;
  assert(m_nextIndexItr != m_orderedIndices.end());
  unsigned int index = *m_nextIndexItr;
  RunLumiToRange runLumiRange = m_runlumiToRange[index];
  //NOTE: this could be a lumi instead of the actual run. We need to find the time for the run
  // so we will scan forward
  while(runLumiRange.m_lumi !=0 && ++index<m_runlumiToRange.size()) {
    const RunLumiToRange& next = m_runlumiToRange[index];
    if(runLumiRange.m_run == next.m_run) {
      runLumiRange = next;
    } else {
      break;
    }
  }

  //NOTE: the setBeginTime and setEndTime functions of RunAuxiliary only work if the previous value was invalid
  // therefore we must copy
  m_runAux = edm::RunAuxiliary(runLumiRange.m_run,edm::Timestamp(runLumiRange.m_beginTime),edm::Timestamp(runLumiRange.m_endTime));
  assert(m_historyIDs.size() > runLumiRange.m_historyIDIndex);
  //std::cout <<"readRunAuxiliary_ "<<m_historyIDs[runLumiRange.m_historyIDIndex]<<std::endl;
  m_runAux.setProcessHistoryID(m_historyIDs[runLumiRange.m_historyIDIndex]);    
  return boost::shared_ptr<edm::RunAuxiliary>( new edm::RunAuxiliary(m_runAux) );
}
boost::shared_ptr<edm::LuminosityBlockAuxiliary> 
DQMRootSource::readLuminosityBlockAuxiliary_()
{
  //std::cout <<"readLuminosityBlockAuxiliary_"<<std::endl;
  assert(m_nextIndexItr != m_orderedIndices.end());
  const RunLumiToRange runLumiRange = m_runlumiToRange[*m_nextIndexItr];
  m_lumiAux = edm::LuminosityBlockAuxiliary(edm::LuminosityBlockID(runLumiRange.m_run,runLumiRange.m_lumi),
                                            edm::Timestamp(runLumiRange.m_beginTime),
                                            edm::Timestamp(runLumiRange.m_endTime));
  assert(m_historyIDs.size() > runLumiRange.m_historyIDIndex);
  //std::cout <<"lumi "<<m_lumiAux.beginTime().value()<<" "<<runLumiRange.m_beginTime<<std::endl;
  m_lumiAux.setProcessHistoryID(m_historyIDs[runLumiRange.m_historyIDIndex]);    
  
  return boost::shared_ptr<edm::LuminosityBlockAuxiliary>(new edm::LuminosityBlockAuxiliary(m_lumiAux));
}
boost::shared_ptr<edm::RunPrincipal> 
DQMRootSource::readRun_(boost::shared_ptr<edm::RunPrincipal> rpCache) 
{
  m_justOpenedFileSoNeedToGenerateRunTransition = false;
  unsigned int runID =rpCache->id().run();
  unsigned int lastRunID = m_lastSeenRun;
  m_lastSeenRun = runID;
  readNextItemType();
  //std::cout <<"readRun_"<<std::endl;
  
  //NOTE: need to reset all run elements at this point
  if(lastRunID != runID) {
    for(std::set<MonitorElement*>::iterator it = m_runElements.begin(), itEnd = m_runElements.end();
        it != itEnd;
        ++it) {
          //std::cout <<"RESETTING "<<(*it)->getName()<<std::endl;
          (*it)->Reset();        
    }
  }

  rpCache->addToProcessHistory();

  if(m_presentIndexItr != m_orderedIndices.end()) {
    RunLumiToRange runLumiRange = m_runlumiToRange[*m_presentIndexItr];
    //NOTE: it is possible to have an Run when all we have stored is lumis
    if(runLumiRange.m_lumi == 0 && 
       runLumiRange.m_run == rpCache->id().run()) {
      readElements();
    }
  }
  
  edm::Service<edm::JobReport> jr;
  jr->reportInputRunNumber(rpCache->id().run());
  return rpCache;
}
boost::shared_ptr<edm::LuminosityBlockPrincipal> 
DQMRootSource::readLuminosityBlock_( boost::shared_ptr<edm::LuminosityBlockPrincipal> lbCache) 
{
  //NOTE: need to reset all lumi block elements at this point
  for(std::set<MonitorElement*>::iterator it = m_lumiElements.begin(), itEnd = m_lumiElements.end();
      it != itEnd;
      ++it) {
        //std::cout <<"RESETTING "<<(*it)->getName()<<std::endl;      
      (*it)->Reset();
  }
  readNextItemType();
  //std::cout <<"readLuminosityBlock_"<<std::endl;
  RunLumiToRange runLumiRange = m_runlumiToRange[*m_presentIndexItr];
  if(runLumiRange.m_run == lbCache->id().run() &&
     runLumiRange.m_lumi == lbCache->id().luminosityBlock()) {
       readElements();
  }
  
  edm::Service<edm::JobReport> jr;
  jr->reportInputLumiSection(lbCache->id().run(),lbCache->id().luminosityBlock());
  
  return lbCache;
}

boost::shared_ptr<edm::FileBlock>
DQMRootSource::readFile_() {
  //std::cout <<"readFile_"<<std::endl;
  setupFile(m_fileIndex);
  ++m_fileIndex;
  readNextItemType();

  edm::Service<edm::JobReport> jr;
  m_jrToken = jr->inputFileOpened(m_catalog.fileNames()[m_fileIndex-1],
                                  m_catalog.logicalFileNames()[m_fileIndex-1],
                                  std::string(),
                                  std::string(),
                                  "DQMRootSource",
                                  "source",
                                  m_file->GetUUID().AsString(),//edm::createGlobalIdentifier(),
                                  std::vector<std::string>()
                                  );
  
  m_doNotReadRemainingPartsOfFileSinceFrameworkTerminating = false;
  return boost::shared_ptr<edm::FileBlock>(new edm::FileBlock);
}


void 
DQMRootSource::endLuminosityBlock(edm::LuminosityBlock& iLumi) {
  //std::cout <<"DQMRootSource::endLumi"<<std::endl;
}
void 
DQMRootSource::endRun(edm::Run& iRun){
  //std::cout <<"DQMRootSource::endRun"<<std::endl;
  //NOTE: the framework will call endRun before closeFile in the case
  // where the frameworks is terminating
  m_doNotReadRemainingPartsOfFileSinceFrameworkTerminating=true;
}


void
DQMRootSource::closeFile_() {
  //std::cout <<"closeFile_"<<std::endl;
  //when going from one file to the next the framework does not call
  // 'endRun' or 'endLumi' until it looks to see if the other file contains
  // a new run or lumi. If the other file doesn't then  
  if(not m_doNotReadRemainingPartsOfFileSinceFrameworkTerminating) {
    while(m_presentIndexItr != m_orderedIndices.end()) {
      readElements();
    }
  }
  edm::Service<edm::JobReport> jr;
  jr->inputFileClosed(m_jrToken);
}

void 
DQMRootSource::readElements() {
  edm::Service<DQMStore> store;
  RunLumiToRange runLumiRange = m_runlumiToRange[*m_presentIndexItr];
  bool shouldContinue = false;
  do {
    shouldContinue = false;
    if(runLumiRange.m_type == kNoTypesStored) {continue;}
    boost::shared_ptr<TreeReaderBase> reader = m_treeReaders[runLumiRange.m_type];
    for(ULong64_t index = runLumiRange.m_firstIndex, endIndex=runLumiRange.m_lastIndex+1;
    index != endIndex;
    ++index) {
      bool isLumi = runLumiRange.m_lumi !=0;
      MonitorElement* element = reader->read(index,*store,isLumi);
      if(isLumi) {
        //std::cout <<"  lumi element "<< element->getName()<<" "<<index<<std::endl;
        m_lumiElements.insert(element);
      } else {
        //std::cout <<"  run element "<< element->getName()<<" "<<index<<std::endl;
        m_runElements.insert(element);
      }
    }
    ++m_presentIndexItr;
    if(m_presentIndexItr != m_orderedIndices.end()) {
      //are there more parts to this same run/lumi?
      const RunLumiToRange nextRunLumiRange = m_runlumiToRange[*m_presentIndexItr];
      //continue to the next item if that item is either
      if( (nextRunLumiRange.m_run == runLumiRange.m_run) && 
          (nextRunLumiRange.m_lumi == runLumiRange.m_lumi) ) {
           shouldContinue= true;
           runLumiRange = nextRunLumiRange;
      } 
    }
  }while(shouldContinue);
}


void 
DQMRootSource::readNextItemType()
{
  //Do the work of actually figuring out where next to go
  RunLumiToRange runLumiRange = m_runlumiToRange[*m_nextIndexItr];
  if(m_nextItemType !=edm::InputSource::IsFile) {
    assert(m_nextIndexItr != m_orderedIndices.end());

    if(runLumiRange.m_lumi ==0) {
      //std::cout <<"reading run "<<runLumiRange.m_run<<std::endl;
      m_runAux.id() = edm::RunID(runLumiRange.m_run);    
    } else {
      if(m_nextItemType == edm::InputSource::IsRun) {
        //std::cout <<" proceeding with dummy run";
        m_nextItemType = edm::InputSource::IsLumi;
        return;
      }
      //std::cout <<"reading lumi "<<runLumiRange.m_run<<","<<runLumiRange.m_lumi<<std::endl;
      m_lumiAux.id() = edm::LuminosityBlockID(runLumiRange.m_run,runLumiRange.m_lumi);
    }
    ++m_nextIndexItr;
  } else {
    //NOTE: the following causes the iterator to move to before 'begin' but that is OK
    // since the first thing in the 'do while' loop is to advance the iterator which puts
    // us at the first entry in the file
    runLumiRange.m_run=0;
  } 
  
  bool shouldContinue = false;
  do {
    shouldContinue = false;
    if(m_nextIndexItr == m_orderedIndices.end()) {
      //go to next file
      m_nextItemType = edm::InputSource::IsFile;
      //std::cout <<"going to next file"<<std::endl;
      if(m_fileIndex == m_catalog.fileNames().size()) {
        m_nextItemType = edm::InputSource::IsStop;
      }       
      break;
    }
    const RunLumiToRange nextRunLumiRange = m_runlumiToRange[*m_nextIndexItr];
    //continue to the next item if that item is the same run or lumi as we just did
    if( (nextRunLumiRange.m_run == runLumiRange.m_run) && (
         nextRunLumiRange.m_lumi == runLumiRange.m_lumi) ) {
         shouldContinue= true;
         runLumiRange = nextRunLumiRange;
         ++m_nextIndexItr;
         //std::cout <<"  advancing " <<nextRunLumiRange.m_run<<" "<<nextRunLumiRange.m_lumi<<std::endl;
    } 
    
  } while(shouldContinue);
  
  if(m_nextIndexItr != m_orderedIndices.end()) {
    if(m_runlumiToRange[*m_nextIndexItr].m_lumi == 0 && (m_justOpenedFileSoNeedToGenerateRunTransition || m_lastSeenRun != m_runlumiToRange[*m_nextIndexItr].m_run) ) {
      m_nextItemType = edm::InputSource::IsRun;
      //std::cout <<"  next is run"<<std::endl;
    } else {
      if(m_runlumiToRange[*m_nextIndexItr].m_run != m_lastSeenRun || m_justOpenedFileSoNeedToGenerateRunTransition ) {
        //we have to create a dummy Run since we switched to a lumi in a new run
        //std::cout <<"  next is dummy run "<<m_justOpenedFileSoNeedToGenerateRunTransition<<std::endl;
        m_nextItemType = edm::InputSource::IsRun;
      } else {
        m_nextItemType = edm::InputSource::IsLumi;      
      }
    }
  }
}

void 
DQMRootSource::setupFile(unsigned int iIndex)
{
  
  m_file = std::auto_ptr<TFile>(TFile::Open(m_catalog.fileNames()[iIndex].c_str()));
  
  //Check file format version, which is encoded in the Title of the TFile
  assert(0==strcmp(m_file->GetTitle(),"1"));
  
  //Get meta Data
  TDirectory* metaDir = m_file->GetDirectory(kMetaDataDirectoryAbsolute);
  assert(0!=metaDir);
  TTree* parameterSetTree = dynamic_cast<TTree*>(metaDir->Get(kParameterSetTree));
  assert(0!=parameterSetTree);
  
  edm::pset::Registry* psr = edm::pset::Registry::instance();
  assert(0!=psr);
  {
    std::string blob;
    std::string* pBlob = &blob;
    parameterSetTree->SetBranchAddress(kParameterSetBranch,&pBlob);
    for(unsigned int index = 0; index != parameterSetTree->GetEntries();++index) {
      parameterSetTree->GetEntry(index);
      cms::Digest dg(blob);
      edm::ParameterSetID psID(dg.digest().toString());
      edm::ParameterSet temp(blob,psID);
    } 
  }

  {
    TTree* processHistoryTree = dynamic_cast<TTree*>(metaDir->Get(kProcessHistoryTree));
    assert(0!=processHistoryTree);
    unsigned int phIndex = 0;
    processHistoryTree->SetBranchAddress(kPHIndexBranch,&phIndex);
    std::string processName;
    std::string* pProcessName = &processName;
    processHistoryTree->SetBranchAddress(kProcessConfigurationProcessNameBranch,&pProcessName);
    std::string parameterSetIDBlob;
    std::string* pParameterSetIDBlob = &parameterSetIDBlob;
    processHistoryTree->SetBranchAddress(kProcessConfigurationParameterSetIDBranch,&pParameterSetIDBlob);
    std::string releaseVersion;
    std::string* pReleaseVersion = &releaseVersion;
    processHistoryTree->SetBranchAddress(kProcessConfigurationReleaseVersion,&pReleaseVersion);
    std::string passID;
    std::string* pPassID = &passID;
    processHistoryTree->SetBranchAddress(kProcessConfigurationPassID,&pPassID);

    edm::ProcessConfigurationRegistry* pcr = edm::ProcessConfigurationRegistry::instance();
    assert(0!=pcr);
    edm::ProcessHistoryRegistry* phr = edm::ProcessHistoryRegistry::instance();
    assert(0!=phr);
    std::vector<edm::ProcessConfiguration> configs;
    configs.reserve(5);
    for(unsigned int i=0; i != processHistoryTree->GetEntries(); ++i) {
      processHistoryTree->GetEntry();
      if(phIndex==0) {
        if(not configs.empty()) {
          edm::ProcessHistory ph(configs);
          m_historyIDs.push_back(ph.id());
          phr->insertMapped(ph);
        }
        configs.clear();
      }
      edm::ParameterSetID psetID(parameterSetIDBlob);
      edm::ProcessConfiguration pc(processName, psetID,releaseVersion,passID);
      pcr->insertMapped(pc);
      configs.push_back(pc);
    }
    if(not configs.empty()) {
      edm::ProcessHistory ph(configs);
      m_historyIDs.push_back(ph.id());
      phr->insertMapped( ph);
      //std::cout <<"inserted "<<ph.id()<<std::endl;
    }
  }
  
  //Setup the indices
  TTree* indicesTree = dynamic_cast<TTree*>(m_file->Get(kIndicesTree));
  assert(0!=indicesTree);
  
  m_runlumiToRange.clear();
  m_runlumiToRange.reserve(indicesTree->GetEntries());
  m_orderedIndices.clear();

  RunLumiToRange temp;
  indicesTree->SetBranchAddress(kRunBranch,&temp.m_run);
  indicesTree->SetBranchAddress(kLumiBranch,&temp.m_lumi);
  indicesTree->SetBranchAddress(kBeginTimeBranch,&temp.m_beginTime);
  indicesTree->SetBranchAddress(kEndTimeBranch,&temp.m_endTime);
  indicesTree->SetBranchAddress(kProcessHistoryIndexBranch,&temp.m_historyIDIndex);
  indicesTree->SetBranchAddress(kTypeBranch,&temp.m_type);
  indicesTree->SetBranchAddress(kFirstIndex,&temp.m_firstIndex);
  indicesTree->SetBranchAddress(kLastIndex,&temp.m_lastIndex);

  //Need to reorder items since if there was a merge done the same  Run and/or Lumi can appear multiple times
  // but we want to process them all at once
  
  //We use a std::list for m_orderedIndices since inserting into the middle of a std::list does not 
  // disrupt the iterators to already existing entries
  
  //The Map is used to see if a Run/Lumi pair has appeared before
  typedef std::map<std::pair<unsigned int, unsigned int>, std::list<unsigned int>::iterator > RunLumiToLastEntryMap;
  RunLumiToLastEntryMap runLumiToLastEntryMap;
  
  //Need to group all lumis for the same run together and move the run entry to the beginning
  typedef std::map<unsigned int, std::pair<std::list<unsigned int>::iterator,std::list<unsigned int>::iterator> > RunToFirstLastEntryMap;
  RunToFirstLastEntryMap runToFirstLastEntryMap;
  
  for(Long64_t index = 0; index != indicesTree->GetEntries();++index) {
    indicesTree->GetEntry(index);
    //std::cout <<"read r:"<<temp.m_run<<" l:"<<temp.m_lumi<<" b:"<<temp.m_beginTime<<" e:"<<temp.m_endTime<<std::endl;
    m_runlumiToRange.push_back(temp);
    
    std::pair<unsigned int, unsigned int> runLumi(temp.m_run,temp.m_lumi);
    
    RunLumiToLastEntryMap::iterator itFind = runLumiToLastEntryMap.find(runLumi);
    if(itFind == runLumiToLastEntryMap.end()) {
      //does not already exist

      //does the run for this already exist?
      std::list<unsigned int>::iterator itLastOfRun = m_orderedIndices.end();
      RunToFirstLastEntryMap::iterator itRunFirstLastEntryFind = runToFirstLastEntryMap.find(temp.m_run);
      bool needNewEntryInRunFirstLastEntryMap = true;
      if(itRunFirstLastEntryFind != runToFirstLastEntryMap.end()) {
        needNewEntryInRunFirstLastEntryMap=false;
        if(temp.m_lumi!=0) {
          //lumis go to the end
          itLastOfRun = itRunFirstLastEntryFind->second.second;
          //we want to insert after this one so must advanced the iterator
          ++itLastOfRun;
        } else {
          //runs go at the beginning
          itLastOfRun = itRunFirstLastEntryFind->second.first;
        }
      } 
      
      std::list<unsigned int>::iterator iter = m_orderedIndices.insert(itLastOfRun,index);
      runLumiToLastEntryMap[runLumi]=iter;
      if(needNewEntryInRunFirstLastEntryMap) {
        runToFirstLastEntryMap[temp.m_run]=std::make_pair(iter,iter);
      } else {
        if(temp.m_lumi!=0) {
          //lumis go at end
          runToFirstLastEntryMap[temp.m_run].second = iter;
        } else {
          //since we haven't yet seen this run/lumi combination it means we haven't yet seen
          // a run so we can put this first
          runToFirstLastEntryMap[temp.m_run].first = iter;
        }
      }
    } else {
      //We need to do a merge since the run/lumi already appeared. Put it after the existing entry
      //std::cout <<" found a second instance of "<<runLumi.first<<" "<<runLumi.second<<" at "<<index<<std::endl;
      std::list<unsigned int>::iterator itNext = itFind->second;
      ++itNext;
      std::list<unsigned int>::iterator iter = m_orderedIndices.insert(itNext,index);
      RunToFirstLastEntryMap::iterator itRunFirstLastEntryFind = runToFirstLastEntryMap.find(temp.m_run);
      if(itRunFirstLastEntryFind->second.second == itFind->second) {
        //if the previous one was the last in the run, we need to update to make this one the last
        itRunFirstLastEntryFind->second.second = iter;
      }
      itFind->second = iter;
    }
  }
  m_nextIndexItr = m_orderedIndices.begin();
  m_presentIndexItr = m_orderedIndices.begin();
  
  if(m_nextIndexItr != m_orderedIndices.end()) {
    for( size_t index = 0; index < kNIndicies; ++index) {
      m_trees[index] = dynamic_cast<TTree*>(m_file->Get(kTypeNames[index]));
      assert(0!=m_trees[index]);
      m_treeReaders[index]->setTree(m_trees[index]);
    }
  }
  //After a file open, the framework expects to see a new 'IsRun'
  //m_lastSeenRun = 0;
  m_justOpenedFileSoNeedToGenerateRunTransition=true;
}

//
// const member functions
//

//
// static member functions
//
DEFINE_FWK_INPUT_SOURCE(DQMRootSource);
