#ifndef ConfigurableAnalysis_Plotter_H
#define ConfigurableAnalysis_Plotter_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TH1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"
#include "PhysicsTools/UtilAlgos/interface/ConfigurableHisto.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


class Plotter {
 public:
  Plotter(){}
  Plotter(edm::ParameterSet iConfig){}
  virtual ~Plotter(){}

  virtual void setDir(std::string dir) =0;
  virtual void fill(std::string subDir,const edm::Event& iEvent) =0;
  virtual void complete() =0;
};


class VariablePlotter : public Plotter {
 public:
  VariablePlotter(edm::ParameterSet iConfig) : currentDir_("youDidNotSetDirectoryFirst") {
    //create the master copy, never filled, just to make copies

    //    make TH1
    edm::ParameterSet th1=iConfig.getParameter<edm::ParameterSet>("TH1s");
    std::vector<std::string> th1Names;
    th1.getParameterSetNames(th1Names);
    for (unsigned int iH=0;iH!=th1Names.size();++iH){
      std::string hname = th1Names[iH];
      edm::ParameterSet hPset=th1.getParameter<edm::ParameterSet>(hname);
      bool split=hPset.exists("splitter") || hPset.exists("splitters");
      if (split)
	master_[hname]=new SplittingConfigurableHisto(ConfigurableHisto::h1, hname, hPset);
      else
	master_[hname]=new ConfigurableHisto(ConfigurableHisto::h1, hname, hPset);
    }

    //    make profiles
    edm::ParameterSet tprof=iConfig.getParameter<edm::ParameterSet>("TProfiles");
    std::vector<std::string> tprofNames;
    tprof.getParameterSetNames(tprofNames);
    for (unsigned int iH=0;iH!=tprofNames.size();++iH){
      std::string hname = tprofNames[iH];
      edm::ParameterSet hPset=tprof.getParameter<edm::ParameterSet>(hname);
      bool split=hPset.exists("splitter") || hPset.exists("splitters");
      if (split)
	master_[hname]=new SplittingConfigurableHisto(ConfigurableHisto::prof, hname, hPset);
      else
	master_[hname]=new ConfigurableHisto(ConfigurableHisto::prof, hname, hPset);
    }
    
    //    make TH2
    edm::ParameterSet th2=iConfig.getParameter<edm::ParameterSet>("TH2s");
    std::vector<std::string> th2Names;
    th2.getParameterSetNames(th2Names);
    for (unsigned int iH=0;iH!=th2Names.size();++iH){
      std::string hname = th2Names[iH];
      edm::ParameterSet hPset=th2.getParameter<edm::ParameterSet>(hname);
      bool split=hPset.exists("splitter") || hPset.exists("splitters");
      if (split)
	master_[hname]=new SplittingConfigurableHisto(ConfigurableHisto::h2, hname, hPset);
      else
	master_[hname]=new ConfigurableHisto(ConfigurableHisto::h2, hname, hPset);
    }
  }

  void setDir(std::string dir){
    //insert a new one
    Directory & insertedDirectory = directories_[dir];

    //create the actual directory in TFile: name is <dir>
    if (!insertedDirectory.dir){
      insertedDirectory.dir=new TFileDirectory(edm::Service<TFileService>()->mkdir(dir));
      insertedDirectory.dirName=dir;
    }

    //remember which directory name this is
    currentDir_=dir;
  }
  
  void fill(std::string subDir,const edm::Event& iEvent){
    //what is the current directory
    Directory & currentDirectory= directories_[currentDir_];

    //what is the current set of sub directories for this
    SubDirectories & currentSetOfSubDirectories=currentDirectory.subDir;
    
    //find the subDirectory requested:
    SubDirectory * subDirectoryToUse=0;
    SubDirectories::iterator subDirectoryFindIterator=currentSetOfSubDirectories.find(subDir);

    //not found? insert a new directory with this name
    if (subDirectoryFindIterator==currentSetOfSubDirectories.end()){
      edm::LogInfo("VariablePlotter")<<" gonna clone histograms for :"<<subDir<<" in "<<currentDir_;
      SubDirectory & insertedDir = currentSetOfSubDirectories[subDir];
      subDirectoryToUse = &insertedDir;
      if (!insertedDir.dir){
	insertedDir.dir=new TFileDirectory(currentDirectory.dir->mkdir(subDir));
	insertedDir.dirName=subDir;
      }

      //create a copy from the master copy
      DirectoryHistos::iterator masterHistogramIterator=master_.begin();
      DirectoryHistos::iterator masterHistogramIterator_end=master_.end();
      for (; masterHistogramIterator!=masterHistogramIterator_end;++masterHistogramIterator)
	{
	  //clone does not book histogram
	  insertedDir.histos[masterHistogramIterator->first]=masterHistogramIterator->second->clone();
	}
      
      //book all copies of the histos
      DirectoryHistos::iterator clonedHistogramIterator=insertedDir.histos.begin();
      DirectoryHistos::iterator clonedHistogramIterator_end=insertedDir.histos.end();
      for (; clonedHistogramIterator!=clonedHistogramIterator_end;++clonedHistogramIterator)
	{
	  clonedHistogramIterator->second->book(insertedDir.dir);
	}
    }
    else{
      subDirectoryToUse=&subDirectoryFindIterator->second;
    }
    
    //now that you have the subdirectory: fill histograms for this sub directory
    DirectoryHistos::iterator histogramIterator=subDirectoryToUse->histos.begin();
    DirectoryHistos::iterator histogramIterator_end=subDirectoryToUse->histos.end();
    for(; histogramIterator!=histogramIterator_end;++histogramIterator)
      { histogramIterator->second->fill(iEvent); }
  }

  ~VariablePlotter(){
    // CANNOT DO THAT because of TFileService holding the histograms
    /*    //loop over all subdirectories and delete all ConfigurableHistograms
	  Directories::iterator dir_It = directories_.begin();
	  Directories::iterator dir_It_end = directories_.end();
	  // loop directories
	  for (;dir_It!=dir_It_end;++dir_It){
	  Directory & currentDirectory=dir_It->second;
	  SubDirectories & currentSetOfSubDirectories=currentDirectory.subDir;
	  SubDirectories::iterator subDir_It = currentSetOfSubDirectories.begin();
	  SubDirectories::iterator subDir_It_end = currentSetOfSubDirectories.end();
	  //loop subdirectories
	  for (;subDir_It!=subDir_It_end;++subDir_It){
	  DirectoryHistos::iterator histogramIterator=subDir_It->second.histos.begin();
	  DirectoryHistos::iterator histogramIterator_end=subDir_It->second.histos.end();
	  //loop configurable histograms
	  for(; histogramIterator!=histogramIterator_end;++histogramIterator){
	  // by doing that you are removing the histogram from the TFileService too. and this will crash
	  //	  delete histogramIterator->second;
	  }
	  }
	  }
    */
  }
  void complete(){
    
    //loop over all subdirectories and call complete() on all ConfigurableHistograms
    
    Directories::iterator dir_It = directories_.begin();
    Directories::iterator dir_It_end = directories_.end();
    // loop directories
    for (;dir_It!=dir_It_end;++dir_It){
      Directory & currentDirectory=dir_It->second;
      SubDirectories & currentSetOfSubDirectories=currentDirectory.subDir;
      SubDirectories::iterator subDir_It = currentSetOfSubDirectories.begin();
      SubDirectories::iterator subDir_It_end = currentSetOfSubDirectories.end();
      //loop subdirectories
      for (;subDir_It!=subDir_It_end;++subDir_It){
	DirectoryHistos::iterator histogramIterator=subDir_It->second.histos.begin();
	DirectoryHistos::iterator histogramIterator_end=subDir_It->second.histos.end();
	//loop configurable histograms
	for(; histogramIterator!=histogramIterator_end;++histogramIterator)
	  { histogramIterator->second->complete(); }
      }
    }
  }

 private:
  typedef std::map<std::string, ConfigurableHisto *> DirectoryHistos;
  DirectoryHistos master_;

  class SubDirectory {
  public:
    SubDirectory() : dirName(""),dir(0){}
    std::string dirName;
    DirectoryHistos histos;
    TFileDirectory * dir;
  };
  typedef std::map<std::string, SubDirectory> SubDirectories;

  class Directory {
  public:
    Directory() : dirName(""),dir(0){}
    std::string dirName;
    SubDirectories subDir;
    TFileDirectory * dir;
  };
  typedef std::map<std::string, Directory> Directories;

  std::string currentDir_;
  Directories directories_;
};


#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

typedef edmplugin::PluginFactory< Plotter* (const edm::ParameterSet&) > PlotterFactory;

#endif
