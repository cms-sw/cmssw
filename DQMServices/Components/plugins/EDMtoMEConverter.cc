/** \file EDMtoMEConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2008/03/29 19:37:13 $
 *  $Revision: 1.9 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DQMServices/Components/plugins/EDMtoMEConverter.h"

EDMtoMEConverter::EDMtoMEConverter(const edm::ParameterSet & iPSet) :
  verbosity(0), frequency(0)
{
  std::string MsgLoggerCat = "EDMtoMEConverter_EDMtoMEConverter";
  
  // get information from parameter set
  name = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  
  // reset the release tag
  releaseTag = false;
  
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;
  
  // get dqm info
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if (dbe) {
    if (verbosity) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }
  
  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << name << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "===============================\n";
  }
  
  classtypes.clear();
  classtypes.push_back("TH1F");
  classtypes.push_back("TH2F");
  classtypes.push_back("TH3F");
  classtypes.push_back("TProfile");
  classtypes.push_back("TProfile2D");
  classtypes.push_back("Float");
  classtypes.push_back("Int");
  classtypes.push_back("String");

  count.clear();
  countf = 0;
  
} // end constructor

EDMtoMEConverter::~EDMtoMEConverter() {} 

void EDMtoMEConverter::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void EDMtoMEConverter::endJob()
{
  std::string MsgLoggerCat = "EDMtoMEConverter_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count.size() << " runs across " 
      << countf << " files.";
  return;
}

void EDMtoMEConverter::respondToOpenInputFile(const edm::FileBlock& iFb)
{
  ++countf;

  return;
}

void EDMtoMEConverter::beginRun(const edm::Run& iRun, 
				const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "EDMtoMEConverter_beginRun";
  
  int nrun = iRun.run();
  
  // keep track of number of unique runs processed
  ++count[nrun];

  if (verbosity) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << count.size() << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || count.size() == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << " (" << count.size() << " runs total)";
    }
  }

  return;
}

void EDMtoMEConverter::endRun(const edm::Run& iRun, 
			      const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "EDMtoMEConverter_endRun";
  
  if (verbosity >= 0)
    edm::LogInfo (MsgLoggerCat) << "\nRestoring MonitorElements.";
  
  for (unsigned int ii = 0; ii < classtypes.size(); ++ii) {    
    if (classtypes[ii] == "TH1F") {
      
      edm::Handle<MEtoEDM<TH1F> > metoedm;
      iRun.getByType(metoedm);
      
      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH1F> doesn't exist in run";
        continue;
      }
      
      std::vector<MEtoEDM<TH1F>::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject(); 
      
      me1.resize(metoedmobject.size());
      
      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {
	
	    me1[i] = 0;

        // get full path of monitor element
	    std::string pathname = metoedmobject[i].name;
	    if (verbosity) std::cout << pathname << std::endl;

        // set the release tag if it has not be yet done
	    if (!releaseTag)
	    {
	      dbe->cd();	
	      dbe->bookString(
	        "ReleaseTag",
	        metoedmobject[i].release.substr(1,metoedmobject[i].release.size()-2)
	      );
	      releaseTag = true;
	    }
	      
	    std::string dir;
	    
	    // deconstruct path from fullpath
	    StringList fulldir = StringOps::split(pathname,"/");
	    
	    for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
	      dir += fulldir[j];
	      if (j != fulldir.size() - 2) dir += "/";
	    }
	    
	    // define new monitor element
	    if (dbe) {
	      dbe->setCurrentFolder(dir);
	      me1[i] = dbe->book1D(metoedmobject[i].object.GetName(), &metoedmobject[i].object);
	    } // end define new monitor elements
	    
	    // attach taglist
	    TagList tags = metoedmobject[i].tags;
	    
	    for (unsigned int j = 0; j < tags.size(); ++j) {
	      dbe->tag(me1[i]->getFullname(),tags[j]);
	    }
	  } // end loop thorugh metoedmobject
	} // end TH1F creation
    
    if (classtypes[ii] == "TH2F") {
    	    
      edm::Handle<MEtoEDM<TH2F> > metoedm;
      iRun.getByType(metoedm);
      
      if (!metoedm.isValid()) {
      	//edm::LogWarning(MsgLoggerCat)
      	//  << "MEtoEDM<TH2F> doesn't exist in run";
      	continue;
      }
      
      std::vector<MEtoEDM<TH2F>::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject(); 
      
      me2.resize(metoedmobject.size());
      
      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {
      
        me2[i] = 0;
	
	    // get full path of monitor element
	    std::string pathname = metoedmobject[i].name;
	    if (verbosity) std::cout << pathname << std::endl;

        // set the release tag if it has not be yet done
	    if (!releaseTag)
	    {
	      dbe->cd();	
	      dbe->bookString(
	        "ReleaseTag",
	        metoedmobject[i].release.substr(1,metoedmobject[i].release.size()-2)
	      );
	      releaseTag = true;
	    }

        std::string dir;
        
        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        
        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }
	
	    // define new monitor element
	    if (dbe) {
	      dbe->setCurrentFolder(dir);
	      me2[i] = dbe->book2D(metoedmobject[i].object.GetName(), &metoedmobject[i].object);
        } // end define new monitor elements
        
        // attach taglist
        TagList tags = metoedmobject[i].tags;
        
        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me2[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH2F creation
    
    if (classtypes[ii] == "TH3F") {
      
      edm::Handle<MEtoEDM<TH3F> > metoedm;
      iRun.getByType(metoedm);
      
      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH3F> doesn't exist in run";
        continue;
      }
      
      std::vector<MEtoEDM<TH3F>::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject(); 
      
      me3.resize(metoedmobject.size());
 
      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {
        
        me3[i] = 0;
        
        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        
        if (verbosity) std::cout << pathname << std::endl;

        // set the release tag if it has not be yet done
	    if (!releaseTag)
	    {
	      dbe->cd();
	      dbe->bookString(
	        "ReleaseTag",
	        metoedmobject[i].release.substr(1,metoedmobject[i].release.size()-2)
	      );
	      releaseTag = true;
	    }
        
        std::string dir;
        
        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }
        
        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          me3[i] = dbe->book3D(metoedmobject[i].object.GetName(), &metoedmobject[i].object);
        } // end define new monitor elements
        
        // attach taglist
        TagList tags = metoedmobject[i].tags;
        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me3[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH3F creation
    
    if (classtypes[ii] == "TProfile") {
      edm::Handle<MEtoEDM<TProfile> > metoedm;
      iRun.getByType(metoedm);
      
      if (!metoedm.isValid()) {
      	//edm::LogWarning(MsgLoggerCat)
      	//  << "MEtoEDM<TProfile> doesn't exist in run";
      	continue;
      }
      
      std::vector<MEtoEDM<TProfile>::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject(); 

      me4.resize(metoedmobject.size());
      
      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me4[i] = 0;
	   
	    // get full path of monitor element
	    std::string pathname = metoedmobject[i].name;
	    if (verbosity) std::cout << pathname << std::endl;

        // set the release tag if it has not be yet done
	    if (!releaseTag)
	    {
	      dbe->cd();
	      dbe->bookString(
	        "ReleaseTag",
	        metoedmobject[i].release.substr(1,metoedmobject[i].release.size()-2)
	      );
	      releaseTag = true;
	    }
	    
	    std::string dir;
	    
	    // deconstruct path from fullpath
	    StringList fulldir = StringOps::split(pathname,"/");
	    
	    for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
	      dir += fulldir[j];
	      if (j != fulldir.size() - 2) dir += "/";
	    }
	
	    // define new monitor element
	    if (dbe) {
	      dbe->setCurrentFolder(dir);
	      me4[i] = dbe->bookProfile(metoedmobject[i].object.GetName(), &metoedmobject[i].object);
	    } // end define new monitor elements
	
        // attach taglist
        TagList tags = metoedmobject[i].tags;
        
        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me4[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TProfile creation

    if (classtypes[ii] == "TProfile2D") {
      edm::Handle<MEtoEDM<TProfile2D> > metoedm;
      iRun.getByType(metoedm);
      
      if (!metoedm.isValid()) {
      	//edm::LogWarning(MsgLoggerCat)
      	//  << "MEtoEDM<TProfile2D> doesn't exist in run";
      	continue;
      }
      
      std::vector<MEtoEDM<TProfile2D>::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject(); 
      
      me5.resize(metoedmobject.size());
      
      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me5[i] = 0;
        
        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity) std::cout << pathname << std::endl;

        // set the release tag if it has not be yet done
	    if (!releaseTag)
	    {
	      dbe->cd();
	      dbe->bookString(
	        "ReleaseTag",
	        metoedmobject[i].release.substr(1,metoedmobject[i].release.size()-2)
	      );
	      releaseTag = true;
	    }
        
        std::string dir;
        
        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }
        
        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          me5[i] = dbe->bookProfile2D(metoedmobject[i].object.GetName(), &metoedmobject[i].object);
        } // end define new monitor elements
        
        // attach taglist
        TagList tags = metoedmobject[i].tags;
        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me5[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TProfile2D creation

    if (classtypes[ii] == "Float") {
      edm::Handle<MEtoEDM<double> > metoedm;
      iRun.getByType(metoedm);
      
      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<double> doesn't exist in run";
        continue;
      }
      
      std::vector<MEtoEDM<double>::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject(); 
      
      me6.resize(metoedmobject.size());
      
      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {
        
        me6[i] = 0;
        
        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity) std::cout << pathname << std::endl;

        // set the release tag if it has not be yet done
	    if (!releaseTag)
	    {
	      dbe->cd();	
	      dbe->bookString(
	        "ReleaseTag",
	        metoedmobject[i].release.substr(1,metoedmobject[i].release.size()-2)
	      );
	      releaseTag = true;
	    }

        std::string dir;
        std::string name;
        
        // deconstruct path from fullpath
        
        StringList fulldir = StringOps::split(pathname,"/");
        name = *(fulldir.end() - 1);
        
        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }
        
        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          me6[i] = dbe->bookFloat(name);
          me6[i]->Fill(metoedmobject[i].object);
        } // end define new monitor elements
	
	    // attach taglist
	    TagList tags = metoedmobject[i].tags;
	    
	    for (unsigned int j = 0; j < tags.size(); ++j) {
	      dbe->tag(me6[i]->getFullname(),tags[j]);
	    }
      } // end loop thorugh metoedmobject      
    } // end Float creation

    if (classtypes[ii] == "Int") {
      edm::Handle<MEtoEDM<int> > metoedm;
      iRun.getByType(metoedm);
      
      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<int> doesn't exist in run";
        continue;
      }
      
      std::vector<MEtoEDM<int>::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject(); 
      
      me7.resize(metoedmobject.size());
      
      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {
        
        me7[i] = 0;
        
        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        
        if (verbosity) std::cout << pathname << std::endl;

        // set the release tag if it has not be yet done
	    if (!releaseTag)
	    {
	      dbe->cd();	
	      dbe->bookString(
	        "ReleaseTag",
	        metoedmobject[i].release.substr(1,metoedmobject[i].release.size()-2)
	      );
	      releaseTag = true;
	    }

        std::string dir;        
        std::string name;
        
        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        
        name = *(fulldir.end() - 1);
        
        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }
        
        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          me7[i] = dbe->bookInt(name);
          me7[i]->Fill(metoedmobject[i].object);
        } // end define new monitor elements
	
	    // attach taglist
	    TagList tags = metoedmobject[i].tags;
	    
	    for (unsigned int j = 0; j < tags.size(); ++j) {
	      dbe->tag(me7[i]->getFullname(),tags[j]);
	    }
      } // end loop thorugh metoedmobject      
    } // end Int creation

    if (classtypes[ii] == "String") {
      edm::Handle<MEtoEDM<TString> > metoedm;
      iRun.getByType(metoedm);
      
      if (!metoedm.isValid()) {
      	//edm::LogWarning(MsgLoggerCat)
      	//  << "MEtoEDM<TString> doesn't exist in run";
      	continue;
      }
      
      std::vector<MEtoEDM<TString>::MEtoEDMObject> metoedmobject = metoedm->getMEtoEdmObject(); 
      
      me8.resize(metoedmobject.size());
      
      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {
	
	    me8[i] = 0;
	
	    // get full path of monitor element
	    std::string pathname = metoedmobject[i].name;
	    if (verbosity) std::cout << pathname << std::endl;
	    
	    // set the release tag if it has not be yet done
	    if (!releaseTag)
	    {
	      dbe->cd();	
	      dbe->bookString(
	        "ReleaseTag",
	        metoedmobject[i].release.substr(1,metoedmobject[i].release.size()-2)
	      );
	      releaseTag = true;
	    }

	    std::string dir;
	    std::string name;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        name = *(fulldir.end() - 1);
        
        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }
        
        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          std::string scont = metoedmobject[i].object.Data();
          me8[i] = dbe->bookString(name,scont);
        } // end define new monitor elements
	
	    // attach taglist
	    TagList tags = metoedmobject[i].tags;
	    
	    for (unsigned int j = 0; j < tags.size(); ++j) {
	      dbe->tag(me8[i]->getFullname(),tags[j]);
	    }
      } // end loop thorugh metoedmobject 
    } // end String creation
  }

  // verify tags stored properly
  if (verbosity) {
    std::vector<std::string> stags;
    dbe->getAllTags(stags);
    for (unsigned int i = 0; i < stags.size(); ++i) {
      std::cout << "Tags: " << stags[i] << std::endl;
    }
  }
  
  return;
}

void EDMtoMEConverter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  return;
}

