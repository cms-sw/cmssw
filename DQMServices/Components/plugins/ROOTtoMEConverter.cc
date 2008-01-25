/** \file ROOTtoMEConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2008/01/25 22:06:48 $
 *  $Revision: 1.8 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DQMServices/Components/plugins/ROOTtoMEConverter.h"

ROOTtoMEConverter::ROOTtoMEConverter(const edm::ParameterSet & iPSet) :
  fName(""), verbosity(0), frequency(0), count(0)
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_ROOTtoMEConverter";
  
  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  outputfile = iPSet.getParameter<std::string>("Outputfile");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
 
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
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
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Outputfile    = " << outputfile << "\n"
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

} // end constructor

ROOTtoMEConverter::~ROOTtoMEConverter() 
{
  if (outputfile.size() != 0 && dbe) dbe->save(outputfile);
} // end destructor

void ROOTtoMEConverter::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void ROOTtoMEConverter::endJob()
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " runs.";
  return;
}

void ROOTtoMEConverter::beginRun(const edm::Run& iRun, 
				 const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "ROOTtoMEConverter_beginRun";
  
  // keep track of number of events processed
  ++count;
  
  int nrun = iRun.run();
  
  if (verbosity) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << count << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || count == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << " (" << count << " runs total)";
    }
  }
  
  return;
}

void ROOTtoMEConverter::endRun(const edm::Run& iRun, 
			       const edm::EventSetup& iSetup)
{
  
  std::string MsgLoggerCat = "ROOTtoMEConverter_endRun";
  
  if (verbosity >= 0)
    edm::LogInfo (MsgLoggerCat)
      << "\nRestoring MonitorElements.";
    
  for (unsigned int ii = 0; ii < classtypes.size(); ++ii) {

    if (classtypes[ii] == "TH1F") {

      edm::Handle<MEtoROOT<TH1F> > metoroot;
      iRun.getByType(metoroot);
      
      if (!metoroot.isValid()) {
	//edm::LogWarning(MsgLoggerCat)
	//  << "MEtoROOT<TH1F> doesn't exist in run";
	continue;
      }
      
      std::vector<MEtoROOT<TH1F>::MEROOTObject> merootobject = 
	metoroot->getMERootObject(); 
      
      me1.resize(merootobject.size());
      
      for (unsigned int i = 0; i < merootobject.size(); ++i) {
	
	me1[i] = 0;
	
	// get full path of monitor element
	std::string pathname = merootobject[i].name;
	if (verbosity) std::cout << pathname << std::endl;
	
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

	  me1[i] = dbe->clone1D(merootobject[i].object.GetName(),
				&merootobject[i].object);
	} // end define new monitor elements

	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me1[i]->getFullname(),tags[j]);
	}

      } // end loop thorugh merootobject
    } // end TH1F creation

    if (classtypes[ii] == "TH2F") {
      
      edm::Handle<MEtoROOT<TH2F> > metoroot;
      iRun.getByType(metoroot);
      
      if (!metoroot.isValid()) {
	//edm::LogWarning(MsgLoggerCat)
	//  << "MEtoROOT<TH2F> doesn't exist in run";
	continue;
      }
      
      std::vector<MEtoROOT<TH2F>::MEROOTObject> merootobject = 
	metoroot->getMERootObject(); 
      
      me2.resize(merootobject.size());
      
      for (unsigned int i = 0; i < merootobject.size(); ++i) {
	
	me2[i] = 0;
	
	// get full path of monitor element
	std::string pathname = merootobject[i].name;
	if (verbosity) std::cout << pathname << std::endl;
	
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

	  me2[i] = dbe->clone2D(merootobject[i].object.GetName(),
				&merootobject[i].object);

	} // end define new monitor elements

	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me2[i]->getFullname(),tags[j]);
	}

      } // end loop thorugh merootobject
    } // end TH2F creation

    if (classtypes[ii] == "TH3F") {

      edm::Handle<MEtoROOT<TH3F> > metoroot;
      iRun.getByType(metoroot);
      
      if (!metoroot.isValid()) {
	//edm::LogWarning(MsgLoggerCat)
	//  << "MEtoROOT<TH3F> doesn't exist in run";
	continue;
      }
      
      std::vector<MEtoROOT<TH3F>::MEROOTObject> merootobject = 
	metoroot->getMERootObject(); 
      
      me3.resize(merootobject.size());
      
      for (unsigned int i = 0; i < merootobject.size(); ++i) {
	
	me3[i] = 0;
	
	// get full path of monitor element
	std::string pathname = merootobject[i].name;
	if (verbosity) std::cout << pathname << std::endl;
	
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
	  
	  me3[i] = dbe->clone3D(merootobject[i].object.GetName(),
				&merootobject[i].object);
	} // end define new monitor elements

	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me3[i]->getFullname(),tags[j]);
	}

      } // end loop thorugh merootobject
    } // end TH3F creation
    
    if (classtypes[ii] == "TProfile") {
      edm::Handle<MEtoROOT<TProfile> > metoroot;
      iRun.getByType(metoroot);
      
      if (!metoroot.isValid()) {
	//edm::LogWarning(MsgLoggerCat)
	//  << "MEtoROOT<TProfile> doesn't exist in run";
	continue;
      }
      
      std::vector<MEtoROOT<TProfile>::MEROOTObject> merootobject = 
	metoroot->getMERootObject(); 
      
      me4.resize(merootobject.size());
      
      for (unsigned int i = 0; i < merootobject.size(); ++i) {
	
	me4[i] = 0;
	
	// get full path of monitor element
	std::string pathname = merootobject[i].name;
	if (verbosity) std::cout << pathname << std::endl;
	
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
	  
	  me4[i] = dbe->cloneProfile(merootobject[i].object.GetName(),
				     &merootobject[i].object);
	} // end define new monitor elements

	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me4[i]->getFullname(),tags[j]);
	}

      } // end loop thorugh merootobject
    } // end TProfile creation

    if (classtypes[ii] == "TProfile2D") {
      edm::Handle<MEtoROOT<TProfile2D> > metoroot;
      iRun.getByType(metoroot);
      
      if (!metoroot.isValid()) {
	//edm::LogWarning(MsgLoggerCat)
	//  << "MEtoROOT<TProfile2D> doesn't exist in run";
	continue;
      }
      
      std::vector<MEtoROOT<TProfile2D>::MEROOTObject> merootobject = 
	metoroot->getMERootObject(); 
      
      me5.resize(merootobject.size());
      
      for (unsigned int i = 0; i < merootobject.size(); ++i) {
	
	me5[i] = 0;
	
	// get full path of monitor element
	std::string pathname = merootobject[i].name;
	if (verbosity) std::cout << pathname << std::endl;
	
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
	  
	  me5[i] = dbe->cloneProfile2D(merootobject[i].object.GetName(),
				       &merootobject[i].object);
	} // end define new monitor elements
	
	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me5[i]->getFullname(),tags[j]);
	}

      } // end loop thorugh merootobject
    } // end TProfile2D creation

    if (classtypes[ii] == "Float") {
      edm::Handle<MEtoROOT<float> > metoroot;
      iRun.getByType(metoroot);
      
      if (!metoroot.isValid()) {
	//edm::LogWarning(MsgLoggerCat)
	//  << "MEtoROOT<float> doesn't exist in run";
	continue;
      }
      
      std::vector<MEtoROOT<float>::MEROOTObject> merootobject = 
	metoroot->getMERootObject(); 
      
      me6.resize(merootobject.size());
      
      for (unsigned int i = 0; i < merootobject.size(); ++i) {
	
	me6[i] = 0;
	
	// get full path of monitor element
	std::string pathname = merootobject[i].name;
	if (verbosity) std::cout << pathname << std::endl;
	
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
	  me6[i]->Fill(merootobject[i].object);

	} // end define new monitor elements
	
	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me6[i]->getFullname(),tags[j]);
	}
	
      } // end loop thorugh merootobject      
      
    } // end Float creation

    if (classtypes[ii] == "Int") {
      edm::Handle<MEtoROOT<int> > metoroot;
      iRun.getByType(metoroot);
      
      if (!metoroot.isValid()) {
	//edm::LogWarning(MsgLoggerCat)
	//  << "MEtoROOT<int> doesn't exist in run";
	continue;
      }
      
      std::vector<MEtoROOT<int>::MEROOTObject> merootobject = 
	metoroot->getMERootObject(); 
      
      me7.resize(merootobject.size());
      
      for (unsigned int i = 0; i < merootobject.size(); ++i) {
	
	me7[i] = 0;
	
	// get full path of monitor element
	std::string pathname = merootobject[i].name;
	if (verbosity) std::cout << pathname << std::endl;
	
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
	  me7[i]->Fill(merootobject[i].object);
	  
	} // end define new monitor elements
	
	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me7[i]->getFullname(),tags[j]);
	}
      } // end loop thorugh merootobject      
    } // end Int creation

    if (classtypes[ii] == "String") {
      edm::Handle<MEtoROOT<TString> > metoroot;
      iRun.getByType(metoroot);
      
      if (!metoroot.isValid()) {
	//edm::LogWarning(MsgLoggerCat)
	//  << "MEtoROOT<TString> doesn't exist in run";
	continue;
      }
      
      std::vector<MEtoROOT<TString>::MEROOTObject> merootobject = 
	metoroot->getMERootObject(); 
      
      me8.resize(merootobject.size());
      
      for (unsigned int i = 0; i < merootobject.size(); ++i) {
	
	me8[i] = 0;
	
	// get full path of monitor element
	std::string pathname = merootobject[i].name;
	if (verbosity) std::cout << pathname << std::endl;
	
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
	  
	  std::string scont = merootobject[i].object.Data();
	  me8[i] = dbe->bookString(name,scont);
	  
	} // end define new monitor elements
	
	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me8[i]->getFullname(),tags[j]);
	}

      } // end loop thorugh merootobject 
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

void ROOTtoMEConverter::analyze(const edm::Event& iEvent, 
				const edm::EventSetup& iSetup)
{
  return;
}

