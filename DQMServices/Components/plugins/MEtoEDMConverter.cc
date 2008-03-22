
/** \file MEtoEDMConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2008/03/04 19:17:06 $
 *  $Revision: 1.4.2.4 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "DQMServices/Components/plugins/MEtoEDMConverter.h"

MEtoEDMConverter::MEtoEDMConverter(const edm::ParameterSet & iPSet) :
  fName(""), verbosity(0), frequency(0)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_MEtoEDMConverter";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  
  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;
  
  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "===============================\n";
  }

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

  // clear out the vector holders and information flags
  items.clear();
  pkgvec.clear();
  pathvec.clear();
  mevec.clear();
  metype.clear();
  packages.clear();
  hasTH1F = false; nTH1F = 0;
  hasTH2F = false; nTH2F = 0;
  hasTH3F = false; nTH3F = 0;
  hasTProfile = false; nTProfile = 0;
  hasTProfile2D = false; nTProfile2D = 0;
  hasFloat = false; nFloat = 0;
  hasInt = false; nInt = 0;
  hasString = false; nString = 0;

  // get contents out of DQM
  std::string classname;
  dbe->getContents(items);
 
  if (!items.empty()) {
    for (i = items.begin (), e = items.end (); i != e; ++i) {

      // verify I have the expected string format
      assert(StringOps::contains(*i,':') == 1);

      // get list of things seperated by :
      StringList item = StringOps::split(*i, ":");

      // get list of directories
      StringList dir = StringOps::split(item[0],"/");

      // keep track of leading directory (i.e. package)
      n = dir.begin();
      std::string package = *n;
      ++packages[package];

      // get list of monitor elements
      StringList me = StringOps::split(item[1],",");

      // keep track of package, path, me name and type for each valid monitor 
      // element
      for (n = me.begin(), m = me.end(); n != m; ++n) {

	//get full path
	std::string fullpath; 
	fullpath.reserve(item[0].size() + me.size() + 2);
	fullpath += item[0];
	if (!item[0].empty()) fullpath += "/";
	fullpath += *n;
	if (verbosity >1 ) std::cout << "Full path is: " << fullpath 
				     << std::endl;
    
	// verify valid monitor elements by type
	bool validME = false;
	if (verbosity > 1) std::cout << "MEobject:" << std::endl;
	if (MonitorElement *me = dbe->get(fullpath)) {

	  // extract classname
	  if (ROOTObj *ob = dynamic_cast<ROOTObj *>(me)) {
	    if (TObject *tobj = ob->operator->()){
	      validME = true;
	      if (verbosity > 1) std::cout << "   normal: " << tobj->GetName();
	      classname = tobj->ClassName();
	      metype.push_back(classname);
	      if (verbosity > 1) std::cout << " is of type " << classname 
					   << std::endl;
	    } 
	  } else if (FoldableMonitor *ob = 
		     dynamic_cast<FoldableMonitor *>(me)) {
	    if (TObject *tobj = ob->getTagObject()) {
	      validME = true;
	      if (verbosity > 1) std::cout << "   foldable: " 
					   << tobj->GetName();
	      classname = tobj->ClassName();
	      if (classname == "TObjString") {
		if (TObjString* histogram = 
		    dynamic_cast<TObjString*>(ob->getTagObject())) {

		  // get contents of TObjString
		  TString contents = histogram->GetName();
		  std::string scont = contents.Data();

		  // verify I have the expected string format		
		  assert(StringOps::contains(scont,'=') == 1);
		  
		  // get list of things seperated by =
		  StringList sitem = StringOps::split(scont, "=");

		  // get front item separated by >
		  StringList sitem1 = StringOps::split(sitem[0], ">");
		  
		  if (sitem1[1] == "f") classname = "Float";
		  if (sitem1[1] == "i") classname = "Int";
		  if (sitem1[1] == "s") classname = "String";
		}
	      }
	      metype.push_back(classname);
	      if (verbosity > 1) std::cout << " is of type " << classname 
				       << std::endl;
	    }
	  }
	  if (!validME) {
	    edm::LogError(MsgLoggerCat)
	      << "ERROR: The DQM object '" << fullpath
	      << "' is neither a ROOT object nor a recognised "
	      << "simple object.\n";
	   continue;
	  }

	  if (classname == "TH1F") {
	    hasTH1F = true;
	    ++nTH1F;
	  }
	  if (classname == "TH2F") {
	    hasTH2F = true;
	    ++nTH2F;
	  }
	  if (classname == "TH3F") {
	    hasTH3F = true;
	    ++nTH3F;
	  }
	  if (classname == "TProfile") {
	    hasTProfile = true;
	    ++nTProfile;
	  }
	  if (classname == "TProfile2D") {
	    hasTProfile2D = true;
	    ++nTProfile2D;
	  }
	  if (classname == "Float") {
	    hasFloat = true;
	    ++nFloat;
	  }
	  if (classname == "Int") {
	    hasInt = true;
	    ++nInt;
	  }
	  if (classname == "String") {
	    hasString = true;
	    ++nString;
	  }
	} // end loop through monitor elements
	
	fullpathvec.push_back(fullpath);
	pkgvec.push_back(package);
	pathvec.push_back(item[0]);
	mevec.push_back(*n);
      }
    } // end loop through me items
  } // end check that items exist

  if (verbosity) {
    // list unique packages
    std::cout << "Packages accessing DQM:" << std::endl;
    for (pkgIter = packages.begin(); pkgIter != packages.end(); ++pkgIter) {
      std::cout << "  " << pkgIter->first << std::endl;
    }
    
    std::cout << "Monitor Elements detected:" << std::endl;
    for (unsigned int a = 0; a < pkgvec.size(); ++a) {
      std::cout << "   " << pkgvec[a] << " " << pathvec[a] << " " << mevec[a] 
		<< std::endl;
    }

    std::cout << "We have " << nTH1F << " TH1F objects" << std::endl;
    std::cout << "We have " << nTH2F << " TH2F objects" << std::endl;
    std::cout << "We have " << nTH3F << " TH3F objects" << std::endl;
    std::cout << "We have " << nTProfile << " TProfile objects" << std::endl;
    std::cout << "We have " << nTProfile2D << " TProfile2D objects" 
	      << std::endl;
    std::cout << "We have " << nFloat << " Float objects" << std::endl;
    std::cout << "We have " << nInt << " Int objects" << std::endl;
    std::cout << "We have " << nString << " String objects" << std::endl;
  }
      
  // create persistent objects
  if (hasTH1F)
    produces<MEtoEDM<TH1F>, edm::InRun>(fName);
  if (hasTH2F)
    produces<MEtoEDM<TH2F>, edm::InRun>(fName);
  if (hasTH3F)
    produces<MEtoEDM<TH3F>, edm::InRun>(fName);
  if (hasTProfile)
    produces<MEtoEDM<TProfile>, edm::InRun>(fName);
  if (hasTProfile2D)
    produces<MEtoEDM<TProfile2D>, edm::InRun>(fName);
  if (hasFloat)
    produces<MEtoEDM<float>, edm::InRun>(fName);
  if (hasInt)
    produces<MEtoEDM<int>, edm::InRun>(fName);
  if (hasString)
    produces<MEtoEDM<TString>, edm::InRun>(fName);

  count.clear();

} // end constructor

MEtoEDMConverter::~MEtoEDMConverter() 
{
} // end destructor

void MEtoEDMConverter::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void MEtoEDMConverter::endJob()
{
  std::string MsgLoggerCat = "MEtoEDMConverter_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count.size() << " runs.";
  return;
}

void MEtoEDMConverter::beginRun(edm::Run& iRun, 
				 const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "MEtoEDMConverter_beginRun";
    
  int nrun = iRun.run();
  
  // keep track of number of runs processed
  ++count[nrun];

  if (verbosity) {  // keep track of number of runs processed
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << count.size() << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || count.size() == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << " (" << count.size() << " runs total)";
    }
  }
  
  // clear out object holders
  TH1FME.name.clear();
  TH1FME.tags.clear();
  TH1FME.object.clear();

  TH2FME.name.clear();
  TH2FME.tags.clear();
  TH2FME.object.clear();

  TH3FME.name.clear();
  TH3FME.tags.clear();
  TH3FME.object.clear();

  TProfileME.name.clear();
  TProfileME.tags.clear();
  TProfileME.object.clear();

  TProfile2DME.name.clear();
  TProfile2DME.tags.clear();
  TProfile2DME.object.clear();

  FloatME.name.clear();
  FloatME.tags.clear();
  FloatME.object.clear();

  IntME.name.clear();
  IntME.tags.clear();
  IntME.object.clear();

  StringME.name.clear();
  StringME.tags.clear();
  StringME.object.clear();

  taglist.clear();

  // reset monitor elements
  for (unsigned int a = 0; a < pkgvec.size(); ++a) {
   
    if (MonitorElement *me = dbe->get(fullpathvec[a])) {
      
      // reset the ROOT object.  This is either a genuine ROOT object,
      // or a scalar one that stores its value as TObjString.
      if (ROOTObj *ob = dynamic_cast<ROOTObj *>(me)) {
	if (dynamic_cast<TH1F*>(ob->operator->())) {
          me->Reset();
	}
	if (dynamic_cast<TH2F*>(ob->operator->())) {
          me->Reset();
	}	
	if (dynamic_cast<TH3F*>(ob->operator->())) {
          me->Reset();
	}  
	if (dynamic_cast<TProfile*>(ob->operator->())) {
          me->Reset();
	}  	
	if (dynamic_cast<TProfile2D*>(ob->operator->())) {
          me->Reset();
	}
      }
    } // end get monitor element
  } // end loop through all monitor elements

  return;
}

void MEtoEDMConverter::endRun(edm::Run& iRun, const edm::EventSetup& iSetup)
{
  
  std::string MsgLoggerCat = "MEtoEDMConverter_endRun";
  
  if (verbosity)
    edm::LogInfo (MsgLoggerCat) << "\nStoring MEtoEDM dataformat histograms.";

  // extract ME information into vectors
  for (unsigned int a = 0; a < pkgvec.size(); ++a) {

    taglist.clear();

    // already extracted full path name in constructor
    if (verbosity > 1) std::cout << "name: " << fullpathvec[a] << std::endl;

    // get tags
    bool foundtags  = false;
    dqm::me_util::dirt_it idir = dbe->allTags.find(pathvec[a]);
    if (idir != dbe->allTags.end()) {
      dqm::me_util::tags_it itag = idir->second.find(mevec[a]);
      if (itag != idir->second.end()) {
	taglist.resize(itag->second.size());
	std::copy(itag->second.begin(), itag->second.end(), taglist.begin());
	foundtags = true;
      }
    }
    if (!foundtags) taglist.clear();
    if (verbosity > 1) {
      std::cout << "taglist:" << std::endl;
      for (unsigned int ii = 0; ii < taglist.size(); ++ii) {
	std::cout << "   " << taglist[ii] << std::endl;
      }
    }

    // get monitor elements
    bool validME = false;
   
    if (verbosity > 1) std::cout << "MEobject:" << std::endl;
    if (MonitorElement *me = dbe->get(fullpathvec[a])) {
      
      // Save the ROOT object.  This is either a genuine ROOT object,
      // or a scalar one that stores its value as TObjString.
      if (ROOTObj *ob = dynamic_cast<ROOTObj *>(me)) {
	if (TH1F* histogram = dynamic_cast<TH1F*>(ob->operator->())) {
	  validME = true;
	  if (verbosity > 1) {
	    std::cout << "   normal: " << histogram->GetName() << std::endl;
	    std::cout << "      classname: " << metype[a] << std::endl;
	  }
	  TH1FME.object.push_back(*histogram);
	  TH1FME.name.push_back(fullpathvec[a]);
	  TH1FME.tags.push_back(taglist);
	}
	if (TH2F* histogram = dynamic_cast<TH2F*>(ob->operator->())) {
	  validME = true;
	  if (verbosity > 1) {
	    std::cout << "   normal: " << histogram->GetName() << std::endl;
	    std::cout << "      classname: " << metype[a] << std::endl;
	  }
	  TH2FME.object.push_back(*histogram);
	  TH2FME.name.push_back(fullpathvec[a]);
	  TH2FME.tags.push_back(taglist);
	}	
	if (TH3F* histogram = dynamic_cast<TH3F*>(ob->operator->())) {
	  validME = true;
	  if (verbosity > 1) {
	    std::cout << "   normal: " << histogram->GetName() << std::endl;
	    std::cout << "      classname: " << metype[a] << std::endl;
	  }
	  TH3FME.object.push_back(*histogram);
	  TH3FME.name.push_back(fullpathvec[a]);
	  TH3FME.tags.push_back(taglist);
	}  
	if (TProfile* histogram = dynamic_cast<TProfile*>(ob->operator->())) {
	  validME = true;
	  if (verbosity > 1) {
	    std::cout << "   normal: " << histogram->GetName() << std::endl;
	    std::cout << "      classname: " << metype[a] << std::endl;
	  }
	  TProfileME.object.push_back(*histogram);
	  TProfileME.name.push_back(fullpathvec[a]);
	  TProfileME.tags.push_back(taglist);
	}  	
	if (TProfile2D* histogram = 
	    dynamic_cast<TProfile2D*>(ob->operator->())) {
	  validME = true;
	  if (verbosity > 1) {
	    std::cout << "   normal: " << histogram->GetName() << std::endl;
	    std::cout << "      classname: " << metype[a] << std::endl;
	  }
	  TProfile2DME.object.push_back(*histogram);
	  TProfile2DME.name.push_back(fullpathvec[a]);
	  TProfile2DME.tags.push_back(taglist);
	}
      } else if (FoldableMonitor *ob = dynamic_cast<FoldableMonitor *>(me)) {
	if (TObjString* histogram = 
	    dynamic_cast<TObjString*>(ob->getTagObject())) {
	  validME = true;
	  if (verbosity > 1) std::cout << "   foldable: " 
				       << histogram->GetName() << std::endl;
	
	  // get contents of TObjString
	  TString contents = histogram->GetName();
	  std::string scont = contents.Data();	  

	  // verify I have the expected string format		
	  assert(StringOps::contains(scont,'=') == 1);
	  
	  // get list of things seperated by =
	  StringList sitem = StringOps::split(scont, "=");
	  
	  // get front item separated by >
	  StringList sitem1 = StringOps::split(sitem[0], ">");
	  
	  std::string classname;
	  if (sitem1[1] == "f") classname = "Float";
	  if (sitem1[1] == "i") classname = "Int";
	  if (sitem1[1] == "s") classname = "String";
	  if (verbosity > 1) std::cout << "      classname: " << classname 
				       << std::endl;
	  
	  // get back item separated by <
	  StringList sitem2 = StringOps::split(sitem[1], "<");
	  
	  if (classname == "Float") {
	    FloatME.object.push_back(atof(sitem2[0].c_str()));
	    if (verbosity > 1)
	      std::cout << "      value: " << atof(sitem2[0].c_str()) 
			<< std::endl;
	    FloatME.name.push_back(fullpathvec[a]);
	    FloatME.tags.push_back(taglist);
	  }
	  if (classname == "Int") {
	    IntME.object.push_back(atoi(sitem2[0].c_str()));
	    if (verbosity > 1)
	      std::cout << "      value: " << atoi(sitem2[0].c_str()) 
			<< std::endl;
	    IntME.name.push_back(fullpathvec[a]);
	    IntME.tags.push_back(taglist);
	  }
	  if (classname == "String") {
	    StringME.object.push_back(sitem2[0]);
	    if (verbosity > 1) 
	      std::cout << "      value: " << sitem2[0]
			<< std::endl;
	    StringME.name.push_back(fullpathvec[a]);
	    StringME.tags.push_back(taglist);
	  }
	}  
      }
      if (!validME) {
	edm::LogError(MsgLoggerCat)
	  << "ERROR: The DQM object '" << fullpathvec[a]
	  << "' is neither a ROOT object nor a recognised "
	  << "simple object.\n";
	return;
      }
    } // end get monitor element
  } // end loop through all monitor elements

  // produce objects to put in events
  if (hasTH1F) {
    std::auto_ptr<MEtoEDM<TH1F> > pOut1(new MEtoEDM<TH1F>);
    pOut1->putMEtoEdmObject(TH1FME.name,TH1FME.tags,TH1FME.object);
    iRun.put(pOut1,fName);
  }
  if (hasTH2F) {
    std::auto_ptr<MEtoEDM<TH2F> > pOut2(new MEtoEDM<TH2F>);
    pOut2->putMEtoEdmObject(TH2FME.name,TH2FME.tags,TH2FME.object);
    iRun.put(pOut2,fName);
  }
  if (hasTH3F) {
    std::auto_ptr<MEtoEDM<TH3F> > pOut3(new MEtoEDM<TH3F>);
    pOut3->putMEtoEdmObject(TH3FME.name,TH3FME.tags,TH3FME.object);
    iRun.put(pOut3,fName);
  }
  if (hasTProfile) {
    std::auto_ptr<MEtoEDM<TProfile> > pOut4(new MEtoEDM<TProfile>);
    pOut4->putMEtoEdmObject(TProfileME.name,TProfileME.tags,TProfileME.object);
    iRun.put(pOut4,fName);
  }
  if (hasTProfile2D) {
    std::auto_ptr<MEtoEDM<TProfile2D> > pOut5(new MEtoEDM<TProfile2D>);
    pOut5->putMEtoEdmObject(TProfile2DME.name,TProfile2DME.tags,
			   TProfile2DME.object);
    iRun.put(pOut5,fName);
  }
  if (hasFloat) {
    std::auto_ptr<MEtoEDM<float> > pOut6(new MEtoEDM<float>);
    pOut6->putMEtoEdmObject(FloatME.name,FloatME.tags,FloatME.object);
    iRun.put(pOut6,fName);
  }
  if (hasInt) {
    std::auto_ptr<MEtoEDM<int> > pOut7(new MEtoEDM<int>);
    pOut7->putMEtoEdmObject(IntME.name,IntME.tags,IntME.object);
    iRun.put(pOut7,fName);
  }
  if (hasString) {
    std::auto_ptr<MEtoEDM<TString> > 
      pOut8(new MEtoEDM<TString>);
    pOut8->putMEtoEdmObject(StringME.name,StringME.tags,StringME.object);
    iRun.put(pOut8,fName);
  }

  TH1FME.name.clear();
  TH1FME.tags.clear();
  TH1FME.object.clear();

  TH2FME.name.clear();
  TH2FME.tags.clear();
  TH2FME.object.clear();

  TH3FME.name.clear();
  TH3FME.tags.clear();
  TH3FME.object.clear();

  TProfileME.name.clear();
  TProfileME.tags.clear();
  TProfileME.object.clear();

  TProfile2DME.name.clear();
  TProfile2DME.tags.clear();
  TProfile2DME.object.clear();

  FloatME.name.clear();
  FloatME.tags.clear();
  FloatME.object.clear();

  IntME.name.clear();
  IntME.tags.clear();
  IntME.object.clear();

  StringME.name.clear();
  StringME.tags.clear();
  StringME.object.clear();

  taglist.clear();

  return;
}

void MEtoEDMConverter::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup)
{
  return;
}
