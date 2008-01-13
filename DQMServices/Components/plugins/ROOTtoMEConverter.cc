/** \file ROOTtoMEConverter.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2008/01/13 00:14:15 $
 *  $Revision: 1.5 $
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
    if (verbosity > 0 ) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }

  //if (dbe) {
  //  if (verbosity > 0 ) dbe->showDirStructure();
  //}
  
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
  
  if (verbosity > 0) {
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
	//std::cout << pathname << std::endl;
	
	std::string dir;
	
	// deconstruct path from fullpath
	StringList fulldir = StringOps::split(pathname,"/");
	for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
	  dir += fulldir[j];
	  if (j != fulldir.size() - 2) dir += "/";
	}
	//std::cout << dir << std::endl;    
	
	// define new monitor element
	if (dbe) {
	  dbe->setCurrentFolder(dir);

	  me1[i] = dbe->clone1D(merootobject[i].object.GetName(),
				&merootobject[i].object);
	  
	  // fill new monitor element
	  Int_t nbins = merootobject[i].object.GetXaxis()->GetNbins();
	  for (Int_t x = 1; x <= nbins; ++x) {
	    Double_t xbincenter = merootobject[i].object.GetBinCenter(x);
	    Double_t value = merootobject[i].object.GetBinContent(x);
	    //Double_t error = merootobject[i].object.GetBinError(x);

	    //me1[i]->setBinContent(x,value);
	    //me1[i]->setBinError(x,error);
	    
	    for (Int_t val = 0; val < value; ++val) {
	      me1[i]->Fill(xbincenter);
	    }
	  } // end fill
   
	} // end define new monitor elements

	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me1[i],tags[j]);
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
	//std::cout << pathname << std::endl;
	
	std::string dir;
	
	// deconstruct path from fullpath
	StringList fulldir = StringOps::split(pathname,"/");
	for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
	  dir += fulldir[j];
	  if (j != fulldir.size() - 2) dir += "/";
	}
	//std::cout << dir << std::endl;    
	
	// define new monitor element
	if (dbe) {
	  dbe->setCurrentFolder(dir);

	  me2[i] = dbe->clone2D(merootobject[i].object.GetName(),
				&merootobject[i].object);
	  
	  // fill new monitor element
	  Int_t nxbins = merootobject[i].object.GetXaxis()->GetNbins();
	  Int_t nybins = merootobject[i].object.GetYaxis()->GetNbins();
	  for (Int_t x = 1; x <= nxbins; ++x) {
	    Double_t xbincenter = merootobject[i].object.
	      GetXaxis()->GetBinCenter(x);
	    for (Int_t y = 1; y <= nybins; ++y) {
	      Double_t ybincenter = merootobject[i].object.
		GetYaxis()->GetBinCenter(y);
	      Double_t value = 
		merootobject[i].object.GetBinContent(x,y);
	      //Double_t error = merootobject[i].object.GetBinError(x,y);

	      //me2[i]->setBinContent(x,y,value);
	      //me2[i]->setBinError(x,y,error);
	      
	      for (Int_t val = 0; val < value; ++val) {
		me2[i]->Fill(xbincenter,ybincenter);
	      }
	    } // end loop through y
	  } // end loop through x

	} // end define new monitor elements

	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me2[i],tags[j]);
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
	//std::cout << pathname << std::endl;
	
	std::string dir;
	
	// deconstruct path from fullpath
	StringList fulldir = StringOps::split(pathname,"/");
	for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
	  dir += fulldir[j];
	  if (j != fulldir.size() - 2) dir += "/";
	}
	//std::cout << dir << std::endl;    
	
	// define new monitor element
	if (dbe) {
	  dbe->setCurrentFolder(dir);
	  
	  me3[i] = dbe->clone3D(merootobject[i].object.GetName(),
				&merootobject[i].object);

	  // fill new monitor element
	  Int_t nxbins = merootobject[i].object.GetXaxis()->GetNbins();
	  Int_t nybins = merootobject[i].object.GetYaxis()->GetNbins();
	  Int_t nzbins = merootobject[i].object.GetZaxis()->GetNbins();
	  for (Int_t x = 1; x <= nxbins; ++x) {
	    Double_t xbincenter = merootobject[i].object.
	      GetXaxis()->GetBinCenter(x);
	    for (Int_t y = 1; y <= nybins; ++y) {
	      Double_t ybincenter = merootobject[i].object.
		GetYaxis()->GetBinCenter(y);
	      for (Int_t z = 1; z <= nzbins; ++z) {
		Double_t zbincenter = merootobject[i].object.
		  GetZaxis()->GetBinCenter(z);
		Double_t value = merootobject[i].object.GetBinContent(x,y,z);
		//Double_t error = merootobject[i].object.GetBinError(x,y,z);
	      
		//me3[i]->setBinContent(x,y,z,value);
		//me3[i]->setBinError(x,y,z,error);		  
		for (Int_t val = 0; val < value; ++val) {
		  me3[i]->Fill(xbincenter,ybincenter,zbincenter);
		}
	      } // end loop through z
	    } // end loop through y
	  } // end loop through x
   
	} // end define new monitor elements

	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me3[i],tags[j]);
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
	//std::cout << pathname << std::endl;
	
	std::string dir;
	
	// deconstruct path from fullpath
	StringList fulldir = StringOps::split(pathname,"/");
	for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
	  dir += fulldir[j];
	  if (j != fulldir.size() - 2) dir += "/";
	}
	//std::cout << dir << std::endl;    
	
	// define new monitor element
	if (dbe) {
	  dbe->setCurrentFolder(dir);
	  
	  me4[i] = dbe->cloneProfile(merootobject[i].object.GetName(),
				     &merootobject[i].object);

	  // doesn't work properly as y information is lost
	  // fill new monitor element
	  Int_t nxbins = merootobject[i].object.GetXaxis()->GetNbins();
	  for (Int_t x = 1; x <= nxbins; ++x) {
	    Double_t xbincenter = merootobject[i].object.
	      GetXaxis()->GetBinCenter(x);
	    Double_t value = merootobject[i].object.GetBinContent(x);
	    //Double_t error = merootobject[i].object.GetBinError(x);
	      
	    //me4[i]->setBinContent(x,value);
	    //me4[i]->setBinError(x,error);
	    for (Int_t val = 0; val < value; ++val) {
	      me4[i]->Fill(xbincenter,1);
	    }
	  } // end loop through x
 
	} // end define new monitor elements

	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me4[i],tags[j]);
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
	//std::cout << pathname << std::endl;
	
	std::string dir;
	
	// deconstruct path from fullpath
	StringList fulldir = StringOps::split(pathname,"/");
	for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
	  dir += fulldir[j];
	  if (j != fulldir.size() - 2) dir += "/";
	}
	//std::cout << dir << std::endl;    
	
	// define new monitor element
	if (dbe) {
	  dbe->setCurrentFolder(dir);
	  
	  me5[i] = dbe->cloneProfile2D(merootobject[i].object.GetName(),
				       &merootobject[i].object);

	  // doesn't work as y and z information are lost
	  // fill new monitor element
	  Int_t nxbins = merootobject[i].object.GetXaxis()->GetNbins();
	  Int_t nybins = merootobject[i].object.GetYaxis()->GetNbins();
	  for (Int_t x = 1; x <= nxbins; ++x) {
	    Double_t xbincenter = merootobject[i].object.
	      GetXaxis()->GetBinCenter(x);
	    for (Int_t y = 1; y <= nybins; ++y) {
	      Double_t ybincenter = merootobject[i].object.
		GetYaxis()->GetBinCenter(y);
	      Double_t value = 
		merootobject[i].object.GetBinContent(x,y);
	      //Double_t error = merootobject[i].object.GetBinError(x,y);
	      
	      //me5[i]->setBinContent(x,y,value);
	      //me5[i]->setBinError(x,y,error);
	      for (Int_t val = 0; val < value; ++val) {
		me5[i]->Fill(xbincenter,ybincenter,1);
	      }	      
	    } // end loop through y
	  } // end loop through x

	} // end define new monitor elements
	
	// attach taglist
	TagList tags = merootobject[i].tags;
	for (unsigned int j = 0; j < tags.size(); ++j) {
	  dbe->tag(me5[i],tags[j]);
	}

      } // end loop thorugh merootobject
    } // end TProfile2D creation

    if (classtypes[ii] == "Float") {

    } // end Float creation

    if (classtypes[ii] == "Int") {

    } // end Int creation

    if (classtypes[ii] == "String") {

    } // end String creation
  }

  return;
}

void ROOTtoMEConverter::analyze(const edm::Event& iEvent, 
				const edm::EventSetup& iSetup)
{
  return;
}

